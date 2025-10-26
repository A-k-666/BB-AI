#!/usr/bin/env node

/**
 * BB-AI 2.0 - Intelligent n8n Workflow Automation Agent
 * Uses OpenAI GPT-4o for DOM reasoning, selector finding, and self-healing
 */

import { config } from 'dotenv';
import { chromium } from 'playwright';
import OpenAI from 'openai';
import { readFile, writeFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { N8nUIExpert } from './n8n_ui_expert.js';
import sharp from 'sharp';
import levenshtein from 'levenshtein';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables
config({ path: join(__dirname, '..', 'config', 'credentials.env') });

// ============================================================================
// 🛠️ HELPER FUNCTIONS - Image compression, retry logic, fuzzy matching
// ============================================================================

/**
 * Compress base64 PNG to JPEG to reduce OpenAI payload
 */
async function compressBase64Png(base64png, widthLimit = 1280, quality = 70) {
    try {
        const buffer = Buffer.from(base64png, 'base64');
        let img = sharp(buffer).rotate(); // fix orientation if needed
        const meta = await img.metadata();
        if (meta.width && meta.width > widthLimit) {
            img = img.resize({ width: widthLimit });
        }
        const compressedBuffer = await img.jpeg({ quality }).toBuffer();
        return compressedBuffer.toString('base64');
    } catch (error) {
        console.warn('⚠️ Image compression failed:', error.message);
        return base64png; // fallback to original
    }
}

/**
 * OpenAI call with retry/backoff for handling 500 errors
 */
async function openaiCallWithRetries(client, params, tries = 3, backoffMs = 1000) {
    for (let i = 0; i < tries; i++) {
        try {
            return await client.chat.completions.create(params);
        } catch (err) {
            const status = err?.status || err?.statusCode || (err?.response?.status);
            console.warn(`⚠️ OpenAI error ${status || err.message} (attempt ${i+1}/${tries})`);
            if (i === tries - 1) throw err;
            await new Promise(r => setTimeout(r, backoffMs * Math.pow(2, i)));
        }
    }
}

/**
 * Structured logging with levels
 */
function log(level, message, ...args) {
    const prefix = {
        info: "ℹ️ ",
        success: "✅ ",
        warn: "⚠️ ",
        error: "❌ ",
        ai: "🧠 ",
        debug: "🔍 "
    }[level] || "";
    console.log(`${prefix}${message}`, ...args);
}

class IntelligentN8nAgent {
    constructor() {
        this.browser = null;
        this.context = null;
        this.page = null;
        this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
        this.n8nExpert = new N8nUIExpert();  // 🧠 N8N UI Expert
        this.nodeIdMap = {};
        this.actionLog = [];
        this.debugMode = true;
        this.screenshotCounter = 0;
        
        // 🧠 Context Memory - stores recent decisions for learning
        this.contextMemory = [];
        this.maxContextMemory = 20; // Increased for better pattern recognition
        
        // ⚙️ AI Configuration
        this.aiModel = process.env.AI_MODEL || 'gpt-4o-mini'; // Use gpt-4o for better reasoning
        this.explainMode = process.env.AI_EXPLAIN_MODE === 'true';
        this.useN8nExpert = true;  // Enable n8n expert mode
        
        // Screenshot folder structure
        this.screenshotDirs = {
            login: '../output/screenshots/login',
            nodes: '../output/screenshots/nodes',
            connections: '../output/screenshots/connections',
            errors: '../output/screenshots/errors',
            final: '../output/screenshots/final'
        };
        
        // Ensure output directories exist
        this.ensureDirectories();
    }
    
    /**
     * 📁 Ensure all output directories exist
     */
    async ensureDirectories() {
        const { mkdir } = await import('fs/promises');
        for (const dir of Object.values(this.screenshotDirs)) {
            await mkdir(dir, { recursive: true }).catch(() => {});
        }
        await mkdir('./output', { recursive: true }).catch(() => {});
    }
    
    /**
     * 🧹 Clean old screenshots (except final folder)
     */
    async cleanOldScreenshots() {
        const { readdir, unlink } = await import('fs/promises');
        
        console.log('🧹 Cleaning old screenshots (keeping final folder)...');
        
        const foldersToClean = ['login', 'nodes', 'connections', 'errors'];
        let totalCleaned = 0;
        
        for (const folder of foldersToClean) {
            const dirPath = this.screenshotDirs[folder];
            try {
                const files = await readdir(dirPath);
                for (const file of files) {
                    if (file.endsWith('.png')) {
                        await unlink(`${dirPath}/${file}`);
                        totalCleaned++;
                    }
                }
            } catch (e) {
                // Directory doesn't exist or empty, skip
            }
        }
        
        if (totalCleaned > 0) {
            console.log(`   ✅ Cleaned ${totalCleaned} old screenshots`);
        }
        console.log(`   💾 Kept all screenshots in: ${this.screenshotDirs.final}\n`);
    }
    
    /**
     * 🧠 Add to context memory (short-term learning)
     */
    addToContextMemory(action, selector, reasoning, success) {
        this.contextMemory.push({
            action,
            selector,
            reasoning,
            success,
            timestamp: new Date().toISOString()
        });
        
        // Keep only last N memories
        if (this.contextMemory.length > this.maxContextMemory) {
            this.contextMemory.shift();
        }
        
        if (this.explainMode) {
            console.log(`\n💭 Learning: ${action} → ${selector} (${success ? '✅' : '❌'})`);
            console.log(`   Reason: ${reasoning}`);
        }
    }
    

    /**
     * 📸 Take organized screenshot
     */
    /**
     * 🧠 INTELLIGENT NODE SELECTION - Hybrid local + AI approach
     * After typing into search, use this to pick correct option from search dropdown
     */
    async selectFromSearchResults(page, desiredLabel, opts = {}) {
        const {
            candidateSelectors = [
                '[data-test-id="node-creator-node-item"]',
                '[data-test-id="node-creator-list"] [role="option"]',
                '[data-test-id="node-creator-list"] li',
                '.node-creator .list-item',
                '.node-creator .option, .listbox__option'
            ],
            timeout = 6000,
            openaiClient = this.openai,
            aiModel = this.aiModel
        } = opts;

        // Wait for debounce tolerance
        await page.waitForTimeout(300);

        // Find options by trying candidate selectors
        let options = [];
        for (const sel of candidateSelectors) {
            try {
                options = await page.$$eval(sel, els => els.map((e, i) => {
                    return {
                        idx: i,
                        text: (e.innerText || e.textContent || '').trim(),
                        html: e.outerHTML,
                    };
                }));
                if (options && options.length > 0) {
                    options._usedSelector = sel;
                    break;
                }
            } catch (e) {
                // ignore and try next selector
            }
        }

        if (!options || options.length === 0) {
            return { ok: false, reason: 'no_options_found' };
        }

        // Silent operation - only log if AI explain mode is on
        if (this.aiExplainMode) {
            log('debug', `Found ${options.length} options`);
        }

        // 🎯 LOCAL HEURISTICS: exact match -> substring -> fuzzy (levenshtein)
        let best = null;
        
        // 1) Exact match (case-insensitive)
        best = options.find(o => o.text.toLowerCase() === desiredLabel.toLowerCase());
        
        if (!best) {
            // 2) Substring match
            best = options.find(o => o.text.toLowerCase().includes(desiredLabel.toLowerCase()));
        }
        
        if (!best) {
            // 3) Fuzzy match by levenshtein distance
            let bestScore = Infinity;
            for (const o of options) {
                const score = levenshtein(o.text.toLowerCase(), desiredLabel.toLowerCase());
                if (score < bestScore) {
                    bestScore = score;
                    best = o;
                }
            }
            // Threshold to avoid wrong picks
            if (bestScore > Math.max(3, Math.floor(desiredLabel.length/2))) {
                best = null;
            }
        }

        // If local heuristics found a candidate, click it
        if (best) {
            if (this.aiExplainMode) {
                log('success', `Local match found: "${best.text}"`);
            }
            const usedSel = options._usedSelector;
            await page.waitForSelector(usedSel, { timeout: 2000 }).catch(()=>{});
            const elements = await page.$$(usedSel);
            if (elements && elements[best.idx]) {
                try {
                    await elements[best.idx].click({ timeout: 3000 });
                    return { ok: true, method: 'local', chosen: best };
                } catch (err) {
                    // Fallback to keyboard navigation
                    await page.keyboard.press('ArrowDown');
                    for (let i=0; i<best.idx; i++) { 
                        await page.keyboard.press('ArrowDown'); 
                    }
                    await page.keyboard.press('Enter');
                    return { ok: true, method: 'keyboard_fallback', chosen: best };
                }
            }
        }

        // 🧠 AI FALLBACK: If local heuristics failed and we have OpenAI
        if (openaiClient) {
            try {
                if (this.aiExplainMode) {
                    log('ai', 'Local heuristics failed, asking AI to choose...');
                }
                
                // Gather short list (cap to 10 options for cost)
                const small = options.slice(0, 10);
                const msg = [
                    { 
                        role: 'system', 
                        content: 'You are a helpful assistant that picks the best match from a list of UI options.' 
                    },
                    { 
                        role: 'user', 
                        content: `I want to add the node "${desiredLabel}". Here are visible options (index: text):\n${small.map((o,i)=> `${i}: ${o.text}`).join('\n')}\nReturn a JSON with {"index": <index>}. If none match, return {"index": -1}.` 
                    }
                ];

                const response = await openaiCallWithRetries(openaiClient, {
                    model: aiModel,
                    messages: msg,
                    temperature: 0
                }, 3);

                const content = response.choices?.[0]?.message?.content || '';
                const jsonMatch = content.match(/\{[\s\S]*\}/);
                if (jsonMatch) {
                    let parsed;
                    try { 
                        parsed = JSON.parse(jsonMatch[0]); 
                    } catch(e) { 
                        parsed = null; 
                    }
                    if (parsed && typeof parsed.index === 'number' && parsed.index >=0 && parsed.index < small.length) {
                        const idx = parsed.index;
                        const usedSel = options._usedSelector;
                        const elements = await page.$$(usedSel);
                        if (elements && elements[idx]) {
                            await elements[idx].click();
                            log('success', `AI selected: "${small[idx].text}"`);
                            return { ok: true, method: 'openai', chosen: small[idx] };
                        }
                    }
                }
            } catch (err) {
                log('warn', 'OpenAI selection failed:', err?.message || err);
            }
        }

        log('warn', 'No confident choice found');
        return { ok: false, reason: 'no_confident_choice', optionsCount: options.length };
    }

    async _askAIToSelectNode(screenshot, desiredPurpose, fallbackName) {
        /**
         * Ask GPT-4o Vision to intelligently select the right node from options
         */
        try {
            // Compress image first to avoid 500 errors
            const compressedScreenshot = await compressBase64Png(screenshot);
            
            const prompt = `You are looking at an n8n node selection dialog.

**Goal:** Find and select a node that serves this purpose: "${desiredPurpose || fallbackName}"

**Task:** 
1. Look at the screenshot
2. Identify ALL visible node options (buttons/items you can click)
3. Choose the BEST option that matches the purpose
4. Return JSON with your choice

**Example purposes and correct choices:**
- Purpose: "Slack Trigger" or "trigger for new Slack messages" → Choose: "On new message posted to channel" (NOT just "Slack")
- Purpose: "Webhook" → Choose: "Webhook" or "On webhook call"
- Purpose: "OpenAI Chat" → Choose: "Chat Message" or "OpenAI"

**Important:**
- Don't match exact text - understand the PURPOSE
- If you see "Slack", it might open subcategory - choose trigger option inside
- Triggers have lightning bolt icon ⚡
- Action nodes have different icons

Return JSON:
{
  "action": "click",
  "text": "exact text of option to click",
  "reasoning": "why this is the right choice"
}

If no good option visible:
{
  "action": "enter",
  "reasoning": "no visible options match"
}`;

            const response = await openaiCallWithRetries(this.openai, {
                model: this.aiModel,
                messages: [
                    {
                        role: 'user',
                        content: [
                            { type: 'text', text: prompt },
                            {
                                type: 'image_url',
                                image_url: { url: `data:image/jpeg;base64,${compressedScreenshot}` }
                            }
                        ]
                    }
                ],
                response_format: { type: 'json_object' },
                max_tokens: 500,
                temperature: 0.1
            }, 3);

            const result = JSON.parse(response.choices[0].message.content);
            
            if (this.aiExplainMode) {
                log('ai', `AI Reasoning: ${result.reasoning}`);
            }
            
            return result;
            
        } catch (error) {
            log('warn', `AI selection failed: ${error.message}`);
            return null;
        }
    }

    async _verifyNodeAdded(nodeType, expectedName) {
        /**
         * Verify node actually appeared on canvas - Fast DOM check + AI Vision fallback
         */
        try {
            await this.page.waitForTimeout(1500);
            
            // 🚀 FAST PATH: DOM-based verification first
            const canvasNodes = await this.page.$$eval(
                '[data-test-id="canvas-node"], .node-box, [class*="node"]',
                (els, expected) => {
                    return els.map(el => {
                        const text = (el.innerText || el.textContent || '').toLowerCase();
                        const dataName = (el.getAttribute('data-name') || '').toLowerCase();
                        return { text, dataName, match: text.includes(expected.toLowerCase()) || dataName.includes(expected.toLowerCase()) };
                    });
                },
                expectedName
            ).catch(() => []);
            
            const domMatch = canvasNodes.some(n => n.match);
            
            if (domMatch) {
                log('success', `Fast DOM verification: Node "${expectedName}" found on canvas`);
                return true;
            }
            
            log('debug', 'DOM check failed, trying AI Vision verification...');
            
            // 🧠 AI VISION FALLBACK: Only if DOM check fails
            const screenshot = await this.page.screenshot({ encoding: 'base64' });
            const compressedScreenshot = await compressBase64Png(screenshot);
            
            const prompt = `Look at this n8n workflow canvas.

**Question:** Is there a node visible with type/name similar to "${expectedName}" or "${nodeType}"?

Look for:
- Node boxes on the canvas (NOT in sidebar/modal)
- Text labels on nodes
- Icons representing node types
- Recent additions (might be highlighted or selected)

Return JSON:
{
  "node_present": true/false,
  "node_name": "visible name if found",
  "confidence": 0-100,
  "reasoning": "what you see"
}`;

            const response = await openaiCallWithRetries(this.openai, {
                model: this.aiModel,
                messages: [
                    {
                        role: 'user',
                        content: [
                            { type: 'text', text: prompt },
                            {
                                type: 'image_url',
                                image_url: { url: `data:image/jpeg;base64,${compressedScreenshot}` }
                            }
                        ]
                    }
                ],
                response_format: { type: 'json_object' },
                max_tokens: 300,
                temperature: 0.1
            }, 3);

            const result = JSON.parse(response.choices[0].message.content);
            
            log('debug', `AI Vision: ${result.node_present ? 'Node found' : 'Node NOT found'} (${result.confidence}% confident)`);
            
            if (this.aiExplainMode && result.reasoning) {
                log('ai', result.reasoning);
            }
            
            return result.node_present && result.confidence > 60;
            
        } catch (error) {
            log('warn', `Verification failed: ${error.message}`);
            return false;
        }
    }

    _extractBaseServiceName(nodeName) {
        /**
         * Extract base service name from full node name
         * Examples:
         *   "Slack Trigger" → "Slack"
         *   "OpenAI Chat Model" → "OpenAI"
         *   "Google Sheets" → "Google Sheets"
         *   "HTTP Request" → "HTTP"
         */
        
        // Common patterns to extract base name
        const patterns = [
            /^(Slack|Telegram|Discord|OpenAI|Google|Microsoft|AWS|Azure)\s/i,
            /^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s(?:Trigger|Agent|Model|Tool|Action)/i,
            /^(HTTP|SMTP|FTP|SSH|API)\s/i
        ];
        
        for (const pattern of patterns) {
            const match = nodeName.match(pattern);
            if (match) {
                return match[1];
            }
        }
        
        // Fallback: return first word
        return nodeName.split(' ')[0];
    }

    async takeScreenshot(category, description) {
        this.screenshotCounter++;
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const filename = `${String(this.screenshotCounter).padStart(3, '0')}_${description.replace(/\s+/g, '_')}.png`;
        const path = `${this.screenshotDirs[category]}/${filename}`;
        
        await this.page.screenshot({ path, fullPage: true });
        console.log(`   📸 Screenshot: ${category}/${filename}`);
        return path;
    }

    /**
     * 🧠 Core AI Reasoning: Smart Selector Finder
     */
    async smartFind(intentDescription, domContext = null, screenshot = null) {
        console.log(`🧠 AI Reasoning: "${intentDescription}"`);

        try {
            // Get DOM context if not provided
            if (!domContext) {
                domContext = await this.page.evaluate(() => {
                    return {
                        bodyHTML: document.body.innerHTML.slice(0, 8000),
                        visibleButtons: Array.from(document.querySelectorAll('button')).slice(0, 20).map(b => ({
                            text: b.textContent?.trim(),
                            dataTestId: b.getAttribute('data-test-id'),
                            className: b.className,
                            visible: b.offsetParent !== null
                        })),
                        visibleInputs: Array.from(document.querySelectorAll('input')).slice(0, 10).map(i => ({
                            placeholder: i.placeholder,
                            dataTestId: i.getAttribute('data-test-id'),
                            type: i.type,
                            visible: i.offsetParent !== null
                        })),
                        url: window.location.href
                    };
                });
            }

            // 🧠 Build context from recent memories
            let memoryContext = '';
            if (this.contextMemory.length > 0) {
                memoryContext = '\n\nRecent successful patterns:\n';
                this.contextMemory.filter(m => m.success).slice(-5).forEach(m => {
                    memoryContext += `- For "${m.action}": used ${m.selector} (${m.reasoning})\n`;
                });
            }

            const messages = [
                {
                    role: "system",
                    content: `You are an expert web automation assistant specializing in n8n Cloud UI.
Your job: Given a user intent and DOM snapshot, find the BEST Playwright selector.

Rules:
1. Prefer data-test-id attributes (e.g., [data-test-id="canvas-plus-button"])
2. Use text selectors for buttons with unique text (e.g., text="Add first step")
3. Use CSS selectors for inputs (e.g., input[placeholder*="Search"])
4. Return ONLY valid JSON with: { "selector": "...", "reasoning": "...", "alternatives": [...] }
5. Selector must work with Playwright's page.locator()
6. If multiple options, put best first in alternatives array
7. Learn from past successful patterns when available${memoryContext}`
                },
                {
                    role: "user",
                    content: `Task: ${intentDescription}

DOM Context:
URL: ${domContext.url}

Visible Buttons (first 20):
${JSON.stringify(domContext.visibleButtons, null, 2)}

Visible Inputs (first 10):
${JSON.stringify(domContext.visibleInputs, null, 2)}

Body HTML (first 8000 chars):
${domContext.bodyHTML}

Respond with JSON: { "selector": "best selector", "reasoning": "why", "alternatives": ["backup1", "backup2"] }`
                }
            ];

            // Add screenshot for multimodal reasoning if provided
            if (screenshot) {
                messages[1].content = [
                    { type: "text", text: messages[1].content },
                    { 
                        type: "image_url", 
                        image_url: { url: `data:image/png;base64,${screenshot}` }
                    }
                ];
            }

            const response = await this.openai.chat.completions.create({
                model: this.aiModel,
                messages,
                response_format: { type: "json_object" },
                temperature: 0.2,
                max_tokens: 500
            });

            const result = JSON.parse(response.choices[0].message.content);
            
            console.log(`   ✅ AI Selector: ${result.selector}`);
            console.log(`   💡 Reasoning: ${result.reasoning}`);
            
            this.actionLog.push({
                intent: intentDescription,
                selector: result.selector,
                reasoning: result.reasoning,
                timestamp: new Date().toISOString()
            });

            return result;

        } catch (error) {
            console.log(`   ⚠️  AI reasoning failed: ${error.message}`);
            return { selector: null, reasoning: "AI call failed", alternatives: [] };
        }
    }

    /**
     * 🔍 Vision-Enhanced Selector Finding
     */
    async smartFindWithVision(intentDescription) {
        console.log(`👁️  Using vision-enhanced AI reasoning...`);
        
        const screenshot = await this.page.screenshot({ encoding: 'base64', fullPage: false });
        const domContext = await this.page.evaluate(() => ({
            bodyHTML: document.body.innerHTML.slice(0, 5000),
            url: window.location.href
        }));

        return await this.smartFind(intentDescription, domContext, screenshot);
    }

    /**
     * 🔍 Self-Evaluation: Check if action succeeded
     */
    async evaluateActionSuccess(beforeScreenshot, afterScreenshot, expectedChange) {
        if (!this.explainMode) return true; // Skip if not in explain mode
        
        try {
            const response = await this.openai.chat.completions.create({
                model: this.aiModel,
                messages: [
                    {
                        role: "system",
                        content: "You are an expert at evaluating UI changes. Compare before/after screenshots and determine if expected change occurred."
                    },
                    {
                        role: "user",
                        content: [
                            { type: "text", text: `Expected change: ${expectedChange}\n\nDid this change occur? Respond with JSON: { "success": true/false, "observation": "what changed" }` },
                            { type: "image_url", image_url: { url: `data:image/png;base64,${beforeScreenshot}` } },
                            { type: "image_url", image_url: { url: `data:image/png;base64,${afterScreenshot}` } }
                        ]
                    }
                ],
                response_format: { type: "json_object" },
                max_tokens: 200
            });
            
            const evaluation = JSON.parse(response.choices[0].message.content);
            
            if (this.explainMode) {
                console.log(`\n🔍 Self-Evaluation: ${evaluation.success ? '✅' : '❌'}`);
                console.log(`   Observation: ${evaluation.observation}`);
            }
            
            return evaluation.success;
        } catch (error) {
            console.log(`   ⚠️  Self-evaluation failed: ${error.message}`);
            return true; // Assume success if evaluation fails
        }
    }

    /**
     * 🛡️ Self-Healing Action: Try selector with AI retry
     */
    async smartClick(intentDescription, maxRetries = 3, expectedChange = null) {
        console.log(`🎯 Smart Click: "${intentDescription}"`);
        
        // Take before screenshot for self-evaluation
        const beforeScreenshot = expectedChange ? await this.page.screenshot({ encoding: 'base64' }) : null;

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                const useVision = attempt > 1; // Use vision on retries
                const result = useVision 
                    ? await this.smartFindWithVision(intentDescription)
                    : await this.smartFind(intentDescription);

                if (!result.selector) {
                    throw new Error("No selector found by AI");
                }

                // Try main selector with dynamic timeout
                const isConnectionAction = intentDescription.includes('connection') || intentDescription.includes('circle');
                const waitTimeout = isConnectionAction ? 20000 : 7000; // 20s for connections, 7s for buttons
                
                const locator = this.page.locator(result.selector).first();
                await locator.waitFor({ state: 'visible', timeout: waitTimeout });
                await locator.click({ timeout: 5000 });
                
                console.log(`   ✅ Clicked successfully (attempt ${attempt})`);
                
                // 🧠 Self-evaluation: Did expected change occur?
                if (expectedChange) {
                    await this.page.waitForTimeout(1500);
                    const afterScreenshot = await this.page.screenshot({ encoding: 'base64' });
                    const success = await this.evaluateActionSuccess(beforeScreenshot, afterScreenshot, expectedChange);
                    
                    // Add to context memory
                    this.addToContextMemory(intentDescription, result.selector, result.reasoning, success);
                    
                    if (!success && attempt < maxRetries) {
                        console.log(`   ⚠️  Expected change did not occur, retrying...`);
                        continue;
                    }
                } else {
                    // No evaluation, assume success
                    this.addToContextMemory(intentDescription, result.selector, result.reasoning, true);
                }
                
                return true;

            } catch (error) {
                console.log(`   ⚠️  Attempt ${attempt} failed: ${error.message}`);
                
                // Auto-enable explain mode on first failure for debugging
                if (!this.explainMode && attempt === 1) {
                    console.log(`   💭 Auto-enabling Explain Mode for debugging...`);
                    this.explainMode = true;
                }
                
                // Add failed attempt to memory
                this.addToContextMemory(intentDescription, 'unknown', error.message, false);
                
                if (attempt < maxRetries) {
                    console.log(`   🔄 Retrying with enhanced vision...`);
                    await this.takeScreenshot('errors', `retry_attempt_${attempt}_${intentDescription.slice(0, 30)}`);
                    await this.page.waitForTimeout(1000);
                } else {
                    console.log(`   ❌ All attempts failed`);
                    await this.takeScreenshot('errors', `failed_${intentDescription.slice(0, 50)}`);
                    return false;
                }
            }
        }

        return false;
    }

    /**
     * 🔍 Get n8n Internal State (console inspection)
     */
    async getN8nState() {
        try {
            const state = await this.page.evaluate(() => {
                // Try to access n8n's internal Vue store
                if (window.__VUE_DEVTOOLS_GLOBAL_HOOK__?.apps?.length > 0) {
                    const app = window.__VUE_DEVTOOLS_GLOBAL_HOOK__.apps[0];
                    return {
                        nodes: app._instance?.proxy?.$store?.state?.ndv?.activeNode,
                        workflow: app._instance?.proxy?.$store?.state?.workflows?.workflow
                    };
                }
                
                // Fallback: try reading from DOM
                const canvasNodes = Array.from(document.querySelectorAll('[data-test-id^="canvas"]')).map(el => ({
                    id: el.getAttribute('data-id'),
                    name: el.getAttribute('data-name'),
                    type: el.getAttribute('data-node-type')
                }));

                return { canvasNodes };
            });

            console.log(`🔍 n8n State:`, JSON.stringify(state, null, 2));
            return state;

        } catch (error) {
            console.log(`⚠️  Could not read n8n state: ${error.message}`);
            return null;
        }
    }

    /**
     * 🌐 Connect to Browserbase
     */
    async connect() {
        console.log('\n🚀 Starting AI-Powered n8n Agent\n');
        
        // Load n8n expert knowledge
        if (this.useN8nExpert) {
            console.log('📚 Loading n8n UI Expert knowledge...');
            await this.n8nExpert.loadDocsContext();
            console.log();
        }
        
        // Auto-clean old screenshots before starting
        await this.cleanOldScreenshots();
        
        if (this.explainMode) {
            console.log('💭 Explain Mode: ON - Detailed reasoning will be shown\n');
        }
        
        const sessionUrl = `wss://connect.browserbase.com?apiKey=${process.env.API_KEY}&projectId=${process.env.PROJECT_ID}`;
        
        this.browser = await chromium.connectOverCDP(sessionUrl);
        this.context = this.browser.contexts()[0];
        this.page = this.context.pages()[0];
        
        // Set default timeout to 14 seconds for fast testing
        this.page.setDefaultTimeout(14000);
        
        console.log('✅ Connected to Browserbase');
        console.log('📄 Page ready\n');
    }

    /**
     * 🔐 Login to n8n Cloud
     */
    async login() {
        console.log('🔐 Logging into n8n...');
        
        await this.page.goto('https://pika2223.app.n8n.cloud/workflow/new', { 
            waitUntil: 'domcontentloaded',
            timeout: 60000 
        });

        await this.page.waitForTimeout(2000);
        await this.takeScreenshot('login', '01_initial_page');

        // Check if already logged in
        const isLoggedIn = await this.page.locator('.workflow-canvas, svg').isVisible().catch(() => false);
        
        if (!isLoggedIn) {
            console.log('🔑 Login required, using FAST direct selectors...');
            
            try {
                // ⚡ FAST MODE - Direct selectors from console inspect
                const emailInput = this.page.locator('input[type="email"]').first();
                await emailInput.waitFor({ timeout: 5000 });
                await emailInput.fill(process.env.N8N_EMAIL);
                await emailInput.press('Enter');
                console.log('   ✅ Email submitted (fast mode)');
                
                await this.page.waitForTimeout(1500);
                
                const passwordInput = this.page.locator('input[type="password"]').first();
                await passwordInput.waitFor({ timeout: 5000 });
                await passwordInput.fill(process.env.N8N_PASSWORD);
                await passwordInput.press('Enter');
                console.log('   ✅ Password submitted (fast mode)');
                console.log('✅ Credentials submitted');
                
            } catch (error) {
                console.log('⚠️  Fast login failed, using AI fallback...');
                
                // AI fallback
            await this.takeScreenshot('login', '02_before_email');
            await this.smartClick("Find and click the email input field", 3, "Email input field should be focused");
            await this.page.keyboard.type(process.env.N8N_EMAIL);
            await this.page.keyboard.press('Enter');
            
            await this.page.waitForTimeout(1500);
            
            await this.takeScreenshot('login', '04_before_password');
            await this.smartClick("Find and click the password input field", 3, "Password input field should be focused");
            await this.page.keyboard.type(process.env.N8N_PASSWORD);
            await this.page.keyboard.press('Enter');
            
                console.log('✅ Credentials submitted (AI mode)');
            }
        } else {
            console.log('✅ Already logged in');
        }

        // Wait for canvas
        await this.page.waitForSelector('.workflow-canvas, svg', { timeout: 15000 });
        await this.page.waitForTimeout(3000);
        
        console.log('✅ Logged in, canvas ready\n');
        
        await this.takeScreenshot('login', '06_canvas_ready');
    }

    /**
     * ➕ AI-Powered Node Creation
     */
    async createNodeIntelligent(nodeName, nodeType, jsonNodeId) {
        console.log(`\n➕ Creating node: ${nodeName} (${nodeType}, ID: ${jsonNodeId})`);

        // Step 1: Before clicking + button
        await this.takeScreenshot('nodes', `before_${nodeType}_canvas`);
        
        // Try multiple + button selectors
        let plusClicked = false;
        
        // First try node-creator-plus-button (if already in creator view)
        try {
            const creatorBtn = this.page.locator('[data-test-id="node-creator-plus-button"]');
            if (await creatorBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
                await creatorBtn.click();
                console.log('   ✅ Clicked node-creator + button');
                plusClicked = true;
            }
        } catch {}
        
        // Fallback to canvas + button
        if (!plusClicked) {
            plusClicked = await this.smartClick("Click the + button to add a new node to the workflow canvas");
        }
        
        if (!plusClicked) {
            console.log('⚠️  Could not click + button');
            return null;
        }

        await this.page.waitForTimeout(2000);
        await this.takeScreenshot('nodes', `${nodeType}_modal_opened`);

        // Step 2: Intelligent 2-step node selection
        const searchResult = await this.smartFind("Find the node search input box in the node creator dialog");
        
        if (searchResult.selector) {
            const searchBox = this.page.locator(searchResult.selector).first();
            
            // 🧠 INTELLIGENT SEARCH - Extract base service name
            // e.g., "Slack Trigger" → search "Slack", then select "Slack Trigger"
            const baseSearch = this._extractBaseServiceName(nodeName);
            const exactOption = this.currentNodeHints?.exact_option || nodeName;
            
            log('ai', `Intelligent search: "${baseSearch}" → then select "${exactOption}"`);
            
            // 🚀 HUMAN-LIKE FLOW: Search → Enter → Verify → Fallback if needed
            await searchBox.fill(baseSearch);
            log('info', `Searching: "${baseSearch}"`);
            await this.page.waitForTimeout(800);
            await this.takeScreenshot('nodes', `${nodeType}_search_filled`);
            
            // 1️⃣ FAST PATH: Press Enter immediately (like a human)
            try {
                await searchBox.press('Enter');
                log('debug', 'Pressed Enter (fast path)');
            } catch (e) {
                log('warn', `Enter press failed: ${e.message}`);
            }
            
            await this.page.waitForTimeout(3000);
            await this.takeScreenshot('nodes', `${nodeType}_node_added`);
            
            // 2️⃣ VERIFY: Check if node was added
            let nodeAdded = await this._verifyNodeAdded(nodeType, nodeName);
            
            if (nodeAdded) {
                log('success', `✅ Node added via Enter: "${nodeName}"`);
            } else {
                // 3️⃣ SMART FALLBACK: Enter didn't work, use intelligent selection
                log('warn', `⚠️ Enter didn't work, using AI fallback for "${nodeName}"...`);
                
                const modalStillOpen = await this.page.locator('[data-test-id="node-creator-search-bar"]').isVisible().catch(() => false);
                
                if (modalStillOpen) {
                    // Get available options without logging spam
                    const options = await this.page.$$eval(
                        '[data-test-id="node-creator-node-item"]',
                        els => els.map(el => el.textContent.trim())
                    ).catch(() => []);
                    
                    if (options.length > 0) {
                        // Try fuzzy matching first
                        let match = options.find(opt => 
                            opt.toLowerCase().includes(exactOption.toLowerCase()) ||
                            opt.toLowerCase().includes(nodeName.toLowerCase()) ||
                            opt.toLowerCase().includes(baseSearch.toLowerCase())
                        );
                        
                        if (match) {
                            log('success', `🧠 Found match: "${match}"`);
                            await this.page.locator(`text=${match}`).first().click();
                            await this.page.waitForTimeout(1500);
                            nodeAdded = await this._verifyNodeAdded(nodeType, nodeName);
                        } else {
                            // AI Vision as last resort
                            log('ai', 'Using AI Vision to identify correct option...');
                            const retryScreenshot = await this.page.screenshot({ encoding: 'base64' });
                            const aiChoice = await this._askAIToSelectNode(retryScreenshot, exactOption, nodeName);
                            
                            if (aiChoice && aiChoice.action === 'click') {
                                log('success', `AI selected: "${aiChoice.text}"`);
                                const optionElement = this.page.getByText(aiChoice.text, { exact: false }).first();
                                if (await optionElement.isVisible({ timeout: 2000 }).catch(() => false)) {
                                    await optionElement.click();
                                    await this.page.waitForTimeout(1500);
                                    nodeAdded = await this._verifyNodeAdded(nodeType, nodeName);
                                }
                            }
                        }
                    } else {
                        log('warn', 'No options found, trying Enter again');
                        await searchBox.press('Enter');
                        await this.page.waitForTimeout(1500);
                        nodeAdded = await this._verifyNodeAdded(nodeType, nodeName);
                    }
                } else {
                    log('info', 'Modal closed, waiting for node to render...');
                    await this.page.waitForTimeout(2000);
                    nodeAdded = await this._verifyNodeAdded(nodeType, nodeName);
                }
            }
            
            // 4️⃣ FINAL RESULT
            if (nodeAdded) {
                log('success', `✅ Node successfully added: "${nodeName}"`);
            } else {
                log('error', `❌ Failed to add: "${nodeName}"`);
            }
            
            // Close settings panel - AGGRESSIVE approach
            // Strategy 1: Click "Back to canvas" button
            for (let i = 0; i < 3; i++) {
                const backBtn = this.page.getByText('Back to canvas');
                if (await backBtn.isVisible().catch(() => false)) {
                    await backBtn.click();
                    console.log(`   ✅ Closed settings panel via button`);
                    await this.page.waitForTimeout(1000);
                    break;
                }
                await this.page.waitForTimeout(500);
            }
            
            // Strategy 2: ESC key (multiple times)
            await this.page.keyboard.press('Escape');
            await this.page.waitForTimeout(500);
            await this.page.keyboard.press('Escape');
            await this.page.waitForTimeout(500);
            
            // Strategy 3: Click canvas directly to ensure focus returns
            try {
                await this.page.locator('.vue-flow__pane, .workflow-canvas').click({ timeout: 2000 });
                console.log(`   ✅ Clicked canvas to reset focus`);
            } catch {}
            
            await this.page.waitForTimeout(1500);
            await this.takeScreenshot('nodes', `${nodeType}_settings_closed`);
        }

        // Step 3: Capture node ID - try multiple strategies
        let nodeId = null;
        
        // Strategy 1: Try data-id attribute directly
        try {
            const allNodes = await this.page.locator('[data-test-id^="canvas"][data-id]').all();
            if (allNodes.length > 0) {
                const lastNode = allNodes[allNodes.length - 1];
                nodeId = await lastNode.getAttribute('data-id');
                console.log(`   🆔 Strategy 1 (data-id): ${nodeId}`);
            }
        } catch {}
        
        // Strategy 2: Try data-name attribute
        if (!nodeId) {
            try {
                const allNodes = await this.page.locator('[data-test-id^="canvas"][data-name]').all();
                if (allNodes.length > 0) {
                    const lastNode = allNodes[allNodes.length - 1];
                    nodeId = await lastNode.getAttribute('data-name');
                    console.log(`   🆔 Strategy 2 (data-name): ${nodeId}`);
                }
            } catch {}
        }
        
        // Strategy 3: Vue state inspection
        if (!nodeId) {
            const state = await this.getN8nState();
            const validNodes = state?.canvasNodes?.filter(n => n.id && n.id !== 'null' && !n.id.includes('__EMPTY__'));
            if (validNodes?.length > 0) {
                const lastNode = validNodes[validNodes.length - 1];
                nodeId = lastNode.id || lastNode.name;
                console.log(`   🆔 Strategy 3 (Vue state): ${nodeId}`);
            }
        }
        
        // Strategy 4: Count nodes as fallback
        if (!nodeId) {
            const nodeCount = await this.page.locator('[data-test-id^="canvas-"]').count();
            nodeId = `node_${nodeCount}`;
            console.log(`   🆔 Strategy 4 (fallback): ${nodeId}`);
        }

        console.log(`   ✅ Final Node ID: ${nodeId}\n`);
        
        // Map both JSON ID and node type for connections
        this.nodeIdMap[jsonNodeId] = nodeId;
        this.nodeIdMap[nodeType] = nodeId;
        
        return nodeId;
    }

    /**
     * 🔗 AI-Powered Node Connection (Visual Strategy)
     */
    async connectNodesIntelligent(fromType, toType) {
        console.log(`\n🔗 Connecting: ${fromType} → ${toType}`);

        try {
            await this.takeScreenshot('connections', `before_${fromType}_to_${toType}`);
            
            // Wait for nodes to fully render
            await this.page.waitForTimeout(2000);
            
            // Strategy 1: Try dragTo (Playwright built-in)
            try {
                console.log(`   🎯 Strategy 1: Direct drag connection`);
                
                // Find nodes by visible elements (not IDs)
                const allNodes = await this.page.locator('[data-test-id^="canvas"]').all();
                console.log(`   📊 Found ${allNodes.length} canvas elements`);
                
                if (allNodes.length >= 2) {
                    // Assume first node = source, last node = target
                    const sourceNode = allNodes[0];
                    const targetNode = allNodes[allNodes.length - 1];
                    
                    // Find output/input circles
                    const outputCircle = sourceNode.locator('circle').last();
                    const inputCircle = targetNode.locator('circle').first();
                    
                    const sourceBox = await outputCircle.boundingBox();
                    const targetBox = await inputCircle.boundingBox();
                    
                    if (sourceBox && targetBox) {
                        console.log(`   🎯 Found connection points`);
                        
                        // Manual drag with mouse
                        await this.page.mouse.move(sourceBox.x + sourceBox.width/2, sourceBox.y + sourceBox.height/2);
                        await this.page.mouse.down();
                        await this.page.waitForTimeout(500);
                        await this.page.mouse.move(targetBox.x + targetBox.width/2, targetBox.y + targetBox.height/2, { steps: 20 });
                        await this.page.waitForTimeout(500);
                        await this.page.mouse.up();
                        
                        console.log(`✅ Connected ${fromType} → ${toType}\n`);
                        await this.page.waitForTimeout(1000);
                        await this.takeScreenshot('connections', `success_${fromType}_to_${toType}`);
                        return true;
                    }
                }
            } catch (e) {
                console.log(`   ⚠️  Direct drag failed: ${e.message}`);
            }
            
            // Strategy 2: AI Vision-based connection
            console.log(`   🎯 Strategy 2: AI Vision-based connection`);
            const screenshot = await this.page.screenshot({ encoding: 'base64' });
            
            const response = await this.openai.chat.completions.create({
                model: this.aiModel,
                messages: [
                    {
                        role: "system",
                        content: "You are an expert at analyzing n8n workflow canvases. Find connection points between nodes."
                    },
                    {
                        role: "user",
                        content: [
                            { 
                                type: "text", 
                                text: `Find output circle of "${fromType}" node and input circle of "${toType}" node. Return JSON: { "sourceX": x, "sourceY": y, "targetX": x, "targetY": y, "possible": true/false }` 
                            },
                            { type: "image_url", image_url: { url: `data:image/png;base64,${screenshot}` } }
                        ]
                    }
                ],
                response_format: { type: "json_object" },
                max_tokens: 200
            });
            
            const coords = JSON.parse(response.choices[0].message.content);
            
            if (coords.possible) {
                console.log(`   🎯 AI found coordinates: (${coords.sourceX}, ${coords.sourceY}) → (${coords.targetX}, ${coords.targetY})`);
                
                await this.page.mouse.move(coords.sourceX, coords.sourceY);
                await this.page.mouse.down();
                await this.page.waitForTimeout(500);
                await this.page.mouse.move(coords.targetX, coords.targetY, { steps: 20 });
                await this.page.waitForTimeout(500);
                await this.page.mouse.up();
                
                console.log(`✅ Connected via AI vision ${fromType} → ${toType}\n`);
                await this.takeScreenshot('connections', `success_vision_${fromType}_to_${toType}`);
                return true;
            }

        } catch (error) {
            console.log(`⚠️  All connection strategies failed: ${error.message}\n`);
            await this.takeScreenshot('connections', `failed_all_${fromType}_to_${toType}`);
        }

        return false;
    }

    /**
     * 📊 Run Full Workflow Automation
     */
    async runAutomation() {
        try {
            // Try AI-enhanced workflow first (91% understanding)
            let actionPath = join(__dirname, '..', 'output', 'ai_workflow_complete.json');
            let workflow;
            
            try {
                workflow = JSON.parse(await readFile(actionPath, 'utf-8'));
                console.log('\n🧠 Using AI-Enhanced Workflow (91% understanding)');
            } catch {
                // Fallback to standard action_sequence.json
                actionPath = join(__dirname, '..', 'output', 'action_sequence.json');
                workflow = JSON.parse(await readFile(actionPath, 'utf-8'));
                console.log('\n📋 Using Standard Workflow');
            }

            // Deduplicate nodes (remove duplicates by name)
            const uniqueNodes = [];
            const seenNames = new Set();
            for (const node of workflow.nodes) {
                const key = `${node.name}_${node.type}`;
                if (!seenNames.has(key)) {
                    seenNames.add(key);
                    uniqueNodes.push(node);
                } else {
                    console.log(`   ⚠️  Skipping duplicate node: ${node.name}`);
                }
            }
            workflow.nodes = uniqueNodes;
            
            console.log(`📋 Workflow: ${workflow.workflow_name}`);
            console.log(`📝 Description: ${workflow.description}`);
            console.log(`📊 Nodes: ${workflow.nodes.length} (deduplicated), Connections: ${workflow.connections.length}`);
            
            // Show AI understanding confidence if available
            const confidence = workflow.metadata?.understanding_confidence;
            if (confidence) {
                const level = workflow.metadata?.understanding_level || 'UNKNOWN';
                console.log(`🎯 AI Understanding: ${confidence}% (${level})`);
            }
            console.log();

            // Track success/failure for each step
            const executionLog = {
                workflow_name: workflow.workflow_name,
                started_at: new Date().toISOString(),
                steps: [],
                final_status: 'in_progress'
            };

            // Process nodes (use AI-enhanced search keywords)
            for (const node of workflow.nodes) {
                console.log(`\n[${node.id}] Creating: ${node.name} (${node.type})`);
                
                const stepLog = {
                    step: node.id,
                    action: 'create_node',
                    target: node.name,
                    timestamp: new Date().toISOString(),
                    status: 'pending',
                    ai_generated: node.ai_generated || false
                };
                
                try {
                    // Use search_keyword from ui_hints if available
                    const searchKeyword = node.ui_hints?.search_keyword || node.name;
                    
                    // 🎯 Store ui_hints for exact option selection
                    this.currentNodeHints = node.ui_hints || {};
                    
                    const nodeId = await this.createNodeIntelligent(
                        searchKeyword,
                        node.type,
                        node.id
                    );
                    
                    stepLog.status = nodeId ? 'success' : 'failed';
                    stepLog.node_id = nodeId;
                    stepLog.search_keyword = searchKeyword;
                    
                    // Store node mapping for connections
                    this.nodeIdMap[node.id] = nodeId;
                    
                } catch (error) {
                    stepLog.status = 'failed';
                    stepLog.error = error.message;
                    console.log(`   ❌ Failed: ${error.message}`);
                }
                
                executionLog.steps.push(stepLog);
            }

            // Process connections
            for (const conn of workflow.connections) {
                console.log(`\n[Connection] ${conn.from} → ${conn.to}`);
                
                const stepLog = {
                    step: `${conn.from}_to_${conn.to}`,
                    action: 'connect_nodes',
                    from: conn.from,
                    to: conn.to,
                    timestamp: new Date().toISOString(),
                    status: 'pending'
                };
                
                try {
                    const success = await this.connectNodesIntelligent(conn.from, conn.to);
                    stepLog.status = success ? 'success' : 'failed';
                } catch (error) {
                    stepLog.status = 'failed';
                    stepLog.error = error.message;
                }
                
                executionLog.steps.push(stepLog);
            }

            // Wait for all nodes/connections to settle
            await this.page.waitForTimeout(3000);
            
            // Take final screenshot
            await this.takeScreenshot('final', 'complete_workflow');
            console.log('📸 Final screenshot saved\n');
            
            // 💾 SAVE WORKFLOW IN N8N
            console.log('💾 Saving workflow in n8n...');
            try {
                // Try to find and click Save button
                const saveBtn = this.page.locator('button:has-text("Save"), button[data-test-id="workflow-save-button"]').first();
                if (await saveBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
                    await saveBtn.click();
                    await this.page.waitForTimeout(2000);
                    console.log('   ✅ Workflow saved in n8n');
                    await this.takeScreenshot('final', 'saved_workflow');
                } else {
                    console.log('   ⚠️  Save button not found - workflow may auto-save');
                }
            } catch (error) {
                console.log(`   ⚠️  Could not save: ${error.message}`);
            }
            console.log();

            // Calculate success metrics
            const totalSteps = executionLog.steps.length;
            const successfulSteps = executionLog.steps.filter(s => s.status === 'success').length;
            const successRate = ((successfulSteps / totalSteps) * 100).toFixed(1);
            
            executionLog.completed_at = new Date().toISOString();
            executionLog.final_status = successRate === '100.0' ? 'success' : 'partial_success';
            executionLog.metrics = {
                total_steps: totalSteps,
                successful: successfulSteps,
                failed: totalSteps - successfulSteps,
                success_rate: `${successRate}%`,
                video_understanding: workflow.metadata?.understanding_confidence || 'N/A',
                ai_model_used: workflow.metadata?.ai_model || 'N/A'
            };

            // Save comprehensive execution log
            await writeFile(
                '../output/execution_log.json',
                JSON.stringify(executionLog, null, 2)
            );
            
            // Save AI reasoning log
            await writeFile(
                '../output/ai_action_log.json',
                JSON.stringify(this.actionLog, null, 2)
            );
            
            console.log('📊 Execution Summary:');
            console.log(`   Total Steps: ${totalSteps}`);
            console.log(`   Successful: ${successfulSteps}`);
            console.log(`   Failed: ${totalSteps - successfulSteps}`);
            console.log(`   Success Rate: ${successRate}%\n`);
            
            console.log('📝 Logs saved:');
            console.log(`   - execution_log.json (step-by-step results)`);
            console.log(`   - ai_action_log.json (AI reasoning)\n`);

        } catch (error) {
            console.log(`❌ Automation error: ${error.message}`);
            await this.takeScreenshot('errors', 'fatal_error');
        }
    }

    /**
     * 🧹 Cleanup
     */
    async close() {
        const keepOpen = parseInt(process.env.BROWSER_KEEP_OPEN_SECONDS || '10');
        console.log(`💡 Browser remains open for ${keepOpen}s for inspection...`);
        await this.page.waitForTimeout(keepOpen * 1000);
        
        if (this.context) await this.context.close();
        if (this.browser) await this.browser.close();
    }
}

// Main execution
(async () => {
    const agent = new IntelligentN8nAgent();
    
    try {
        await agent.connect();
        await agent.login();
        await agent.runAutomation();
        console.log('✅ AI-powered automation completed!\n');
    } catch (error) {
        console.error('❌ Fatal error:', error);
    } finally {
        await agent.close();
    }
})();

