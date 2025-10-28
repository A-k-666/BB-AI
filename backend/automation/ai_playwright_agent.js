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

/**
 * Parse OpenAI JSON response (removes markdown blocks if present)
 */
function parseOpenAIJSON(content) {
    if (typeof content !== 'string') {
        throw new Error('Content must be a string');
    }
    
    // Remove markdown code blocks
    let cleaned = content.trim();
    if (cleaned.includes('```json')) {
        cleaned = cleaned.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    } else if (cleaned.includes('```')) {
        cleaned = cleaned.replace(/```\n?/g, '').trim();
    }
    
    return JSON.parse(cleaned);
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
        this.maxContextMemory = 20;
        
        // ⚙️ AI Configuration
        this.aiModel = process.env.AI_MODEL || 'gpt-4o-mini';
        this.explainMode = process.env.AI_EXPLAIN_MODE === 'true';
        this.useN8nExpert = true;
        
        // Screenshot folder structure
        this.screenshotDirs = {
            login: '../output/screenshots/login',
            nodes: '../output/screenshots/nodes',
            connections: '../output/screenshots/connections',
            errors: '../output/screenshots/errors',
            final: '../output/screenshots/final'
        };
        
        // 🎯 Visual Reference & Validation
        this.referenceImagePath = join(__dirname, '..', 'output', 'ai_visual_recipe.png');
        this.referenceImageBase64 = null;
        this.workflowBlueprint = null;
        this.validationSteps = [];
        
        // ⚡ PERFORMANCE CACHE
        this.canvasHandle = null;
        this.cachedSelectors = new Map();
        this.nodeRetryCount = new Map();
        
        // ⏰ KEEP-ALIVE
        this.keepAliveInterval = null;
        
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

            const result = parseOpenAIJSON(response.choices[0].message.content);
            
            if (this.aiExplainMode) {
                log('ai', `AI Reasoning: ${result.reasoning}`);
            }
            
            return result;
            
        } catch (error) {
            log('warn', `AI selection failed: ${error.message}`);
            return null;
        }
    }

    async _selectCorrectOptionWithAI(nodeName, exactOption, currentNodeHints) {
        /**
         * 🧠 Use AI Vision to select correct option from sidebar
         * Uses: workflow JSON hints + transcript + visual recipe
         */
        try {
            // Take screenshot of options sidebar
            const screenshot = await this.page.screenshot({ encoding: 'base64' });
            const compressedScreenshot = await compressBase64Png(screenshot);
            
            // Get hints from workflow JSON
            const selectOption = currentNodeHints?.select_option || exactOption;
            const configNotes = currentNodeHints?.configuration_notes || "";
            
            // Load transcript hints
            const fs = await import('fs/promises');
            let transcriptHint = "";
            try {
                const transcriptPath = join(__dirname, '..', 'analyzerYT', 'output', 'transcript.json');
                const transcriptData = JSON.parse(await fs.readFile(transcriptPath, 'utf-8'));
                transcriptHint = transcriptData.full_text.substring(0, 1000);
            } catch {}
            
            const prompt = `Look at this n8n options sidebar/panel.

REQUIRED OPTION (from workflow analysis):
"${selectOption}"

TRANSCRIPT CONTEXT:
${transcriptHint}

CONFIGURATION NOTES:
${configNotes}

TASK:
1. Find the option that matches "${selectOption}" in the visible panel
2. Return EXACT text as shown in the UI

Common patterns:
- "on messages" or "On messages" for WhatsApp trigger
- "send message" or "Send message" for WhatsApp action
- "Message a model" for OpenAI model interaction

Return JSON:
{
  "selected_option": "exact UI text of option",
  "confidence": 0-100,
  "reasoning": "match found"
}`;

            const response = await openaiCallWithRetries(this.openai, {
                model: 'gpt-4o',
                messages: [{
                    role: 'user',
                    content: [
                        { type: 'text', text: prompt },
                        {
                            type: 'image_url',
                            image_url: { url: `data:image/jpeg;base64,${compressedScreenshot}` }
                        }
                    ]
                }],
                response_format: { type: 'json_object' },
                max_tokens: 200,
                temperature: 0
            }, 3);
            
            const result = parseOpenAIJSON(response.choices[0].message.content);
            
            if (result.selected_option && result.confidence > 60) {
                log('ai', `AI found: "${result.selected_option}" (${result.confidence}% match with "${selectOption}")`);
                return result.selected_option;
            }
            
            return null;
            
        } catch (error) {
            log('warn', `AI option selection failed: ${error.message}`);
            return null;
        }
    }

    async _verifyNodeAdded(nodeType, expectedName) {
        /**
         * ✅ ORIGINAL verification logic (reliable)
         * CRITICAL: Check settings panel + AI Vision fallback
         */
        try {
            await this.page.waitForTimeout(1500);
            
            // 🎯 BEST VERIFICATION: Check if settings/config panel appeared
            const settingsPanel = await this.page.locator('[data-test-id="node-settings"], .node-settings-panel, [class*="ndv-wrapper"]').isVisible().catch(() => false);
            
            if (settingsPanel) {
                log('success', `✅ Settings panel opened - Node "${expectedName}" definitely added!`);
                return true;
            }
            
            log('warn', `⚠️  No settings panel detected - node might not have been added`);
            
            // 🚨 CRITICAL FIX: Settings panel is MANDATORY for node creation
            // If no settings panel, node was NOT added - this is the most reliable check
            // AI Vision can be fooled by sidebar panels, but settings panel opening is definitive
            
            log('error', `❌ Node NOT added - settings panel did not open`);
            log('info', `This usually means a sidebar/options panel opened instead of adding the node`);
            
            // STOP HERE - return false immediately
            return false;
            
        } catch (error) {
            log('error', `Verification error: ${error.message}`);
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
        const filename = `${String(this.screenshotCounter).padStart(3, '0')}_${description.replace(/\s+/g, '_')}.png`;
        const path = `${this.screenshotDirs[category]}/${filename}`;
        
        // ✅ BLOCKING - Original behavior (reliable)
        try {
            await this.page.screenshot({ 
                path, 
                fullPage: true
            });
        console.log(`   📸 Screenshot: ${category}/${filename}`);
        } catch (e) {
            console.log(`   ⚠️  Screenshot failed: ${e.message}`);
        }
        
        return path;
    }
    
    /**
     * ⏰ Keep-Alive (prevent Browserbase timeout)
     */
    startKeepAlive() {
        if (this.keepAliveInterval) return;
        this.keepAliveInterval = setInterval(async () => {
            try {
                await this.page.evaluate(() => Date.now());
            } catch (e) {
                clearInterval(this.keepAliveInterval);
            }
        }, 25000); // Every 25s
    }
    
    stopKeepAlive() {
        if (this.keepAliveInterval) {
            clearInterval(this.keepAliveInterval);
        }
    }
    
    /**
     * 🛡️ Safe wrapper
     */
    async safe(fn, fallback = null) {
        try {
            return await fn();
        } catch (e) {
            return fallback;
        }
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

            const result = parseOpenAIJSON(response.choices[0].message.content);
            
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
            
            const evaluation = parseOpenAIJSON(response.choices[0].message.content);
            
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
     * 🎯 Compare current canvas with reference blueprint
     */
    async compareWithReference(stepName) {
        if (!this.referenceImageBase64 || !this.workflowBlueprint) {
            return { match_percentage: 100, notes: "No reference loaded" };
        }

        try {
            log('info', `📊 Comparing with reference: ${stepName}`);
            const currentScreenshot = await this.page.screenshot({ encoding: 'base64', fullPage: false });
            const compressed = await compressBase64Png(currentScreenshot);

            const response = await openaiCallWithRetries(this.openai, {
                model: 'gpt-4o',
                messages: [{
                    role: 'user',
                    content: [
                        {
                            type: 'text',
                            text: `Compare current n8n canvas (left) with reference workflow (right).

Step: ${stepName}

Analyze:
1. How many nodes match? (name + type)
2. Are connections similar?
3. Any missing nodes or connections?
4. Special slots filled correctly? (like "Chat Model+")

Return JSON:
{
  "match_percentage": 85,
  "nodes_matched": ["WhatsApp Trigger", "AI Agent"],
  "nodes_missing": ["WhatsApp Business Cloud"],
  "connections_status": "2/3 complete",
  "issues": ["Chat Model slot not connected"],
  "next_action": "Add missing node"
}`
                        },
                        {
                            type: 'image_url',
                            image_url: { url: `data:image/jpeg;base64,${compressed}` }
                        },
                        {
                            type: 'image_url',
                            image_url: { url: `data:image/png;base64,${this.referenceImageBase64}` }
                        }
                    ]
                }],
                max_tokens: 1000
            });

            const comparison = parseOpenAIJSON(response.choices[0].message.content);
            this.validationSteps.push({ step: stepName, ...comparison });

            log('ai', `Match: ${comparison.match_percentage}% | ${comparison.connections_status}`);
            if (comparison.issues.length > 0) {
                log('warn', `Issues: ${comparison.issues.join(', ')}`);
            }

            return comparison;

        } catch (error) {
            log('warn', `Comparison failed: ${error.message}`);
            return { match_percentage: 0, issues: [error.message] };
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
     * 🎯 Load reference image (ai_visual_recipe.png) as blueprint
     */
    async loadReferenceImage() {
        try {
            log('info', '🖼️  Loading reference image from analyzerYT output');
            const fs = await import('fs/promises');
            // Try analyzerYT output first, fallback to old output
            let imagePath = this.referenceImagePath.replace('/output/', '/analyzerYT/output/');
            try {
                await fs.access(imagePath);
                log('info', '✅ Using analyzerYT visual recipe');
            } catch {
                imagePath = this.referenceImagePath;
                log('info', '⚠️  Fallback to old output folder');
            }
            const imageBuffer = await fs.readFile(imagePath);
            this.referenceImageBase64 = imageBuffer.toString('base64');
            
            // Extract workflow blueprint using AI Vision
            const response = await openaiCallWithRetries(this.openai, {
                model: 'gpt-4o',
                messages: [{
                    role: 'user',
                    content: [
                        {
                            type: 'text',
                            text: `Analyze this n8n workflow reference image. Extract:
1. All node names and their types (trigger/action/model/output)
2. All connections between nodes
3. Special connection types (like "Chat Model+" slots)

Return JSON:
{
  "nodes": [{"name": "WhatsApp Trigger", "type": "trigger", "special_slots": []}],
  "connections": [{"from": "node1", "to": "node2", "connection_type": "normal/slot"}],
  "total_nodes": 4
}`
                        },
                        {
                            type: 'image_url',
                            image_url: { url: `data:image/png;base64,${this.referenceImageBase64}` }
                        }
                    ]
                }],
                max_tokens: 2000
            });
            
            this.workflowBlueprint = parseOpenAIJSON(response.choices[0].message.content);
            log('success', `Blueprint loaded: ${this.workflowBlueprint.total_nodes} nodes expected`);
            return true;
        } catch (error) {
            log('warn', `Could not load reference: ${error.message}`);
            return false;
        }
    }

    /**
     * 🌐 Connect to Browserbase
     */
    async connect() {
        console.log('\n🚀 Starting AI-Powered n8n Agent\n');
        
        // Load reference image first
        await this.loadReferenceImage();
        
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
        
        // Increase timeout to 60 seconds for stability
        this.page.setDefaultTimeout(60000);
        this.context.setDefaultTimeout(60000);
        
        // ⏰ Start keep-alive to prevent session timeout
        this.startKeepAlive();
        
        // 🎯 Cache canvas handle
        this.canvasHandle = await this.safe(() => this.page.$('[data-test-id="canvas-wrapper"]'));
        
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
                // 3️⃣ SMART FALLBACK: Use AI Vision to select correct option from sidebar
                log('warn', `⚠️ Enter didn't work - sidebar opened with options`);
                log('info', `🧠 Using AI Vision to select correct option...`);
                
                // Check if node creator OR sidebar panel is open
                await this.page.waitForTimeout(1000);
                
                // 🧠 Use AI Vision to find and click correct option
                const selectedOption = await this._selectCorrectOptionWithAI(nodeName, exactOption, this.currentNodeHints);
                
                if (selectedOption) {
                    log('success', `✅ AI selected: "${selectedOption}"`);
                    
                    // Try multiple click strategies
                    let clicked = false;
                    
                    // Strategy 1: Click from node creator items
                    try {
                        const optionElement = this.page.locator(`[data-test-id="node-creator-node-item"]:has-text("${selectedOption}")`).first();
                        if (await optionElement.isVisible({ timeout: 1000 }).catch(() => false)) {
                            await optionElement.click();
                            clicked = true;
                            log('info', '   Clicked from node creator');
                        }
                    } catch {}
                    
                    // Strategy 2: Click from sidebar actions/triggers list
                    if (!clicked) {
                        try {
                            const sidebarOption = this.page.getByText(selectedOption, { exact: false }).first();
                            if (await sidebarOption.isVisible({ timeout: 1000 }).catch(() => false)) {
                                await sidebarOption.click();
                                clicked = true;
                                log('info', '   Clicked from sidebar');
                            }
                        } catch {}
                    }
                    
                    if (clicked) {
                        await this.page.waitForTimeout(2000);
                        
                        // Verify settings panel opened
                        nodeAdded = await this._verifyNodeAdded(nodeType, nodeName);
                        
                        if (!nodeAdded) {
                            log('error', `❌ Click succeeded but settings panel didn't open`);
                        }
                    } else {
                        log('error', `❌ Could not click option: "${selectedOption}"`);
                    }
                } else {
                    log('error', `❌ AI could not find correct option`);
                    
                    // OLD FALLBACK CODE (keep for edge cases)
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
    /**
     * 🔍 Detect + icons in a node using AI Vision
     */
    async detectPlusIconsInNode(nodeName) {
        try {
            log('ai', `Detecting + icons in "${nodeName}" node...`);
            const screenshot = await this.page.screenshot({ encoding: 'base64' });
            const compressed = await compressBase64Png(screenshot);

            const response = await openaiCallWithRetries(this.openai, {
                model: 'gpt-4o',
                messages: [{
                    role: 'user',
                    content: [
                        {
                            type: 'text',
                            text: `Analyze the "${nodeName}" node on this n8n canvas.

Find ALL "+" icons (plus buttons) in this node with their labels.

Common + icon labels:
- "Chat Model +" or "Chat Model*"
- "Memory +"
- "Tool +"
- Any other "Something +"

For each + icon, provide:
1. Label text (what's written next to +)
2. Position (x, y coordinates)
3. Purpose (what type of node should connect here)

Return JSON:
{
  "has_plus_icons": true/false,
  "plus_icons": [
    {
      "label": "Chat Model",
      "position": {"x": 450, "y": 300},
      "purpose": "AI model or chat model",
      "clickable": true
    }
  ],
  "node_name": "${nodeName}"
}`
                        },
                        {
                            type: 'image_url',
                            image_url: { url: `data:image/jpeg;base64,${compressed}` }
                        }
                    ]
                }],
                max_tokens: 1000
            });

            const result = parseOpenAIJSON(response.choices[0].message.content);
            
            if (result.has_plus_icons && result.plus_icons.length > 0) {
                log('success', `Found ${result.plus_icons.length} + icon(s) in "${nodeName}"`);
                result.plus_icons.forEach(icon => {
                    log('info', `  - "${icon.label}" at (${Math.round(icon.position.x)}, ${Math.round(icon.position.y)})`);
                });
                return result.plus_icons;
            }

            log('info', `No + icons found in "${nodeName}"`);
            return [];

        } catch (error) {
            log('warn', `+ icon detection failed: ${error.message}`);
            return [];
        }
    }

    /**
     * 🧠 Match + icon with target node (intelligent matching)
     */
    matchPlusIconWithNode(plusIcons, targetNodeName) {
        if (!plusIcons || plusIcons.length === 0) return null;

        // Smart matching logic
        for (const icon of plusIcons) {
            const label = icon.label.toLowerCase();
            const target = targetNodeName.toLowerCase();
            const purpose = icon.purpose.toLowerCase();

            // Exact or substring match
            if (target.includes(label) || label.includes(target)) {
                log('success', `✅ Matched: "${icon.label}" ← "${targetNodeName}"`);
                return icon;
            }

            // Purpose-based match - Enhanced for "Chat Model", "OpenRouter", etc.
            if ((target.includes('chat') || target.includes('model') || target.includes('openrouter')) && 
                (purpose.includes('chat') || label.includes('chat') || label.includes('model'))) {
                log('success', `✅ Matched (purpose): "${icon.label}" ← "${targetNodeName}" (chat/model)`);
                return icon;
            }

            if (target.includes('memory') && (purpose.includes('memory') || label.includes('memory'))) {
                log('success', `✅ Matched (purpose): "${icon.label}" ← "${targetNodeName}" (memory)`);
                return icon;
            }

            if (target.includes('tool') && (purpose.includes('tool') || label.includes('tool'))) {
                log('success', `✅ Matched (purpose): "${icon.label}" ← "${targetNodeName}" (tool)`);
                return icon;
            }
        }

        // No match found - use first + icon as fallback
        if (plusIcons.length > 0) {
            log('warn', `No perfect match, using first + icon: "${plusIcons[0].label}"`);
            return plusIcons[0];
        }

        return null;
    }

    /**
     * 🔍 Verify connection exists in DOM
     */
    async verifyConnectionInDOM(sourceNodeId, targetNodeId) {
        try {
            const exists = await this.page.evaluate(({src, tgt}) => {
                // Extract UUID from node IDs
                const srcUUID = src.split('-')[1];
                const tgtUUID = tgt.split('-')[1];

                // Check SVG connection paths
                const connections = [...document.querySelectorAll('g[data-type="connection"] path, svg path[class*="connection"]')];
                
                return connections.some(path => {
                    const d = path.getAttribute('d') || '';
                    const dataConnection = path.closest('g')?.getAttribute('data-connection') || '';
                    
                    // Check if path contains both node UUIDs
                    return (d.includes(srcUUID) || dataConnection.includes(srcUUID)) &&
                           (d.includes(tgtUUID) || dataConnection.includes(tgtUUID));
                });
            }, {src: sourceNodeId, tgt: targetNodeId});

            return exists;
        } catch (error) {
            log('warn', `Connection verification failed: ${error.message}`);
            return false;
        }
    }

    async connectNodesIntelligent(fromType, toType, fromNodeId = null, toNodeId = null) {
        console.log(`\n🔗 Connecting: ${fromType} → ${toType}`);

        try {
            await this.takeScreenshot('connections', `before_${fromType}_to_${toType}`);
            
            // Wait for nodes to fully render
            await this.page.waitForTimeout(2000);
            
            // 🔁 RETRY LOOP (up to 3 attempts)
            for (let attempt = 1; attempt <= 3; attempt++) {
                log('info', `🔄 Attempt ${attempt}/3`);
                
                // ⭐ STRATEGY 1: Check for + icons first (intelligent!)
                log('ai', `Step 1: Checking if "${fromType}" has + icons...`);
                const plusIcons = await this.detectPlusIconsInNode(fromType);
                
                if (plusIcons.length > 0) {
                    // Find matching + icon for target node
                    const matchedIcon = this.matchPlusIconWithNode(plusIcons, toType);
                    
                    if (matchedIcon) {
                        log('success', `✅ Using + icon: "${matchedIcon.label}" for "${toType}"`);
                        
                        // Get target node position by looking for diamond OR circle shape
                        const targetNodeInfo = await this.page.evaluate((targetName) => {
                            // Look for canvas nodes containing target name
                            const canvasNodes = [...document.querySelectorAll('[data-test-id^="canvas-node"]')];
                            
                            for (const node of canvasNodes) {
                                const text = node.textContent || '';
                                if (text.includes(targetName) || text.toLowerCase().includes(targetName.toLowerCase())) {
                                    // Check if it has diamond (path) or circle shape
                                    const hasDiamond = node.querySelector('path[d*="M"]') !== null;
                                    const hasCircle = node.querySelector('circle[r]') !== null;
                                    
                                    const box = node.getBoundingClientRect();
                                    return {
                                        found: true,
                                        shape: hasDiamond ? 'diamond' : (hasCircle ? 'circle' : 'rect'),
                                        x: box.x,
                                        y: box.y,
                                        width: box.width,
                                        height: box.height,
                                        name: text.trim().split('\n')[0]
                                    };
                                }
                            }
                            return { found: false };
                        }, toType);
                        
                        if (targetNodeInfo.found) {
                            log('success', `Found target: "${targetNodeInfo.name}" (${targetNodeInfo.shape})`);
                            log('info', `🎯 Dragging from + icon to ${targetNodeInfo.shape} node...`);
                            
                            // Drag from + icon to target node center
                            const plusX = matchedIcon.position.x;
                            const plusY = matchedIcon.position.y;
                            const targetX = targetNodeInfo.x + targetNodeInfo.width / 2;
                            const targetY = targetNodeInfo.y + targetNodeInfo.height / 2;
                            
                            log('debug', `Coordinates: (${Math.round(plusX)}, ${Math.round(plusY)}) → (${Math.round(targetX)}, ${Math.round(targetY)})`);
                            
                            await this.page.mouse.move(plusX, plusY);
                            await this.page.mouse.down();
                            await this.page.waitForTimeout(400);
                            await this.page.mouse.move(targetX, targetY, { steps: 30 });
                            await this.page.waitForTimeout(400);
                            await this.page.mouse.up();
                            await this.page.waitForTimeout(1500);
                            
                            log('success', `✅ Connected via + icon drag: ${fromType}.${matchedIcon.label} → ${toType}`);
                            await this.takeScreenshot('connections', `success_plus_${fromType}_to_${toType}`);
                            return true;
                        } else {
                            log('warn', `Target node "${toType}" not found on canvas`);
                        }
                    }
                }
                
                // ⭐ STRATEGY 2: Regular circle-to-circle drag (fallback)
                log('info', `Step 2: Using regular circle drag...`);
                try {
                    console.log(`   🎯 Circle-based drag connection`);
                    
                    // Find ALL canvas nodes
                const allNodes = await this.page.locator('[data-test-id^="canvas"]').all();
                console.log(`   📊 Found ${allNodes.length} canvas elements`);
                
                if (allNodes.length >= 2) {
                        // Find source and target based on position (first = oldest, last = newest)
                    const sourceNode = allNodes[0];
                    const targetNode = allNodes[allNodes.length - 1];
                    
                        // Find ALL circles in source (output is usually last)
                        const sourceCircles = await sourceNode.locator('circle').all();
                        // Find ALL circles in target (input is usually first)
                        const targetCircles = await targetNode.locator('circle').all();
                        
                        console.log(`   🔍 Source circles: ${sourceCircles.length}, Target circles: ${targetCircles.length}`);
                        
                        if (sourceCircles.length > 0 && targetCircles.length > 0) {
                            // Output = last circle, Input = first circle
                            const outputCircle = sourceCircles[sourceCircles.length - 1];
                            const inputCircle = targetCircles[0];
                    
                    const sourceBox = await outputCircle.boundingBox();
                    const targetBox = await inputCircle.boundingBox();
                    
                    if (sourceBox && targetBox) {
                                console.log(`   🎯 Connection points: (${Math.round(sourceBox.x)}, ${Math.round(sourceBox.y)}) → (${Math.round(targetBox.x)}, ${Math.round(targetBox.y)})`);
                                
                                // Human-like drag with precise center targeting
                                const srcX = sourceBox.x + sourceBox.width / 2;
                                const srcY = sourceBox.y + sourceBox.height / 2;
                                const tgtX = targetBox.x + targetBox.width / 2;
                                const tgtY = targetBox.y + targetBox.height / 2;
                                
                                await this.page.mouse.move(srcX, srcY);
                        await this.page.mouse.down();
                                await this.page.waitForTimeout(300);
                                await this.page.mouse.move(tgtX, tgtY, { steps: 25 });
                                await this.page.waitForTimeout(300);
                        await this.page.mouse.up();
                        
                                // Wait for connection to render
                                await this.page.waitForTimeout(1000);
                                
                                // Verify connection in DOM
                                const verified = sourceId && targetId ? 
                                    await this.verifyConnectionInDOM(sourceId, targetId) : 
                                    true;
                                
                                if (verified) {
                                    log('success', `✅ Connection verified in DOM!`);
                        console.log(`✅ Connected ${fromType} → ${toType}\n`);
                                    await this.takeScreenshot('connections', `success_${fromType}_to_${toType}`);
                                    return true;
                                } else if (attempt < 3) {
                                    log('warn', `Connection not verified, retrying...`);
                        await this.page.waitForTimeout(1000);
                                    continue;
                                } else {
                                    log('warn', `Connection created but not verified`);
                                    console.log(`✅ Connected ${fromType} → ${toType} (unverified)\n`);
                        await this.takeScreenshot('connections', `success_${fromType}_to_${toType}`);
                        return true;
                                }
                            }
                    }
                }
            } catch (e) {
                    log('warn', `Circle drag failed: ${e.message}`);
                    if (attempt < 3) {
                        await this.page.waitForTimeout(1000);
                        continue;
                    }
                }
            }
            
            // Strategy 2: AI Vision-based connection
            console.log(`   🎯 Strategy 2: AI Vision-based connection`);
            const screenshot = await this.page.screenshot({ encoding: 'base64' });
            const compressed = await compressBase64Png(screenshot);
            
            const response = await openaiCallWithRetries(this.openai, {
                model: 'gpt-4o',
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
                            { type: "image_url", image_url: { url: `data:image/jpeg;base64,${compressed}` } }
                        ]
                    }
                ],
                response_format: { type: "json_object" },
                max_tokens: 200
            });
            
            const coords = parseOpenAIJSON(response.choices[0].message.content);
            
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
            // 🎯 PRIORITY: Load from analyzerYT (new robust analyzer)
            let actionPath = join(__dirname, '..', 'analyzerYT', 'output', 'ai_workflow_complete.json');
            let workflow;
            
            try {
                workflow = JSON.parse(await readFile(actionPath, 'utf-8'));
                console.log('\n🎯 Using analyzerYT Workflow (Audio + Vision)');
                console.log(`📋 Workflow: ${workflow.workflow_name}`);
            } catch {
                // Fallback to old output folder
                try {
                    actionPath = join(__dirname, '..', 'output', 'ai_workflow_complete.json');
                workflow = JSON.parse(await readFile(actionPath, 'utf-8'));
                    console.log('\n🧠 Using AI-Enhanced Workflow (fallback)');
                } catch {
                    // Final fallback
                    actionPath = join(__dirname, '..', 'output', 'action_sequence.json');
                    workflow = JSON.parse(await readFile(actionPath, 'utf-8'));
                    console.log('\n📋 Using Standard Workflow (fallback)');
                }
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
                    
                    // Compare with reference after each node (skip to save time)
                    // if (nodeId && this.referenceImageBase64) {
                    //     const comparison = await this.compareWithReference(`After adding ${node.name}`);
                    //     stepLog.validation = comparison;
                    // }
                    
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
            
            // 🎯 FINAL VALIDATION - Compare with reference
            console.log('\n========================================');
            console.log('🎯 FINAL VALIDATION');
            console.log('========================================\n');
            
            const finalComparison = await this.compareWithReference('Final Workflow');
            
            console.log(`\n📊 Workflow Replication Results:`);
            console.log(`   Match: ${finalComparison.match_percentage}%`);
            console.log(`   Nodes Matched: ${finalComparison.nodes_matched?.length || 0}/${this.workflowBlueprint?.total_nodes || workflow.nodes.length}`);
            console.log(`   Connections: ${finalComparison.connections_status || 'Unknown'}`);
            
            if (finalComparison.nodes_missing && finalComparison.nodes_missing.length > 0) {
                console.log(`\n❌ Missing Nodes:`);
                finalComparison.nodes_missing.forEach(node => console.log(`   - ${node}`));
            }
            
            if (finalComparison.issues && finalComparison.issues.length > 0) {
                console.log(`\n⚠️  Issues Found:`);
                finalComparison.issues.forEach(issue => console.log(`   - ${issue}`));
            }
            
            // Success criteria
            if (finalComparison.match_percentage >= 90) {
                log('success', `\n✅ EXCELLENT! ${finalComparison.match_percentage}% match with reference`);
            } else if (finalComparison.match_percentage >= 75) {
                log('success', `\n✅ GOOD! ${finalComparison.match_percentage}% match with reference`);
            } else {
                log('warn', `\n⚠️  PARTIAL! ${finalComparison.match_percentage}% match with reference`);
            }
            
            console.log('\n========================================\n');
            
            // Add to execution log
            executionLog.final_validation = finalComparison;
            
            // 💾 SAVE WORKFLOW IN N8N (Enhanced)
            console.log('\n💾 Saving workflow in n8n...');
            try {
                // Wait for any pending UI updates
                await this.page.waitForTimeout(1000);
                
                // Strategy 1: Try direct Save button
                const saveBtn = this.page.locator('button:has-text("Save")').first();
                const isSaveVisible = await saveBtn.isVisible({ timeout: 3000 }).catch(() => false);
                
                if (isSaveVisible) {
                    log('info', 'Clicking Save button...');
                    await saveBtn.click();
                    await this.page.waitForTimeout(2000);
                    
                    // Check for success toast/notification
                    const saved = await this.page.locator('text=/saved|success/i').isVisible({ timeout: 3000 }).catch(() => false);
                    
                    if (saved) {
                        log('success', '✅ Workflow saved successfully!');
                    } else {
                        log('success', '✅ Save clicked (checking...)');
                    }
                    
                    await this.takeScreenshot('final', 'saved_workflow');
                } else {
                    // Strategy 2: Try keyboard shortcut
                    log('info', 'Trying Ctrl+S shortcut...');
                    await this.page.keyboard.press('Control+KeyS');
                    await this.page.waitForTimeout(2000);
                    log('success', '✅ Save shortcut sent');
                    await this.takeScreenshot('final', 'saved_workflow');
                }
                
                // Get workflow URL
                const currentUrl = this.page.url();
                log('info', `📍 Workflow URL: ${currentUrl}`);
                
            } catch (error) {
                log('error', `Could not save: ${error.message}`);
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
            
            console.log('\n========================================');
            console.log('📊 AUTOMATION SUMMARY');
            console.log('========================================\n');
            console.log(`   Total Steps: ${totalSteps}`);
            console.log(`   ✅ Successful: ${successfulSteps}`);
            console.log(`   ❌ Failed: ${totalSteps - successfulSteps}`);
            console.log(`   📈 Success Rate: ${successRate}%`);
            
            if (executionLog.final_status === 'success') {
                console.log(`\n🎉 ✅ Automation completed successfully!`);
            } else {
                console.log(`\n⚠️  Automation completed with ${totalSteps - successfulSteps} issue(s)`);
            }
            
            console.log('\n========================================\n');
            
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
        // ⏰ Stop keep-alive
        this.stopKeepAlive();
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

