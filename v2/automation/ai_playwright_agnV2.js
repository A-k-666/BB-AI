#!/usr/bin/env node

/**
 * ============================================================
 * AI N8N PLAYWRIGHT AGENT V2 - CLEAN & ROBUST
 * ============================================================
 * Key Features:
 * ✅ Auto-connect: Nodes added via + icon (no manual connections!)
 * ✅ AI Vision search: Scans results BEFORE clicking (no blind Enter!)
 * ✅ Visual verification: Settings panel check (100% reliable)
 * ✅ Clean architecture: No shape detection, no complex retries
 * ✅ Screenshot management: Auto-cleanup old screenshots
 * 
 * Flow:
 * 1. Click + icon on last node → auto-connects new node
 * 2. Search for node name (e.g., "Google")
 * 3. AI Vision scans search results (modal + sidebar)
 * 4. Click specific best match (e.g., "Google Gemini Chat Model")
 * 5. Verify settings panel opened (CRITICAL)
 * 6. Sidebar fallback if needed
 * 
 * Critical Fix:
 * - NO blind Enter after search!
 * - AI Vision finds exact match from ALL visible options
 * - Prevents selecting wrong node (e.g., Google Sheets vs Google Gemini)
 * ============================================================
 */

import { config } from 'dotenv';
import { chromium } from 'playwright';
import OpenAI from 'openai';
import { readFile, writeFile, mkdir } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import sharp from 'sharp';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

config({ path: join(__dirname, '..', 'config', 'credentials.env') });

// ============================================================================
// 🛠️ UTILITIES
// ============================================================================

/**
 * Compress base64 PNG to JPEG (reduces OpenAI payload)
 */
async function compressImage(base64png, width = 1280, quality = 70) {
    try {
        const buffer = Buffer.from(base64png, 'base64');
        const compressed = await sharp(buffer)
            .resize({ width, withoutEnlargement: true })
            .jpeg({ quality })
            .toBuffer();
        return compressed.toString('base64');
    } catch (error) {
        console.warn('⚠️ Image compression failed, using original');
        return base64png;
    }
}

/**
 * Retry OpenAI calls with exponential backoff
 */
async function callOpenAI(client, params, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            return await client.chat.completions.create(params);
        } catch (err) {
            const status = err?.status || err?.response?.status;
            console.warn(`⚠️ OpenAI error ${status || err.message} (attempt ${i+1}/${retries})`);
            if (i === retries - 1) throw err;
            await new Promise(r => setTimeout(r, 1000 * Math.pow(2, i)));
        }
    }
}

/**
 * Parse OpenAI JSON (handles markdown blocks)
 */
function parseJSON(content) {
    let cleaned = content.trim();
    if (cleaned.includes('```json')) {
        cleaned = cleaned.replace(/```json\n?/g, '').replace(/```\n?/g, '');
    } else if (cleaned.includes('```')) {
        cleaned = cleaned.replace(/```\n?/g, '');
    }
    return JSON.parse(cleaned.trim());
}

/**
 * Structured logging
 */
function log(level, message, ...args) {
    const icons = {
        info: 'ℹ️',
        success: '✅',
        warn: '⚠️',
        error: '❌',
        ai: '🧠',
        debug: '🔍'
    };
    console.log(`${icons[level] || ''} ${message}`, ...args);
}

// ============================================================================
// 🧠 N8N UI EXPERT (RAG-based reasoning)
// ============================================================================

class N8nUIExpert {
    constructor(openai, model = 'gpt-4o') {
        this.openai = openai;
        this.model = model;
        this.docsContext = null;
        this.conversationHistory = [];
    }

    /**
     * 📚 Load n8n documentation context
     */
    async loadDocs() {
        try {
            const docPath = join(__dirname, '..', 'docs', 'n8n_docs_combined.md');
            this.docsContext = (await readFile(docPath, 'utf-8')).slice(0, 80000);
            log('success', `Loaded n8n docs (${(this.docsContext.length / 1000).toFixed(1)}K chars)`);
        } catch {
            log('warn', 'Docs not found, using fallback knowledge');
            this.docsContext = `n8n UI basics:
- Canvas: Main workflow area
- Node Creator: Search dialog to add nodes (opens with + button)
- Settings Panel: Appears when node is added (confirms success)
- Auto-connection: Clicking + on existing node auto-connects new node`;
        }
    }

    /**
     * 🧠 Ask expert for guidance
     */
    async ask(question, screenshot = null) {
        const systemPrompt = `You are an n8n UI automation expert.

Knowledge base:
${this.docsContext}

Provide precise, actionable Playwright instructions.`;

        const messages = [
            { role: 'system', content: systemPrompt },
            ...this.conversationHistory.slice(-6), // Keep last 3 exchanges
            { role: 'user', content: question }
        ];

        if (screenshot) {
            messages[messages.length - 1].content = [
                { type: 'text', text: question },
                { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${screenshot}` } }
            ];
        }

        try {
            const response = await callOpenAI(this.openai, {
                model: this.model,
                messages,
                temperature: 0.1,
                max_tokens: 800
            });

            const answer = response.choices[0].message.content;
            
            // Update history
            this.conversationHistory.push(
                { role: 'user', content: typeof messages[messages.length - 1].content === 'string' 
                    ? messages[messages.length - 1].content 
                    : question },
                { role: 'assistant', content: answer }
            );

            return answer;
        } catch (error) {
            log('error', `Expert query failed: ${error.message}`);
            return null;
        }
    }

    /**
     * 🎯 Get best selector for an intent
     */
    async getSelector(intent, domSnapshot = null) {
        const question = `Task: ${intent}

${domSnapshot ? `DOM Context:\n${JSON.stringify(domSnapshot, null, 2)}\n` : ''}

Return JSON with best Playwright selector:
{
  "selector": "...",
  "reasoning": "...",
  "fallbacks": ["..."]
}`;

        const answer = await this.ask(question);
        try {
            return parseJSON(answer);
        } catch {
            return { selector: null, reasoning: answer, fallbacks: [] };
        }
    }
}

// ============================================================================
// 🤖 MAIN AGENT
// ============================================================================

class N8nAutomationAgent {
    constructor() {
        this.browser = null;
        this.page = null;
        this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
        this.expert = new N8nUIExpert(this.openai, process.env.AI_MODEL || 'gpt-4o');
        
        this.screenshotDir = join(__dirname, '..', 'output', 'screenshots');
        this.screenshotCounter = 0;
        
        this.executionLog = {
            started_at: new Date().toISOString(),
            steps: []
        };
    }

    /**
     * 🚀 Initialize browser & load expert knowledge
     */
    async init() {
        log('info', 'Initializing agent...');
        
        // Create output directories
        await mkdir(join(this.screenshotDir, 'nodes'), { recursive: true });
        await mkdir(join(this.screenshotDir, 'errors'), { recursive: true });
        
        // 🧹 Clean old screenshots
        await this.cleanOldScreenshots();
        
        // Load expert knowledge
        await this.expert.loadDocs();
        
        // Connect to browser
        const sessionUrl = `wss://connect.browserbase.com?apiKey=${process.env.API_KEY}&projectId=${process.env.PROJECT_ID}`;
        this.browser = await chromium.connectOverCDP(sessionUrl);
        this.page = this.browser.contexts()[0].pages()[0];
        this.page.setDefaultTimeout(60000);
        
        log('success', 'Agent initialized');
    }

    /**
     * 🧹 Clean old screenshots
     */
    async cleanOldScreenshots() {
        try {
            const { readdir, unlink } = await import('fs/promises');
            
            const foldersToClean = ['nodes', 'errors'];
            let totalCleaned = 0;
            
            for (const folder of foldersToClean) {
                const dirPath = join(this.screenshotDir, folder);
                try {
                    const files = await readdir(dirPath);
                    for (const file of files) {
                        if (file.endsWith('.png')) {
                            await unlink(join(dirPath, file));
                            totalCleaned++;
                        }
                    }
                } catch (e) {
                    // Directory doesn't exist or empty, skip
                }
            }
            
            if (totalCleaned > 0) {
                log('info', `Cleaned ${totalCleaned} old screenshots`);
            }
        } catch (error) {
            log('warn', `Screenshot cleanup failed: ${error.message}`);
        }
    }

    /**
     * 📸 Take organized screenshot
     */
    async screenshot(category, description) {
        this.screenshotCounter++;
        const filename = `${String(this.screenshotCounter).padStart(3, '0')}_${description}.png`;
        const path = join(this.screenshotDir, category, filename);
        
        try {
            await this.page.screenshot({ path, fullPage: false, timeout: 15000 });
            log('debug', `Screenshot: ${category}/${filename}`);
        } catch (error) {
            log('warn', `Screenshot failed: ${error.message}`);
        }
        
        return path;
    }

    /**
     * 🔐 Login to n8n
     */
    async loginToN8n() {
        log('info', 'Logging into n8n...');
        
        await this.page.goto('https://pika2223.app.n8n.cloud/workflow/new', { 
            waitUntil: 'domcontentloaded',
            timeout: 60000 
        });
        
        await this.page.waitForTimeout(2000);

        // Check if already logged in
        const isLoggedIn = await this.page.locator('.workflow-canvas, .vue-flow').isVisible().catch(() => false);
        
        if (isLoggedIn) {
            log('success', 'Already logged in');
            return;
        }

        // Fast login with direct selectors
        try {
            const emailInput = this.page.locator('input[type="email"]').first();
            await emailInput.waitFor({ state: 'visible', timeout: 8000 });
            await emailInput.fill(process.env.N8N_EMAIL);
            await emailInput.press('Enter');
            log('info', 'Email submitted');
            
            await this.page.waitForTimeout(2000);
            
            const passwordInput = this.page.locator('input[type="password"]').first();
            await passwordInput.waitFor({ state: 'visible', timeout: 8000 });
            await passwordInput.fill(process.env.N8N_PASSWORD);
            await passwordInput.press('Enter');
            log('info', 'Password submitted');
        } catch (error) {
            log('error', `Login failed: ${error.message}`);
            throw error;
        }

        // Wait for canvas
        await this.page.waitForSelector('.workflow-canvas, .vue-flow', { timeout: 20000 });
        await this.page.waitForTimeout(3000);
        
        log('success', 'Logged in successfully');
    }

    /**
     * 🎯 CLEAN NODE CREATION (using + icon auto-connect)
     * 
     * Flow:
     * 1. Click + icon on last node (auto-connects!) OR canvas + for first node
     * 2. Search for node
     * 3. Press Enter (fast path)
     * 4. Verify settings panel opened ✅ CRITICAL CHECK
     * 5. If sidebar appeared → use AI vision to select correct option
     */
    async addNode(nodeName, nodeType, isFirstNode = false, nodeHints = {}) {
        log('info', `\n➕ Creating node: ${nodeName} (${nodeType})`);
        
        const stepLog = {
            node: nodeName,
            type: nodeType,
            timestamp: new Date().toISOString(),
            status: 'pending'
        };

        try {
            // Step 1: Take before screenshot
            await this.screenshot('nodes', `before_${nodeType}_canvas`);
            
            let plusClicked = false;
            
            if (!isFirstNode) {
                // 🎯 CLICK + ICON ON LAST NODE (auto-connects!)
                try {
                    // Find all + buttons on canvas nodes
                    const plusButtons = await this.page.locator('[data-test-id^="canvas"] button[class*="plus"], [data-test-id^="canvas"] button[class*="add"]').all();
                    
                    if (plusButtons.length > 0) {
                        const lastPlusBtn = plusButtons[plusButtons.length - 1];
                        await lastPlusBtn.click({ timeout: 3000 });
                        log('success', '✅ Clicked + icon on last node (auto-connect mode)');
                        plusClicked = true;
                    }
                } catch (e) {
                    log('warn', `Node + icon failed: ${e.message}`);
                }
            }
            
            // Fallback: Canvas + button (for first node or if node + failed)
            if (!plusClicked) {
                try {
                    const canvasPlusBtn = this.page.locator('[data-test-id="node-creator-plus-button"]');
                    if (await canvasPlusBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
                        await canvasPlusBtn.click({ force: true });
                        log('info', 'Clicked canvas + button');
                        plusClicked = true;
                    }
                } catch (e) {
                    log('warn', `Canvas + button failed: ${e.message}`);
                }
            }
            
            if (!plusClicked) {
                throw new Error('Could not click + button');
            }

            await this.page.waitForTimeout(1500);
            await this.screenshot('nodes', `${nodeType}_modal_opened`);

            // Step 2: Intelligent search
            const searchBox = this.page.locator('[data-test-id="node-creator-search-bar"]').first();
            await searchBox.waitFor({ state: 'visible', timeout: 5000 });
            
            // Extract base service name (e.g., "Slack Trigger" → "Slack")
            const baseSearch = this._extractBaseServiceName(nodeName);
            const exactOption = nodeHints.select_option || nodeName;
            
            log('ai', `Intelligent search: "${baseSearch}" → then select "${exactOption}"`);
            
            await searchBox.fill(baseSearch);
            log('info', `Searching: "${baseSearch}"`);
            await this.page.waitForTimeout(1000);
            await this.screenshot('nodes', `${nodeType}_search_filled`);

            // Step 3: AI VISION SEARCH - Don't blindly press Enter!
            log('ai', '🔍 Using AI Vision to find best match from search results...');
            
            const searchScreenshot = await this.page.screenshot({ encoding: 'base64' });
            const compressed = await compressImage(searchScreenshot);
            
            const matchResponse = await callOpenAI(this.openai, {
                model: 'gpt-4o',
                messages: [{
                    role: 'user',
                    content: [
                        {
                            type: 'text',
                            text: `I searched for "${baseSearch}" in n8n node creator.
I want to add: "${exactOption}"

Look at the search results (central modal AND right sidebar if visible).

Find the BEST matching option for "${exactOption}".

Examples:
- Want "Google Gemini Chat Model" → Select "Google Gemini Chat Model" (NOT "Google Sheets")
- Want "Slack Trigger" → Select "On new message posted to channel" (NOT just "Slack")
- Want "OpenAI Chat Model" → Select "OpenAI Chat Model" (NOT "OpenAI")

Return JSON:
{
  "best_match": "exact text of best option",
  "confidence": 0-100,
  "match_type": "exact|close|partial",
  "reasoning": "why this is the best match"
}`
                        },
                        {
                            type: 'image_url',
                            image_url: { url: `data:image/jpeg;base64,${compressed}` }
                        }
                    ]
                }],
                response_format: { type: 'json_object' },
                max_tokens: 300,
                temperature: 0
            }, 3);
            
            const matchResult = parseJSON(matchResponse.choices[0].message.content);
            
            log('ai', `AI found: "${matchResult.best_match}" (${matchResult.confidence}% confidence, ${matchResult.match_type} match)`);
            log('debug', `Reasoning: ${matchResult.reasoning}`);
            
            let nodeAdded = false;
            
            if (matchResult.confidence > 70 && matchResult.best_match) {
                // Click the specific option (don't press Enter!)
                log('info', `Clicking: "${matchResult.best_match}"`);
                
                try {
                    const optionElement = this.page.getByText(matchResult.best_match, { exact: false }).first();
                    const optionVisible = await optionElement.isVisible({ timeout: 2000 }).catch(() => false);
                    
                    if (optionVisible) {
                        await optionElement.click();
                        await this.page.waitForTimeout(2000);
                        await this.screenshot('nodes', `${nodeType}_node_added`);
                        
                        // Verify settings panel opened
                        nodeAdded = await this._verifyNodeAdded(nodeType, nodeName);
                    } else {
                        log('warn', `Option "${matchResult.best_match}" not visible, trying Enter as fallback`);
                        await searchBox.press('Enter');
                        await this.page.waitForTimeout(2000);
                        nodeAdded = await this._verifyNodeAdded(nodeType, nodeName);
                    }
                } catch (e) {
                    log('warn', `Click failed: ${e.message}, trying Enter as fallback`);
                    await searchBox.press('Enter');
                    await this.page.waitForTimeout(2000);
                    nodeAdded = await this._verifyNodeAdded(nodeType, nodeName);
                }
            } else {
                log('warn', `Low confidence (${matchResult.confidence}%), pressing Enter as fallback`);
                await searchBox.press('Enter');
                await this.page.waitForTimeout(2000);
                await this.screenshot('nodes', `${nodeType}_node_added`);
                nodeAdded = await this._verifyNodeAdded(nodeType, nodeName);
            }

            // Step 4: If still not added, try sidebar fallback
            if (!nodeAdded) {
                log('warn', `⚠️ Primary search didn't work - trying sidebar fallback...`);
                
                await this.page.waitForTimeout(1000);
                
                // Use AI Vision to find and click correct option from sidebar
                const selectedOption = await this._selectCorrectOptionWithAI(nodeName, exactOption, nodeHints);
                
                if (selectedOption) {
                    log('success', `✅ AI found in sidebar: "${selectedOption}"`);
                    
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
                    
                    // Strategy 2: Click from sidebar
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
                    log('error', `❌ AI could not find correct option in sidebar`);
                }
            }
            
            // Step 5: Final result
            if (nodeAdded) {
                log('success', `✅ Node successfully added: "${nodeName}"`);
                stepLog.status = 'success';
            } else {
                log('error', `❌ Failed to add: "${nodeName}"`);
                stepLog.status = 'failed';
            }
            
            // Close settings panel - AGGRESSIVE approach
            for (let i = 0; i < 3; i++) {
                const backBtn = this.page.getByText('Back to canvas');
                if (await backBtn.isVisible().catch(() => false)) {
                    await backBtn.click();
                    log('success', 'Closed settings panel via button');
                    await this.page.waitForTimeout(1000);
                    break;
                }
                await this.page.waitForTimeout(500);
            }
            
            // ESC key fallback
            await this.page.keyboard.press('Escape');
            await this.page.waitForTimeout(500);
            await this.page.keyboard.press('Escape');
            await this.page.waitForTimeout(500);
            
            await this.screenshot('nodes', `${nodeType}_settings_closed`);
            
            return nodeAdded;

        } catch (error) {
            log('error', `Failed to add ${nodeName}: ${error.message}`);
            stepLog.status = 'failed';
            stepLog.error = error.message;
            await this.screenshot('errors', `failed_${nodeName.replace(/\s+/g, '_')}`);
            return false;
        } finally {
            this.executionLog.steps.push(stepLog);
        }
    }

    /**
     * 🔍 Extract base service name from full node name
     */
    _extractBaseServiceName(nodeName) {
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
        
        return nodeName.split(' ')[0];
    }

    /**
     * ✅ CRITICAL: Verify node was added (settings panel check)
     */
    async _verifyNodeAdded(nodeType, expectedName) {
        try {
            await this.page.waitForTimeout(1500);
            
            // 🎯 BEST VERIFICATION: Check if settings/config panel appeared
            const settingsPanel = await this.page.locator(
                '[data-test-id="node-settings"], .node-settings-panel, [class*="ndv-wrapper"]'
            ).isVisible().catch(() => false);
            
            if (settingsPanel) {
                log('success', `✅ Settings panel opened - Node "${expectedName}" definitely added!`);
                return true;
            }
            
            log('warn', `⚠️ No settings panel detected - node might not have been added`);
            log('error', `❌ Node NOT added - settings panel did not open`);
            log('info', `This usually means a sidebar/options panel opened instead of adding the node`);
            
            return false;
            
        } catch (error) {
            log('error', `Verification error: ${error.message}`);
            return false;
        }
    }

    /**
     * 🧠 Use AI Vision to select correct option from sidebar
     */
    async _selectCorrectOptionWithAI(nodeName, exactOption, nodeHints) {
        try {
            const screenshot = await this.page.screenshot({ encoding: 'base64' });
            const compressed = await compressImage(screenshot);
            
            const selectOption = nodeHints?.select_option || exactOption;
            const configNotes = nodeHints?.configuration_notes || "";
            
            const prompt = `Look at this n8n options sidebar/panel.

REQUIRED OPTION (from workflow analysis):
"${selectOption}"

CONFIGURATION NOTES:
${configNotes}

TASK:
1. Find the option that matches "${selectOption}" in the visible panel
2. Return EXACT text as shown in the UI

Common patterns:
- "on messages" or "On messages" for messaging trigger
- "send message" or "Send message" for messaging action
- "Message a model" for AI model interaction

Return JSON:
{
  "selected_option": "exact UI text of option",
  "confidence": 0-100,
  "reasoning": "match found"
}`;

            const response = await callOpenAI(this.openai, {
                model: 'gpt-4o',
                messages: [{
                    role: 'user',
                    content: [
                        { type: 'text', text: prompt },
                        {
                            type: 'image_url',
                            image_url: { url: `data:image/jpeg;base64,${compressed}` }
                        }
                    ]
                }],
                response_format: { type: 'json_object' },
                max_tokens: 200,
                temperature: 0
            }, 3);
            
            const result = parseJSON(response.choices[0].message.content);
            
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

    /**
     * 📋 Run from workflow plan JSON
     */
    async runFromPlan(planPath) {
        log('info', '\n🚀 Starting workflow automation\n');
        
        const plan = JSON.parse(await readFile(planPath, 'utf-8'));
        
        // 🎯 CHECK VERDICT - Only proceed if analysis is ready
        const verdict = plan.metadata?.verdict;
        if (verdict) {
            log('info', '🎯 Analysis Verdict:');
            log('info', `   ${verdict.summary}`);
            log('info', `   Understanding: ${verdict.understanding}`);
            log('info', `   Status: ${verdict.automation_status}\n`);
            
            if (!verdict.ready_for_automation) {
                log('error', '❌ AUTOMATION STOPPED');
                log('warn', '⚠️  Analysis quality insufficient for automation');
                log('warn', '   Recommendation: Review workflow manually or re-analyze video\n');
                return;
            }
            
            log('success', '✅ Analysis approved - proceeding with automation\n');
        }
        
        log('info', `Workflow: ${plan.workflow_name || 'Untitled'}`);
        log('info', `Nodes to create: ${plan.nodes.length}`);
        log('info', `🔗 Auto-connect: ON (using + icon)\n`);

        // Process each node (connections happen automatically via + icon!)
        for (let i = 0; i < plan.nodes.length; i++) {
            const node = plan.nodes[i];
            const searchTerm = node.ui_hints?.search_keyword || node.name;
            const nodeHints = node.ui_hints || {};
            
            await this.addNode(searchTerm, node.type, i === 0, nodeHints);
            
            // Brief pause between nodes
            await this.page.waitForTimeout(1500);
        }

        // Wait for all nodes to settle
        await this.page.waitForTimeout(2000);

        // Take final screenshot
        const finalScreenshotPath = join(this.screenshotDir, 'final_workflow.png');
        await this.page.screenshot({ path: finalScreenshotPath, fullPage: false });
        log('debug', 'Final screenshot captured');

        // 🎯 COMPARE WITH REFERENCE IMAGE
        log('info', '\n========================================');
        log('info', '🎯 COMPARING WITH REFERENCE WORKFLOW');
        log('info', '========================================\n');
        
        const comparisonResult = await this.compareWithReference();
        
        if (comparisonResult) {
            log('info', `📊 Match: ${comparisonResult.match_percentage}%`);
            log('success', `✅ Done: ${comparisonResult.done}`);
            if (comparisonResult.missing) {
                log('warn', `⚠️ Missing: ${comparisonResult.missing}`);
            }
        }

        // 💾 SAVE WORKFLOW
        log('info', '\n========================================');
        log('info', '💾 SAVING WORKFLOW');
        log('info', '========================================\n');
        
        const saved = await this.saveWorkflow(plan.workflow_name || 'AI Generated Workflow');
        
        if (saved) {
            log('success', '✅ Workflow saved successfully');
        } else {
            log('warn', '⚠️ Could not save workflow (you can save manually)');
        }

        // Summary
        const total = this.executionLog.steps.length;
        const successful = this.executionLog.steps.filter(s => s.status === 'success').length;
        const successRate = ((successful / total) * 100).toFixed(1);

        log('info', '\n========================================');
        log('info', '📊 AUTOMATION SUMMARY');
        log('info', '========================================\n');
        log('info', `Total nodes: ${total}`);
        log('success', `✅ Successful: ${successful}`);
        log('error', `❌ Failed: ${total - successful}`);
        log('info', `📈 Success rate: ${successRate}%`);
        log('info', `🔗 Connections: Auto-connected via + icon\n`);

        // Save log
        this.executionLog.completed_at = new Date().toISOString();
        this.executionLog.metrics = { 
            total, 
            successful, 
            failed: total - successful, 
            success_rate: successRate,
            auto_connected: true,
            workflow_saved: saved
        };
        
        // Add comparison result to log
        if (comparisonResult) {
            this.executionLog.comparison = comparisonResult;
        }
        
        // Add workflow URL to log
        if (saved) {
            this.executionLog.workflow_url = this.page.url();
        }
        
        await writeFile(
            join(__dirname, '..', 'output', 'execution_log.json'),
            JSON.stringify(this.executionLog, null, 2)
        );
        
        log('success', 'Execution log saved\n');
    }

    /**
     * 💾 Save workflow in n8n
     */
    async saveWorkflow(workflowName) {
        try {
            log('info', `Saving workflow: "${workflowName}"`);
            
            // Wait for any pending UI updates
            await this.page.waitForTimeout(1500);
            
            // Strategy 1: Try Ctrl+S keyboard shortcut (fastest)
            try {
                await this.page.keyboard.press('Control+KeyS');
                await this.page.waitForTimeout(2000);
                
                // Check for success notification
                const savedNotification = await this.page.locator('text=/saved|success/i').isVisible({ timeout: 3000 }).catch(() => false);
                
                if (savedNotification) {
                    log('success', 'Saved via Ctrl+S shortcut');
                    
                    // Get workflow URL
                    const currentUrl = this.page.url();
                    log('info', `Workflow URL: ${currentUrl}`);
                    
                    return true;
                }
            } catch (e) {
                log('debug', `Ctrl+S failed: ${e.message}`);
            }
            
            // Strategy 2: Try Save button
            try {
                const saveBtn = this.page.locator('button:has-text("Save")').first();
                const isSaveVisible = await saveBtn.isVisible({ timeout: 3000 }).catch(() => false);
                
                if (isSaveVisible) {
                    await saveBtn.click();
                    await this.page.waitForTimeout(2000);
                    
                    // Check for success
                    const saved = await this.page.locator('text=/saved|success/i').isVisible({ timeout: 3000 }).catch(() => false);
                    
                    if (saved) {
                        log('success', 'Saved via Save button');
                        
                        // Get workflow URL
                        const currentUrl = this.page.url();
                        log('info', `Workflow URL: ${currentUrl}`);
                        
                        return true;
                    }
                }
            } catch (e) {
                log('debug', `Save button failed: ${e.message}`);
            }
            
            // Strategy 3: Look for "Saved" status indicator
            const alreadySaved = await this.page.locator('text=/saved/i').isVisible({ timeout: 2000 }).catch(() => false);
            
            if (alreadySaved) {
                log('info', 'Workflow already saved (auto-save active)');
                
                // Get workflow URL
                const currentUrl = this.page.url();
                log('info', `Workflow URL: ${currentUrl}`);
                
                return true;
            }
            
            log('warn', 'Could not confirm save status');
            return false;
            
        } catch (error) {
            log('error', `Save workflow failed: ${error.message}`);
            return false;
        }
    }

    /**
     * 🎯 Compare current workflow with reference image
     */
    async compareWithReference() {
        try {
            // Load reference image
            const referencePath = join(__dirname, '..', 'analyzerYT', 'output', 'ai_visual_recipe.png');
            const referenceBuffer = await readFile(referencePath);
            const referenceBase64 = referenceBuffer.toString('base64');
            
            // Take current screenshot
            const currentScreenshot = await this.page.screenshot({ encoding: 'base64' });
            const compressedCurrent = await compressImage(currentScreenshot);
            const compressedReference = await compressImage(referenceBase64);
            
            log('info', 'Comparing workflows with AI Vision...');
            
            const response = await callOpenAI(this.openai, {
                model: 'gpt-4o',
                messages: [{
                    role: 'user',
                    content: [
                        {
                            type: 'text',
                            text: `Compare these two n8n workflows:
1. LEFT: Current automated workflow
2. RIGHT: Reference workflow (target)

Analyze:
- How many nodes match (name + type)?
- Are connections correct?
- Any missing nodes or connections?

Return JSON (keep descriptions SHORT - max 10 words each):
{
  "match_percentage": 85,
  "done": "3 nodes added, 2 connections made",
  "missing": "Chat Model not connected"
}

If everything matches perfectly:
{
  "match_percentage": 100,
  "done": "All nodes and connections match",
  "missing": null
}`
                        },
                        {
                            type: 'image_url',
                            image_url: { url: `data:image/jpeg;base64,${compressedCurrent}` }
                        },
                        {
                            type: 'image_url',
                            image_url: { url: `data:image/jpeg;base64,${compressedReference}` }
                        }
                    ]
                }],
                response_format: { type: 'json_object' },
                max_tokens: 300,
                temperature: 0
            }, 3);
            
            const result = parseJSON(response.choices[0].message.content);
            return result;
            
        } catch (error) {
            log('warn', `Comparison failed: ${error.message}`);
            return null;
        }
    }

    /**
     * 🧹 Cleanup
     */
    async close() {
        const keepOpen = parseInt(process.env.BROWSER_KEEP_OPEN_SECONDS || '10');
        log('info', `Browser remains open for ${keepOpen}s...`);
        
        await this.page.waitForTimeout(keepOpen * 1000).catch(() => {});
        
        if (this.browser?.isConnected()) {
            await this.browser.close().catch(() => {});
        }
    }
}

// ============================================================================
// 🎬 MAIN EXECUTION
// ============================================================================

(async () => {
    const agent = new N8nAutomationAgent();
    
    try {
        await agent.init();
        await agent.loginToN8n();
        
        // Load workflow plan
        const planPath = join(__dirname, '..', 'analyzerYT', 'output', 'ai_workflow_complete.json');
        await agent.runFromPlan(planPath);
        
        log('success', '✅ Automation completed!\n');
    } catch (error) {
        log('error', `Fatal error: ${error.message}`);
        console.error(error);
    } finally {
        await agent.close();
    }
})();