/**
 * Robust Connection Test with AI Vision
 * ======================================
 * Paste workflow → Detect shapes → Connect with retry
 */

import { chromium } from 'playwright';
import { readFileSync, writeFileSync, rmSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import OpenAI from 'openai';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load credentials
dotenv.config({ path: join(__dirname, '../../config/credentials.env') });

const config = {
    projectId: process.env.PROJECT_ID,
    apiKey: process.env.API_KEY,
    n8nUrl: process.env.N8N_URL,
    n8nEmail: process.env.N8N_EMAIL,
    n8nPassword: process.env.N8N_PASSWORD,
    openaiKey: process.env.OPENAI_API_KEY
};

const openai = new OpenAI({ apiKey: config.openaiKey });

let screenshotCounter = 1;

// Workflow JSON
const WORKFLOW_JSON = {
    "nodes": [
        {
            "parameters": { "updates": ["account_review_update"], "options": {} },
            "type": "n8n-nodes-base.whatsAppTrigger",
            "typeVersion": 1,
            "position": [-336, 16],
            "id": "8a6e99d6-e19b-426c-82ab-98ff2dd259e4",
            "name": "WhatsApp Trigger",
            "webhookId": "b668885f-4fec-41be-a158-d554b79b8bd0"
        },
        {
            "parameters": { "options": {} },
            "type": "@n8n/n8n-nodes-langchain.agent",
            "typeVersion": 2.2,
            "position": [0, 0],
            "id": "782468ec-699e-41f0-8061-d6b1a7a823b4",
            "name": "AI Agent"
        },
        {
            "parameters": { "options": {} },
            "type": "@n8n/n8n-nodes-langchain.lmChatOpenRouter",
            "typeVersion": 1,
            "position": [0, 208],
            "id": "40000f1d-6612-4748-8119-fe2ae290d599",
            "name": "OpenRouter Chat Model"
        },
        {
            "parameters": { "operation": "send", "additionalFields": {} },
            "type": "n8n-nodes-base.whatsApp",
            "typeVersion": 1,
            "position": [304, 0],
            "id": "d960fbd1-1914-4627-bec1-4bc548b50bae",
            "name": "Send message",
            "webhookId": "21634b01-99f4-4a5b-9a5b-1f55ea407f72"
        }
    ],
    "connections": {},
    "pinData": {},
    "meta": { "instanceId": "8f3920c1c387278711fa29fa8ace65a47b0f874e4e33452f814d9bd09129cbf5" }
};

// Connection map
const CONNECTION_MAP = JSON.parse(readFileSync('./connection_map.json', 'utf-8'));

async function takeScreenshot(page, name) {
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, '_').substring(0, 50);
    const path = `./screenshots/${String(screenshotCounter).padStart(3, '0')}_${safeName}.png`;
    await page.screenshot({ path, fullPage: false });
    console.log(`   📸 ${path}`);
    screenshotCounter++;
    return path;
}

function cleanScreenshots() {
    try {
        rmSync('./screenshots', { recursive: true, force: true });
        mkdirSync('./screenshots', { recursive: true });
        console.log('✅ Screenshots cleaned\n');
    } catch (e) {
        mkdirSync('./screenshots', { recursive: true });
    }
}

/**
 * 🔍 Find node using DOM (fast, no API calls)
 */
async function findNodeWithDOM(page, nodeName) {
    console.log(`   🔍 Finding "${nodeName}" in DOM...`);
    
    const nodeData = await page.evaluate((targetName) => {
        // Find all nodes
        const allNodes = document.querySelectorAll('[data-test-id^="canvas-node"], .vue-flow__node');
        
        let bestMatch = null;
        let bestMatchScore = 0;
        
        for (const nodeEl of allNodes) {
            const name = 
                nodeEl.getAttribute('data-name') ||
                nodeEl.textContent?.trim().split('\n')[0] ||
                '';
            
            // Score the match
            const nameLower = name.toLowerCase();
            const targetLower = targetName.toLowerCase();
            
            let score = 0;
            if (nameLower === targetLower) score = 100;  // Exact match
            else if (nameLower.includes(targetLower)) score = 50;  // Contains
            else if (targetLower.includes(nameLower)) score = 40;  // Partial
            
            // Bonus for key words
            if (targetLower.includes('whatsapp') && nameLower.includes('whatsapp')) score += 10;
            if (targetLower.includes('trigger') && nameLower.includes('trigger')) score += 10;
            if (targetLower.includes('agent') && nameLower.includes('agent')) score += 10;
            if (targetLower.includes('chat') && nameLower.includes('chat')) score += 10;
            if (targetLower.includes('model') && nameLower.includes('model')) score += 10;
            if (targetLower.includes('send') && nameLower.includes('send')) score += 10;
            
            if (score > bestMatchScore) {
                bestMatchScore = score;
                bestMatch = { nodeEl, name };
            }
        }
        
        if (bestMatch && bestMatchScore > 0) {
            const nodeEl = bestMatch.nodeEl;
            const name = bestMatch.name;
                
                const bbox = nodeEl.getBoundingClientRect();
                
                // Find handles
                const handles = {
                    outputs: [],
                    inputs: [],
                    plusIcons: []
                };
                
                // Output handles
                nodeEl.querySelectorAll('[data-handle-type="source"], circle').forEach(h => {
                    const hbox = h.getBoundingClientRect();
                    if (hbox.x > bbox.x + bbox.width * 0.6) {
                        handles.outputs.push({
                            x: hbox.x + hbox.width / 2,
                            y: hbox.y + hbox.height / 2
                        });
                    }
                });
                
                // Input handles  
                nodeEl.querySelectorAll('[data-handle-type="target"], circle').forEach(h => {
                    const hbox = h.getBoundingClientRect();
                    if (hbox.x < bbox.x + bbox.width * 0.4) {
                        handles.inputs.push({
                            x: hbox.x + hbox.width / 2,
                            y: hbox.y + hbox.height / 2
                        });
                    }
                });
                
                // Plus icons (look for text near bottom)
                nodeEl.querySelectorAll('svg, [data-test-id*="plus"]').forEach(el => {
                    const text = el.textContent || el.getAttribute('aria-label') || '';
                    if (text.toLowerCase().includes('chat') || text.toLowerCase().includes('model') || 
                        text.toLowerCase().includes('tool') || text.toLowerCase().includes('memory')) {
                        const ebox = el.getBoundingClientRect();
                        if (ebox.y > bbox.y + bbox.height * 0.5) {
                            handles.plusIcons.push({
                                label: text,
                                x: ebox.x + ebox.width / 2,
                                y: ebox.y + ebox.height / 2
                            });
                        }
                    }
                });
                
            return {
                found: true,
                name: name,
                node_center: {
                    x: bbox.x + bbox.width / 2,
                    y: bbox.y + bbox.height / 2
                },
                output_handle: handles.outputs[0] || null,
                input_handle: handles.inputs[0] || null,
                plus_icons: handles.plusIcons,
                match_score: bestMatchScore
            };
        }
        
        return { found: false };
    }, nodeName);
    
    if (nodeData.found) {
        console.log(`      ✅ Found: ${nodeData.name}`);
        console.log(`      Center: (${nodeData.node_center.x.toFixed(0)}, ${nodeData.node_center.y.toFixed(0)})`);
        if (nodeData.output_handle) {
            console.log(`      Output: (${nodeData.output_handle.x.toFixed(0)}, ${nodeData.output_handle.y.toFixed(0)})`);
        }
        if (nodeData.input_handle) {
            console.log(`      Input: (${nodeData.input_handle.x.toFixed(0)}, ${nodeData.input_handle.y.toFixed(0)})`);
        }
        if (nodeData.plus_icons?.length > 0) {
            nodeData.plus_icons.forEach(icon => {
                console.log(`      + icon: (${icon.x.toFixed(0)}, ${icon.y.toFixed(0)})`);
            });
        }
    } else {
        console.log(`      ❌ Not found`);
    }
    
    return nodeData;
}

/**
 * 🔗 Connect two nodes with retry logic
 */
async function connectNodesRobust(page, connectionSpec, maxRetries = 3) {
    const { from_node, to_node, connection_type, description, strategy } = connectionSpec;
    
    console.log(`\n🔗 Connection: ${from_node} → ${to_node}`);
    console.log(`   Type: ${connection_type}`);
    console.log(`   ${description}`);
    
    const stepName = `${from_node}_to_${to_node}`;
    await takeScreenshot(page, `before_${stepName}`);
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        console.log(`\n   🔄 Attempt ${attempt}/${maxRetries}`);
        
        try {
            // Step 1: Find source node
            const sourceNode = await findNodeWithDOM(page, from_node);
            if (!sourceNode.found) {
                throw new Error(`Source node "${from_node}" not found`);
            }
            
            await page.waitForTimeout(500);
            
            // Step 2: Find target node
            const targetNode = await findNodeWithDOM(page, to_node);
            if (!targetNode.found) {
                throw new Error(`Target node "${to_node}" not found`);
            }
            
            await page.waitForTimeout(500);
            
            // Step 3: Determine connection points
            let sourcePoint, targetPoint;
            
            if (connection_type === 'ai_languageModel') {
                // Special: Connect to AI Agent's Chat Model + icon
                const chatModelIcon = targetNode.plus_icons?.find(icon => 
                    icon.label.toLowerCase().includes('chat') || 
                    icon.label.toLowerCase().includes('model')
                );
                
                if (!chatModelIcon) {
                    throw new Error('Chat Model + icon not found on AI Agent');
                }
                
                sourcePoint = sourceNode.output_handle || sourceNode.node_center;
                targetPoint = chatModelIcon;
                
                console.log(`   📍 Connecting to + icon: "${chatModelIcon.label}"`);
            } else {
                // Regular main connection
                sourcePoint = sourceNode.output_handle || sourceNode.node_center;
                targetPoint = targetNode.input_handle || targetNode.node_center;
            }
            
            console.log(`   📍 Source: (${sourcePoint.x}, ${sourcePoint.y})`);
            console.log(`   📍 Target: (${targetPoint.x}, ${targetPoint.y})`);
            
            // Step 4: Perform drag connection
            await page.mouse.move(sourcePoint.x, sourcePoint.y);
            await page.waitForTimeout(300);
            
            await page.mouse.down();
            await page.waitForTimeout(200);
            
            await page.mouse.move(targetPoint.x, targetPoint.y, { steps: 30 });
            await page.waitForTimeout(300);
            
            await page.mouse.up();
            await page.waitForTimeout(1500);
            
            await takeScreenshot(page, `success_${stepName}`);
            
            // Success!
            console.log(`   ✅ Connection completed`);
            return true;
            
        } catch (error) {
            console.log(`   ❌ Attempt ${attempt} failed: ${error.message}`);
            
            if (attempt < maxRetries) {
                await page.waitForTimeout(2000);
            }
        }
    }
    
    console.log(`   ❌ All ${maxRetries} attempts failed`);
    await takeScreenshot(page, `failed_${stepName}`);
    return false;
}

/**
 * 🚀 Main test
 */
async function main() {
    console.log('🚀 Robust Connection Test with AI Vision\n');
    console.log('='*60 + '\n');
    
    cleanScreenshots();
    
    // Connect to Browserbase
    const browser = await chromium.connectOverCDP(
        `wss://connect.browserbase.com?apiKey=${config.apiKey}&projectId=${config.projectId}`
    );
    
    const context = browser.contexts()[0];
    const page = context.pages()[0];
    
    console.log('✅ Connected to Browserbase\n');
    
    try {
        // Step 1: Login
        console.log('🔐 Logging into n8n...');
        await page.goto(config.n8nUrl, { timeout: 60000 });
        await page.waitForTimeout(3000);
        await takeScreenshot(page, 'initial_page');
        
        const canvasVisible = await page.locator('.vue-flow__pane, svg').isVisible({ timeout: 3000 }).catch(() => false);
        
        if (!canvasVisible) {
            const emailInput = page.locator('input[type="email"]').first();
            await emailInput.fill(config.n8nEmail);
            await emailInput.press('Tab');
            await page.waitForTimeout(500);
            
            const passwordInput = page.locator('input[type="password"]').first();
            await passwordInput.fill(config.n8nPassword);
            await passwordInput.press('Enter');
            await page.waitForTimeout(5000);
        }
        
        await page.waitForSelector('.vue-flow__pane, svg', { timeout: 15000 });
        await page.waitForTimeout(2000);
        await takeScreenshot(page, 'canvas_ready');
        console.log('✅ Logged in\n');
        
        // Step 2: Paste workflow
        console.log('📋 Pasting workflow JSON...');
        
        await page.evaluate((data) => {
            navigator.clipboard.writeText(JSON.stringify(data));
        }, WORKFLOW_JSON);
        
        await page.waitForTimeout(1000);
        await page.locator('.vue-flow__pane, canvas, svg').first().click();
        await page.waitForTimeout(500);
        await page.keyboard.press('Control+v');
        await page.waitForTimeout(5000);
        
        await takeScreenshot(page, 'workflow_pasted');
        console.log('✅ Workflow pasted\n');
        
        // Step 3: Connect nodes
        console.log('='*60);
        console.log('🔗 CONNECTING NODES');
        console.log('='*60);
        
        const results = {
            total: CONNECTION_MAP.connections.length,
            successful: 0,
            failed: 0
        };
        
        for (const connection of CONNECTION_MAP.connections) {
            const success = await connectNodesRobust(page, connection);
            if (success) {
                results.successful++;
            } else {
                results.failed++;
            }
        }
        
        // Step 4: Save workflow
        console.log('\n💾 Saving workflow...');
        await page.keyboard.press('Control+s');
        await page.waitForTimeout(2000);
        await takeScreenshot(page, 'workflow_saved');
        console.log('✅ Saved\n');
        
        // Final summary
        console.log('='*60);
        console.log('📊 RESULTS');
        console.log('='*60);
        console.log(`   Total Connections: ${results.total}`);
        console.log(`   ✅ Successful: ${results.successful}`);
        console.log(`   ❌ Failed: ${results.failed}`);
        console.log(`   📈 Success Rate: ${((results.successful/results.total)*100).toFixed(0)}%`);
        console.log('='*60);
        
        await page.waitForTimeout(10000);
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        await takeScreenshot(page, 'error');
    } finally {
        await browser.close();
        console.log('\n✅ Test complete');
    }
}

main();

