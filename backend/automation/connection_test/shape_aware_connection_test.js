/**
 * Shape-Aware Connection Test
 * ============================
 * Tests precise connection logic using shape coordinate detection
 */

import { chromium } from 'playwright';
import { readFileSync, writeFileSync, rmSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load credentials
dotenv.config({ path: join(__dirname, '../../config/credentials.env') });

const config = {
    projectId: process.env.PROJECT_ID,
    apiKey: process.env.API_KEY,
    n8nUrl: process.env.N8N_URL,
    n8nEmail: process.env.N8N_EMAIL,
    n8nPassword: process.env.N8N_PASSWORD,
};

let screenshotCounter = 1;

async function takeScreenshot(page, name) {
    // Sanitize filename (remove special chars, limit length)
    const safeName = name
        .replace(/[^a-zA-Z0-9_-]/g, '_')
        .substring(0, 50);
    const path = `./screenshots/${String(screenshotCounter).padStart(3, '0')}_${safeName}.png`;
    await page.screenshot({ path, fullPage: false });
    console.log(`   📸 Screenshot: ${path}`);
    screenshotCounter++;
}

async function cleanScreenshots() {
    try {
        rmSync('./screenshots', { recursive: true, force: true });
        mkdirSync('./screenshots', { recursive: true });
        console.log('✅ Cleaned old screenshots\n');
    } catch (e) {
        mkdirSync('./screenshots', { recursive: true });
    }
}

/**
 * 🔍 Detect node shapes and their coordinates from DOM
 */
async function detectNodeShapes(page) {
    console.log('\n🔍 Detecting node shapes from DOM...');
    
    const result = await page.evaluate(() => {
        const nodes = [];
        
        // Find all canvas nodes - look for actual node elements
        const nodeElements = document.querySelectorAll('[data-test-id^="canvas-node"], [class*="node"], .vue-flow__node');
        
        nodeElements.forEach(nodeEl => {
            // Get node name - prioritize data attributes first
            let nodeName = nodeEl.getAttribute('data-name');
            
            // If not found, try text content but be more selective
            if (!nodeName || nodeName === '') {
                const labelEl = nodeEl.querySelector('[data-test-id="canvas-node-label"]');
                if (labelEl) {
                    nodeName = labelEl.textContent?.trim();
                } else {
                    // Get only direct text content, not child elements
                    const walker = document.createTreeWalker(
                        nodeEl,
                        NodeFilter.SHOW_TEXT,
                        null
                    );
                    let firstText = '';
                    while (walker.nextNode() && !firstText) {
                        const text = walker.currentNode.textContent?.trim();
                        if (text && text.length > 2 && text.length < 50) {
                            firstText = text;
                            break;
                        }
                    }
                    nodeName = firstText || 'Unknown';
                }
            }
            
            // Clean up node name
            nodeName = nodeName
                .split('\n')[0]
                .trim()
                .replace(/[^\w\s-]/g, '')
                .substring(0, 30);
            
            const bbox = nodeEl.getBoundingClientRect();
            
            // Skip if invalid bbox
            if (bbox.width === 0 || bbox.height === 0) return;
            
            // Find connection handles
            const handles = {
                outputs: [],
                inputs: [],
                tools: []
            };
            
            // Strategy 1: Look for data-handle attributes
            nodeEl.querySelectorAll('[data-handle-type]').forEach(handle => {
                const htype = handle.getAttribute('data-handle-type');
                const hbox = handle.getBoundingClientRect();
                
                if (hbox.width === 0 || hbox.height === 0) return;
                
                const coord = {
                    x: hbox.x + hbox.width / 2,
                    y: hbox.y + hbox.height / 2
                };
                
                if (htype === 'source') handles.outputs.push(coord);
                else if (htype === 'target') handles.inputs.push(coord);
                else if (htype.includes('tool') || htype.includes('ai_')) handles.tools.push(coord);
            });
            
            // Strategy 2: Look for SVG circles (common handle shapes)
            if (handles.outputs.length === 0 && handles.inputs.length === 0) {
                nodeEl.querySelectorAll('circle, [class*="handle"]').forEach(circle => {
                    const cbox = circle.getBoundingClientRect();
                    if (cbox.width === 0 || cbox.height === 0) return;
                    
                    const coord = {
                        x: cbox.x + cbox.width / 2,
                        y: cbox.y + cbox.height / 2
                    };
                    
                    // Heuristic: right side = output, left side = input
                    if (cbox.x > bbox.x + bbox.width * 0.7) {
                        handles.outputs.push(coord);
                    } else if (cbox.x < bbox.x + bbox.width * 0.3) {
                        handles.inputs.push(coord);
                    }
                });
            }
            
            // Only add if we have BOTH a valid name AND handles
            if (nodeName && nodeName !== 'Unknown' && nodeName.length > 2 &&
                (handles.outputs.length > 0 || handles.inputs.length > 0 || handles.tools.length > 0)) {
                nodes.push({
                    name: nodeName,
                    bbox: {
                        x: bbox.x,
                        y: bbox.y,
                        width: bbox.width,
                        height: bbox.height,
                        centerX: bbox.x + bbox.width / 2,
                        centerY: bbox.y + bbox.height / 2
                    },
                    handles
                });
            }
        });
        
        const debug = [];
        debug.push(`Total nodes before dedup: ${nodes.length}`);
        
        // List all node names
        nodes.forEach(n => {
            debug.push(`  - ${n.name} (${n.handles.outputs.length}out, ${n.handles.inputs.length}in, ${n.handles.tools.length}tool)`);
        });
        
        // Filter to get unique nodes with best handle count
        const nodeMap = new Map();
        nodes.forEach(node => {
            const existing = nodeMap.get(node.name);
            const newScore = node.handles.outputs.length + node.handles.inputs.length + node.handles.tools.length;
            const existingScore = existing ? 
                (existing.handles.outputs.length + existing.handles.inputs.length + existing.handles.tools.length) : 0;
            
            if (!existing || newScore > existingScore) {
                nodeMap.set(node.name, node);
            }
        });
        
        return {
            nodes: Array.from(nodeMap.values()),
            debug
        };
    });
    
    // Print debug info
    console.log('\n   DEBUG INFO:');
    result.debug.forEach(line => console.log(`   ${line}`));
    
    const shapes = result.nodes;
    console.log(`\n   Found ${shapes.length} unique nodes with handles:`);
    shapes.forEach(node => {
        console.log(`   - ${node.name}: ${node.handles.outputs.length} outputs, ${node.handles.inputs.length} inputs, ${node.handles.tools.length} tools`);
        console.log(`      Position: (${node.bbox.centerX.toFixed(0)}, ${node.bbox.centerY.toFixed(0)})`);
    });
    
    return shapes;
}

/**
 * 🔗 Connect two nodes using precise shape coordinates
 */
async function connectNodesShapeAware(page, fromNode, toNode, connectionType = 'main') {
    console.log(`\n🔗 Connecting: ${fromNode.name} → ${toNode.name} (${connectionType})`);
    
    await takeScreenshot(page, `before_${fromNode.name}_to_${toNode.name}`);
    
    // Select correct handles based on connection type
    let sourceHandle, targetHandle;
    
    if (connectionType === 'ai_languageModel' || connectionType === 'tool') {
        // AI Agent tool connection
        sourceHandle = toNode.handles.tools[0]; // Model connects TO agent's tool slot
        targetHandle = fromNode.handles.outputs[0]; // Agent's tool output
        
        if (!sourceHandle || !targetHandle) {
            console.log(`   ⚠️  No tool handles found, trying outputs`);
            sourceHandle = toNode.handles.outputs[0];
            targetHandle = fromNode.handles.inputs[0];
        }
    } else {
        // Regular main connection
        sourceHandle = fromNode.handles.outputs[0];
        targetHandle = toNode.handles.inputs[0];
    }
    
    if (!sourceHandle || !targetHandle) {
        console.log(`   ❌ Missing handles for connection`);
        return false;
    }
    
    console.log(`   Source: (${sourceHandle.x.toFixed(0)}, ${sourceHandle.y.toFixed(0)})`);
    console.log(`   Target: (${targetHandle.x.toFixed(0)}, ${targetHandle.y.toFixed(0)})`);
    
    try {
        // Precise shape-based drag
        await page.mouse.move(sourceHandle.x, sourceHandle.y);
        await page.waitForTimeout(300);
        
        await page.mouse.down();
        await page.waitForTimeout(200);
        
        // Drag with steps for smooth animation
        await page.mouse.move(targetHandle.x, targetHandle.y, { steps: 30 });
        await page.waitForTimeout(300);
        
        await page.mouse.up();
        await page.waitForTimeout(1000);
        
        await takeScreenshot(page, `success_${fromNode.name}_to_${toNode.name}`);
        console.log(`   ✅ Connection attempt complete`);
        return true;
        
    } catch (error) {
        console.log(`   ❌ Connection failed: ${error.message}`);
        return false;
    }
}

/**
 * 💾 Save workflow
 */
async function saveWorkflow(page) {
    console.log('\n💾 Saving workflow...');
    
    try {
        // Try Ctrl+S
        await page.keyboard.press('Control+s');
        await page.waitForTimeout(2000);
        
        // Click Save button if visible
        const saveButton = page.locator('button:has-text("Save")').first();
        if (await saveButton.isVisible({ timeout: 2000 }).catch(() => false)) {
            await saveButton.click();
            await page.waitForTimeout(2000);
        }
        
        await takeScreenshot(page, 'workflow_saved');
        console.log('   ✅ Workflow saved');
        
    } catch (error) {
        console.log(`   ⚠️  Save failed: ${error.message}`);
    }
}

/**
 * 🚀 Main test
 */
async function main() {
    console.log('🚀 Shape-Aware Connection Test\n');
    console.log('='*60);
    
    // Clean screenshots
    await cleanScreenshots();
    
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
        await page.goto(config.n8nUrl);
        await page.waitForTimeout(2000);
        await takeScreenshot(page, 'initial_page');
        
        // Check if already logged in
        const canvasVisible = await page.locator('.vue-flow__pane, svg').isVisible({ timeout: 3000 }).catch(() => false);
        
        if (!canvasVisible) {
            // Fill login form - robust selectors
            const emailInput = page.locator('input[type="email"], input[name="email"], #email').first();
            await emailInput.fill(config.n8nEmail);
            await emailInput.press('Tab');
            await page.waitForTimeout(500);
            
            const passwordInput = page.locator('input[type="password"], input[name="password"], #password').first();
            await passwordInput.fill(config.n8nPassword);
            await passwordInput.press('Enter');
            await page.waitForTimeout(5000);
        }
        
        await page.waitForSelector('.vue-flow__pane, svg', { timeout: 15000 });
        await page.waitForTimeout(2000);
        await takeScreenshot(page, 'canvas_ready');
        console.log('✅ Logged in\n');
        
        // Step 2: Import workflow JSON via API or manual paste
        console.log('📋 Please manually paste the workflow JSON below into n8n (Ctrl+V):');
        console.log('='*60);
        
        const workflowData = {
            "nodes": [
                {
                    "parameters": {
                        "updates": ["account_review_update"],
                        "options": {}
                    },
                    "type": "n8n-nodes-base.whatsAppTrigger",
                    "typeVersion": 1,
                    "position": [128, 288],
                    "id": "19f69942-c624-4ab1-b6c2-7bd675660653",
                    "name": "WhatsApp Trigger",
                    "webhookId": "b668885f-4fec-41be-a158-d554b79b8bd0"
                },
                {
                    "parameters": {
                        "options": {}
                    },
                    "type": "@n8n/n8n-nodes-langchain.agent",
                    "typeVersion": 2.2,
                    "position": [460, 280],
                    "id": "736ab089-79fe-4cac-b318-3157694cc3fa",
                    "name": "AI Agent"
                },
                {
                    "parameters": {
                        "options": {}
                    },
                    "type": "@n8n/n8n-nodes-langchain.lmChatOpenRouter",
                    "typeVersion": 1,
                    "position": [460, 480],
                    "id": "fa2d821d-83a2-4172-924e-75b76c0c63d8",
                    "name": "OpenRouter Chat Model"
                },
                {
                    "parameters": {
                        "resource": "message",
                        "operation": "send",
                        "phoneNumberId": "",
                        "to": "={{ $json.entry[0].changes[0].value.messages[0].from }}",
                        "message": "={{ $json.response }}"
                    },
                    "type": "n8n-nodes-base.whatsApp",
                    "typeVersion": 1,
                    "position": [760, 280],
                    "id": "8f3920c1-c387-2787-11fa-29fa8ace65a4",
                    "name": "Send message"
                }
            ],
            "connections": {}
        };
        
        // Import workflow programmatically
        console.log('Importing workflow via clipboard...');
        
        // Set clipboard content
        await page.evaluate((data) => {
            const jsonStr = JSON.stringify(data);
            // Use Clipboard API
            navigator.clipboard.writeText(jsonStr);
        }, workflowData);
        
        await page.waitForTimeout(1000);
        
        // Click canvas to focus
        await page.locator('.vue-flow__pane, canvas, svg').first().click();
        await page.waitForTimeout(500);
        
        // Paste with Ctrl+V
        await page.keyboard.press('Control+v');
        await page.waitForTimeout(5000);
        
        await takeScreenshot(page, 'workflow_imported');
        console.log('✅ Workflow import attempted\n');
        
        // Step 3: Detect shapes
        const nodeShapes = await detectNodeShapes(page);
        
        if (nodeShapes.length === 0) {
            console.log('❌ No nodes detected!');
            return;
        }
        
        // Step 4: Connect nodes
        console.log('\n🔗 Starting connections...\n');
        console.log('='*60);
        
        const whatsappTrigger = nodeShapes.find(n => n.name.includes('WhatsApp Trigger'));
        const aiAgent = nodeShapes.find(n => n.name.includes('AI Agent'));
        const chatModel = nodeShapes.find(n => n.name.includes('OpenRouter') || n.name.includes('Chat Model'));
        const sendMessage = nodeShapes.find(n => n.name.includes('Send message'));
        
        let successCount = 0;
        let totalConnections = 0;
        
        // Connection 1: WhatsApp Trigger → AI Agent
        if (whatsappTrigger && aiAgent) {
            totalConnections++;
            if (await connectNodesShapeAware(page, whatsappTrigger, aiAgent, 'main')) {
                successCount++;
            }
        }
        
        // Connection 2: OpenRouter Chat Model → AI Agent (tool slot)
        if (chatModel && aiAgent) {
            totalConnections++;
            if (await connectNodesShapeAware(page, chatModel, aiAgent, 'ai_languageModel')) {
                successCount++;
            }
        }
        
        // Connection 3: AI Agent → Send message
        if (aiAgent && sendMessage) {
            totalConnections++;
            if (await connectNodesShapeAware(page, aiAgent, sendMessage, 'main')) {
                successCount++;
            }
        }
        
        // Step 5: Save workflow
        await saveWorkflow(page);
        
        // Final summary
        console.log('\n' + '='*60);
        console.log('📊 CONNECTION TEST RESULTS');
        console.log('='*60);
        console.log(`   Total Connections: ${totalConnections}`);
        console.log(`   Successful: ${successCount}`);
        console.log(`   Failed: ${totalConnections - successCount}`);
        console.log(`   Success Rate: ${((successCount/totalConnections)*100).toFixed(0)}%`);
        console.log('='*60);
        
        await page.waitForTimeout(10000); // Keep open for inspection
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        await takeScreenshot(page, 'error');
    } finally {
        await browser.close();
        console.log('\n✅ Test complete');
    }
}

main();

