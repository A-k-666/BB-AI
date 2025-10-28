/**
 * Vision-based Shape Detector
 * ============================
 * Uses GPT-4o Vision to find exact coordinates of connection shapes
 */

import { chromium } from 'playwright';
import { rmSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import OpenAI from 'openai';

const __dirname = dirname(fileURLToPath(import.meta.url));

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

// Model fallback chain (cheapest to most expensive)
const VISION_MODELS = ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'];

let screenshotCounter = 1;

const WORKFLOW_JSON = {
    "nodes": [
        {"parameters": {"updates": ["account_review_update"], "options": {}}, "type": "n8n-nodes-base.whatsAppTrigger", "typeVersion": 1, "position": [-336, 16], "id": "8a6e99d6-e19b-426c-82ab-98ff2dd259e4", "name": "WhatsApp Trigger", "webhookId": "b668885f-4fec-41be-a158-d554b79b8bd0"},
        {"parameters": {"options": {}}, "type": "@n8n/n8n-nodes-langchain.agent", "typeVersion": 2.2, "position": [0, 0], "id": "782468ec-699e-41f0-8061-d6b1a7a823b4", "name": "AI Agent"},
        {"parameters": {"options": {}}, "type": "@n8n/n8n-nodes-langchain.lmChatOpenRouter", "typeVersion": 1, "position": [0, 208], "id": "40000f1d-6612-4748-8119-fe2ae290d599", "name": "OpenRouter Chat Model"},
        {"parameters": {"operation": "send", "additionalFields": {}}, "type": "n8n-nodes-base.whatsApp", "typeVersion": 1, "position": [304, 0], "id": "d960fbd1-1914-4627-bec1-4bc548b50bae", "name": "Send message", "webhookId": "21634b01-99f4-4a5b-9a5b-1f55ea407f72"}
    ],
    "connections": {}, "pinData": {}, "meta": {"instanceId": "8f3920c1c387278711fa29fa8ace65a47b0f874e4e33452f814d9bd09129cbf5"}
};

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
 * 🎯 Find ALL connection shapes in the canvas
 */
async function findAllConnectionShapes(screenshotBase64) {
    console.log('\n🔍 Analyzing canvas for connection shapes...');
    
    // Try each model in fallback chain
    for (let modelIdx = 0; modelIdx < VISION_MODELS.length; modelIdx++) {
        const model = VISION_MODELS[modelIdx];
        console.log(`   Trying model: ${model}...`);
        
        try {
            const response = await openai.chat.completions.create({
                model: model,
                messages: [{
            role: 'user',
            content: [
                {
                    type: 'text',
                    text: `Analyze this n8n workflow canvas and find ALL connection shapes (circles, plus icons, handles).

For each node, identify:
1. Node name
2. OUTPUT shape (circle on RIGHT side of node) - exact pixel coordinates
3. INPUT shape (circle on LEFT side of node) - exact pixel coordinates  
4. PLUS icons (+ at bottom of node, if any) - with label and coordinates

Return JSON array:
{
  "nodes": [
    {
      "name": "WhatsApp Trigger",
      "output_shape": {"x": 150, "y": 300, "type": "circle"},
      "input_shape": null,
      "plus_icons": []
    },
    {
      "name": "AI Agent",
      "output_shape": {"x": 450, "y": 300, "type": "circle"},
      "input_shape": {"x": 350, "y": 300, "type": "circle"},
      "plus_icons": [
        {"label": "Chat Model", "x": 400, "y": 350},
        {"label": "Tool", "x": 420, "y": 350}
      ]
    },
    {
      "name": "OpenRouter Chat Model",
      "output_shape": {"x": 450, "y": 500, "type": "circle"},
      "input_shape": null,
      "plus_icons": []
    },
    {
      "name": "Send message",
      "output_shape": null,
      "input_shape": {"x": 550, "y": 300, "type": "rectangle"},
      "plus_icons": []
    }
  ]
}

IMPORTANT:
- Return EXACT pixel coordinates where you see the shapes
- output_shape is on the RIGHT edge of node
- input_shape is on the LEFT edge of node  
- plus_icons are at BOTTOM of node (look for "+")
- If shape doesn't exist, set to null`
                },
                {
                    type: 'image_url',
                    image_url: { url: `data:image/png;base64,${screenshotBase64}` }
                }
            ]
        }],
                response_format: { type: 'json_object' },
                max_tokens: 1000,
                temperature: 0
            });
            
            const result = JSON.parse(response.choices[0].message.content);
            
            console.log(`   ✅ Success! Found ${result.nodes.length} nodes:`);
            result.nodes.forEach(node => {
                console.log(`\n   📦 ${node.name}`);
                if (node.output_shape) {
                    console.log(`      ➡️  Output: (${node.output_shape.x}, ${node.output_shape.y})`);
                }
                if (node.input_shape) {
                    console.log(`      ⬅️  Input: (${node.input_shape.x}, ${node.input_shape.y})`);
                }
                if (node.plus_icons && node.plus_icons.length > 0) {
                    node.plus_icons.forEach(icon => {
                        console.log(`      ➕ ${icon.label}: (${icon.x}, ${icon.y})`);
                    });
                }
            });
            
            return result.nodes;
            
        } catch (error) {
            console.log(`   ❌ Model ${model} failed: ${error.message.substring(0, 100)}`);
            
            if (modelIdx < VISION_MODELS.length - 1) {
                console.log(`   ⏳ Trying next model...`);
                await new Promise(resolve => setTimeout(resolve, 2000));
            } else {
                console.log(`   ❌ All models failed`);
                throw error;
            }
        }
    }
}

/**
 * 🔗 Connect using exact shape coordinates
 */
async function connectShapes(page, fromCoord, toCoord, label) {
    console.log(`\n🔗 Connecting: ${label}`);
    console.log(`   From: (${fromCoord.x}, ${fromCoord.y})`);
    console.log(`   To: (${toCoord.x}, ${toCoord.y})`);
    
    const stepName = label.replace(/[^a-zA-Z0-9]/g, '_');
    await takeScreenshot(page, `before_${stepName}`);
    
    try {
        // Move to source shape
        await page.mouse.move(fromCoord.x, fromCoord.y);
        await page.waitForTimeout(500);
        
        // Press mouse down
        await page.mouse.down();
        await page.waitForTimeout(300);
        
        // Drag to target with smooth steps
        await page.mouse.move(toCoord.x, toCoord.y, { steps: 40 });
        await page.waitForTimeout(500);
        
        // Release
        await page.mouse.up();
        await page.waitForTimeout(2000);
        
        await takeScreenshot(page, `after_${stepName}`);
        console.log(`   ✅ Connection completed`);
        return true;
        
    } catch (error) {
        console.log(`   ❌ Failed: ${error.message}`);
        return false;
    }
}

/**
 * 🚀 Main
 */
async function main() {
    console.log('🚀 Vision Shape Detector Test\n');
    console.log('='*60 + '\n');
    
    cleanScreenshots();
    
    const browser = await chromium.connectOverCDP(
        `wss://connect.browserbase.com?apiKey=${config.apiKey}&projectId=${config.projectId}`
    );
    
    const context = browser.contexts()[0];
    const page = context.pages()[0];
    
    console.log('✅ Connected to Browserbase\n');
    
    try {
        // Login
        console.log('🔐 Logging in...');
        await page.goto(config.n8nUrl, { timeout: 60000 });
        await page.waitForTimeout(3000);
        await takeScreenshot(page, 'initial');
        
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
        
        // Paste workflow
        console.log('📋 Pasting workflow...');
        await page.evaluate((data) => {
            navigator.clipboard.writeText(JSON.stringify(data));
        }, WORKFLOW_JSON);
        
        await page.waitForTimeout(1000);
        await page.locator('.vue-flow__pane, canvas, svg').first().click();
        await page.waitForTimeout(500);
        await page.keyboard.press('Control+v');
        await page.waitForTimeout(5000);
        
        const workflowScreenshot = await takeScreenshot(page, 'workflow_pasted');
        console.log('✅ Workflow pasted\n');
        
        // Analyze shapes - use COMPRESSED screenshot of ONLY canvas area
        console.log('\n📸 Capturing canvas area only...');
        
        // Get canvas bounding box to crop
        const canvasBbox = await page.locator('.vue-flow__pane, canvas, svg').first().boundingBox();
        
        const screenshotBase64 = await page.screenshot({ 
            encoding: 'base64',
            quality: 50,  // Heavy compression
            type: 'jpeg',
            clip: canvasBbox || undefined  // Crop to canvas only
        });
        console.log(`   Image size: ~${(screenshotBase64.length / 1024).toFixed(0)} KB (compressed)`);
        
        const nodes = await findAllConnectionShapes(screenshotBase64);
        
        if (nodes.length === 0) {
            console.log('❌ No nodes detected');
            return;
        }
        
        // Make connections
        console.log('\n' + '='*60);
        console.log('🔗 MAKING CONNECTIONS');
        console.log('='*60);
        
        const results = { total: 0, successful: 0, failed: 0 };
        
        // Connection 1: WhatsApp Trigger → AI Agent
        const whatsappTrigger = nodes.find(n => n.name.toLowerCase().includes('whatsapp') && n.name.toLowerCase().includes('trigger'));
        const aiAgent = nodes.find(n => n.name.toLowerCase().includes('ai') && n.name.toLowerCase().includes('agent'));
        
        if (whatsappTrigger?.output_shape && aiAgent?.input_shape) {
            results.total++;
            const success = await connectShapes(
                page,
                whatsappTrigger.output_shape,
                aiAgent.input_shape,
                'WhatsApp_to_AI_Agent'
            );
            if (success) results.successful++;
            else results.failed++;
        }
        
        // Connection 2: OpenRouter Chat Model → AI Agent (Chat Model + icon)
        const chatModel = nodes.find(n => n.name.toLowerCase().includes('openrouter') || n.name.toLowerCase().includes('chat model'));
        const chatModelIcon = aiAgent?.plus_icons?.find(icon => 
            icon.label.toLowerCase().includes('chat') || 
            icon.label.toLowerCase().includes('model')
        );
        
        if (chatModel?.output_shape && chatModelIcon) {
            results.total++;
            const success = await connectShapes(
                page,
                chatModel.output_shape,
                chatModelIcon,
                'ChatModel_to_AI_Agent_plus'
            );
            if (success) results.successful++;
            else results.failed++;
        } else {
            console.log('\n⚠️  Chat Model + icon not found by Vision');
        }
        
        // Connection 3: AI Agent → Send message
        const sendMessage = nodes.find(n => n.name.toLowerCase().includes('send'));
        
        if (aiAgent?.output_shape && sendMessage?.input_shape) {
            results.total++;
            const success = await connectShapes(
                page,
                aiAgent.output_shape,
                sendMessage.input_shape,
                'AI_Agent_to_Send_message'
            );
            if (success) results.successful++;
            else results.failed++;
        }
        
        // Save
        console.log('\n💾 Saving...');
        await page.keyboard.press('Control+s');
        await page.waitForTimeout(2000);
        await takeScreenshot(page, 'saved');
        
        // Results
        console.log('\n' + '='*60);
        console.log('📊 RESULTS');
        console.log('='*60);
        console.log(`   Total: ${results.total}`);
        console.log(`   ✅ Success: ${results.successful}`);
        console.log(`   ❌ Failed: ${results.failed}`);
        console.log(`   📈 Rate: ${results.total > 0 ? ((results.successful/results.total)*100).toFixed(0) : 0}%`);
        console.log('='*60);
        
        await page.waitForTimeout(10000);
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        await takeScreenshot(page, 'error');
    } finally {
        await browser.close();
        console.log('\n✅ Complete');
    }
}

main();

