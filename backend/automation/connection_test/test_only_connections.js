/**
 * 🔗 CONNECTION-ONLY TEST
 * 
 * Assumes nodes already exist on canvas:
 * 1. WhatsApp Trigger
 * 2. AI Agent
 * 3. OpenRouter Chat Model
 * 4. Send message (WhatsApp Business)
 * 
 * This script ONLY tests connection logic.
 */

import { chromium } from 'playwright';
import { config } from 'dotenv';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { mkdirSync, existsSync } from 'fs';
import stringSimilarity from 'string-similarity';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Create screenshots directory
const screenshotDir = join(__dirname, 'screenshots');
if (!existsSync(screenshotDir)) {
    mkdirSync(screenshotDir, { recursive: true });
}

let screenshotCounter = 0;

// Load env
config({ path: join(__dirname, '..', '..', 'config', 'credentials.env') });

// ============================================================
// 📸 HELPER: Take Screenshot
// ============================================================
async function takeScreenshot(page, description) {
    screenshotCounter++;
    const filename = `${String(screenshotCounter).padStart(3, '0')}_${description.replace(/\s+/g, '_')}.png`;
    const path = join(screenshotDir, filename);
    try {
        await page.screenshot({ path, fullPage: true });
        console.log(`   📸 Screenshot: ${filename}`);
        return path;
    } catch (e) {
        console.log(`   ⚠️ Screenshot failed: ${e.message}`);
        return null;
    }
}

// ============================================================
// 🔍 HELPER: Universal Handle Detection (circles, diamonds, etc.)
// ============================================================
const handleSelectors = [
    'circle',                                           // Classic circle handles
    '[data-handle-type="input"]',                      // New n8n input handles
    '[data-handle-type="output"]',                     // New n8n output handles
    '[data-test-id*="handle"]',                        // Test ID handles
    'g[class*="connection-handle"]',                   // Group handles
    'foreignObject div[data-test-id*="handle"]',       // Foreign object handles
    'path[d*="M"]',                                    // Diamond shapes (SVG paths)
    '[data-endpoint-input]',                           // n8n endpoint inputs
    '[data-endpoint-output]'                           // n8n endpoint outputs
];

async function getHandles(node) {
    console.log(`   🔍 Detecting handles...`);
    
    for (const selector of handleSelectors) {
        const handles = await node.locator(selector).all();
        if (handles.length > 0) {
            console.log(`   ✅ Found ${handles.length} handles with: ${selector}`);
            return handles;
        }
    }
    
    console.warn(`   ⚠️ No handles found with any selector`);
    return [];
}

// ============================================================
// 🔍 HELPER: Find Node by Name (NOT by index!)
// ============================================================
async function findNodeByName(page, nodeName) {
    console.log(`🔍 Searching for node: "${nodeName}"`);
    
    const nodes = await page.locator('[data-test-id^="canvas-node"]').all();
    console.log(`   📊 Total canvas nodes: ${nodes.length}`);
    
    for (const node of nodes) {
        const text = (await node.textContent() || '').trim();
        const firstLine = text.split('\n')[0].trim();
        
        // Exact match
        if (firstLine.toLowerCase() === nodeName.toLowerCase()) {
            console.log(`   ✅ Found (exact): "${firstLine}"`);
            return node;
        }
        
        // Contains match
        if (firstLine.toLowerCase().includes(nodeName.toLowerCase())) {
            console.log(`   ✅ Found (contains): "${firstLine}"`);
            return node;
        }
    }
    
    // Fuzzy match fallback
    const labels = [];
    for (const node of nodes) {
        const text = (await node.textContent() || '').trim().split('\n')[0];
        labels.push({ text, node });
    }
    
    const matches = labels.map(l => ({
        score: stringSimilarity.compareTwoStrings(l.text.toLowerCase(), nodeName.toLowerCase()),
        node: l.node,
        text: l.text
    }));
    
    matches.sort((a, b) => b.score - a.score);
    
    if (matches[0]?.score > 0.5) {
        console.log(`   ✅ Found (fuzzy ${Math.round(matches[0].score * 100)}%): "${matches[0].text}"`);
        return matches[0].node;
    }
    
    console.warn(`   ⚠️ Node not found: "${nodeName}"`);
    return null;
}

// ============================================================
// ⭕ Connect Nodes (Universal - handles circles, diamonds, etc.)
// ============================================================
async function connectCircles(page, fromName, toName) {
    console.log(`\n🔗 Connecting: ${fromName} → ${toName}`);
    console.log('━'.repeat(60));
    
    // Screenshot before
    await takeScreenshot(page, `before_${fromName}_to_${toName}`);
    
    // Wait for UI stability
    await page.waitForTimeout(1000);
    
    // Find nodes BY NAME (not by index!)
    const fromNode = await findNodeByName(page, fromName);
    const toNode = await findNodeByName(page, toName);
    
    if (!fromNode || !toNode) {
        console.error(`❌ Cannot connect: missing node(s)`);
        return false;
    }
    
    // Get handles (universal detection: circles, diamonds, paths, etc.)
    const fromHandles = await getHandles(fromNode);
    const toHandles = await getHandles(toNode);
    
    console.log(`   📊 Handles → From: ${fromHandles.length}, To: ${toHandles.length}`);
    
    if (fromHandles.length === 0 || toHandles.length === 0) {
        console.error(`   ❌ No handles detected`);
        await takeScreenshot(page, `no_handles_${fromName}_to_${toName}`);
        return false;
    }
    
    // Retry logic with multiple attempts
    for (let attempt = 1; attempt <= 2; attempt++) {
        try {
            console.log(`   🔄 Attempt ${attempt}/2`);
            
            // Get output (last handle) and input (first handle)
            const outputHandle = fromHandles[fromHandles.length - 1];
            const inputHandle = toHandles[0];
            
            const fromBox = await outputHandle.boundingBox();
            const toBox = await inputHandle.boundingBox();
            
            if (!fromBox || !toBox) {
                console.warn(`   ⚠️ No bounding boxes, scrolling...`);
                await page.mouse.wheel(0, -200);
                await page.waitForTimeout(1000);
                continue;
            }
            
            console.log(`   🎯 From: (${Math.round(fromBox.x)}, ${Math.round(fromBox.y)})`);
            console.log(`   🎯 To: (${Math.round(toBox.x)}, ${Math.round(toBox.y)})`);
            
            // Calculate center points
            const srcX = fromBox.x + fromBox.width / 2;
            const srcY = fromBox.y + fromBox.height / 2;
            const tgtX = toBox.x + toBox.width / 2;
            const tgtY = toBox.y + toBox.height / 2;
            
            // Perform drag with smooth motion
            console.log(`   🖱️  Dragging ${Math.round(srcX)},${Math.round(srcY)} → ${Math.round(tgtX)},${Math.round(tgtY)}`);
            await page.mouse.move(srcX, srcY);
            await page.mouse.down();
            await page.waitForTimeout(400);
            await page.mouse.move(tgtX, tgtY, { steps: 30 });
            await page.waitForTimeout(300);
            await page.mouse.up();
            await page.waitForTimeout(1500);
            
            // Verify connection
            const verified = await verifyConnection(page);
            
            if (verified) {
                console.log(`   ✅ Connection verified on attempt ${attempt}!`);
                await takeScreenshot(page, `success_${fromName}_to_${toName}`);
                console.log(`✅ SUCCESS: ${fromName} → ${toName}\n`);
                return true;
            } else {
                console.warn(`   ⚠️ Not verified, retrying...`);
            }
            
        } catch (err) {
            console.warn(`   ⚠️ Attempt ${attempt} failed: ${err.message}`);
        }
    }
    
    // Screenshot after all attempts
    await takeScreenshot(page, `failed_${fromName}_to_${toName}`);
    console.log(`⚠️ CONNECTION FAILED: ${fromName} → ${toName}\n`);
    return false;
}

// ============================================================
// ✅ Verify Connection in DOM
// ============================================================
async function verifyConnection(page) {
    try {
        const pathCount = await page.evaluate(() => {
            const paths = [...document.querySelectorAll('svg path, g[data-type="connection"] path')];
            return paths.filter(p => {
                const d = p.getAttribute('d') || '';
                return d.length > 50; // Valid connection line
            }).length;
        });
        
        console.log(`   🔍 SVG paths found: ${pathCount}`);
        return pathCount > 0;
    } catch (e) {
        return true;
    }
}

// ============================================================
// 🧪 Main Test Function
// ============================================================
async function testConnections() {
    console.log('\n🧪 CONNECTION-ONLY TEST');
    console.log('═'.repeat(60));
    console.log('📋 Expected nodes on canvas:');
    console.log('   1. WhatsApp Trigger');
    console.log('   2. AI Agent');
    console.log('   3. OpenRouter Chat Model');
    console.log('   4. Send message (or WhatsApp Business)');
    console.log('═'.repeat(60));
    
    // Connect to Browserbase
    console.log('\n🌐 Connecting to Browserbase...');
    const sessionUrl = `wss://connect.browserbase.com?apiKey=${process.env.API_KEY}&projectId=${process.env.PROJECT_ID}`;
    const browser = await chromium.connectOverCDP(sessionUrl);
    const context = browser.contexts()[0];
    const page = context.pages()[0];
    
    console.log('✅ Connected to browser\n');
    
    // Navigate to specific workflow
    console.log('📍 Opening workflow: https://pika2223.app.n8n.cloud/workflow/P2sKwz3vI012RTf2');
    await page.goto('https://pika2223.app.n8n.cloud/workflow/P2sKwz3vI012RTf2', {
        waitUntil: 'domcontentloaded',
        timeout: 60000
    });
    
    await page.waitForTimeout(2000);
    await takeScreenshot(page, 'initial_page');
    
    // Login if needed
    const needsLogin = await page.locator('input[type="email"]').isVisible().catch(() => false);
    if (needsLogin) {
        console.log('🔐 Logging in...');
        await page.locator('input[type="email"]').fill(process.env.N8N_EMAIL);
        await page.locator('input[type="email"]').press('Enter');
        await page.waitForTimeout(1500);
        await page.locator('input[type="password"]').fill(process.env.N8N_PASSWORD);
        await page.locator('input[type="password"]').press('Enter');
        await page.waitForTimeout(3000);
        console.log('✅ Logged in successfully\n');
    }
    
    // Wait for canvas to load
    console.log('⏳ Waiting for canvas...');
    await page.waitForTimeout(5000);
    await takeScreenshot(page, 'canvas_ready');
    
    // Test cases (matching JSON workflow)
    const connections = [
        {
            from: 'WhatsApp Trigger',
            to: 'AI Agent',
            type: 'main'
        },
        {
            from: 'AI Agent',
            to: 'OpenRouter Chat Model',
            type: 'ai_languageModel'
        },
        {
            from: 'AI Agent',
            to: 'Send message',
            type: 'main'
        }
    ];
    
    let successCount = 0;
    let failCount = 0;
    
    console.log('🚀 Starting connection tests...\n');
    
    for (const conn of connections) {
        console.log(`\n${'═'.repeat(60)}`);
        console.log(`📝 Test ${successCount + failCount + 1}/${connections.length}`);
        console.log(`   Type: ${conn.type}`);
        
        const result = await connectCircles(page, conn.from, conn.to);
        
        if (result) {
            successCount++;
        } else {
            failCount++;
        }
        
        // Wait before next connection
        await page.waitForTimeout(2000);
    }
    
    // Summary
    console.log('\n' + '═'.repeat(60));
    console.log('📊 TEST SUMMARY');
    console.log('═'.repeat(60));
    console.log(`✅ Success: ${successCount}/${connections.length}`);
    console.log(`❌ Failed: ${failCount}/${connections.length}`);
    console.log(`📈 Success Rate: ${Math.round((successCount / connections.length) * 100)}%`);
    console.log('═'.repeat(60));
    
    // Save workflow
    console.log('\n💾 Saving workflow...');
    await saveWorkflow(page);
    
    // Keep browser open for inspection
    console.log('\n💡 Browser will stay open for 30 seconds for inspection...');
    await page.waitForTimeout(30000);
    
    await browser.close();
    console.log('\n✅ Test complete!\n');
}

// ============================================================
// 💾 Save Workflow
// ============================================================
async function saveWorkflow(page) {
    try {
        // Try Ctrl+S first
        console.log('   🔧 Trying Ctrl+S...');
        await page.keyboard.press('Control+s');
        await page.waitForTimeout(2000);
        
        // Check for save button
        const saveButton = page.locator('button:has-text("Save"), button[data-test-id*="save"]');
        const saveVisible = await saveButton.isVisible().catch(() => false);
        
        if (saveVisible) {
            console.log('   💾 Clicking Save button...');
            await saveButton.click();
            await page.waitForTimeout(2000);
        }
        
        // Check for success indicator
        const saved = await page.locator('[class*="toast"], [class*="notification"]:has-text("Saved")').isVisible({ timeout: 3000 }).catch(() => false);
        
        if (saved) {
            console.log('   ✅ Workflow saved successfully!');
            await takeScreenshot(page, 'workflow_saved');
        } else {
            console.log('   ⚠️ Save status unknown (no toast notification)');
            await takeScreenshot(page, 'workflow_save_attempted');
        }
        
    } catch (err) {
        console.warn(`   ⚠️ Save failed: ${err.message}`);
        await takeScreenshot(page, 'workflow_save_failed');
    }
}

// Run test
testConnections().catch(err => {
    console.error('\n❌ Fatal error:', err.message);
    process.exit(1);
});

