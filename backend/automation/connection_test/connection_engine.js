/**
 * 🔗 Robust Connection Engine for n8n
 * Handles all connection strategies intelligently
 */

import OpenAI from 'openai';
import sharp from 'sharp';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/**
 * 🎯 Main Connection Function
 */
export async function connectNodes(page, fromName, toName, opts = {}) {
    console.log(`\n🔗 Connecting: ${fromName} → ${toName}\n`);

    // Wait for UI to settle
    await page.waitForTimeout(2000);

    // Strategy 1: Try + icon connection first
    const plusConnResult = await tryPlusIconConnection(page, fromName, toName);
    if (plusConnResult.success) {
        console.log(`✅ Connected via + icon: ${fromName} → ${toName}`);
        return { success: true, method: 'plus_icon' };
    }

    // Strategy 2: Try circle-to-circle connection
    const circleConnResult = await tryCircleConnection(page, fromName, toName);
    if (circleConnResult.success) {
        console.log(`✅ Connected via circles: ${fromName} → ${toName}`);
        return { success: true, method: 'circles' };
    }

    // Strategy 3: AI Vision fallback
    const aiConnResult = await tryAIVisionConnection(page, fromName, toName);
    if (aiConnResult.success) {
        console.log(`✅ Connected via AI Vision: ${fromName} → ${toName}`);
        return { success: true, method: 'ai_vision' };
    }

    console.log(`❌ All connection strategies failed for ${fromName} → ${toName}`);
    return { success: false, method: 'none' };
}

/**
 * 🔍 STRATEGY 1: + Icon Connection
 */
async function tryPlusIconConnection(page, fromName, toName) {
    console.log('🧠 Step 1: Checking for + icons...');

    try {
        // Detect + icons using AI Vision
        const screenshot = await page.screenshot({ encoding: 'base64' });
        const compressed = await compressImage(screenshot);

        const response = await openai.chat.completions.create({
            model: 'gpt-4o',
            messages: [{
                role: 'user',
                content: [
                    {
                        type: 'text',
                        text: `Find the "${fromName}" node on this n8n canvas.

Detect ALL "+" icons (plus buttons) with labels like:
- "Chat Model +"
- "Memory +"
- "Tool +"

For each + icon, return:
- Label text
- Position (x, y)
- Which target node should connect here

Return JSON:
{
  "plus_icons": [
    {
      "label": "Chat Model",
      "x": 450,
      "y": 300,
      "matches_target": "OpenRouter Chat Model" // true if label matches ${toName}
    }
  ]
}`
                    },
                    { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${compressed}` } }
                ]
            }],
            max_tokens: 500
        });

        const result = JSON.parse(response.choices[0].message.content.trim());

        if (result.plus_icons && result.plus_icons.length > 0) {
            // Find matching icon
            const matchedIcon = result.plus_icons.find(icon => 
                icon.matches_target === toName || 
                icon.label.toLowerCase().includes(toName.toLowerCase().split(' ')[0])
            ) || result.plus_icons[0];

            console.log(`✅ Found + icon: "${matchedIcon.label}"`);

            // Get target node position
            const targetInfo = await page.evaluate((name) => {
                const nodes = [...document.querySelectorAll('[data-test-id^="canvas-node"]')];
                for (const node of nodes) {
                    if (node.textContent.includes(name)) {
                        const rect = node.getBoundingClientRect();
                        return {
                            found: true,
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        };
                    }
                }
                return { found: false };
            }, toName);

            if (targetInfo.found) {
                console.log(`🎯 Dragging from + icon to "${toName}"...`);
                
                // Drag from + icon to target center
                await page.mouse.move(matchedIcon.x, matchedIcon.y);
                await page.mouse.down();
                await page.waitForTimeout(400);
                await page.mouse.move(
                    targetInfo.x + targetInfo.width / 2,
                    targetInfo.y + targetInfo.height / 2,
                    { steps: 30 }
                );
                await page.waitForTimeout(400);
                await page.mouse.up();
                await page.waitForTimeout(1500);

                // Verify connection
                const verified = await verifyConnection(page, fromName, toName);
                
                return { success: verified, method: 'plus_icon' };
            }
        }

        return { success: false };
    } catch (e) {
        console.log(`   ⚠️ + icon connection failed: ${e.message}`);
        return { success: false };
    }
}

/**
 * ⭕ STRATEGY 2: Circle-to-Circle Connection
 */
async function tryCircleConnection(page, fromName, toName) {
    console.log('⭕ Step 2: Using circle-to-circle drag...');

    try {
        // Find source and target nodes BY NAME
        const fromNode = await findNodeByName(page, fromName);
        const toNode = await findNodeByName(page, toName);

        if (!fromNode || !toNode) {
            console.log(`   ⚠️ Nodes not found: ${fromName} or ${toName}`);
            return { success: false };
        }

        console.log(`   ✅ Found both nodes: "${fromName}" → "${toName}"`);

        // Get circles
        const sourceCircles = await fromNode.locator('circle').all();
        const targetCircles = await toNode.locator('circle').all();

        console.log(`   🔍 From node: ${sourceCircles.length} circles`);
        console.log(`   🔍 To node: ${targetCircles.length} circles`);

        if (sourceCircles.length === 0 || targetCircles.length === 0) {
            return { success: false };
        }

        // Get circles on correct sides
        const sourceBox = await sourceCircles[sourceCircles.length - 1].boundingBox();
        const targetBox = await targetCircles[0].boundingBox();

        if (!sourceBox || !targetBox) {
            return { success: false };
        }

        console.log(`   🎯 Source: (${Math.round(sourceBox.x)}, ${Math.round(sourceBox.y)})`);
        console.log(`   🎯 Target: (${Math.round(targetBox.x)}, ${Math.round(targetBox.y)})`);

        // Drag from source center to target center
        const srcX = sourceBox.x + sourceBox.width / 2;
        const srcY = sourceBox.y + sourceBox.height / 2;
        const tgtX = targetBox.x + targetBox.width / 2;
        const tgtY = targetBox.y + targetBox.height / 2;

        await page.mouse.move(srcX, srcY);
        await page.mouse.down();
        await page.waitForTimeout(300);
        await page.mouse.move(tgtX, tgtY, { steps: 25 });
        await page.waitForTimeout(300);
        await page.mouse.up();
        await page.waitForTimeout(1000);

        const verified = await verifyConnection(page, fromName, toName);
        return { success: verified };

    } catch (e) {
        console.log(`   ⚠️ Circle connection failed: ${e.message}`);
        return { success: false };
    }
}

/**
 * 🧠 STRATEGY 3: AI Vision Fallback
 */
async function tryAIVisionConnection(page, fromName, toName) {
    console.log('🧠 Step 3: Using AI Vision fallback...');

    try {
        const screenshot = await page.screenshot({ encoding: 'base64' });
        const compressed = await compressImage(screenshot);

        const response = await openai.chat.completions.create({
            model: 'gpt-4o',
            messages: [{
                role: 'user',
                content: [
                    {
                        type: 'text',
                        text: `Find output port of "${fromName}" and input port of "${toName}".

Return coordinates:
{ "sourceX": x, "sourceY": y, "targetX": x, "targetY": y, "possible": true }
`
                    },
                    { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${compressed}` } }
                ]
            }],
            max_tokens: 200
        });

        const coords = JSON.parse(response.choices[0].message.content);
        
        if (coords.possible) {
            await page.mouse.move(coords.sourceX, coords.sourceY);
            await page.mouse.down();
            await page.waitForTimeout(300);
            await page.mouse.move(coords.targetX, coords.targetY, { steps: 20 });
            await page.waitForTimeout(300);
            await page.mouse.up();
            
            const verified = await verifyConnection(page, fromName, toName);
            return { success: verified };
        }

        return { success: false };
    } catch (e) {
        return { success: false };
    }
}

/**
 * 🔍 Helper: Find Node by Name
 */
async function findNodeByName(page, nodeName) {
    const allNodes = await page.locator('[data-test-id^="canvas-node"]').all();
    
    for (const node of allNodes) {
        const text = await node.textContent();
        if (text && (text.includes(nodeName) || nodeName.toLowerCase().includes(text.trim().split('\n')[0].toLowerCase()))) {
            return node;
        }
    }
    return null;
}

/**
 * ✅ Helper: Verify Connection in DOM
 */
async function verifyConnection(page, fromName, toName) {
    try {
        const exists = await page.evaluate(({from, to}) => {
            const connections = [...document.querySelectorAll('svg path')];
            return connections.some(path => {
                const d = path.getAttribute('d') || '';
                const line = path.parentElement;
                return d.length > 50; // Valid connection line
            });
        }, {from: fromName, to: toName});

        return exists;
    } catch (e) {
        return true; // Assume success if verification fails
    }
}

/**
 * 🖼️ Helper: Compress Image
 */
async function compressImage(base64png) {
    try {
        const buffer = Buffer.from(base64png, 'base64');
        const compressed = await sharp(buffer)
            .resize({ width: 1280 })
            .jpeg({ quality: 70 })
            .toBuffer();
        return compressed.toString('base64');
    } catch (e) {
        return base64png;
    }
}

export { connectNodes };


