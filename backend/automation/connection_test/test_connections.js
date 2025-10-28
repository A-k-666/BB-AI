/**
 * 🧪 Test Connection Engine
 * Run this to test connection logic without full automation
 */

import { chromium } from 'playwright';
import { connectNodes } from './connection_engine.js';
import { config } from 'dotenv';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load env
config({ path: join(__dirname, '..', '..', 'config', 'credentials.env') });

async function testConnections() {
    console.log('\n🧪 Testing Connection Engine\n');
    console.log('========================================');

    // Connect to Browserbase
    const sessionUrl = `wss://connect.browserbase.com?apiKey=${process.env.API_KEY}&projectId=${process.env.PROJECT_ID}`;
    const browser = await chromium.connectOverCDP(sessionUrl);
    const context = browser.contexts()[0];
    const page = context.pages()[0];

    await page.goto(`${process.env.N8N_URL}/workflow/new`, {
        waitUntil: 'domcontentloaded',
        timeout: 60000
    });

    await page.waitForTimeout(3000);

    // Login
    if (await page.locator('input[type="email"]').isVisible().catch(() => false)) {
        await page.locator('input[type="email"]').fill(process.env.N8N_EMAIL);
        await page.locator('input[type="email"]').press('Enter');
        await page.waitForTimeout(1500);
        await page.locator('input[type="password"]').fill(process.env.N8N_PASSWORD);
        await page.locator('input[type="password"]').press('Enter');
        await page.waitForTimeout(3000);
    }

    console.log('\n✅ Logged in\n');

    // Test cases
    const tests = [
        {
            from: 'WhatsApp Trigger',
            to: 'AI Agent',
            description: 'Circle to Circle connection'
        },
        {
            from: 'AI Agent',
            to: 'OpenRouter Chat Model',
            description: '+ Icon to Diamond connection'
        },
        {
            from: 'AI Agent',
            to: 'WhatsApp Business Cloud',
            description: '+ Icon to Circle connection'
        }
    ];

    for (const test of tests) {
        console.log(`\n🎯 Test: ${test.description}`);
        console.log(`   Connecting: ${test.from} → ${test.to}\n`);

        const result = await connectNodes(page, test.from, test.to);

        if (result.success) {
            console.log(`\n✅ PASSED: ${test.description}\n`);
        } else {
            console.log(`\n❌ FAILED: ${test.description}\n`);
        }

        await page.waitForTimeout(2000);
    }

    console.log('\n========================================');
    console.log('✅ All tests completed!');
    console.log('========================================\n');

    await browser.close();
}

testConnections().catch(console.error);


