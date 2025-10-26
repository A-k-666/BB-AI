#!/usr/bin/env node

/**
 * N8N UI Expert - RAG-based intelligent UI understanding
 * Trained on complete n8n documentation for human-like interactions
 */

import { readFile } from 'fs/promises';
import { config } from 'dotenv';
import OpenAI from 'openai';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

config({ path: join(__dirname, '..', 'config', 'credentials.env') });

class N8nUIExpert {
    constructor() {
        this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
        this.model = 'gpt-4o';  // Use full GPT-4o for expert reasoning
        this.docsContext = null;
        this.conversationHistory = [];
    }
    
    /**
     * 📚 Load n8n documentation context
     */
    async loadDocsContext() {
        try {
            let combinedDocs = '';
            
            // Load multiple docs sources
            const docSources = [
                join(__dirname, '..', 'docs', 'n8n_docs_combined.md'),
                join(__dirname, '..', 'docs', 'n8n_ui_guide.txt')
            ];
            
            for (const docPath of docSources) {
                try {
                    const content = await readFile(docPath, 'utf-8');
                    combinedDocs += content + '\n\n';
                } catch (err) {
                    // Skip if file doesn't exist
                }
            }
            
            // Extract key sections (limit to ~80K chars for context window)
            this.docsContext = combinedDocs.slice(0, 80000);
            
            console.log('[N8N EXPERT] Loaded n8n documentation context');
            console.log(`   Context size: ${(this.docsContext.length / 1000).toFixed(1)}K characters`);
            
            return true;
        } catch (error) {
            console.log('[WARN] n8n docs not found - using base knowledge');
            this.docsContext = `n8n is a workflow automation tool. Common nodes:
- Webhook (trigger): Receives HTTP requests
- Code: JavaScript/Python code execution
- HTTP Request: Make API calls
- Set: Set/modify data
- IF: Conditional branching
- OpenAI: AI text generation

UI Elements:
- Canvas: Main workflow area with drag-drop nodes
- Node Creator: Search dialog to add nodes
- Settings Panel: Configure node properties
- Connections: Drag from output circle to input circle`;
            
            return false;
        }
    }
    
    /**
     * 🧠 Ask n8n UI expert for guidance
     */
    async askExpert(question, currentScreenshot = null) {
        console.log(`\n[EXPERT] Question: "${question}"`);
        
        // Build context-aware prompt
        const systemPrompt = `You are an expert n8n workflow automation specialist with deep knowledge of the n8n UI.

Your knowledge includes:
${this.docsContext}

You help automate workflows by:
1. Understanding the n8n Cloud UI structure
2. Providing exact Playwright selectors
3. Explaining step-by-step how to interact with UI elements
4. Handling edge cases and modal behaviors

Always respond with actionable, precise instructions.`;

        const messages = [
            { role: "system", content: systemPrompt },
            ...this.conversationHistory,
            { role: "user", content: question }
        ];
        
        // Add screenshot if provided
        if (currentScreenshot) {
            messages[messages.length - 1].content = [
                { type: "text", text: question },
                { 
                    type: "image_url", 
                    image_url: { url: `data:image/png;base64,${currentScreenshot}` }
                }
            ];
        }
        
        try {
            const response = await this.openai.chat.completions.create({
                model: this.model,
                messages,
                temperature: 0.1,
                max_tokens: 1000
            });
            
            const answer = response.choices[0].message.content;
            
            // Update conversation history (keep last 5 exchanges)
            this.conversationHistory.push(
                { role: "user", content: question },
                { role: "assistant", content: answer }
            );
            
            if (this.conversationHistory.length > 10) {
                this.conversationHistory = this.conversationHistory.slice(-10);
            }
            
            console.log(`[EXPERT] Answer: ${answer.slice(0, 200)}...`);
            
            return answer;
            
        } catch (error) {
            console.log(`[ERROR] Expert query failed: ${error.message}`);
            return null;
        }
    }
    
    /**
     * 🎯 Get selector recommendation from expert
     */
    async getSelector(intent, domSnapshot = null, screenshot = null) {
        const question = `Task: ${intent}

${domSnapshot ? `DOM Context:\n${JSON.stringify(domSnapshot, null, 2)}\n` : ''}

Provide the BEST Playwright selector and explain why. Respond with JSON:
{
  "selector": "best selector",
  "reasoning": "why this selector",
  "alternatives": ["backup1", "backup2"],
  "interaction_steps": ["step 1", "step 2"]
}`;

        const answer = await this.askExpert(question, screenshot);
        
        if (answer) {
            try {
                return JSON.parse(answer);
            } catch {
                // Not JSON, return as text reasoning
                return { selector: null, reasoning: answer, alternatives: [] };
            }
        }
        
        return null;
    }
    
    /**
     * 🔧 Get troubleshooting advice
     */
    async troubleshoot(problem, screenshot = null) {
        const question = `Problem: ${problem}

What should I try next to fix this issue in n8n Cloud UI? Provide 3 concrete solutions.`;

        return await this.askExpert(question, screenshot);
    }
    
    /**
     * 📖 Explain workflow step
     */
    async explainStep(stepDescription, screenshot = null) {
        const question = `Explain how to: ${stepDescription}

Provide exact UI interaction steps for n8n Cloud.`;

        return await this.askExpert(question, screenshot);
    }
}

export { N8nUIExpert };

// Test
if (import.meta.url === `file://${process.argv[1]}`) {
    (async () => {
        const expert = new N8nUIExpert();
        await expert.loadDocsContext();
        
        const answer = await expert.askExpert(
            "How do I add a Webhook trigger node in n8n Cloud? Provide exact steps."
        );
        
        console.log('\n[FULL ANSWER]:', answer);
    })();
}

