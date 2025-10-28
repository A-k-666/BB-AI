"""
Multimodal Fuser Module
=======================
Combines audio transcript + visual detections.
"""

import json
from openai import OpenAI


class MultimodalFuser:
    """Fuse audio + vision using AI"""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def analyze_frame_with_context(self, frame_b64: str, transcript_text: str, timestamp: float) -> dict:
        """
        Analyze single frame with transcript context
        
        Returns: {"nodes": [...], "connections": [...]}
        """
        
        # Find relevant transcript around this timestamp
        context = self._get_transcript_context(transcript_text, timestamp)
        
        prompt = f"""Analyze this n8n workflow screenshot with audio context.

AUDIO CONTEXT (what speaker said):
"{context}"

TASK: Extract EXACT node names and connections from the screenshot.

Common n8n nodes you might see:
- WhatsApp Trigger, Slack, Telegram, Email
- AI Agent, OpenAI Chat Model, Gemini
- HTTP Request, Code, Set, IF/Switch
- Send message, Send email

Return ONLY JSON:
{{
  "nodes": [
    {{"label": "exact node name from screenshot", "type": "trigger/action/model"}}
  ],
  "connections": [
    {{"from": "node1 label", "to": "node2 label"}}
  ],
  "workflow_purpose": "what this workflow does"
}}

CRITICAL: Read exact text from nodes - use transcript to help identify."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                        }
                    ]
                }],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"      Vision error: {e}")
            return {"nodes": [], "connections": []}
    
    def _get_transcript_context(self, full_text: str, timestamp: float, window: float = 10.0) -> str:
        """Get relevant transcript text around timestamp"""
        
        # Simple approach: return full text if short
        if len(full_text) < 500:
            return full_text
        
        # Otherwise return a reasonable chunk
        return full_text[:500] + "..."


if __name__ == "__main__":
    pass

