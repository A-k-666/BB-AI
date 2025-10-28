"""
Audio Transcriber Module
========================
Transcribes video audio using AssemblyAI.
"""

import assemblyai as aai
import json
import os


class AudioTranscriber:
    """Transcribe video audio with word-level timestamps"""
    
    def __init__(self, api_key: str):
        """Initialize with AssemblyAI API key"""
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
    
    def transcribe_video(self, video_path: str, output_dir: str = './output') -> dict:
        """
        Transcribe video audio
        
        Returns:
        {
            "full_text": "complete transcript",
            "segments": [
                {
                    "start": 5.3,
                    "end": 10.2,
                    "text": "Add a webhook node",
                    "confidence": 0.95,
                    "words": [...]
                }
            ]
        }
        """
        
        print("Step 1/4: Transcribing audio with AssemblyAI...")
        
        try:
            # Transcribe
            transcript_result = self.transcriber.transcribe(video_path)
            
            if transcript_result.status == aai.TranscriptStatus.error:
                print(f"   Error: {transcript_result.error}")
                return {"full_text": "", "segments": []}
            
            # Parse into structured format
            segments = []
            for utterance in transcript_result.utterances or []:
                segments.append({
                    "start": utterance.start / 1000.0,  # ms to seconds
                    "end": utterance.end / 1000.0,
                    "text": utterance.text.strip(),
                    "confidence": utterance.confidence,
                    "words": [
                        {
                            "text": w.text,
                            "start": w.start / 1000.0,
                            "end": w.end / 1000.0,
                            "confidence": w.confidence
                        }
                        for w in (utterance.words or [])
                    ]
                })
            
            result = {
                "full_text": transcript_result.text,
                "segments": segments,
                "summary": self._extract_workflow_keywords(transcript_result.text)
            }
            
            # Save transcript
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/transcript.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"   Transcribed: {len(result['full_text'])} chars, {len(segments)} segments")
            print(f"   Saved: {output_dir}/transcript.json")
            
            return result
            
        except Exception as e:
            print(f"   Transcription failed: {e}")
            return {"full_text": "", "segments": []}
    
    def _extract_workflow_keywords(self, text: str) -> dict:
        """Extract workflow-related keywords from transcript"""
        
        text_lower = text.lower()
        
        keywords = {
            "nodes_mentioned": [],
            "actions_mentioned": [],
            "tools_mentioned": []
        }
        
        # Common n8n nodes
        node_keywords = [
            "webhook", "whatsapp", "slack", "telegram", "email", "gmail",
            "http request", "code", "function", "set", "if", "switch",
            "openai", "ai agent", "chat model", "gemini", "anthropic"
        ]
        
        for keyword in node_keywords:
            if keyword in text_lower:
                keywords["nodes_mentioned"].append(keyword)
        
        # Common actions
        action_keywords = ["connect", "add", "create", "configure", "set up", "drag"]
        for keyword in action_keywords:
            if keyword in text_lower:
                keywords["actions_mentioned"].append(keyword)
        
        return keywords


if __name__ == "__main__":
    # Test
    transcriber = AudioTranscriber(api_key="YOUR_KEY")
    result = transcriber.transcribe_video("../temp_video.mp4", output_dir="./output")
    print(f"\nKeywords found: {result['summary']}")

