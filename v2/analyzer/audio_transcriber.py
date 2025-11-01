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
            # Transcribe with utterances + auto_chapters for better segmentation
            config = aai.TranscriptionConfig(
                speaker_labels=False,
                punctuate=True,
                format_text=True,
                auto_chapters=True  # Enable chapter detection for better segments
            )
            
            transcript_result = self.transcriber.transcribe(video_path, config=config)
            
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
        """Extract workflow-related keywords from transcript (GENERIC)"""
        
        text_lower = text.lower()
        
        keywords = {
            "workflow_actions": [],
            "has_workflow_content": False
        }
        
        # GENERIC workflow action indicators (not specific to any service)
        action_keywords = [
            "add node", "create node", "connect", "drag", "workflow",
            "trigger", "configure", "set up", "automation", "integrate"
        ]
        
        for keyword in action_keywords:
            if keyword in text_lower:
                keywords["workflow_actions"].append(keyword)
        
        # Check if this is workflow-related content
        workflow_indicators = [
            "workflow", "n8n", "automation", "node", "trigger", "connect"
        ]
        keywords["has_workflow_content"] = any(ind in text_lower for ind in workflow_indicators)
        
        return keywords


if __name__ == "__main__":
    # Test
    transcriber = AudioTranscriber(api_key="YOUR_KEY")
    result = transcriber.transcribe_video("../temp_video.mp4", output_dir="./output")
    print(f"\nKeywords found: {result['summary']}")




