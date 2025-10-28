"""
Vision-based Frame Finder
==========================
Uses GPT-4o Vision to find the BEST complete workflow frame.
"""

import cv2
import base64
import json
from openai import OpenAI


class VisionFrameFinder:
    """Find best workflow frame using AI Vision"""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def find_best_workflow_frame(self, video_path: str, sample_count: int = 20) -> dict:
        """
        Find frame with most complete n8n workflow
        
        Strategy:
        1. Sample 20 frames across video
        2. Ask GPT-4o Vision to score each frame
        3. Return frame with highest workflow completeness score
        """
        
        print("Step 2/4: Finding best complete workflow frame (AI Vision)...")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames evenly across video
        sample_indices = [int(i * total_frames / sample_count) for i in range(1, sample_count)]
        
        print(f"   Scanning {len(sample_indices)} frames...")
        
        best_frame = None
        best_score = 0
        best_idx = 0
        best_timestamp = 0
        
        for i, idx in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Check every 3rd frame (balance between coverage and cost)
            if i % 3 != 0:
                continue
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Ask AI to score this frame
            score = self._score_frame_with_vision(frame_b64, idx / fps)
            
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_idx = idx
                best_timestamp = idx / fps
        
        cap.release()
        
        print(f"   Best frame: #{best_idx} (t={best_timestamp:.1f}s)")
        print(f"   Completeness score: {best_score}/10")
        
        return {
            "frame": best_frame,
            "frame_idx": best_idx,
            "timestamp": best_timestamp,
            "score": best_score
        }
    
    def _score_frame_with_vision(self, frame_b64: str, timestamp: float) -> float:
        """
        Use GPT-4o to score frame completeness
        
        Returns: 0-10 score (10 = perfect complete workflow, 0 = menu/overlay)
        """
        
        prompt = """Score this frame for n8n workflow completeness (0-10).

SCORING CRITERIA:
10 = Perfect complete workflow visible (all nodes, connections, clean canvas, no overlays)
8-9 = Most workflow visible, minor overlays or partial view
5-7 = Some workflow visible but incomplete or has overlays (YouTube player, menus)
3-4 = Mostly menus/settings, minimal workflow
0-2 = No workflow (loading, login, YouTube player covering everything)

CHECK FOR:
✓ Multiple n8n nodes visible (boxes with icons)
✓ Connection lines/arrows between nodes
✓ Clean canvas (no YouTube player overlay at bottom)
✓ No menu panels covering workflow
✗ Dark bottom region (YouTube player)
✗ Settings panels open
✗ Node creator menu open

Return ONLY JSON:
{
  "score": 8.5,
  "reason": "Complete workflow with 3 nodes visible, slight YouTube overlay at bottom"
}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Faster, cheaper for scoring
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
                max_tokens=100,
                temperature=0
            )
            
            result = json.loads(response.choices[0].message.content)
            score = result.get("score", 0)
            
            print(f"      t={timestamp:.1f}s: score={score}/10 - {result.get('reason', '')[:50]}")
            
            return float(score)
            
        except Exception as e:
            print(f"      Vision scoring error: {e}")
            return 0.0


if __name__ == "__main__":
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    
    load_dotenv('../config/credentials.env')
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    finder = VisionFrameFinder(client)
    
    result = finder.find_best_workflow_frame("./temp_video.mp4", sample_count=20)
    
    if result:
        cv2.imwrite("best_frame_vision.png", result['frame'])
        print(f"\nSaved best frame to best_frame_vision.png")

