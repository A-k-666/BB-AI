"""
Smart Frame Selector
====================
Finds the BEST complete workflow frame using OCR + text detection.
"""

import cv2
import numpy as np
import pytesseract
import os


# Set Tesseract path for Windows
if os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class SmartFrameSelector:
    """Find best complete workflow frame"""
    
    def __init__(self):
        self.n8n_keywords = [
            'whatsapp', 'trigger', 'agent', 'model', 'send', 'message',
            'slack', 'email', 'http', 'code', 'openai', 'gemini', 'claude',
            'workflow', 'execute', 'connection'
        ]
    
    def find_best_workflow_frame(self, video_path: str, sample_count: int = 30) -> dict:
        """
        Find frame with MOST workflow text visible
        
        Strategy:
        1. Sample 30 frames across video
        2. Run OCR on each
        3. Count n8n-related keywords
        4. Check for overlays (dark bottom = YouTube player)
        5. Return frame with most workflow content
        """
        
        print("Finding best complete workflow frame...")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames evenly
        sample_indices = [int(i * total_frames / sample_count) for i in range(1, sample_count)]
        
        best_frame = None
        best_score = 0
        best_info = {}
        
        print(f"   Scanning {len(sample_indices)} frames for complete workflow...")
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Score this frame
            score, info = self._score_frame(frame)
            
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_info = {
                    "frame_idx": idx,
                    "timestamp": idx / fps,
                    "score": score,
                    **info
                }
        
        cap.release()
        
        if best_frame is not None:
            print(f"   Best frame: #{best_info['frame_idx']} (t={best_info['timestamp']:.1f}s)")
            print(f"   Score: {best_info['score']:.2f}")
            print(f"   Keywords: {best_info.get('keyword_count', 0)}")
            print(f"   Has overlay: {best_info.get('has_overlay', False)}")
            
            return {
                "frame": best_frame,
                "frame_idx": best_info["frame_idx"],
                "timestamp": best_info["timestamp"]
            }
        
        return None
    
    def _score_frame(self, frame: np.ndarray) -> tuple:
        """
        Score frame quality for workflow completeness
        
        Returns: (score, info_dict)
        """
        h, w = frame.shape[:2]
        score = 0.0
        info = {}
        
        # 1. Check for YouTube overlay (dark bottom = bad)
        bottom_region = frame[int(h*0.8):, :]
        bottom_brightness = np.mean(bottom_region)
        has_overlay = bottom_brightness < 100
        info['has_overlay'] = has_overlay
        
        if has_overlay:
            score -= 10.0  # Heavy penalty
        
        # 2. OCR text detection
        try:
            # Preprocess for better OCR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract text
            text = pytesseract.image_to_string(gray).lower()
            
            # Count workflow-related keywords
            keyword_count = sum(1 for keyword in self.n8n_keywords if keyword in text)
            info['keyword_count'] = keyword_count
            info['text_sample'] = text[:100]
            
            # More keywords = more complete workflow
            score += keyword_count * 2.0
            
        except Exception as e:
            info['ocr_error'] = str(e)
            keyword_count = 0
        
        # 3. Visual complexity (workflow has nodes/connections)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        info['edge_density'] = edge_density
        
        # Moderate edge density = good (nodes present)
        if 0.02 < edge_density < 0.12:
            score += 3.0
        
        # 4. Canvas brightness (n8n canvas is light)
        center_region = frame[int(h*0.2):int(h*0.7), int(w*0.1):int(w*0.9)]
        center_brightness = np.mean(center_region)
        info['center_brightness'] = center_brightness
        
        if center_brightness > 150:
            score += 2.0
        
        info['total_score'] = score
        return score, info


if __name__ == "__main__":
    # Test
    selector = SmartFrameSelector()
    result = selector.find_best_workflow_frame("./temp_video.mp4", sample_count=30)
    
    if result:
        print(f"\nBest frame found at {result['timestamp']:.1f}s")
        cv2.imwrite("best_workflow_frame.png", result['frame'])
        print("Saved as best_workflow_frame.png")

