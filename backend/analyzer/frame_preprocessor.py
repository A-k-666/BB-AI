"""
Frame Preprocessing for Better AI Vision Understanding
Upscales, crops, and enhances frames before GPT-4o analysis
"""

import cv2
import numpy as np
from typing import Tuple


class FramePreprocessor:
    """Prepare video frames for optimal AI Vision analysis"""
    
    def __init__(self):
        self.target_width = 1920  # HD resolution
        self.target_height = 1080
        
    def preprocess_for_ai(self, frame: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for AI Vision
        
        Steps:
        1. Upscale to HD
        2. Crop to n8n canvas area
        3. Enhance contrast
        4. Sharpen text
        
        Returns:
            Enhanced frame optimized for GPT-4o Vision
        """
        # Step 1: Upscale if needed
        h, w = frame.shape[:2]
        if w < self.target_width:
            scale = self.target_width / w
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Step 2: Auto-crop to workflow canvas
        frame = self._crop_to_canvas(frame)
        
        # Step 3: Enhance contrast
        frame = self._enhance_contrast(frame)
        
        # Step 4: Sharpen text
        frame = self._sharpen_text(frame)
        
        return frame
    
    def _crop_to_canvas(self, frame: np.ndarray) -> np.ndarray:
        """
        Auto-detect and crop to n8n workflow canvas
        Removes sidebars, headers, footers
        """
        h, w = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find canvas area (usually has workflow nodes/grid)
        # Look for vertical edges (sidebar boundaries)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find horizontal line sums to detect canvas boundaries
        vertical_sum = np.sum(edges, axis=0)
        horizontal_sum = np.sum(edges, axis=1)
        
        # Canvas typically starts after sidebar (10-20% from left)
        left_boundary = int(w * 0.05)  # Skip leftmost 5%
        right_boundary = int(w * 0.95)  # Skip rightmost 5%
        top_boundary = int(h * 0.05)   # Skip top 5%
        bottom_boundary = int(h * 0.95) # Skip bottom 5%
        
        # Crop
        canvas = frame[top_boundary:bottom_boundary, left_boundary:right_boundary]
        
        return canvas
    
    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Makes node text more readable
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _sharpen_text(self, frame: np.ndarray) -> np.ndarray:
        """
        Sharpen filter to make text crisp for AI Vision
        """
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        
        sharpened = cv2.filter2D(frame, -1, kernel)
        
        return sharpened
    
    def extract_best_frames(self, video_path: str, num_frames: int = 10) -> list:
        """
        Intelligently select best frames from video
        
        Criteria:
        - High sharpness (not blurry)
        - High variance (content-rich)
        - Evenly distributed across video
        
        Returns:
            List of best preprocessed frames
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample candidates (3x more than needed)
        candidate_indices = np.linspace(0, total_frames - 1, num_frames * 3, dtype=int)
        
        candidates = []
        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Calculate frame quality score
            quality = self._calculate_frame_quality(frame)
            candidates.append((idx, frame, quality))
        
        cap.release()
        
        # Sort by quality and pick top N, ensuring distribution
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Pick best frames with min distance between them
        selected = []
        min_gap = total_frames // (num_frames + 1)
        
        for idx, frame, quality in candidates:
            # Check if this frame is far enough from already selected
            if all(abs(idx - s[0]) > min_gap for s in selected):
                selected.append((idx, frame, quality))
                if len(selected) >= num_frames:
                    break
        
        # Sort by index to maintain temporal order
        selected.sort(key=lambda x: x[0])
        
        # Preprocess all selected frames
        best_frames = [self.preprocess_for_ai(frame) for _, frame, _ in selected]
        
        return best_frames
    
    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """
        Calculate frame quality score
        
        Metrics:
        - Sharpness (Laplacian variance)
        - Content richness (pixel variance)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sharpness: higher variance = sharper
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Content: higher variance = more content
        content = gray.var()
        
        # Combined score
        quality = (sharpness * 0.7) + (content * 0.3)
        
        return quality


# Example usage:
if __name__ == "__main__":
    preprocessor = FramePreprocessor()
    
    # Extract 10 best frames
    frames = preprocessor.extract_best_frames('./temp_video.mp4', num_frames=10)
    
    print(f"Extracted {len(frames)} high-quality frames for AI analysis")
    
    # Save for inspection
    for i, frame in enumerate(frames):
        cv2.imwrite(f'./output/debug/best_frame_{i:03d}.png', frame)



