"""
Frame Detector Module
=====================
Extracts clean workflow frames from video.
"""

import cv2
import numpy as np
import os


class FrameDetector:
    """Extract best workflow frames from video"""
    
    def __init__(self):
        pass
    
    def extract_workflow_frames(self, video_path: str, max_frames: int = 10) -> list:
        """
        Extract clean workflow frames
        
        Returns: List of dicts with frame, timestamp, frame_idx
        """
        
        print("Step 2/4: Extracting workflow frames...")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        print(f"   Video: {duration:.1f}s, {total_frames} frames, {fps:.0f} FPS")
        
        # Extract evenly spaced frames
        frame_indices = []
        step = total_frames // (max_frames + 1)
        for i in range(1, max_frames + 1):
            frame_indices.append(i * step)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                timestamp = idx / fps
                frames.append({
                    "frame_idx": idx,
                    "timestamp": timestamp,
                    "frame": frame
                })
        
        cap.release()
        
        print(f"   Extracted {len(frames)} frames")
        
        return frames
    
    def find_best_workflow_frame(self, frames: list, detections: list) -> dict:
        """
        Find frame with most complete workflow
        
        Args:
            frames: List of frame dicts
            detections: List of AI detection results
            
        Returns: Best frame dict
        """
        
        best_idx = 0
        max_nodes = 0
        
        for i, detection in enumerate(detections):
            node_count = len(detection.get("nodes", []))
            if node_count > max_nodes:
                max_nodes = node_count
                best_idx = i
        
        if max_nodes == 0:
            # Default to last frame
            best_idx = len(frames) - 1
        
        print(f"   Best frame: #{best_idx} with {max_nodes} nodes")
        
        return frames[best_idx] if frames else None


if __name__ == "__main__":
    # Test
    detector = FrameDetector()
    frames = detector.extract_workflow_frames("../temp_video.mp4")
    print(f"Extracted {len(frames)} frames")

