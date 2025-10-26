"""
Robust video analyzer for YouTube n8n tutorial videos.
Analyzes video frames to detect nodes, connections, and workflow steps.
"""

import cv2
import numpy as np
import logging
import json
import os
import yt_dlp
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import requests
import pytesseract
from .utils import ActionSequenceGenerator

# Import robust detector
try:
    from .robust_node_detector import detect_nodes_robust
    ROBUST_DETECTION_AVAILABLE = True
except ImportError:
    ROBUST_DETECTION_AVAILABLE = False

# Set Tesseract path for Windows
import platform
if platform.system() == 'Windows':
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe'
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

# Try to import scikit-image, fallback if not available
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    # Fallback SSIM implementation
    def ssim(img1, img2):
        """Simple fallback SSIM calculation."""
        return 0.5  # Placeholder


class VideoAnalyzer:
    """Analyzes YouTube tutorial videos to extract n8n workflow information."""
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize video analyzer.
        
        Args:
            openai_api_key: OpenAI API key for AI reasoning (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.openai_api_key = openai_api_key
        self.action_generator = ActionSequenceGenerator()
        
        # Node detection patterns
        self.node_patterns = {
            'webhook': ['webhook', 'trigger', 'http'],
            'openai': ['openai', 'gpt', 'ai', 'completion'],
            'http_request': ['http', 'request', 'api', 'fetch'],
            'email': ['email', 'mail', 'smtp'],
            'slack': ['slack', 'notification'],
            'google_sheets': ['sheets', 'spreadsheet', 'google'],
            'schedule': ['schedule', 'cron', 'timer']
        }
    
    def download_video(self, video_url: str, output_path: str = "./temp_video.mp4") -> bool:
        """
        Download YouTube video.
        
        Args:
            video_url: YouTube video URL
            output_path: Local path to save video
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            ydl_opts = {
                'outtmpl': output_path,
                # Download BEST quality for AI analysis (HD 1080p+)
                'format': 'bestvideo[ext=mp4][height>=1080]+bestaudio[ext=m4a]/best[ext=mp4][height>=1080]/best',
                'merge_output_format': 'mp4',
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            self.logger.info(f"Video downloaded successfully (HD): {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to download HD video: {e}")
            # Fallback to any quality
            try:
                ydl_opts_fallback = {
                    'outtmpl': output_path,
                    'format': 'best',
                }
                with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
                    ydl.download([video_url])
                self.logger.info(f"Video downloaded (fallback quality): {output_path}")
                return True
            except:
                return False
    
    def detect_keyframes(self, video_path: str, threshold: float = 0.25) -> List[int]:
        """
        Detect keyframes using SSIM-based scene change detection.
        
        Args:
            video_path: Path to video file
            threshold: SSIM threshold for scene change detection
            
        Returns:
            List[int]: List of keyframe indices
        """
        cap = cv2.VideoCapture(video_path)
        keyframes = []
        prev_frame = None
        idx = 0
        
        self.logger.info("Detecting keyframes using SSIM...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is None:
                keyframes.append(idx)
            else:
                try:
                    if SKIMAGE_AVAILABLE:
                        score = ssim(prev_frame, gray)
                        if score < 1 - threshold:
                            keyframes.append(idx)
                            self.logger.debug(f"Keyframe detected at {idx}, SSIM: {score:.3f}")
                    else:
                        # Fallback: use simple frame difference
                        diff = cv2.absdiff(prev_frame, gray)
                        mean_diff = np.mean(diff)
                        if mean_diff > 30:  # Threshold for significant change
                            keyframes.append(idx)
                            self.logger.debug(f"Keyframe detected at {idx}, diff: {mean_diff:.1f}")
                except Exception as e:
                    self.logger.warning(f"Keyframe detection failed at frame {idx}: {e}")
                    # Fallback: add frame if detection fails
                    keyframes.append(idx)
            
            prev_frame = gray
            idx += 1
        
        cap.release()
        self.logger.info(f"Detected {len(keyframes)} keyframes from {idx} total frames")
        return keyframes
    
    def extract_frames(self, video_path: str, frame_interval: int = 30) -> List[np.ndarray]:
        """
        Extract frames from video at specified intervals or keyframes.
        
        Args:
            video_path: Path to video file
            frame_interval: Extract every Nth frame (fallback if keyframe detection fails)
            
        Returns:
            List[np.ndarray]: List of extracted frames
        """
        # Try keyframe detection first
        try:
            keyframes = self.detect_keyframes(video_path)
            if len(keyframes) > 0:
                return self._extract_frames_at_indices(video_path, keyframes)
        except Exception as e:
            self.logger.warning(f"Keyframe detection failed: {e}, using interval method")
        
        # Fallback to interval-based extraction
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return frames
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame.copy())
            
            frame_count += 1
        
        cap.release()
        self.logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    def _extract_frames_at_indices(self, video_path: str, indices: List[int]) -> List[np.ndarray]:
        """
        Extract frames at specific indices.
        
        Args:
            video_path: Path to video file
            indices: List of frame indices to extract
            
        Returns:
            List[np.ndarray]: List of extracted frames
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return frames
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame.copy())
            else:
                self.logger.warning(f"Failed to read frame at index {idx}")
        
        cap.release()
        return frames
    
    def detect_workflow_nodes(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect n8n workflow nodes in frame.
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            List[Dict]: Detected nodes with positions and types
        """
        nodes = []
        
        # Check if frame is valid
        if frame is None or frame.size == 0:
            self.logger.warning("Invalid frame provided to detect_workflow_nodes")
            return nodes
        
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            self.logger.warning(f"Failed to convert frame to grayscale: {e}")
            return nodes
        
        # Detect rectangular shapes (potential nodes)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area and aspect ratio
            area = cv2.contourArea(contour)
            if area < 1000:  # Too small
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check if it looks like a node (rectangular, reasonable size)
            if 0.5 < aspect_ratio < 3.0 and w > 100 and h > 50:
                # Extract text from region
                roi = frame[y:y+h, x:x+w]
                node_text = self._extract_text_from_region(roi)
                
                if node_text:
                    node_type = self._classify_node_type(node_text)
                    nodes.append({
                        'position': {'x': x, 'y': y, 'width': w, 'height': h},
                        'text': node_text,
                        'label': node_text,  # Add label field
                        'type': node_type,
                        'confidence': 0.8  # Placeholder confidence
                    })
        
        return nodes
    
    def _extract_text_from_region(self, roi: np.ndarray) -> str:
        """
        Extract text from image region using pytesseract with preprocessing.
        
        Args:
            roi: Image region of interest
            
        Returns:
            str: Extracted text
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Adaptive thresholding for better text extraction
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Resize small text for better OCR
            h, w = morph.shape
            if h < 50 or w < 100:
                morph = cv2.resize(morph, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
            
            # OCR with optimized config
            text = pytesseract.image_to_string(
                morph, 
                config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
            )
            
            return text.strip()
            
        except Exception as e:
            self.logger.warning(f"OCR failed: {e}")
            return "Node"  # Fallback
    
    def _classify_node_type(self, text: str) -> str:
        """
        Classify node type based on text content.
        
        Args:
            text: Node text content
            
        Returns:
            str: Classified node type
        """
        text_lower = text.lower()
        
        for node_type, keywords in self.node_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return node_type
        
        return "unknown"
    
    def detect_connections(self, frame: np.ndarray, nodes: List[Dict]) -> List[Dict]:
        """
        Detect curved connections between nodes using skeletonization.
        
        Args:
            frame: Video frame
            nodes: List of detected nodes
            
        Returns:
            List[Dict]: Detected connections
        """
        connections = []
        
        try:
            # Convert to grayscale and create binary image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create binary image focusing on potential connection lines
            # Look for thin lines that could be connectors
            edges = cv2.Canny(gray, 50, 150)
            
            # Skeletonize to find connection paths
            skeleton = self._skeletonize(edges)
            
            # Find potential connections between nodes
            for i, from_node in enumerate(nodes):
                for j, to_node in enumerate(nodes):
                    if i != j:
                        # Check if there's a skeleton path between nodes
                        if self._has_connection_path(skeleton, from_node, to_node):
                            connections.append({
                                'from': from_node.get('id', f'node_{i}'),
                                'to': to_node.get('id', f'node_{j}'),
                                'type': 'data',
                                'confidence': 0.85,
                                'anchors': [
                                    from_node['position'],
                                    to_node['position']
                                ]
                            })
        
        except Exception as e:
            self.logger.warning(f"Connection detection failed: {e}")
        
        return connections
    
    def _skeletonize(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Skeletonize binary image to find connection paths.
        
        Args:
            binary_image: Binary image
            
        Returns:
            np.ndarray: Skeletonized image
        """
        size = np.size(binary_image)
        skel = np.zeros(binary_image.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        img = binary_image.copy()
        
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True
        
        return skel
    
    def _has_connection_path(self, skeleton: np.ndarray, from_node: Dict, to_node: Dict) -> bool:
        """
        Check if there's a skeleton path between two nodes.
        
        Args:
            skeleton: Skeletonized image
            from_node: Source node
            to_node: Target node
            
        Returns:
            bool: True if connection path exists
        """
        try:
            # Get node centers
            from_center = (
                from_node['position']['x'] + from_node['position']['width'] // 2,
                from_node['position']['y'] + from_node['position']['height'] // 2
            )
            to_center = (
                to_node['position']['x'] + to_node['position']['width'] // 2,
                to_node['position']['y'] + to_node['position']['height'] // 2
            )
            
            # Check if skeleton pixels exist along the path
            # Simple heuristic: check if skeleton pixels are near the line
            distance = np.sqrt((to_center[0] - from_center[0])**2 + (to_center[1] - from_center[1])**2)
            
            if distance < 50:  # Nodes too close
                return False
            
            # Sample points along the line
            steps = int(distance / 10)
            for i in range(steps):
                t = i / steps
                x = int(from_center[0] + t * (to_center[0] - from_center[0]))
                y = int(from_center[1] + t * (to_center[1] - from_center[1]))
                
                if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]:
                    if skeleton[y, x] > 0:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Connection path check failed: {e}")
            return False
    
    def _find_node_at_position(self, nodes: List[Dict], position: Tuple[int, int]) -> Optional[int]:
        """
        Find node at given position.
        
        Args:
            nodes: List of nodes
            position: (x, y) position
            
        Returns:
            int: Node index or None
        """
        x, y = position
        
        for i, node in enumerate(nodes):
            node_x = node['position']['x']
            node_y = node['position']['y']
            node_w = node['position']['width']
            node_h = node['position']['height']
            
            if node_x <= x <= node_x + node_w and node_y <= y <= node_y + node_h:
                return i
        
        return None
    
    def merge_nodes_across_frames(self, all_nodes_per_frame: List[List[Dict]], iou_thresh: float = 0.5) -> List[Dict]:
        """
        Merge nodes detected across frames using IoU tracking.
        
        Args:
            all_nodes_per_frame: List of node lists from each frame
            iou_thresh: IoU threshold for matching nodes
            
        Returns:
            List[Dict]: Stable nodes with aggregated information
        """
        def calculate_iou(boxA: Dict, boxB: Dict) -> float:
            """Calculate IoU between two bounding boxes."""
            xA = max(boxA['x'], boxB['x'])
            yA = max(boxA['y'], boxB['y'])
            xB = min(boxA['x'] + boxA['width'], boxB['x'] + boxB['width'])
            yB = min(boxA['y'] + boxA['height'], boxB['y'] + boxB['height'])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = boxA['width'] * boxA['height']
            boxBArea = boxB['width'] * boxB['height']
            
            return interArea / float(boxAArea + boxBArea - interArea)
        
        stable_nodes = []
        node_id = 1
        
        for frame_idx, frame_nodes in enumerate(all_nodes_per_frame):
            for node in frame_nodes:
                node['frame_last_seen'] = frame_idx
                matched = False
                
                for stable_node in stable_nodes:
                    if calculate_iou(node['position'], stable_node['position']) > iou_thresh:
                        # Merge information
                        stable_node['frame_last_seen'] = frame_idx
                        stable_node['confidence'] = (stable_node['confidence'] + node['confidence']) / 2
                        stable_node['label'] = node.get('label', node.get('text', 'Node'))  # Use latest label
                        matched = True
                        break
                
                if not matched:
                    node['id'] = f"n{node_id}"
                    stable_nodes.append(node)
                    node_id += 1
        
        self.logger.info(f"Merged {len(stable_nodes)} stable nodes from {len(all_nodes_per_frame)} frames")
        return stable_nodes
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze complete video and generate action sequence.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict: Analysis results with action sequence
        """
        self.logger.info(f"Starting video analysis: {video_path}")
        
        # Extract frames
        frames = self.extract_frames(video_path)
        if not frames:
            return {"error": "Failed to extract frames"}
        
        all_nodes_per_frame = []
        all_connections = []
        
        # Analyze all frames
        for i, frame in enumerate(frames):
            self.logger.info(f"Analyzing frame {i+1}/{len(frames)}")
            
            # 🧠 Try robust detection first (AI Vision)
            if ROBUST_DETECTION_AVAILABLE and i == 0:  # Use robust on first frame
                try:
                    nodes, connections = detect_nodes_robust(frame)
                    if nodes:
                        self.logger.info(f"[ROBUST] Detected {len(nodes)} nodes via AI Vision")
                        all_nodes_per_frame.append(nodes)
                        all_connections.extend(connections)
                        continue  # Skip traditional detection for this frame
                except Exception as e:
                    self.logger.warning(f"Robust detection failed: {e}")
            
            # Fallback: Traditional CV+OCR detection
            nodes = self.detect_workflow_nodes(frame)
            all_nodes_per_frame.append(nodes)
            
            # Detect connections
            connections = self.detect_connections(frame, nodes)
            all_connections.extend(connections)
        
        # Merge nodes across frames
        stable_nodes = self.merge_nodes_across_frames(all_nodes_per_frame)
        
        # Remove duplicate connections
        unique_connections = self._remove_duplicate_connections(all_connections)
        
        # Generate action sequence
        action_sequence = self._generate_action_sequence(stable_nodes, unique_connections)
        
        # Save debug output
        self._save_debug_output(frames, stable_nodes, unique_connections)
        
        return {
            "nodes": stable_nodes,
            "connections": unique_connections,
            "action_sequence": action_sequence,
            "total_frames_analyzed": len(frames),
            "stable_nodes_count": len(stable_nodes),
            "keyframes": frames,  # 🔥 Add keyframes for AI Vision
            "video_path": video_path  # 🔥 Add video path for reference
        }
    
    def _remove_duplicate_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """Remove duplicate nodes based on position and type."""
        unique_nodes = []
        seen = set()
        
        for node in nodes:
            key = (node['type'], node['position']['x'], node['position']['y'])
            if key not in seen:
                seen.add(key)
                unique_nodes.append(node)
        
        return unique_nodes
    
    def _remove_duplicate_connections(self, connections: List[Dict]) -> List[Dict]:
        """Remove duplicate connections."""
        unique_connections = []
        seen = set()
        
        for conn in connections:
            key = (conn['from'], conn['to'])
            if key not in seen:
                seen.add(key)
                unique_connections.append(conn)
        
        return unique_connections
    
    def _save_debug_output(self, frames: List[np.ndarray], nodes: List[Dict], connections: List[Dict]) -> None:
        """
        Save debug output with annotated frames.
        
        Args:
            frames: List of video frames
            nodes: Detected nodes
            connections: Detected connections
        """
        try:
            debug_dir = "./output/debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save annotated frames
            for i, frame in enumerate(frames[:5]):  # Save first 5 frames
                annotated_frame = self._annotate_frame(frame, nodes, connections)
                cv2.imwrite(f"{debug_dir}/annotated_frame_{i:03d}.png", annotated_frame)
            
            self.logger.info(f"Debug output saved to {debug_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save debug output: {e}")
    
    def _annotate_frame(self, frame: np.ndarray, nodes: List[Dict], connections: List[Dict]) -> np.ndarray:
        """
        Annotate frame with detected nodes and connections.
        
        Args:
            frame: Video frame
            nodes: Detected nodes
            connections: Detected connections
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated = frame.copy()
        
        # Draw nodes
        for node in nodes:
            pos = node['position']
            x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
            
            # Draw rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = node.get('label', 'Node')
            cv2.putText(annotated, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw connections
        for conn in connections:
            if 'anchors' in conn and len(conn['anchors']) >= 2:
                from_pos = conn['anchors'][0]
                to_pos = conn['anchors'][1]
                
                from_center = (
                    from_pos['x'] + from_pos['width'] // 2,
                    from_pos['y'] + from_pos['height'] // 2
                )
                to_center = (
                    to_pos['x'] + to_pos['width'] // 2,
                    to_pos['y'] + to_pos['height'] // 2
                )
                
                cv2.line(annotated, from_center, to_center, (255, 0, 0), 2)
        
        return annotated
    
    def _generate_action_sequence(self, nodes: List[Dict], connections: List[Dict]) -> Dict[str, Any]:
        """
        Generate structured action sequence with proper JSON schema for Playwright.
        
        Args:
            nodes: List of detected nodes
            connections: List of detected connections
            
        Returns:
            Dict: Complete action sequence with schema
        """
        from datetime import datetime
        
        # Create structured action sequence
        action_sequence = {
            "schema_version": "1.0.0",
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "analyzer_version": "video_analyzer_v0.3",
                "total_nodes": len(nodes),
                "total_connections": len(connections)
            },
            "nodes": [],
            "connections": [],
            "actions": []
        }
        
        # Process nodes
        for node in nodes:
            node_data = {
                "id": node.get('id', f"node_{len(action_sequence['nodes'])}"),
                "type": node.get('type', 'unknown'),
                "label": node.get('label', node.get('text', 'Node')),
                "position": node['position'],
                "confidence": node.get('confidence', 0.8),
                "config": self._get_node_config(node)
            }
            action_sequence['nodes'].append(node_data)
            
            # Add create action
            action_sequence['actions'].append({
                "action": "create_node",
                "node_id": node_data['id'],
                "node_type": node_data['type'],
                "label": node_data['label'],
                "position": node_data['position'],
                "config": node_data['config']
            })
        
        # Process connections
        for conn in connections:
            conn_data = {
                "from": conn.get('from'),
                "to": conn.get('to'),
                "type": conn.get('type', 'data'),
                "confidence": conn.get('confidence', 0.85),
                "anchors": conn.get('anchors', [])
            }
            action_sequence['connections'].append(conn_data)
            
            # Add connect action
            action_sequence['actions'].append({
                "action": "connect_nodes",
                "from": conn_data['from'],
                "to": conn_data['to'],
                "connection_type": conn_data['type']
            })
        
        return action_sequence
    
    def _get_node_config(self, node: Dict) -> Dict[str, Any]:
        """
        Get node configuration based on type.
        
        Args:
            node: Node data
            
        Returns:
            Dict: Node configuration
        """
        node_type = node.get('type', 'unknown')
        
        configs = {
            'webhook': {
                'httpMethod': 'POST',
                'path': '/webhook',
                'responseMode': 'responseNode'
            },
            'openai': {
                'model': 'gpt-3.5-turbo',
                'temperature': 0.7
            },
            'http_request': {
                'method': 'GET',
                'url': 'https://api.example.com'
            },
            'email': {
                'fromEmail': 'noreply@example.com',
                'subject': 'Workflow Notification'
            },
            'slack': {
                'channel': '#general',
                'text': 'Workflow executed'
            }
        }
        
        return configs.get(node_type, {})
    
    def analyze_youtube_video(self, video_url: str) -> Dict[str, Any]:
        """
        Analyze YouTube video directly from URL.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Dict: Analysis results
        """
        temp_video_path = "./temp_video.mp4"
        
        # Download video
        if not self.download_video(video_url, temp_video_path):
            return {"error": "Failed to download video"}
        
        try:
            # Analyze video
            result = self.analyze_video(temp_video_path)
            result['video_path'] = temp_video_path  # Ensure video path is set
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
