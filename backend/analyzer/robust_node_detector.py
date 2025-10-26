#!/usr/bin/env python3
"""
Robust Node Detection - Multi-strategy approach
Combines: Template matching, AI Vision (GPT-4o), Spatial heuristics
"""

import os
import cv2
import base64
import json
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv('./config/credentials.env')


class RobustNodeDetector:
    """
    Multi-strategy node detection for n8n workflows
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = os.getenv('AI_MODEL', 'gpt-4o-mini')
        
        # Common n8n node types and their visual characteristics
        self.node_types = {
            'webhook': {'color_range': ([0, 100, 100], [20, 255, 255])},  # Reddish
            'code': {'color_range': ([100, 100, 100], [130, 255, 255])},  # Blueish
            'http_request': {'color_range': ([90, 100, 100], [110, 255, 255])},  # Cyan
        }
    
    def detect_nodes_multiStrategy(self, frame: np.ndarray) -> List[Dict]:
        """
        Main detection function using multiple strategies
        
        Args:
            frame: Video frame (BGR format)
            
        Returns:
            List of detected nodes with metadata
        """
        nodes = []
        
        # Strategy 1: GPT-4o Vision (Most robust)
        print("   [DETECT] Strategy 1: AI Vision-based detection...")
        ai_nodes = self._detect_with_ai_vision(frame)
        if ai_nodes:
            nodes.extend(ai_nodes)
            print(f"   [OK] AI Vision detected {len(ai_nodes)} nodes")
        
        # Strategy 2: Color-based detection (n8n nodes have distinct colors)
        if len(nodes) == 0:
            print("   [DETECT] Strategy 2: Color-based detection...")
            color_nodes = self._detect_by_color(frame)
            if color_nodes:
                nodes.extend(color_nodes)
                print(f"   [OK] Color detection found {len(color_nodes)} nodes")
        
        # Strategy 3: Contour detection with better filtering
        if len(nodes) == 0:
            print("   [DETECT] Strategy 3: Enhanced contour detection...")
            contour_nodes = self._detect_by_contours_enhanced(frame)
            if contour_nodes:
                nodes.extend(contour_nodes)
                print(f"   [OK] Contour detection found {len(contour_nodes)} nodes")
        
        return nodes
    
    def _detect_with_ai_vision(self, frame: np.ndarray) -> List[Dict]:
        """
        Use GPT-4o Vision to detect and classify nodes
        """
        try:
            # Encode frame to base64
            _, buffer = cv2.imencode('.png', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            prompt = """Analyze this n8n workflow screenshot. Identify all workflow nodes.

For each node, provide:
- Node type (Webhook, Code, HTTP Request, OpenAI, Set, IF, etc.)
- Approximate position (x, y as percentage of image width/height)
- Label/name if visible

Return JSON array:
[
  {
    "type": "Webhook",
    "label": "Webhook Trigger",
    "position_percent": {"x": 15, "y": 30},
    "confidence": 0.95
  }
]

If no n8n workflow visible, return empty array []."""

            response = self.client.chat.completions.create(
                model='gpt-4o',  # Use full gpt-4o for vision
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            detected = result.get('nodes', [])
            
            # Convert percentage positions to pixels
            h, w = frame.shape[:2]
            nodes = []
            
            for idx, node in enumerate(detected):
                pos_pct = node.get('position_percent', {})
                x = int((pos_pct.get('x', 0) / 100) * w)
                y = int((pos_pct.get('y', 0) / 100) * h)
                
                nodes.append({
                    'id': f'n{idx + 1}',
                    'type': node.get('type', 'unknown').lower().replace(' ', '_'),
                    'label': node.get('label', node.get('type', 'Node')),
                    'position': {
                        'x': x,
                        'y': y,
                        'width': 150,  # Default size
                        'height': 80
                    },
                    'confidence': node.get('confidence', 0.9),
                    'detected_by': 'ai_vision'
                })
            
            return nodes
            
        except Exception as e:
            print(f"   [WARN] AI Vision detection failed: {e}")
            return []
    
    def _detect_by_color(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect nodes by their characteristic colors in n8n UI
        """
        try:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            detected_regions = []
            
            # Detect colored regions (n8n nodes have distinct colors)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            nodes = []
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filter by size (n8n nodes are medium-sized boxes)
                if 5000 < area < 50000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Aspect ratio check (nodes are roughly rectangular)
                    aspect_ratio = w / float(h) if h > 0 else 0
                    if 0.8 < aspect_ratio < 3.0:
                        nodes.append({
                            'id': f'n{idx + 1}',
                            'type': 'unknown',
                            'label': f'Node {idx + 1}',
                            'position': {'x': x, 'y': y, 'width': w, 'height': h},
                            'confidence': 0.7,
                            'detected_by': 'color_filter'
                        })
            
            return nodes[:10]  # Limit to 10 most likely
            
        except Exception as e:
            print(f"   [WARN] Color detection failed: {e}")
            return []
    
    def _detect_by_contours_enhanced(self, frame: np.ndarray) -> List[Dict]:
        """
        Enhanced contour detection with better preprocessing
        """
        try:
            # Multiple preprocessing approaches
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Approach 1: Adaptive threshold
            thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
            
            # Approach 2: Otsu's thresholding
            _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Combine both
            combined = cv2.bitwise_or(thresh1, thresh2)
            
            # Morphological operations to clean noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            nodes = []
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # n8n nodes are medium-sized rectangles
                if 8000 < area < 60000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Aspect ratio + position filtering
                    aspect_ratio = w / float(h) if h > 0 else 0
                    
                    if 1.0 < aspect_ratio < 2.5 and y > 50:  # Skip top UI bars
                        nodes.append({
                            'id': f'n{idx + 1}',
                            'type': 'unknown',
                            'label': f'Node {idx + 1}',
                            'position': {'x': x, 'y': y, 'width': w, 'height': h},
                            'confidence': 0.65,
                            'detected_by': 'enhanced_contours'
                        })
            
            return nodes[:10]
            
        except Exception as e:
            print(f"   [WARN] Enhanced contour detection failed: {e}")
            return []
    
    def predict_connections_spatial(self, nodes: List[Dict]) -> List[Dict]:
        """
        Predict connections using spatial analysis + AI reasoning
        """
        if len(nodes) < 2:
            return []
        
        connections = []
        
        # Sort by X position (left to right workflow)
        sorted_nodes = sorted(nodes, key=lambda n: n['position']['x'])
        
        # Connect sequential nodes if on similar Y level
        for i in range(len(sorted_nodes) - 1):
            from_node = sorted_nodes[i]
            to_node = sorted_nodes[i + 1]
            
            y_diff = abs(from_node['position']['y'] - to_node['position']['y'])
            x_diff = to_node['position']['x'] - from_node['position']['x']
            
            # If nodes are horizontally aligned and close
            if y_diff < 150 and 100 < x_diff < 600:
                connections.append({
                    'from': from_node['id'],
                    'to': to_node['id'],
                    'type': 'data',
                    'confidence': 0.8,
                    'predicted': True,
                    'reasoning': 'Horizontal spatial proximity'
                })
        
        print(f"   [PREDICT] Spatial analysis: {len(connections)} connections")
        return connections
    
    def draw_debug_visualization(self, frame: np.ndarray, nodes: List[Dict], 
                                 connections: List[Dict], output_path: str):
        """
        Draw comprehensive debug visualization
        """
        annotated = frame.copy()
        
        # Draw connections
        for conn in connections:
            from_n = next((n for n in nodes if n['id'] == conn['from']), None)
            to_n = next((n for n in nodes if n['id'] == conn['to']), None)
            
            if from_n and to_n:
                from_center = (
                    int(from_n['position']['x'] + from_n['position']['width'] / 2),
                    int(from_n['position']['y'] + from_n['position']['height'] / 2)
                )
                to_center = (
                    int(to_n['position']['x'] + to_n['position']['width'] / 2),
                    int(to_n['position']['y'] + to_n['position']['height'] / 2)
                )
                
                color = (0, 255, 0) if not conn.get('predicted') else (255, 165, 0)
                cv2.arrowedLine(annotated, from_center, to_center, color, 3, tipLength=0.02)
        
        # Draw nodes
        for node in nodes:
            pos = node['position']
            x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
            
            # Color based on detection method
            colors = {
                'ai_vision': (0, 255, 0),      # Green
                'color_filter': (255, 165, 0),  # Orange
                'enhanced_contours': (255, 255, 0)  # Yellow
            }
            color = colors.get(node.get('detected_by', 'unknown'), (128, 128, 128))
            
            # Box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
            
            # Label
            label = f"{node['label']} ({node['confidence']:.0%})"
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # ID
            cv2.putText(annotated, node['id'], (x + 5, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Legend
        cv2.putText(annotated, "Detection Methods:", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.rectangle(annotated, (10, 40), (30, 60), (0, 255, 0), -1)
        cv2.putText(annotated, "AI Vision", (40, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)
        print(f"   [VIZ] Debug visualization: {output_path}")
        
        return output_path


def detect_nodes_robust(frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
    """
    Robust node detection entry point
    
    Args:
        frame: Video frame
        
    Returns:
        Tuple of (nodes, connections)
    """
    detector = RobustNodeDetector()
    
    # Detect nodes using multi-strategy
    nodes = detector.detect_nodes_multiStrategy(frame)
    
    # Predict connections
    connections = detector.predict_connections_spatial(nodes) if nodes else []
    
    # Debug visualization
    if nodes:
        detector.draw_debug_visualization(
            frame, nodes, connections,
            './output/robust_detection_debug.png'
        )
    
    return nodes, connections


if __name__ == "__main__":
    # Test with first debug frame
    import glob
    debug_frames = glob.glob('./output/debug/annotated_frame_*.png')
    
    if debug_frames:
        test_frame = cv2.imread(debug_frames[0])
        print(f"\n[TEST] Testing robust detection on: {debug_frames[0]}")
        
        nodes, connections = detect_nodes_robust(test_frame)
        
        print(f"\n[RESULT] Detected {len(nodes)} nodes, {len(connections)} connections")
        for node in nodes:
            print(f"  - {node['id']}: {node['label']} ({node['type']})")
    else:
        print("[ERROR] No debug frames found. Run video analyzer first.")

