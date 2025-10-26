#!/usr/bin/env python3
"""
AI Enhancement Layer for Video Analyzer
Enhances raw CV/OCR output with GPT-4o Vision intelligence
"""

import os
import json
import base64
from typing import Dict, List, Any
from datetime import datetime
import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv('./config/credentials.env')

class WorkflowAIEnhancer:
    """
    AI-powered enhancement of raw video analysis results
    """
    
    def __init__(self, openai_api_key: str = None):
        """Initialize AI enhancer with OpenAI client"""
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        self.model = os.getenv('AI_MODEL', 'gpt-4o-mini')
        self.version = "workflow_ai_v1.2"
        
    def enhance_workflow(self, raw_json: Dict, reference_frame: np.ndarray = None) -> Dict:
        """
        Main enhancement function
        
        Args:
            raw_json: Raw output from video_analyzer
            reference_frame: Best keyframe for visual analysis
            
        Returns:
            Enhanced JSON with AI predictions
        """
        print("\n[AI] Enhancing workflow with AI intelligence...")
        
        enhanced = {
            "workflow_name": "AI-Analyzed n8n Workflow",
            "description": "Automatically enhanced with GPT-4o Vision",
            "metadata": {
                **raw_json.get('metadata', {}),
                "ai_model_version": self.version,
                "ai_model": self.model,
                "enhanced_at": datetime.now().isoformat()
            },
            "nodes": [],
            "connections": [],
            "actions": [],
            "expected_output": {}
        }
        
        # 1. Enhance nodes with AI classification
        nodes = raw_json.get('nodes', [])
        stable_count = 0
        
        print(f"   [CLASSIFY] Processing {len(nodes)} nodes...")
        for node in nodes:
            enhanced_node = self._enhance_node(node, reference_frame)
            enhanced['nodes'].append(enhanced_node)
            if enhanced_node.get('stable'):
                stable_count += 1
        
        # 2. Predict connections if missing
        connections = raw_json.get('connections', [])
        if len(connections) == 0 and len(nodes) > 1:
            print(f"   [PREDICT] Predicting connections (none detected)...")
            connections = self._predict_connections(enhanced['nodes'])
        else:
            connections = self._enhance_connections(connections, enhanced['nodes'])
        
        enhanced['connections'] = connections
        
        # 3. Generate smart action sequence
        enhanced['actions'] = self._generate_smart_actions(enhanced['nodes'], connections)
        
        # 4. Calculate analysis quality
        total_nodes = len(enhanced['nodes'])
        predicted_nodes = sum(1 for n in enhanced['nodes'] if n.get('predicted', False))
        type_confidence_avg = sum(n.get('type_confidence', 0) for n in enhanced['nodes']) / max(total_nodes, 1)
        
        enhanced['metadata']['analysis_quality'] = self._calculate_quality_score(
            total_nodes, stable_count, len(connections), type_confidence_avg
        )
        enhanced['metadata']['stable_nodes_count'] = stable_count
        enhanced['metadata']['unstable_nodes_count'] = total_nodes - stable_count
        
        # 5. Expected output
        enhanced['expected_output'] = {
            "total_nodes": total_nodes,
            "total_connections": len(connections),
            "workflow_status": "ready_to_execute" if type_confidence_avg > 0.7 else "needs_review"
        }
        
        print(f"   [OK] Enhanced: {total_nodes} nodes, {len(connections)} connections")
        print(f"   [SCORE] Quality Score: {enhanced['metadata']['analysis_quality']:.1f}%")
        
        return enhanced
    
    def _enhance_node(self, node: Dict, reference_frame: np.ndarray = None) -> Dict:
        """
        Enhance single node with AI classification
        """
        label = node.get('label', node.get('text', 'Unknown'))
        node_type = node.get('type', 'unknown')
        
        # AI classify if type is unknown or label is garbage
        if node_type == 'unknown' or len(label) <= 3:
            predicted_type, confidence = self._ai_classify_node(label, node.get('position'))
        else:
            predicted_type = node_type
            confidence = 0.95
        
        # Determine stability based on frame presence
        frame_count = node.get('frame_last_seen', 1)
        stable = frame_count >= 3  # Seen in at least 3 frames
        
        enhanced = {
            "id": node.get('id'),
            "name": self._get_clean_label(label, predicted_type),
            "label": label,  # Keep original OCR
            "type": predicted_type,
            "type_confidence": confidence,
            "predicted": predicted_type != node.get('type', 'unknown'),
            "stable": stable,
            "category": self._get_node_category(predicted_type),
            "position": node.get('position', {}),
            "confidence": node.get('confidence', 0.8),
            "frame_last_seen": frame_count,
            "config": self._get_smart_config(predicted_type),
            "ui_hints": {
                "search_keyword": self._get_search_keyword(predicted_type),
                "needs_manual_setup": self._needs_manual_setup(predicted_type)
            }
        }
        
        return enhanced
    
    def _ai_classify_node(self, label: str, position: Dict) -> tuple:
        """
        Use AI to classify node type from OCR text + position
        """
        try:
            prompt = f"""You are an n8n workflow expert. Classify this node:

OCR Label: "{label}"
Position: x={position.get('x', 0)}, y={position.get('y', 0)}

Common n8n nodes: Webhook, HTTP Request, Code, Set, IF, Switch, OpenAI, Schedule Trigger, Email Send

Return JSON: {{"type": "node_type", "confidence": 0.0-1.0, "reasoning": "why"}}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=150
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('type', 'unknown'), result.get('confidence', 0.5)
            
        except Exception as e:
            print(f"   [WARN] AI classification failed: {e}")
            return 'unknown', 0.3
    
    def _predict_connections(self, nodes: List[Dict]) -> List[Dict]:
        """
        Predict connections based on node positions and types
        """
        connections = []
        
        # Sort nodes by x position (left to right flow)
        sorted_nodes = sorted(nodes, key=lambda n: n['position'].get('x', 0))
        
        # Simple heuristic: connect adjacent nodes in sequence
        for i in range(len(sorted_nodes) - 1):
            from_node = sorted_nodes[i]
            to_node = sorted_nodes[i + 1]
            
            # Check if they're on similar Y level (horizontal flow)
            y_diff = abs(from_node['position'].get('y', 0) - to_node['position'].get('y', 0))
            
            if y_diff < 200:  # Same row
                confidence = 0.85 if from_node.get('stable') and to_node.get('stable') else 0.6
                
                connections.append({
                    "from": from_node['id'],
                    "to": to_node['id'],
                    "type": "data",
                    "predicted": True,
                    "confidence": confidence,
                    "reasoning": "Horizontal flow detected"
                })
        
        print(f"   [AI] Predicted {len(connections)} connections")
        return connections
    
    def _enhance_connections(self, connections: List[Dict], nodes: List[Dict]) -> List[Dict]:
        """
        Enhance detected connections with additional metadata
        """
        enhanced = []
        for conn in connections:
            enhanced.append({
                **conn,
                "predicted": False,  # Detected by CV, not predicted
                "reasoning": "Detected from video analysis"
            })
        return enhanced
    
    def _generate_smart_actions(self, nodes: List[Dict], connections: List[Dict]) -> List[Dict]:
        """
        Generate intelligent action sequence with dependencies
        """
        actions = []
        step = 1
        
        # Sort nodes by position (triggers first, then left-to-right)
        trigger_nodes = [n for n in nodes if n.get('type') == 'trigger']
        action_nodes = [n for n in nodes if n.get('type') != 'trigger']
        sorted_nodes = trigger_nodes + sorted(action_nodes, key=lambda n: n['position'].get('x', 0))
        
        # Create node actions
        for node in sorted_nodes:
            actions.append({
                "step": step,
                "action": "create_node",
                "node_id": node['id'],
                "label": node['name'],
                "type": node['type'],
                "depends_on": [],
                "predicted": node.get('predicted', False),
                "intent": f"Add {node['name']} node to workflow"
            })
            step += 1
        
        # Create connection actions
        for conn in connections:
            from_label = next((n['name'] for n in nodes if n['id'] == conn['from']), 'Unknown')
            to_label = next((n['name'] for n in nodes if n['id'] == conn['to']), 'Unknown')
            
            actions.append({
                "step": step,
                "action": "connect_nodes",
                "from": conn['from'],
                "to": conn['to'],
                "depends_on": [conn['from'], conn['to']],
                "predicted": conn.get('predicted', False),
                "intent": f"Connect {from_label} output to {to_label} input"
            })
            step += 1
        
        return actions
    
    def _get_clean_label(self, ocr_label: str, predicted_type: str) -> str:
        """
        Clean OCR garbage and map to proper node name
        """
        if len(ocr_label) <= 3 or not ocr_label.replace(' ', '').isalpha():
            # OCR is garbage, use predicted type as label
            return predicted_type
        
        # OCR is reasonable, keep it
        return ocr_label.title()
    
    def _get_node_category(self, node_type: str) -> str:
        """
        Categorize node by type
        """
        categories = {
            'trigger': 'Input',
            'webhook': 'Input',
            'schedule': 'Input',
            'action': 'Data',
            'code': 'Data',
            'set': 'Data',
            'if': 'Logic',
            'switch': 'Logic',
            'openai': 'AI',
            'http_request': 'Network',
            'email': 'Communication',
            'output': 'Output'
        }
        return categories.get(node_type.lower(), 'Utility')
    
    def _get_smart_config(self, node_type: str) -> Dict:
        """
        Generate smart default config based on node type
        """
        configs = {
            'webhook': {
                "path": "/webhook",
                "method": "POST",
                "responseMode": "lastNode"
            },
            'http_request': {
                "method": "GET",
                "url": "https://api.example.com",
                "responseFormat": "json"
            },
            'openai': {
                "model": "gpt-4o-mini",
                "prompt": "{{ $json.input }}"
            },
            'code': {
                "mode": "runOnceForAllItems",
                "code": "// Process items\nreturn items;"
            },
            'set': {
                "mode": "manual",
                "values": {}
            }
        }
        return configs.get(node_type.lower(), {})
    
    def _get_search_keyword(self, node_type: str) -> str:
        """
        Get n8n UI search keyword for node type
        """
        keywords = {
            'webhook': 'Webhook',
            'http_request': 'HTTP Request',
            'openai': 'OpenAI',
            'code': 'Code',
            'set': 'Set',
            'if': 'IF',
            'switch': 'Switch',
            'schedule': 'Schedule Trigger',
            'email': 'Email Send',
            'trigger': 'Webhook'
        }
        return keywords.get(node_type.lower(), node_type.title())
    
    def _needs_manual_setup(self, node_type: str) -> bool:
        """
        Check if node requires manual configuration
        """
        manual_setup_nodes = ['openai', 'email', 'http_request', 'database']
        return node_type.lower() in manual_setup_nodes
    
    def _calculate_quality_score(self, total_nodes: int, stable_nodes: int, 
                                  total_connections: int, avg_confidence: float) -> float:
        """
        Calculate overall analysis quality percentage
        """
        stability_score = (stable_nodes / max(total_nodes, 1)) * 100
        connection_score = min((total_connections / max(total_nodes - 1, 1)) * 100, 100)
        confidence_score = avg_confidence * 100
        
        # Weighted average
        quality = (stability_score * 0.3 + connection_score * 0.3 + confidence_score * 0.4)
        return round(quality, 1)
    
    def draw_analysis_visualization(self, frame: np.ndarray, nodes: List[Dict], 
                                   connections: List[Dict], output_path: str) -> str:
        """
        Draw annotated frame with bounding boxes, labels, and connections
        
        Args:
            frame: Video frame
            nodes: Enhanced nodes
            connections: Enhanced connections
            output_path: Path to save visualization
            
        Returns:
            Path to saved image
        """
        print("   [VIZ] Drawing analysis visualization...")
        
        if frame is None or frame.size == 0:
            print("   [WARN] Invalid frame, skipping visualization")
            return None
        
        annotated = frame.copy()
        
        # Draw connections first (behind nodes)
        for conn in connections:
            from_node = next((n for n in nodes if n['id'] == conn['from']), None)
            to_node = next((n for n in nodes if n['id'] == conn['to']), None)
            
            if from_node and to_node:
                from_pos = from_node['position']
                to_pos = to_node['position']
                
                # Calculate center points
                from_center = (
                    int(from_pos['x'] + from_pos['width'] / 2),
                    int(from_pos['y'] + from_pos['height'] / 2)
                )
                to_center = (
                    int(to_pos['x'] + to_pos['width'] / 2),
                    int(to_pos['y'] + to_pos['height'] / 2)
                )
                
                # Draw connection line
                color = (0, 255, 0) if not conn.get('predicted') else (255, 165, 0)  # Green = detected, Orange = predicted
                cv2.arrowedLine(annotated, from_center, to_center, color, 3, tipLength=0.03)
        
        # Draw nodes
        for node in nodes:
            pos = node['position']
            x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
            
            # Color based on stability and prediction
            if node.get('stable') and not node.get('predicted'):
                color = (0, 255, 0)  # Green = stable + detected
            elif node.get('predicted'):
                color = (255, 165, 0)  # Orange = AI predicted
            else:
                color = (0, 165, 255)  # Yellow = unstable
            
            # Draw rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
            
            # Draw label with type and confidence
            label_text = f"{node['name']} ({node['type_confidence']:.0%})"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
            
            # Text
            cv2.putText(annotated, label_text, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Node ID
            cv2.putText(annotated, node['id'], (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add legend
        legend_y = 30
        cv2.putText(annotated, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.rectangle(annotated, (10, legend_y + 10), (30, legend_y + 30), (0, 255, 0), -1)
        cv2.putText(annotated, "Stable + Detected", (40, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.rectangle(annotated, (10, legend_y + 40), (30, legend_y + 60), (255, 165, 0), -1)
        cv2.putText(annotated, "AI Predicted", (40, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)
        print(f"   [OK] Visualization saved: {output_path}")
        
        return output_path
    
    def _ai_classify_node(self, label: str, position: Dict) -> tuple:
        """
        AI-powered node type classification
        """
        # Common n8n node patterns
        patterns = {
            'webhook': ['webhook', 'web', 'hook', 'trigger', 'http', 'incoming'],
            'code': ['code', 'function', 'script', 'js', 'python'],
            'http_request': ['http', 'request', 'api', 'get', 'post', 'rest'],
            'openai': ['openai', 'gpt', 'ai', 'chat', 'completion'],
            'set': ['set', 'variable', 'assign', 'data'],
            'if': ['if', 'condition', 'branch', 'switch'],
            'email': ['email', 'mail', 'send', 'smtp']
        }
        
        # Simple pattern matching first
        label_lower = label.lower()
        for node_type, keywords in patterns.items():
            if any(kw in label_lower for kw in keywords):
                return node_type, 0.85
        
        # Position-based heuristic
        x = position.get('x', 0)
        if x < 300:
            return 'trigger', 0.6  # Left side = likely trigger
        elif x > 800:
            return 'action', 0.5  # Right side = likely output
        
        return 'unknown', 0.3
    
    def _get_clean_label(self, ocr_label: str, predicted_type: str) -> str:
        """
        Map OCR garbage to clean label
        """
        if len(ocr_label) <= 3 or not ocr_label.replace(' ', '').isalpha():
            # OCR is garbage
            return self._get_search_keyword(predicted_type)
        return ocr_label.title()
    
    def _get_node_category(self, node_type: str) -> str:
        """Get semantic category"""
        categories = {
            'trigger': 'Input',
            'webhook': 'Input',
            'schedule': 'Input',
            'action': 'Data',
            'code': 'Data',
            'set': 'Data',
            'if': 'Logic',
            'switch': 'Logic',
            'openai': 'AI',
            'http_request': 'Network',
            'email': 'Communication'
        }
        return categories.get(node_type.lower(), 'Utility')
    
    def _get_smart_config(self, node_type: str) -> Dict:
        """Generate default config"""
        configs = {
            'webhook': {"path": "/webhook", "method": "POST"},
            'http_request': {"method": "GET", "url": "https://api.example.com"},
            'openai': {"model": "gpt-4o-mini", "prompt": "{{ $json.input }}"},
            'code': {"mode": "runOnceForAllItems", "code": "return items;"},
            'set': {"mode": "manual", "values": {}}
        }
        return configs.get(node_type.lower(), {})
    
    def _get_search_keyword(self, node_type: str) -> str:
        """Get n8n search keyword"""
        keywords = {
            'webhook': 'Webhook',
            'http_request': 'HTTP Request',
            'openai': 'OpenAI',
            'code': 'Code',
            'set': 'Set',
            'if': 'IF',
            'email': 'Email Send',
            'trigger': 'Webhook'
        }
        return keywords.get(node_type.lower(), node_type.title())
    
    def _needs_manual_setup(self, node_type: str) -> bool:
        """Check if manual setup needed"""
        return node_type.lower() in ['openai', 'email', 'database', 'http_request']
    
    def _generate_smart_actions(self, nodes: List[Dict], connections: List[Dict]) -> List[Dict]:
        """Generate action sequence with dependencies"""
        actions = []
        step = 1
        
        # Create nodes first
        for node in nodes:
            actions.append({
                "step": step,
                "action": "create_node",
                "label": node['name'],
                "type": node['type'],
                "depends_on": [],
                "intent": f"Add {node['name']} node"
            })
            step += 1
        
        # Then connections
        for conn in connections:
            actions.append({
                "step": step,
                "action": "connect_nodes",
                "from": conn['from'],
                "to": conn['to'],
                "intent": "Connect nodes"
            })
            step += 1
        
        return actions


def enhance_video_analysis(raw_json_path: str, output_path: str, 
                           reference_frame_path: str = None) -> Dict:
    """
    Main enhancement function
    
    Args:
        raw_json_path: Path to raw analysis JSON
        output_path: Path to save enhanced JSON
        reference_frame_path: Optional path to reference frame
        
    Returns:
        Enhanced workflow JSON
    """
    # Load raw JSON
    with open(raw_json_path, 'r') as f:
        raw_json = json.load(f)
    
    # Load reference frame if provided
    reference_frame = None
    if reference_frame_path and os.path.exists(reference_frame_path):
        reference_frame = cv2.imread(reference_frame_path)
    
    # Enhance
    enhancer = WorkflowAIEnhancer()
    enhanced = enhancer.enhance_workflow(raw_json, reference_frame)
    
    # Save enhanced JSON
    with open(output_path, 'w') as f:
        json.dump(enhanced, f, indent=2)
    
    print(f"\n[OK] Enhanced JSON saved: {output_path}")
    
    # Draw visualization if frame available
    if reference_frame is not None:
        viz_path = output_path.replace('.json', '_visualization.png')
        enhancer.draw_analysis_visualization(
            reference_frame, 
            enhanced['nodes'], 
            enhanced['connections'],
            viz_path
        )
    
    return enhanced


if __name__ == "__main__":
    # Test enhancement
    enhance_video_analysis(
        './output/analysis_results.json',
        './output/action_sequence_enhanced.json'
    )

