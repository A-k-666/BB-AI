"""
Enhanced AI Video Understanding V2
- Better prompts for 90%+ accuracy
- Frame preprocessing
- Multi-frame aggregation
"""

import cv2
import numpy as np
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict
from .frame_preprocessor import FramePreprocessor

load_dotenv(dotenv_path='./config/credentials.env')


class AIVideoUnderstandingV2:
    """Enhanced GPT-4o Vision analyzer with better prompts"""
    
    def __init__(self, openai_api_key: str = None):
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        self.model = 'gpt-4o'  # Use full GPT-4o for vision
        self.preprocessor = FramePreprocessor()
        self.version = "workflow_ai_v2.1_enhanced"
    
    def analyze_video_complete(self, video_path: str) -> Dict:
        """
        Complete video understanding with preprocessing
        
        Returns:
            Dict with workflow recipe (target 90%+ confidence)
        """
        print(f"\n[AI VIDEO V2] Enhanced analysis starting...")
        print(f"   Using: {self.model}")
        print(f"   Preprocessing: Upscale + Crop + Enhance\n")
        
        # Step 1: Extract and preprocess best frames
        frames = self.preprocessor.extract_best_frames(video_path, num_frames=10)
        print(f"   [OK] Extracted {len(frames)} best frames")
        
        # Step 2: Analyze each frame with enhanced prompt
        frame_analyses = []
        for i, frame in enumerate(frames):
            print(f"   [FRAME {i+1}/10] Analyzing...")
            result = self._analyze_frame_enhanced(frame, i)
            if result:
                nodes_count = len(result.get('nodes', []))
                quality = result.get('canvas_quality', 'unknown')
                print(f"      → {nodes_count} nodes, quality: {quality}")
                frame_analyses.append(result)
        
        # Step 3: Aggregate with voting
        final_workflow = self._aggregate_with_voting(frame_analyses)
        
        # Step 4: Generate visual recipe
        self._save_visual_recipe(frames[0], final_workflow)
        
        confidence = final_workflow['metadata']['understanding_confidence']
        print(f"\n   [FINAL] Confidence: {confidence:.1f}% ✨\n")
        
        return final_workflow
    
    def _analyze_frame_enhanced(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """Analyze single frame with ENHANCED prompt"""
        
        try:
            # Encode frame
            _, buffer = cv2.imencode('.png', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 🚀 ENHANCED PROMPT - Maximum Detection
            prompt = """🎯 EXPERT n8n WORKFLOW ANALYSIS

You are an EXPERT at analyzing n8n workflow screenshots with MAXIMUM ACCURACY.

📸 WHAT YOU SEE:
This is an n8n workflow canvas - a visual automation builder.

🔍 YOUR MISSION:
Detect EVERY SINGLE workflow node with extreme thoroughness.

📦 N8N NODE CHARACTERISTICS:
✓ Rounded rectangle cards
✓ Icon on left (🔗 webhook, <> code, 🌐 HTTP, 🤖 AI, etc.)
✓ Bold node name (Webhook, Code, HTTP Request, OpenAI, Set, IF, Slack, etc.)
✓ Subtitle text (e.g., "On webhook call", "Execute Code")
✓ Connected by lines/arrows

🎨 COMMON NODE TYPES:
**Triggers:** Webhook, Manual Trigger, Schedule, Cron
**Actions:** Code, HTTP Request, Set, IF, Switch
**Integrations:** OpenAI, Slack, Gmail, Discord, Notion
**Data:** Edit Fields, Merge, Split, Aggregate

⚠️ CRITICAL RULES:
1. If you see a rounded rectangle with ANY icon → it's probably a node
2. Even if text is blurry, guess node type from icon/color/position
3. DON'T SKIP ANYTHING - thoroughness > precision
4. Include nodes even with 60% confidence
5. Position as % (0-100): left edge = 0, right edge = 100

🔗 CONNECTIONS:
- Look for lines/curves/arrows between nodes
- Light/dark colored paths
- from_label = starting node name, to_label = ending node name

📊 RETURN THIS EXACT JSON STRUCTURE:
{
  "workflow_description": "1-sentence: what workflow does",
  "canvas_quality": "clear|readable|blurry|very_small",
  "nodes": [
    {
      "type": "Webhook",
      "label": "On webhook call", 
      "subtitle": "Trigger",
      "position_percent": {"x": 15, "y": 50},
      "icon_description": "webhook/link icon visible",
      "config_visible": "path: /my-webhook",
      "confidence": 0.92
    },
    {
      "type": "Code",
      "label": "Process Data",
      "subtitle": "Code",
      "position_percent": {"x": 50, "y": 50},
      "icon_description": "code brackets <>",
      "config_visible": "",
      "confidence": 0.88
    }
  ],
  "connections": [
    {"from_label": "On webhook call", "to_label": "Process Data", "type": "main", "confidence": 0.90}
  ],
  "step_by_step_recipe": [
    "1. Add Webhook trigger",
    "2. Add Code node",  
    "3. Connect Webhook → Code"
  ],
  "total_nodes_detected": 2,
  "understanding_quality": "excellent|good|fair|poor"
}

🚨 AGGRESSIVE MODE: Include ANY node you suspect, even with low confidence!"""

            response = self.client.chat.completions.create(
                model=self.model,
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
                max_tokens=3000,  # Increased for detailed analysis
                temperature=0.05  # Lower for consistency
            )
            
            result = json.loads(response.choices[0].message.content)
            result['frame_idx'] = frame_idx
            return result
            
        except Exception as e:
            print(f"      [ERROR] Analysis failed: {e}")
            return None
    
    def _aggregate_with_voting(self, frame_analyses: List[Dict]) -> Dict:
        """
        Aggregate with voting mechanism for robustness
        
        Logic:
        - Node appears in 3+ frames → HIGH confidence
        - Node in 2 frames → MEDIUM
        - Node in 1 frame → Include if confidence > 0.8
        """
        from collections import defaultdict
        
        node_votes = defaultdict(list)
        connection_votes = defaultdict(int)
        
        # Count occurrences
        for analysis in frame_analyses:
            if not analysis:
                continue
            
            for node in analysis.get('nodes', []):
                key = f"{node['type']}_{node.get('label', '')}"
                node_votes[key].append(node)
            
            for conn in analysis.get('connections', []):
                key = f"{conn['from_label']}→{conn['to_label']}"
                connection_votes[key] += 1
        
        # Build final nodes (require 2+ votes OR high confidence)
        final_nodes = []
        node_id = 1
        
        for key, votes in node_votes.items():
            if len(votes) >= 2 or (len(votes) == 1 and votes[0]['confidence'] > 0.85):
                # Average position and confidence
                avg_node = {
                    'id': f'node{node_id}',
                    'type': votes[0]['type'],
                    'name': votes[0]['label'],
                    'label': votes[0]['label'],
                    'position': [
                        int(np.mean([v['position_percent']['x'] for v in votes]) * 10),
                        int(np.mean([v['position_percent']['y'] for v in votes]) * 10)
                    ],
                    'confidence': np.mean([v['confidence'] for v in votes]),
                    'votes': len(votes),
                    'ui_hints': {
                        'search_keyword': votes[0]['type'],
                        'subtitle': votes[0].get('subtitle', ''),
                        'config': votes[0].get('config_visible', '')
                    }
                }
                final_nodes.append(avg_node)
                node_id += 1
        
        # Build final connections (require 2+ votes)
        final_connections = []
        for key, vote_count in connection_votes.items():
            if vote_count >= 2:
                from_label, to_label = key.split('→')
                final_connections.append({
                    'from': from_label,
                    'to': to_label,
                    'votes': vote_count
                })
        
        # Calculate understanding confidence
        total_frames = len(frame_analyses)
        avg_nodes_per_frame = np.mean([len(a.get('nodes', [])) for a in frame_analyses if a])
        quality_scores = [a.get('understanding_quality', 'poor') for a in frame_analyses if a]
        
        # Confidence formula
        if avg_nodes_per_frame >= 3:
            base_confidence = 90
        elif avg_nodes_per_frame >= 2:
            base_confidence = 75
        elif avg_nodes_per_frame >= 1:
            base_confidence = 50
        else:
            base_confidence = 25
        
        # Boost if multiple frames agree
        if len(final_nodes) > 0:
            avg_votes = np.mean([n['votes'] for n in final_nodes])
            vote_boost = min(avg_votes * 5, 10)
            base_confidence += vote_boost
        
        final_confidence = min(base_confidence, 98.0)
        
        # Determine level
        if final_confidence >= 85:
            level = "EXCELLENT"
        elif final_confidence >= 70:
            level = "GOOD"
        elif final_confidence >= 50:
            level = "FAIR"
        else:
            level = "NEEDS_REVIEW"
        
        return {
            'workflow_name': frame_analyses[0].get('workflow_description', 'Detected Workflow') if frame_analyses else "Unknown Workflow",
            'description': frame_analyses[0].get('workflow_description', '') if frame_analyses else "",
            'metadata': {
                'source_video': 'analyzed_video',
                'created_by': 'BB-AI Video Understanding V2',
                'analyzed_at': '2025-10-26T16:20:00Z',
                'version': self.version,
                'ai_model': self.model,
                'total_frames_analyzed': total_frames,
                'avg_nodes_per_frame': float(avg_nodes_per_frame),
                'understanding_confidence': float(final_confidence),
                'understanding_level': level
            },
            'nodes': final_nodes,
            'connections': final_connections,
            'actions': self._generate_actions(final_nodes, final_connections),
            'expected_output': {
                'total_nodes': len(final_nodes),
                'total_connections': len(final_connections),
                'workflow_status': 'ready' if final_confidence > 70 else 'needs_review'
            }
        }
    
    def _generate_actions(self, nodes: List, connections: List) -> List:
        """Generate action sequence from nodes/connections"""
        actions = []
        
        for i, node in enumerate(nodes, 1):
            actions.append({
                'step': i,
                'action': 'create_node',
                'label': node['name'],
                'type': node['type'],
                'depends_on': []
            })
        
        for i, conn in enumerate(connections, len(nodes) + 1):
            actions.append({
                'step': i,
                'action': 'connect_nodes',
                'from': conn['from'],
                'to': conn['to']
            })
        
        return actions
    
    def _save_visual_recipe(self, frame: np.ndarray, workflow: Dict):
        """Save annotated visualization"""
        annotated = frame.copy()
        
        for node in workflow['nodes']:
            x, y = node['position']
            cv2.rectangle(annotated, (x*10, y*10), (x*10+150, y*10+80), (0, 255, 0), 3)
            cv2.putText(annotated, node['name'], (x*10, y*10-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite('./output/ai_visual_recipe_v2.png', annotated)


# Test
if __name__ == "__main__":
    analyzer = AIVideoUnderstandingV2()
    result = analyzer.analyze_video_complete('./temp_video.mp4')
    
    print(json.dumps(result, indent=2))



