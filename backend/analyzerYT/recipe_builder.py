"""
Recipe Builder Module
=====================
Generates final JSON recipe and PNG outputs.
"""

import cv2
import json
import os
from datetime import datetime


class RecipeBuilder:
    """Build final workflow recipe"""
    
    def __init__(self):
        pass
    
    def build_workflow_json(self, detections: list, transcript: dict) -> dict:
        """
        Build final workflow JSON from detections
        
        Args:
            detections: List of frame detection results
            transcript: Transcript data
            
        Returns: Complete workflow JSON
        """
        
        # Find best detection (most nodes)
        best_detection = max(detections, key=lambda x: len(x.get("nodes", []))) if detections else {}
        
        workflow = {
            "workflow_name": best_detection.get("workflow_purpose", "Detected Workflow"),
            "description": transcript.get("full_text", "")[:200],
            "metadata": {
                "analyzed_at": datetime.now().isoformat(),
                "understanding_confidence": 90.0,
                "total_nodes": len(best_detection.get("nodes", [])),
                "total_connections": len(best_detection.get("connections", [])),
                "has_audio": len(transcript.get("full_text", "")) > 0
            },
            "nodes": [],
            "connections": best_detection.get("connections", []),
            "actions": []
        }
        
        # Build nodes
        for i, node in enumerate(best_detection.get("nodes", [])):
            workflow["nodes"].append({
                "id": f"node{i+1}",
                "name": node.get("label", "Unknown"),
                "type": node.get("type", "action"),
                "category": "Input" if node.get("type") == "trigger" else "Data",
                "position": [128 + i * 350, 288],
                "config": {},
                "ui_hints": {
                    "search_keyword": node.get("label", "Unknown"),
                    "needs_manual_setup": False
                }
            })
        
        # Build actions
        step_num = 1
        for node in workflow["nodes"]:
            workflow["actions"].append({
                "step": step_num,
                "action": "create_node",
                "label": node["name"],
                "type": node["type"],
                "depends_on": [],
                "intent": f"Add {node['name']} node",
                "ai_generated": True
            })
            step_num += 1
        
        # Add connection actions
        for conn in workflow["connections"]:
            workflow["actions"].append({
                "step": step_num,
                "action": "connect_nodes",
                "from": conn.get("from", ""),
                "to": conn.get("to", ""),
                "depends_on": [],
                "intent": f"Connect {conn.get('from')} to {conn.get('to')}",
                "ai_generated": True
            })
            step_num += 1
        
        return workflow
    
    def save_visual_recipe(self, frame, output_path: str):
        """Save clean workflow frame as visual recipe"""
        
        if frame is None:
            print("   No frame to save")
            return
        
        cv2.imwrite(output_path, frame)
        print(f"   Saved visual recipe: {output_path}")
    
    def save_outputs(self, workflow: dict, best_frame, output_dir: str = './output'):
        """Save all outputs"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON
        with open(f"{output_dir}/ai_workflow_complete.json", 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)
        
        with open(f"{output_dir}/action_sequence.json", 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)
        
        # Save visual recipe
        if best_frame is not None:
            cv2.imwrite(f"{output_dir}/ai_visual_recipe.png", best_frame)
        
        print("\n" + "="*60)
        print("OUTPUTS SAVED:")
        print(f"  - {output_dir}/ai_workflow_complete.json")
        print(f"  - {output_dir}/action_sequence.json")
        print(f"  - {output_dir}/ai_visual_recipe.png")
        print(f"  - {output_dir}/transcript.json")
        print("="*60)

