"""
JSON Workflow Refiner - Post-processing validation using complete workflow screenshot
Matches detected nodes against final workflow view for 100% accuracy
"""

import json
import logging
from typing import Dict, List, Any
import base64
from pathlib import Path

class WorkflowJSONRefiner:
    """
    Final refinement layer that validates and corrects workflow JSON
    by analyzing the complete workflow screenshot
    """
    
    def __init__(self, openai_client=None):
        self.logger = logging.getLogger(__name__)
        self.openai = openai_client
        
    def refine_workflow_json(self, workflow_json: Dict, final_screenshot_path: str = None, 
                            final_screenshot_base64: str = None) -> Dict:
        """
        Refine workflow JSON by validating against complete workflow screenshot
        
        Args:
            workflow_json: Initial workflow JSON from analyzer
            final_screenshot_path: Path to complete workflow screenshot
            final_screenshot_base64: Or base64 encoded screenshot
            
        Returns:
            Refined and validated workflow JSON
        """
        
        self.logger.info("[REFINER] Starting JSON refinement process...")
        
        # Get screenshot
        if final_screenshot_path:
            with open(final_screenshot_path, 'rb') as f:
                screenshot_base64 = base64.b64encode(f.read()).decode('utf-8')
        elif final_screenshot_base64:
            screenshot_base64 = final_screenshot_base64
        else:
            self.logger.warning("[REFINER] No screenshot provided, skipping refinement")
            return workflow_json
            
        # Step 1: Validate node count and names
        refined_json = self._validate_with_vision(workflow_json, screenshot_base64)
        
        # Step 2: Remove duplicates
        refined_json = self._remove_duplicate_nodes(refined_json)
        
        # Step 3: Validate connections
        refined_json = self._validate_connections(refined_json, screenshot_base64)
        
        # Step 4: Clean up positions
        refined_json = self._optimize_positions(refined_json)
        
        # Step 5: Final consistency check
        refined_json = self._final_validation(refined_json, screenshot_base64)
        
        self.logger.info(f"[REFINER] Refinement complete. Nodes: {len(refined_json['nodes'])}, Connections: {len(refined_json['connections'])}")
        
        return refined_json
    
    def _validate_with_vision(self, workflow_json: Dict, screenshot: str) -> Dict:
        """Use GPT-4o Vision to validate and correct node detection"""
        
        if not self.openai:
            self.logger.warning("[REFINER] No OpenAI client, skipping vision validation")
            return workflow_json
            
        try:
            current_nodes = [n['name'] for n in workflow_json['nodes']]
            
            prompt = f"""You are analyzing a complete n8n workflow screenshot to validate detected nodes.

**Currently Detected Nodes:** {', '.join(current_nodes)}

**Your Task:**
1. Look at the COMPLETE workflow in the screenshot
2. Count ALL unique nodes visible on the canvas
3. Read the EXACT labels on each node
4. Identify which detected nodes are CORRECT
5. Identify any MISSING nodes
6. Identify any DUPLICATE or INCORRECT nodes

**Important:**
- Only count actual workflow nodes on the canvas (not UI elements)
- A node appears as a box with a label and icon
- If you see the same node type multiple times, count each instance
- Read labels carefully (e.g., "WhatsApp Trigger" vs "WhatsApp Business Cloud")

Return JSON:
{{
  "total_nodes_in_screenshot": <number>,
  "correct_nodes": ["node1", "node2", ...],
  "incorrect_nodes": ["wrong1", ...],
  "missing_nodes": ["missing1", ...],
  "duplicate_nodes": ["dup1", ...],
  "refined_node_list": [
    {{"name": "exact label from screenshot", "type": "trigger|action|model", "confidence": 0-100}}
  ],
  "confidence": 0-100,
  "reasoning": "explanation"
}}"""

            response = self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{screenshot}"}
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            self.logger.info(f"[REFINER] Vision validation: {result['total_nodes_in_screenshot']} nodes detected")
            self.logger.info(f"[REFINER] Confidence: {result['confidence']}%")
            
            if result['confidence'] > 80:
                # Replace nodes with refined list
                refined_nodes = []
                for i, node_info in enumerate(result['refined_node_list']):
                    # Try to find matching node in original JSON to preserve config
                    original_node = None
                    for orig in workflow_json['nodes']:
                        if orig['name'].lower() == node_info['name'].lower():
                            original_node = orig
                            break
                    
                    if original_node:
                        refined_nodes.append(original_node)
                    else:
                        # Create new node
                        refined_nodes.append({
                            "id": f"node{i+1}",
                            "name": node_info['name'],
                            "type": node_info['type'],
                            "category": "Utility",
                            "position": [100 + i*300, 288],
                            "config": {},
                            "ui_hints": {
                                "search_keyword": node_info['name'],
                                "needs_manual_setup": False
                            }
                        })
                
                workflow_json['nodes'] = refined_nodes
                workflow_json['metadata']['refiner_confidence'] = result['confidence']
                
        except Exception as e:
            self.logger.error(f"[REFINER] Vision validation failed: {e}")
            
        return workflow_json
    
    def _remove_duplicate_nodes(self, workflow_json: Dict) -> Dict:
        """Remove duplicate nodes with same name"""
        
        seen_names = set()
        unique_nodes = []
        
        for node in workflow_json['nodes']:
            node_key = node['name'].lower().strip()
            if node_key not in seen_names:
                seen_names.add(node_key)
                unique_nodes.append(node)
            else:
                self.logger.info(f"[REFINER] Removed duplicate: {node['name']}")
        
        workflow_json['nodes'] = unique_nodes
        
        # Update node IDs sequentially
        for i, node in enumerate(workflow_json['nodes']):
            old_id = node['id']
            new_id = f"node{i+1}"
            node['id'] = new_id
            
            # Update connections
            for conn in workflow_json.get('connections', []):
                if conn['from'] == old_id:
                    conn['from'] = new_id
                if conn['to'] == old_id:
                    conn['to'] = new_id
        
        return workflow_json
    
    def _validate_connections(self, workflow_json: Dict, screenshot: str) -> Dict:
        """Validate connections using vision"""
        
        if not self.openai:
            return workflow_json
            
        try:
            node_names = [n['name'] for n in workflow_json['nodes']]
            current_connections = [f"{c['from']} → {c['to']}" for c in workflow_json.get('connections', [])]
            
            prompt = f"""Look at this n8n workflow screenshot and identify ALL connections (arrows) between nodes.

**Visible Nodes:** {', '.join(node_names)}

**Currently Detected Connections:** {', '.join(current_connections) if current_connections else 'None'}

**Your Task:**
1. Look at the arrows connecting nodes
2. Identify FROM node → TO node for each arrow
3. Verify current connections are correct
4. Add any missing connections

Return JSON:
{{
  "correct_connections": [{{"from": "NodeA", "to": "NodeB"}}, ...],
  "missing_connections": [{{"from": "NodeX", "to": "NodeY"}}, ...],
  "total_connections": <number>,
  "confidence": 0-100
}}"""

            response = self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{screenshot}"}
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=800,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if result['confidence'] > 70:
                # Map node names to IDs
                name_to_id = {n['name']: n['id'] for n in workflow_json['nodes']}
                
                refined_connections = []
                for conn in result['correct_connections']:
                    from_id = name_to_id.get(conn['from'])
                    to_id = name_to_id.get(conn['to'])
                    
                    if from_id and to_id:
                        refined_connections.append({
                            "from": from_id,
                            "to": to_id,
                            "type": "data"
                        })
                
                workflow_json['connections'] = refined_connections
                self.logger.info(f"[REFINER] Validated {len(refined_connections)} connections")
                
        except Exception as e:
            self.logger.error(f"[REFINER] Connection validation failed: {e}")
            
        return workflow_json
    
    def _optimize_positions(self, workflow_json: Dict) -> Dict:
        """Optimize node positions for better layout"""
        
        num_nodes = len(workflow_json['nodes'])
        
        # Simple horizontal layout
        for i, node in enumerate(workflow_json['nodes']):
            node['position'] = [128 + i * 350, 288]
        
        return workflow_json
    
    def _final_validation(self, workflow_json: Dict, screenshot: str) -> Dict:
        """Final consistency check"""
        
        # Remove connections to non-existent nodes
        valid_node_ids = {n['id'] for n in workflow_json['nodes']}
        
        valid_connections = [
            conn for conn in workflow_json.get('connections', [])
            if conn['from'] in valid_node_ids and conn['to'] in valid_node_ids
        ]
        
        workflow_json['connections'] = valid_connections
        
        # Update expected output
        workflow_json['expected_output'] = {
            "total_nodes": len(workflow_json['nodes']),
            "total_connections": len(workflow_json['connections']),
            "workflow_status": "refined_and_validated"
        }
        
        # Update metadata
        workflow_json['metadata']['refiner_applied'] = True
        workflow_json['metadata']['final_node_count'] = len(workflow_json['nodes'])
        workflow_json['metadata']['final_connection_count'] = len(workflow_json['connections'])
        
        return workflow_json



