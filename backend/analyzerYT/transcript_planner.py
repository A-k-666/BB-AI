"""
Transcript-Based Workflow Planner
==================================
Uses GPT-4o to understand transcript and generate precise workflow plan.
"""

import json
from openai import OpenAI


class TranscriptPlanner:
    """Generate workflow plan from transcript using GPT-4o"""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
    
    def generate_workflow_plan(self, transcript_text: str, visual_frame_b64: str = None) -> dict:
        """
        Analyze transcript and generate detailed workflow plan
        
        Returns complete workflow JSON with exact node names and configuration
        """
        
        print("\nStep 3.5/4: Generating workflow plan from transcript...")
        
        prompt = f"""You are an n8n workflow expert. Read this tutorial transcript and generate a PERFECT workflow plan.

TRANSCRIPT (Complete tutorial explanation):
{transcript_text}

YOUR TASK:
1. Identify EXACT node names mentioned in the ORDER they appear in tutorial
2. Understand the workflow purpose
3. Map out ALL connections in CORRECT sequence (trigger → process → output)
4. Identify which specific options/variants to select for each node
5. Look for wrapper nodes like "AI Agent" that contain sub-nodes

Return this EXACT JSON structure:
{{
  "workflow_name": "short descriptive name",
  "workflow_purpose": "what this workflow does",
  "nodes": [
    {{
      "name": "EXACT node name from transcript",
      "type": "trigger|action|model|output",
      "search_term": "what to search in n8n",
      "select_option": "specific option if multiple variants exist",
      "configuration_notes": "what settings are mentioned in transcript",
      "order": 1
    }}
  ],
  "connections": [
    {{
      "from": "exact node name",
      "to": "exact node name",
      "connection_type": "main|ai_languageModel|other",
      "notes": "any special connection details from transcript"
    }}
  ],
  "step_by_step_guide": [
    "1. Add WhatsApp Trigger node and configure with credentials",
    "2. Add AI Agent node",
    "3. Connect WhatsApp Trigger to AI Agent main input",
    "etc..."
  ]
}}

CRITICAL RULES:
1. Extract EXACT node names from transcript (don't guess)
2. If transcript says "WhatsApp Business Cloud", use that exact name
3. If multiple variants mentioned, note in "select_option"
4. Order nodes by sequence mentioned in tutorial
5. Include ALL connections mentioned
6. Use transcript details for configuration notes"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at understanding n8n tutorials. Extract exact node names, connections, and workflow structure from tutorial transcripts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=3000,
                temperature=0
            )
            
            plan = json.loads(response.choices[0].message.content)
            
            print(f"   Generated plan:")
            print(f"     Workflow: {plan.get('workflow_name')}")
            print(f"     Nodes: {len(plan.get('nodes', []))}")
            print(f"     Connections: {len(plan.get('connections', []))}")
            
            return plan
            
        except Exception as e:
            print(f"   Planning error: {e}")
            return {"nodes": [], "connections": []}
    
    def convert_plan_to_automation_json(self, plan: dict) -> dict:
        """
        Convert transcript plan to automation-ready JSON format
        """
        
        workflow = {
            "workflow_name": plan.get("workflow_name", "Workflow"),
            "description": plan.get("workflow_purpose", ""),
            "metadata": {
                "understanding_confidence": 95.0,  # High confidence from transcript
                "total_nodes": len(plan.get("nodes", [])),
                "total_connections": len(plan.get("connections", [])),
                "has_audio": True,
                "source": "transcript_analysis"
            },
            "nodes": [],
            "connections": [],
            "actions": []
        }
        
        # Convert nodes
        for i, node in enumerate(plan.get("nodes", [])):
            workflow["nodes"].append({
                "id": f"node{i+1}",
                "name": node.get("name"),
                "type": node.get("type", "action"),
                "category": "Input" if node.get("type") == "trigger" else "Data",
                "position": [128 + i * 350, 288],
                "config": {},
                "ui_hints": {
                    "search_keyword": node.get("search_term", node.get("name")),
                    "select_option": node.get("select_option", ""),
                    "needs_manual_setup": False,
                    "configuration_notes": node.get("configuration_notes", "")
                }
            })
        
        # Convert connections - map names to node IDs
        node_name_to_id = {node.get("name"): f"node{i+1}" 
                          for i, node in enumerate(plan.get("nodes", []))}
        
        for conn in plan.get("connections", []):
            from_id = node_name_to_id.get(conn.get("from"))
            to_id = node_name_to_id.get(conn.get("to"))
            
            if from_id and to_id:
                workflow["connections"].append({
                    "from": from_id,
                    "to": to_id,
                    "type": conn.get("connection_type", "main")
                })
        
        # Build actions
        step_num = 1
        
        # Create nodes
        for node in workflow["nodes"]:
            workflow["actions"].append({
                "step": step_num,
                "action": "create_node",
                "label": node["name"],
                "type": node["type"],
                "search_term": node["ui_hints"]["search_keyword"],
                "select_option": node["ui_hints"]["select_option"],
                "depends_on": [],
                "intent": f"Add {node['name']} node",
                "ai_generated": True
            })
            step_num += 1
        
        # Create connections
        for conn in workflow["connections"]:
            from_node = next((n for n in workflow["nodes"] if n["id"] == conn["from"]), None)
            to_node = next((n for n in workflow["nodes"] if n["id"] == conn["to"]), None)
            
            if from_node and to_node:
                workflow["actions"].append({
                    "step": step_num,
                    "action": "connect_nodes",
                    "from": conn["from"],
                    "to": conn["to"],
                    "connection_type": conn.get("type", "main"),
                    "depends_on": [conn["from"], conn["to"]],
                    "intent": f"Connect {from_node['name']} to {to_node['name']}",
                    "ai_generated": True
                })
                step_num += 1
        
        return workflow


if __name__ == "__main__":
    pass

