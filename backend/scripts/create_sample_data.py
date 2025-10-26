"""
Create sample action sequence JSON for testing.
"""

import json
import os
from datetime import datetime

# Sample workflow data
sample_data = {
    "schema_version": "1.0.0",
    "metadata": {
        "generated_at": datetime.now().isoformat(),
        "analyzer_version": "video_analyzer_v0.3",
        "total_nodes": 3,
        "total_connections": 2
    },
    "nodes": [
        {
            "id": "n1",
            "type": "webhook",
            "label": "Webhook Trigger",
            "position": {"x": 100, "y": 200, "width": 150, "height": 60},
            "confidence": 0.95,
            "config": {
                "httpMethod": "POST",
                "path": "/webhook",
                "responseMode": "responseNode"
            }
        },
        {
            "id": "n2",
            "type": "openai",
            "label": "OpenAI",
            "position": {"x": 350, "y": 200, "width": 150, "height": 60},
            "confidence": 0.92,
            "config": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            }
        },
        {
            "id": "n3",
            "type": "http_request",
            "label": "HTTP Request",
            "position": {"x": 600, "y": 200, "width": 150, "height": 60},
            "confidence": 0.88,
            "config": {
                "method": "GET",
                "url": "https://api.example.com"
            }
        }
    ],
    "connections": [
        {
            "from": "n1",
            "to": "n2",
            "type": "data",
            "confidence": 0.90,
            "anchors": []
        },
        {
            "from": "n2",
            "to": "n3",
            "type": "data",
            "confidence": 0.85,
            "anchors": []
        }
    ],
    "actions": [
        {
            "action": "create_node",
            "node_id": "n1",
            "node_type": "webhook",
            "label": "Webhook Trigger",
            "position": {"x": 100, "y": 200, "width": 150, "height": 60},
            "config": {
                "httpMethod": "POST",
                "path": "/webhook",
                "responseMode": "responseNode"
            }
        },
        {
            "action": "create_node",
            "node_id": "n2",
            "node_type": "openai",
            "label": "OpenAI",
            "position": {"x": 350, "y": 200, "width": 150, "height": 60},
            "config": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            }
        },
        {
            "action": "create_node",
            "node_id": "n3",
            "node_type": "http_request",
            "label": "HTTP Request",
            "position": {"x": 600, "y": 200, "width": 150, "height": 60},
            "config": {
                "method": "GET",
                "url": "https://api.example.com"
            }
        },
        {
            "action": "connect_nodes",
            "from": "n1",
            "to": "n2",
            "connection_type": "data"
        },
        {
            "action": "connect_nodes",
            "from": "n2",
            "to": "n3",
            "connection_type": "data"
        }
    ]
}

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Save sample data
with open('output/action_sequence.json', 'w', encoding='utf-8') as f:
    json.dump(sample_data, f, indent=2, ensure_ascii=False)

print('Sample action_sequence.json created successfully!')
print(f'Created workflow with:')
print(f'   - {len(sample_data["nodes"])} nodes')
print(f'   - {len(sample_data["connections"])} connections')
print(f'   - {len(sample_data["actions"])} actions')
print(f'\nReady for automation testing!')
print(f'\nNext step: run_automation.bat')

