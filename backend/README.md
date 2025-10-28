# 🤖 BB-AI - YouTube to n8n Workflow Automation

**Automatically analyzes YouTube tutorial videos and replicates workflows in n8n**

---

## 📂 Project Structure

```
backend/
├── 📁 analyzer/           ⭐ Python video analysis engine
│   ├── video_analyzer.py           # CV/OCR frame processing
│   ├── ai_video_understanding.py   # GPT-4o Vision analysis
│   ├── ai_enhancer.py              # AI classification layer
│   ├── robust_node_detector.py     # Multi-strategy detection
│   ├── utils.py                    # Helper functions
│   └── __init__.py                 # Package initialization
│
├── 📁 automation/         ⭐ Node.js n8n automation
│   ├── ai_playwright_agent.js      # Main AI automation agent
│   ├── n8n_ui_expert.js            # RAG-based UI expert
│   ├── playwright_agent.js         # Legacy agent (backup)
│   ├── package.json                # npm dependencies
│   └── README.md                   # Automation guide
│
├── 📁 config/             🔒 Configuration & credentials
│   └── credentials.env             # API keys, n8n credentials
│
├── 📁 output/             📊 Analysis results & screenshots
│   ├── ai_workflow_complete.json   # AI-generated workflow
│   ├── action_sequence.json        # Enhanced workflow JSON
│   ├── execution_log.json          # Automation results
│   ├── ai_action_log.json          # AI reasoning logs
│   └── screenshots/                # Debug screenshots
│       ├── login/
│       ├── nodes/
│       ├── connections/
│       └── final/                  # Preserved final workflows
│
├── 📁 docs/               📚 Documentation
│   ├── n8n_docs_combined.md        # Complete n8n docs (3.9MB)
│   ├── COMMANDS.md                 # Quick reference
│   └── AI_AGENT_README.md          # AI features guide
│
├── 📁 scripts/            🛠️ Utility scripts
│   ├── download_n8n_docs.bat       # Clone n8n documentation
│   └── create_sample_data.py       # Test data generator
│
├── 📁 tests/              🧪 Test suite
│   └── test_*.py                   # Unit/integration tests
│
├── 📁 archive/            📦 Old files & backups
│
├── agent.py               🎯 Main entry point (orchestrator)
├── requirements.txt       📦 Python dependencies
├── RUN.bat               🚀 Master control menu
└── README.md             📖 This file
```

---

