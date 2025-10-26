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

## 🚀 Quick Start

### **1️⃣ One Command (Everything):**

```bash
RUN.bat
# Select [1] Run Complete Pipeline
```

### **2️⃣ Step by Step:**

```bash
# Activate venv (if not already)
..\venv\Scripts\Activate.ps1

# Install dependencies
RUN.bat → [4]

# Run analysis
python agent.py

# Run automation
cd automation
node ai_playwright_agent.js
```

---

## 📊 System Flow

```
YouTube Video
    ↓
┌─────────────────────────────────┐
│  PART 1: Video Analysis         │
│  (analyzer/ - Python)            │
├─────────────────────────────────┤
│  video_analyzer.py               │ Extract 30 keyframes (SSIM)
│  ↓                                │
│  ai_video_understanding.py       │ GPT-4o Vision → detect nodes
│  ↓                                │
│  ai_enhancer.py                  │ Classify & enhance
│  ↓                                │
│  ai_workflow_complete.json       │ 24-95% understanding
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  PART 2: n8n Automation          │
│  (automation/ - Node.js)         │
├─────────────────────────────────┤
│  ai_playwright_agent.js          │ AI-powered automation
│  ├── n8n_ui_expert.js            │ RAG guidance (3.9MB docs)
│  ├── Smart selectors (GPT-4o)    │ DOM reasoning
│  ├── Self-healing retry          │ 3x attempts
│  └── Visual verification         │ Screenshot comparison
│  ↓                                │
│  n8n Workflow Created ✅         │
└─────────────────────────────────┘
```

---

## 🎯 Current Status

### ✅ Working:
- Video download (yt-dlp)
- Keyframe extraction (30 frames, SSIM)
- GPT-4o Vision analysis (integrated)
- AI enhancement (classification)
- Playwright automation (100% on simple workflows)
- n8n Expert RAG (3.9MB docs loaded)
- Screenshot management

### ⚠️ Known Issues:
- Low video quality → low detection (24% vs target 91%)
- OCR garbage output (fallback mode)
- Need HD tutorial videos for best results

---

## 📦 Dependencies

### Python:
- opencv-python, pillow, numpy
- pytesseract, yt-dlp
- openai, python-dotenv

### Node.js:
- playwright, openai
- dotenv

---

## 🔧 Configuration

Edit `config/credentials.env`:

```env
# Browserbase
PROJECT_ID=your_project_id
API_KEY=your_api_key

# n8n Cloud
N8N_EMAIL=your_email
N8N_PASSWORD=your_password

# OpenAI
OPENAI_API_KEY=sk-...

# Video to analyze
VIDEO_URL=https://youtube.com/...
```

---

## 📝 Commands Reference

See `docs/COMMANDS.md` for detailed command list.

---

**Built with ❤️ for n8n automation** 🚀



