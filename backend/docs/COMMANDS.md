# 🚀 **Quick Commands for BB-AI**

## **For You to Run Later:**

### **1️⃣ Complete AI Analysis + Automation (Best):**

```bash
# Step 1: Activate Python environment
.\venv\Scripts\Activate.ps1

# Step 2: Go to backend
cd backend

# Step 3: Run video analysis with GPT-4o Vision (fixes needed)
python agent.py

# Step 4: Run automation 
node ai_playwright_agent.js
```

---

### **2️⃣ Download n8n Docs (One Time):**

```bash
cd ..
git clone https://github.com/n8n-io/n8n-docs.git
cd n8n-docs
Get-ChildItem -Recurse -Filter *.md | Get-Content | Out-File -FilePath ..\backend\n8n_docs_combined.md -Encoding UTF8
cd ..\backend
```

---

## **📊 Current Status:**

### **What's Working ✅:**
- Python backend: Video download, keyframe extraction (30 frames)
- AI enhancer: Classifies nodes, predicts connections (34.5% quality)
- Playwright agent: Login, node creation, connections (100% success on simple workflows)
- n8n Expert: RAG with 3.9MB docs loaded
- Screenshot management: Auto-cleanup system

### **What's NOT Working ❌:**
- **GPT-4o Vision integration**: Keyframes not passed correctly from `video_analyzer.py` to `agent.py`
- **OCR**: Tesseract giving garbage ("a", "zi", "Gp" instead of real node names)
- **Result**: 34.5% quality instead of target 91%+

---

## **🔧 Fix Needed:**

The issue: `video_analyzer.py` uses `ai_video_understanding.py` but doesn't return keyframes in the result dict.

**Quick workaround**: Use existing `ai_workflow_complete.json` (manually created) and run:

```bash
node ai_playwright_agent.js
```

This will use the 95% confidence workflow we created earlier!

---

## **📂 Key Files:**

| File | Purpose | Status |
|------|---------|--------|
| `agent.py` | Main orchestrator | ✅ Working (but keyframe issue) |
| `video_analyzer.py` | CV/OCR engine | ⚠️  OCR garbage |
| `ai_video_understanding.py` | GPT-4o Vision | ✅ Works (91-95%) |
| `ai_enhancer.py` | Classifier | ✅ Works (34.5% on garbage input) |
| `ai_playwright_agent.js` | Automation | ✅ 100% success |
| `n8n_ui_expert.js` | RAG expert | ✅ Docs loaded |

---

## **🎯 Next Steps:**

1. Fix keyframe passing in `video_analyzer.py`
2. Test complete pipeline: `python agent.py` → `node ai_playwright_agent.js`
3. Expected result: **91%+ understanding → 100% automation** 🚀

