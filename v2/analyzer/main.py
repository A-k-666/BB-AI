"""
YouTube Workflow Analyzer - Main Entry Point
============================================
Simple, fast, robust analysis in 4 steps.
"""

import os
import sys
import base64
import cv2
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from audio_transcriber import AudioTranscriber
from recipe_builder import RecipeBuilder
from video_downloader import VideoDownloader
from transcript_planner import TranscriptPlanner
from vision_frame_finder import VisionFrameFinder


def _generate_verdict(workflow: dict, openai_client) -> dict:
    """
    Generate AI verdict on analysis quality and automation readiness
    Returns: dict with summary, understanding, and ready_for_automation flag
    """
    try:
        nodes = workflow.get('nodes', [])
        connections = workflow.get('connections', [])
        confidence = workflow.get('metadata', {}).get('understanding_confidence', 0)
        workflow_name = workflow.get('workflow_name', 'Unknown')
        
        # Build context for AI
        nodes_summary = ', '.join([n.get('name', 'Unknown') for n in nodes[:10]])
        if len(nodes) > 10:
            nodes_summary += f' + {len(nodes) - 10} more'
        
        prompt = f"""Analyze this n8n workflow extraction result:

Workflow: {workflow_name}
Nodes detected: {len(nodes)} ({nodes_summary})
Connections: {len(connections)}
Confidence: {confidence}%

Evaluate:
1. Are nodes clear and well-defined?
2. Are connections logical?
3. Is this ready for automated recreation?

Return JSON (keep VERY SHORT - max 10 words per field):
{{
  "summary": "Detected X-node workflow: [brief description]",
  "understanding": "Clear trigger->action flow with Y integrations",
  "automation_status": "Ready - all nodes identifiable" or "Needs review - missing details",
  "ready_for_automation": true/false
}}

Ready if:
- Nodes >= 2
- Confidence >= 70%
- Clear node types (trigger/action/model)

Not ready if:
- Too few nodes (< 2)
- Low confidence (< 70%)
- Missing critical info"""

        response = openai_client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system", "content": "You are an n8n workflow analysis expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0
        )
        
        import json
        verdict = json.loads(response.choices[0].message.content)
        
        # Safety check: force not ready if critical thresholds not met
        if len(nodes) < 2 or confidence < 70:
            verdict['ready_for_automation'] = False
            if len(nodes) < 2:
                verdict['automation_status'] = "Needs review - insufficient nodes detected"
        
        return verdict
        
    except Exception as e:
        print(f"   Verdict generation failed: {e}")
        # Return safe default
        return {
            "summary": f"Detected {len(nodes)}-node workflow",
            "understanding": "Basic workflow structure identified",
            "automation_status": "Review recommended",
            "ready_for_automation": len(nodes) >= 2 and confidence >= 70
        }


def analyze_youtube_workflow(video_source: str, output_dir: str = './output'):
    """
    Complete YouTube workflow analysis
    
    Steps:
    1. Transcribe audio (AssemblyAI)
    2. Extract frames
    3. Analyze frames with AI Vision + transcript context
    4. Build recipe + save outputs
    """
    
    # Load config - try multiple paths
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'config', 'credentials.env'),
        '../config/credentials.env',
        './config/credentials.env',
        '../../config/credentials.env'
    ]
    
    env_loaded = False
    for env_path in possible_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            env_loaded = True
            print(f"Loaded config from: {env_path}")
            break
    
    openai_key = os.getenv('OPENAI_API_KEY')
    assemblyai_key = os.getenv('ASSEMBLYAI_API_KEY')
    
    # Strip quotes if present
    if assemblyai_key:
        assemblyai_key = assemblyai_key.strip('"\'')
    if openai_key:
        openai_key = openai_key.strip('"\'')
    
    if not openai_key or not assemblyai_key:
        print(f"ERROR: Missing API keys")
        print(f"  OPENAI_API_KEY: {'found' if openai_key else 'MISSING'}")
        print(f"  ASSEMBLYAI_API_KEY: {'found' if assemblyai_key else 'MISSING'}")
        return None
    
    print("\n" + "="*60)
    print("YOUTUBE WORKFLOW ANALYZER")
    print("="*60 + "\n")
    
    # Initialize modules
    downloader = VideoDownloader()
    audio_transcriber = AudioTranscriber(assemblyai_key)
    openai_client = OpenAI(api_key=openai_key)
    frame_finder = VisionFrameFinder(openai_client)  # NEW: AI Vision frame finder
    planner = TranscriptPlanner(openai_client)
    recipe_builder = RecipeBuilder()
    
    try:
        # Step 0: Download if URL
        if video_source.startswith(('http://', 'https://')):
            video_path = downloader.download_youtube(video_source, './temp_video.mp4')
            if not video_path:
                print("ERROR: Download failed")
                return None
        else:
            video_path = video_source
        
        # Step 1: Transcribe audio
        transcript = audio_transcriber.transcribe_video(video_path, output_dir)
        
        # Step 2: Find BEST complete workflow frame using AI Vision
        best_frame_result = frame_finder.find_best_workflow_frame(video_path, sample_count=30)
        
        if not best_frame_result:
            print("   ERROR: No workflow frame found")
            return None
        
        # Step 3: Generate plan from TRANSCRIPT + Vision validation
        print("\nStep 3/4: Generating workflow plan (transcript + vision)...")
        
        # Encode best frame
        _, buffer = cv2.imencode('.jpg', best_frame_result['frame'], 
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Step 3A: Transcript analysis
        workflow_plan = planner.generate_workflow_plan(
            transcript.get("full_text", ""),
            frame_b64
        )
        
        # Step 3B: Vision analysis on best frame to catch missing nodes
        print("\n   Running Vision on best frame to catch sub-nodes...")
        vision_nodes = planner.extract_nodes_from_frame(frame_b64)
        
        # Merge vision nodes with transcript nodes
        if vision_nodes:
            plan_node_names = {n.get("name", "").lower() for n in workflow_plan.get("nodes", [])}
            
            for vnode in vision_nodes:
                vname = vnode.get('name', '').strip()
                if not vname:
                    continue
                
                # Check if already exists
                is_similar = any(
                    fuzz.ratio(vname.lower(), existing.lower()) > 75
                    for existing in plan_node_names
                )
                
                if not is_similar:
                    workflow_plan.setdefault("nodes", []).append({
                        "name": vname,
                        "type": vnode.get("type", "model"),
                        "search_term": vname,
                        "select_option": "",
                        "configuration_notes": "Detected via Vision (sub-node)",
                        "order": 999
                    })
                    print(f"      + Added from Vision: {vname}")
        
        # Step 4: Convert plan to automation JSON
        print("\nStep 4/4: Building automation-ready workflow...")
        
        workflow = planner.convert_plan_to_automation_json(workflow_plan)
        
        # Save best OCR-selected frame
        best_frame = best_frame_result['frame']
        
        # Save all outputs
        recipe_builder.save_outputs(workflow, best_frame, output_dir)
        
        print(f"\nSUCCESS!")
        print(f"  Nodes: {len(workflow['nodes'])}")
        print(f"  Connections: {len(workflow['connections'])}")
        print(f"  Confidence: {workflow['metadata']['understanding_confidence']}%")
        print(f"  Has audio: {workflow['metadata']['has_audio']}")
        
        # 🎯 FINAL VERDICT - Automation Readiness Check
        print("\n" + "="*60)
        print("🎯 ANALYSIS VERDICT")
        print("="*60)
        
        verdict = _generate_verdict(workflow, openai_client)
        
        print(f"\n{verdict['summary']}")
        print(f"\n✅ Understanding: {verdict['understanding']}")
        print(f"{'✅' if verdict['ready_for_automation'] else '⚠️'} Automation Ready: {verdict['automation_status']}")
        
        if not verdict['ready_for_automation']:
            print(f"\n⚠️  WARNING: Analysis quality insufficient for automation")
            print(f"   Recommendation: Review workflow manually or re-analyze video")
        
        print("\n" + "="*60 + "\n")
        
        # Add verdict to workflow metadata
        workflow['metadata']['verdict'] = verdict
        
        # Cleanup: Delete temp video after successful analysis
        if video_source.startswith(('http://', 'https://')) and os.path.exists('./temp_video.mp4'):
            try:
                os.remove('./temp_video.mp4')
                print(f"Cleaned up temp_video.mp4\n")
            except:
                pass
        
        return workflow
        
    except Exception as e:
        print(f"\nERROR: Analysis failed - {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error too
        if os.path.exists('./temp_video.mp4'):
            try:
                os.remove('./temp_video.mp4')
                print(f"  Cleaned up temp_video.mp4")
            except:
                pass
        
        return None


if __name__ == "__main__":
    # Test mode - use YouTube URL from credentials
    load_dotenv('../config/credentials.env')
    video_url = os.getenv('VIDEO_URL')
    
    if not video_url:
        print("ERROR: No VIDEO_URL in credentials.env")
        sys.exit(1)
    
    analyze_youtube_workflow(video_url, output_dir="./output")

