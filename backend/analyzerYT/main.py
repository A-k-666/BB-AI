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

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from audio_transcriber import AudioTranscriber
from recipe_builder import RecipeBuilder
from video_downloader import VideoDownloader
from transcript_planner import TranscriptPlanner
from vision_frame_finder import VisionFrameFinder


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
        
        # Step 3: PRIMARY - Generate plan from TRANSCRIPT (most accurate!)
        print("\nStep 3/4: Generating workflow plan from transcript (PRIMARY)...")
        
        # Use best OCR-selected frame for visual reference
        _, buffer = cv2.imencode('.jpg', best_frame_result['frame'], 
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Generate plan using transcript (this is the KEY step!)
        workflow_plan = planner.generate_workflow_plan(
            transcript.get("full_text", ""),
            frame_b64
        )
        
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
        
        return workflow
        
    except Exception as e:
        print(f"\nERROR: Analysis failed - {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test mode - use YouTube URL from credentials
    load_dotenv('../config/credentials.env')
    video_url = os.getenv('VIDEO_URL')
    
    if not video_url:
        print("ERROR: No VIDEO_URL in credentials.env")
        sys.exit(1)
    
    analyze_youtube_workflow(video_url, output_dir="./output")

