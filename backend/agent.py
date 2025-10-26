"""
Main agent controller for Browserbase + n8n automation.
Orchestrates video analysis and action sequence generation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from analyzer.video_analyzer import VideoAnalyzer
from analyzer.utils import setup_logging, save_action_sequence, AutomationUtils
from analyzer.ai_enhancer import WorkflowAIEnhancer
from analyzer.ai_video_understanding import AIVideoUnderstanding
from analyzer.json_refiner import WorkflowJSONRefiner
from analyzer.fast_analyzer import FastVideoAnalyzer

# Try enhanced V2 analyzer
try:
    from analyzer.ai_video_understanding_v2 import AIVideoUnderstandingV2
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# Try multimodal analyzer (Vision + Audio)
try:
    from analyzer.multimodal_analyzer import MultimodalWorkflowAnalyzer
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False


class BrowserbaseN8nAgent:
    """Main agent for orchestrating video analysis and n8n workflow replication."""
    
    def __init__(self, config_path: str = "./config/credentials.env"):
        """
        Initialize the agent.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.video_analyzer = VideoAnalyzer(
            openai_api_key=self.config.get('OPENAI_API_KEY')
        )
        
        # Setup logging
        setup_logging(
            log_level=self.config.get('LOG_LEVEL', 'INFO'),
            log_file=self.config.get('LOG_FILE', 'agent.log')
        )
    
    def _load_config(self, config_path: str) -> Dict[str, str]:
        """
        Load configuration from environment file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Dict: Configuration dictionary
        """
        config = {}
        
        if os.path.exists(config_path):
            load_dotenv(config_path)
        else:
            self.logger.warning(f"Config file not found: {config_path}")
        
        # Load required configuration
        required_configs = [
            'PROJECT_ID', 'API_KEY', 'N8N_URL', 'N8N_EMAIL', 
            'N8N_PASSWORD', 'OPENAI_API_KEY'
        ]
        
        for key in required_configs:
            value = os.getenv(key)
            if not value or value.startswith('your_'):
                self.logger.warning(f"Missing or placeholder value for {key}")
                config[key] = None
            else:
                config[key] = value
        
        # Load optional configuration
        optional_configs = {
            'VIDEO_URL': None,
            'LOCAL_VIDEO_PATH': None,
            'OUTPUT_DIR': './output',
            'LOG_LEVEL': 'INFO',
            'LOG_FILE': 'agent.log'
        }
        
        for key, default in optional_configs.items():
            config[key] = os.getenv(key, default)
        
        return config
    
    def validate_config(self) -> bool:
        """
        Validate configuration and check for missing credentials.
        
        Returns:
            bool: True if config is valid, False otherwise
        """
        missing_configs = []
        
        for key, value in self.config.items():
            if value is None and key in ['PROJECT_ID', 'API_KEY', 'N8N_URL', 'N8N_EMAIL', 'N8N_PASSWORD']:
                missing_configs.append(key)
        
        if missing_configs:
            self.logger.error(f"Missing required configurations: {missing_configs}")
            return False
        
        self.logger.info("Configuration validated successfully")
        return True
    
    def prompt_user_for_credentials(self) -> None:
        """Prompt user for missing credentials."""
        print("\n" + "="*50)
        print("CREDENTIALS REQUIRED")
        print("="*50)
        
        credentials_to_prompt = {
            'PROJECT_ID': 'Browserbase Project ID',
            'API_KEY': 'Browserbase API Key',
            'N8N_URL': 'n8n Instance URL (e.g., https://your-n8n.com)',
            'N8N_EMAIL': 'n8n Login Email',
            'N8N_PASSWORD': 'n8n Login Password',
            'OPENAI_API_KEY': 'OpenAI API Key'
        }
        
        updated_config = {}
        for key, description in credentials_to_prompt.items():
            if not self.config.get(key) or self.config[key].startswith('your_'):
                value = input(f"Enter {description}: ").strip()
                if value:
                    self.config[key] = value
                    updated_config[key] = value
                    self.logger.info(f"Updated {key}")
        
        # Save updated config to file
        if updated_config:
            self._save_config_to_file(updated_config)
        
        print("="*50)
    
    def _save_config_to_file(self, updated_config: Dict[str, str], config_file: str = "./config/credentials.env") -> None:
        """
        Save updated configuration to file.
        
        Args:
            updated_config: Dictionary of updated configuration values
            config_file: Path to config file (optional)
        """
        try:
            
            # Read existing file
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            else:
                lines = []
            
            # Update lines with new values
            updated_lines = []
            for line in lines:
                updated = False
                for key, value in updated_config.items():
                    if line.strip().startswith(f"{key}="):
                        updated_lines.append(f"{key}={value}\n")
                        updated = True
                        break
                if not updated:
                    updated_lines.append(line)
            
            # Add new keys that weren't in the file
            existing_keys = set()
            for line in lines:
                if '=' in line and not line.strip().startswith('#'):
                    key = line.split('=')[0].strip()
                    existing_keys.add(key)
            
            for key, value in updated_config.items():
                if key not in existing_keys:
                    updated_lines.append(f"{key}={value}\n")
            
            # Write back to file
            with open(config_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            
            self.logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def _extract_keyframes_for_ai(self, video_path: str, max_frames: int = 5) -> List:
        """
        Extract keyframes for AI Vision analysis
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract (cost control)
            
        Returns:
            List of numpy arrays (frames)
        """
        import cv2
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample evenly across video
            frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
            
            keyframes = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    keyframes.append(frame)
            
            cap.release()
            self.logger.info(f"[AI] Extracted {len(keyframes)} keyframes for GPT-4o Vision")
            return keyframes
            
        except Exception as e:
            self.logger.warning(f"Keyframe extraction failed: {e}")
            return []
    
    def analyze_video(self, video_source: str) -> Dict[str, Any]:
        """
        Analyze video and generate action sequence with GPT-4o Vision.
        
        Args:
            video_source: Video URL or local file path
            
        Returns:
            Dict: Analysis results
        """
        self.logger.info(f"Starting video analysis: {video_source}")
        
        try:
            # Step 1: Extract keyframes using traditional analyzer
            if video_source.startswith(('http://', 'https://')):
                result = self.video_analyzer.analyze_youtube_video(video_source)
            else:
                result = self.video_analyzer.analyze_video(video_source)
            
            if 'error' in result:
                self.logger.error(f"Video analysis failed: {result['error']}")
                return result
            
            # Step 2: Get keyframes for AI Vision analysis
            video_path = result.get('video_path', './temp_video.mp4')
            
            # Use keyframes from video_analyzer if available
            if 'keyframes' in result and result['keyframes']:
                keyframes = result['keyframes']
                self.logger.info(f"[AI] Using {len(keyframes)} keyframes from video analyzer")
            else:
                keyframes = self._extract_keyframes_for_ai(video_path)
            
            # Step 3: Use FAST analyzer (5-10x faster)
            if keyframes:
                self.logger.info("[FAST] Using parallel fast analyzer...")
                try:
                    import openai as openai_module
                    openai_client = openai_module.OpenAI(api_key=self.config.get('OPENAI_API_KEY'))
                    
                    fast_analyzer = FastVideoAnalyzer(openai_client=openai_client)
                    ai_workflow = fast_analyzer.analyze_parallel(keyframes, video_path)
                    
                    result['action_sequence'] = ai_workflow
                    result['ai_enhanced'] = True
                    result['understanding_confidence'] = ai_workflow['metadata'].get('understanding_confidence', 0)
                    
                    if 'keyframes' in result:
                        del result['keyframes']
                    
                    self.logger.info(f"[FAST] ✅ Complete in parallel mode: {result['understanding_confidence']}%")
                    
                except Exception as e:
                    self.logger.warning(f"Fast analyzer failed: {e}, using fallback...")
            
            # Fallback: Use best available analyzer
            if not result.get('ai_enhanced') and MULTIMODAL_AVAILABLE:
                self.logger.info("[MULTIMODAL] Using Vision + Audio analyzer (BEST)...")
                try:
                    multimodal = MultimodalWorkflowAnalyzer()
                    ai_workflow = multimodal.analyze_complete(video_path)
                    
                    result['action_sequence'] = ai_workflow
                    result['ai_enhanced'] = True
                    result['understanding_confidence'] = ai_workflow['metadata'].get('understanding_confidence', 0)
                    
                    if 'keyframes' in result:
                        del result['keyframes']
                    
                    self.logger.info(f"[MULTIMODAL] Understanding: {result['understanding_confidence']:.1f}% (with audio)")
                    
                except Exception as e:
                    self.logger.warning(f"Multimodal failed: {e}, trying V2...")
                    
            if not result.get('ai_enhanced') and V2_AVAILABLE:
                self.logger.info("[AI V2] Using ENHANCED GPT-4o Vision analyzer...")
                try:
                    ai_analyzer = AIVideoUnderstandingV2()
                    ai_workflow = ai_analyzer.analyze_video_complete(video_path)
                    
                    # Replace traditional result with AI-enhanced workflow
                    result['action_sequence'] = ai_workflow
                    result['ai_enhanced'] = True
                    result['understanding_confidence'] = ai_workflow['metadata'].get('understanding_confidence', 0)
                    
                    # Remove numpy arrays before JSON serialization
                    if 'keyframes' in result:
                        del result['keyframes']  # Remove numpy arrays
                    
                    self.logger.info(f"[AI V2] Understanding: {result['understanding_confidence']:.1f}%")
                    
                except Exception as e:
                    self.logger.warning(f"AI V2 failed: {e}")
                    
                    # Fallback to V1
                    if keyframes:
                        self.logger.info("[AI V1] Trying original analyzer...")
                        try:
                            ai_analyzer = AIVideoUnderstanding()
                            ai_workflow = ai_analyzer.analyze_video_complete(video_path, keyframes)
                            result['action_sequence'] = ai_workflow
                            result['ai_enhanced'] = True
                            result['understanding_confidence'] = ai_workflow['metadata'].get('understanding_confidence', 0)
                            if 'keyframes' in result:
                                del result['keyframes']
                            self.logger.info(f"[AI V1] Understanding: {result['understanding_confidence']:.1f}%")
                        except Exception as e2:
                            self.logger.warning(f"AI V1 also failed: {e2}")
            
            elif keyframes:
                self.logger.info("[AI] Using GPT-4o Vision for intelligent analysis...")
                try:
                    ai_analyzer = AIVideoUnderstanding()
                    ai_workflow = ai_analyzer.analyze_video_complete(video_path, keyframes)
                    result['action_sequence'] = ai_workflow
                    result['ai_enhanced'] = True
                    result['understanding_confidence'] = ai_workflow['metadata'].get('understanding_confidence', 0)
                    if 'keyframes' in result:
                        del result['keyframes']
                    self.logger.info(f"[AI] Understanding: {result['understanding_confidence']:.1f}%")
                except Exception as e:
                    self.logger.warning(f"AI Vision analysis failed: {e}")
            
            # Save results
            self._save_analysis_results(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Video analysis error: {e}")
            return {"error": str(e)}
    
    def _save_analysis_results(self, results: Dict[str, Any]) -> None:
        """
        Save analysis results to output directory.
        
        Args:
            results: Analysis results dictionary
        """
        output_dir = self.config.get('OUTPUT_DIR', './output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save action sequence
        if 'action_sequence' in results:
            # Check if AI-enhanced (from ai_video_understanding.py)
            if results.get('ai_enhanced', False):
                # 🔧 REFINEMENT LAYER: Validate and refine JSON using complete workflow screenshot
                final_screenshot_path = os.path.join(output_dir, 'ai_visual_recipe.png')
                
                if os.path.exists(final_screenshot_path):
                    self.logger.info("[REFINER] Applying final refinement layer...")
                    
                    # Import OpenAI client
                    import openai as openai_module
                    openai_client = openai_module.OpenAI(api_key=self.config.get('OPENAI_API_KEY'))
                    
                    refiner = WorkflowJSONRefiner(openai_client=openai_client)
                    
                    results['action_sequence'] = refiner.refine_workflow_json(
                        results['action_sequence'],
                        final_screenshot_path=final_screenshot_path
                    )
                    
                    self.logger.info(f"[REFINER] ✅ JSON refined and validated")
                else:
                    self.logger.warning("[REFINER] No final screenshot found, skipping refinement")
                
                # Already AI-enhanced with GPT-4o Vision
                enhanced_path = os.path.join(output_dir, 'ai_workflow_complete.json')
                with open(enhanced_path, 'w', encoding='utf-8') as f:
                    json.dump(results['action_sequence'], f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"[OK] AI Vision workflow saved: {enhanced_path}")
                self.logger.info(f"   Understanding: {results.get('understanding_confidence', 0):.1f}%")
                
                # Also save as action_sequence.json for compatibility
                action_path = os.path.join(output_dir, 'action_sequence.json')
                with open(action_path, 'w', encoding='utf-8') as f:
                    json.dump(results['action_sequence'], f, indent=2, ensure_ascii=False)
                
            else:
                # Traditional CV/OCR - save raw and enhance
                raw_sequence_path = os.path.join(output_dir, 'action_sequence_raw.json')
                save_action_sequence(results['action_sequence'], raw_sequence_path)
                self.logger.info(f"Raw action sequence saved: {raw_sequence_path}")
                
                # AI Enhancement Layer (fallback)
                try:
                    self.logger.info("Enhancing with AI classifier...")
                    enhancer = WorkflowAIEnhancer()
                    enhanced = enhancer.enhance_workflow(results['action_sequence'])
                    
                    enhanced_path = os.path.join(output_dir, 'action_sequence.json')
                    with open(enhanced_path, 'w', encoding='utf-8') as f:
                        json.dump(enhanced, f, indent=2, ensure_ascii=False)
                    
                    self.logger.info(f"[OK] Enhanced sequence saved: {enhanced_path}")
                    self.logger.info(f"   Quality: {enhanced['metadata'].get('analysis_quality', 0):.1f}%")
                    
                except Exception as e:
                    self.logger.warning(f"Enhancement failed: {e}")
                    sequence_path = os.path.join(output_dir, 'action_sequence.json')
                    save_action_sequence(results['action_sequence'], sequence_path)
        
        # Save full results
        results_path = os.path.join(output_dir, 'analysis_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Full results saved: {results_path}")
    
    def generate_workflow_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate human-readable workflow summary.
        
        Args:
            analysis_results: Analysis results from video
            
        Returns:
            str: Workflow summary
        """
        if 'error' in analysis_results:
            return f"❌ Analysis failed: {analysis_results['error']}"
        
        nodes = analysis_results.get('nodes', [])
        connections = analysis_results.get('connections', [])
        action_sequence = analysis_results.get('action_sequence', [])
        
        summary = f"""
WORKFLOW ANALYSIS SUMMARY
{'='*50}

DETECTED NODES ({len(nodes)}):
"""
        
        for i, node in enumerate(nodes):
            summary += f"  {i+1}. {node.get('text', 'Unknown')} ({node.get('type', 'unknown')})\n"
        
        summary += f"\nDETECTED CONNECTIONS ({len(connections)}):\n"
        
        for i, conn in enumerate(connections):
            summary += f"  {i+1}. Node {conn.get('from', '?')} -> Node {conn.get('to', '?')}\n"
        
        summary += f"\nACTION SEQUENCE ({len(action_sequence)} steps):\n"
        
        for i, action in enumerate(action_sequence):
            if isinstance(action, dict):
                step_type = action.get('action', 'unknown')
                if step_type == 'create_node':
                    node_name = action.get('node', {}).get('name', 'Unknown')
                    summary += f"  {i+1}. Create node: {node_name}\n"
                elif step_type == 'connect_nodes':
                    conn = action.get('connection', {})
                    summary += f"  {i+1}. Connect: {conn.get('from', '?')} -> {conn.get('to', '?')}\n"
            else:
                summary += f"  {i+1}. {action}\n"
        
        summary += f"\nReady for Node.js Playwright automation!"
        return summary
    
    def run(self, video_source: str = None) -> Dict[str, Any]:
        """
        Main execution method.
        
        Args:
            video_source: Video URL or path (optional, uses config if not provided)
            
        Returns:
            Dict: Execution results
        """
        self.logger.info("Starting Browserbase + n8n Agent")
        
        # Validate configuration
        if not self.validate_config():
            self.logger.info("Configuration validation failed. Prompting for credentials...")
            self.prompt_user_for_credentials()
            
            if not self.validate_config():
                return {"error": "Configuration still invalid after user input"}
        
        # Get video source
        if not video_source:
            video_source = self.config.get('VIDEO_URL') or self.config.get('LOCAL_VIDEO_PATH')
            if not video_source:
                video_source = input("Enter video URL or local file path: ").strip()
        
        if not video_source:
            return {"error": "No video source provided"}
        
        # Analyze video
        self.logger.info(f"Analyzing video: {video_source}")
        analysis_results = self.analyze_video(video_source)
        
        if 'error' in analysis_results:
            return analysis_results
        
        # Generate summary
        summary = self.generate_workflow_summary(analysis_results)
        print(summary)
        
        # Save summary
        output_dir = self.config.get('OUTPUT_DIR', './output')
        summary_path = os.path.join(output_dir, 'workflow_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        self.logger.info("Agent execution completed successfully")
        return {
            "status": "success",
            "analysis_results": analysis_results,
            "summary": summary,
            "output_directory": output_dir
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Browserbase + n8n Automation Agent')
    parser.add_argument('--video', help='Video URL or local file path')
    parser.add_argument('--config', default='./config/credentials.env', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = BrowserbaseN8nAgent(args.config)
    
    # Run agent
    results = agent.run(args.video)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return 1
    else:
        print("Agent completed successfully!")
        return 0


if __name__ == "__main__":
    exit(main())
