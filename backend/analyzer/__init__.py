"""
BB-AI Video Analyzer Module
============================

Analyzes YouTube tutorial videos to extract n8n workflow structure.

Core Components:
- video_analyzer: CV/OCR-based frame analysis
- ai_video_understanding: GPT-4o Vision-based intelligent analysis
- ai_enhancer: AI classification and enhancement layer
- robust_node_detector: Multi-strategy node detection
- utils: Helper functions and utilities
"""

from .video_analyzer import VideoAnalyzer
from .ai_video_understanding import AIVideoUnderstanding
from .ai_enhancer import WorkflowAIEnhancer
from .utils import setup_logging, save_action_sequence, AutomationUtils

__all__ = [
    'VideoAnalyzer',
    'AIVideoUnderstanding',
    'WorkflowAIEnhancer',
    'setup_logging',
    'save_action_sequence',
    'AutomationUtils'
]

__version__ = '2.0.0'



