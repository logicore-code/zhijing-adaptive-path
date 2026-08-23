"""情感模块"""
from src.emotion.sentiment import (
    EmotionState,
    EmotionDetector,
    FatigueTracker,
    detect_emotion,
    recommend_pedagogy,
)

__all__ = [
    "EmotionState",
    "EmotionDetector",
    "FatigueTracker",
    "detect_emotion",
    "recommend_pedagogy",
]
