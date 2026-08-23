"""对话引擎模块"""
from src.dialogue.scaffolding import ScaffoldingEngine, ScaffoldingDecision, ScaffoldingLevel
from src.dialogue.socratic_engine import SocraticEngine, SocraticResponse, DialogueStage, DialogueTurn

__all__ = [
    "ScaffoldingEngine", "ScaffoldingDecision", "ScaffoldingLevel",
    "SocraticEngine", "SocraticResponse", "DialogueStage", "DialogueTurn",
]
