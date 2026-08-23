"""
Teaching Agent (教学智能体 - 苏格拉底式)
==========================================

职责：
- 苏格拉底式反诘引导学生
- 动态脚手架
- 永远不直接给答案
- 维护 Bloom 认知层级适配
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from src.cognitive.student_model import CognitiveStateNetwork
from src.dialogue.socratic_engine import SocraticEngine, SocraticResponse, DialogueStage
from src.emotion.sentiment import EmotionState
from src.knowledge.knowledge_graph import KnowledgeGraph


@dataclass
class TeachingDecision:
    """教学决策"""
    response: SocraticResponse
    next_action: str  # "继续对话"/"进入测试"/"切换知识点"
    should_give_answer: bool


class TeachingAgent:
    """教学智能体"""
    def __init__(
        self,
        csn: CognitiveStateNetwork,
        knowledge_graph: KnowledgeGraph,
        llm=None,
    ):
        self.csn = csn
        self.kg = knowledge_graph
        self.engine = SocraticEngine()
        if llm:
            self.engine.set_llm(llm)

    def respond(
        self,
        student_input: str,
        student_id: str,
        skill_id: str,
        emotion: Optional[EmotionState] = None,
        is_correct: Optional[bool] = None,
        stage: DialogueStage = DialogueStage.PROBE,
    ) -> TeachingDecision:
        """
        回应学生。
        """
        mastery = self.csn.get_mastery(student_id, skill_id)
        confidence = self.csn.get_confidence(student_id, skill_id)

        # 获取最近答题记录
        recent = self.csn.history.get(student_id, [])
        recent_results = [c for _, c, _ in recent[-5:]]

        fatigue = emotion.fatigue if emotion else 0.0

        # 学生是否请求提示
        request_hint = self._detect_hint_request(student_input)

        response = self.engine.respond(
            student_input=student_input,
            current_skill=self.kg.get_node(skill_id).name if self.kg.get_node(skill_id) else skill_id,
            mastery=mastery,
            recent_results=recent_results,
            stage=stage,
            fatigue=fatigue,
            student_request_hint=request_hint,
        )

        # 决定下一步行动
        if mastery > 0.85 and len(recent_results) >= 3 and all(recent_results[-3:]):
            next_action = "进入测试"
        elif emotion and emotion.frustration > 0.7:
            next_action = "切换知识点"  # 逃避困境
        else:
            next_action = "继续对话"

        return TeachingDecision(
            response=response,
            next_action=next_action,
            should_give_answer=response.should_end,
        )

    def _detect_hint_request(self, text: str) -> bool:
        """检测学生是否主动求提示"""
        keywords = ["提示", "hint", "help", "help me", "给我答案", "告诉我"]
        return any(kw in text.lower() for kw in keywords)
