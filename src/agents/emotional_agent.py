"""
Emotional Agent (情感智能体)
==============================

职责：
- 实时监测学生情感状态
- 检测困惑、挫败、兴趣、疲劳
- 推荐情感适配的教学策略
- 把情感信号反向输入到教学节奏
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.emotion.sentiment import (
    EmotionState, EmotionDetector, FatigueTracker, detect_emotion, recommend_pedagogy
)


@dataclass
class EmotionalAnalysis:
    """情感分析结果"""
    state: EmotionState
    dominant: str
    recommendation: str
    should_intervene: bool  # 是否需要教师主动干预
    intervention_type: Optional[str]  # "鼓励" / "降低难度" / "建议休息" / "增加挑战"


class EmotionalAgent:
    """情感智能体"""
    def __init__(self):
        self.states: Dict[str, EmotionState] = {}
        self.trackers: Dict[str, FatigueTracker] = {}
        self.detector = EmotionDetector()

    def analyze(
        self,
        student_id: str,
        text: str,
        is_correct: Optional[bool] = None,
    ) -> EmotionalAnalysis:
        """分析学生情感"""
        state = self.states.setdefault(student_id, EmotionState())
        tracker = self.trackers.setdefault(student_id, FatigueTracker())
        state = detect_emotion(
            text,
            state=state,
            is_correct=is_correct,
            input_length=len(text),
            fatigue_tracker=tracker,
        )
        rec = recommend_pedagogy(state)

        # 决定是否干预
        should_intervene = False
        intervention = None
        if state.frustration > 0.7:
            should_intervene = True
            intervention = "鼓励 + 降低难度"
        elif state.fatigue > 0.6:
            should_intervene = True
            intervention = "建议休息或切换到轻量任务"
        elif state.confusion > 0.7:
            should_intervene = True
            intervention = "用类比或图示重新讲解"
        elif state.confidence < 0.2:
            should_intervene = True
            intervention = "肯定已有进步"
        elif state.engagement < 0.3:
            should_intervene = True
            intervention = "增加趣味性 / 引入实例"

        return EmotionalAnalysis(
            state=state,
            dominant=state.dominant(),
            recommendation=rec,
            should_intervene=should_intervene,
            intervention_type=intervention,
        )

    def get_state(self, student_id: str) -> EmotionState:
        return self.states.setdefault(student_id, EmotionState())

    def reset(self, student_id: Optional[str] = None):
        if student_id:
            self.states.pop(student_id, None)
            self.trackers.pop(student_id, None)
        else:
            self.states.clear()
            self.trackers.clear()
