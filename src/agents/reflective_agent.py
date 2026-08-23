"""
Reflective Agent (反思智能体)
=================================

职责：
- 学习后的元认知培养
- 生成反思笔记
- 总结学习模式
- 给出下次学习建议
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.cognitive.student_model import CognitiveStateNetwork
from src.knowledge.knowledge_graph import KnowledgeGraph
from src.memory.long_term_memory import (
    LongTermMemory, Episode, ReflectiveNote, generate_reflection
)


@dataclass
class ReflectionResult:
    """反思结果"""
    summary: str
    achievements: List[str]
    challenges: List[str]
    next_steps: List[str]
    reflective_note: ReflectiveNote


class ReflectiveAgent:
    """反思智能体"""
    def __init__(
        self,
        csn: CognitiveStateNetwork,
        knowledge_graph: KnowledgeGraph,
        memory: LongTermMemory,
    ):
        self.csn = csn
        self.kg = knowledge_graph
        self.memory = memory

    def reflect(self, student_id: str) -> ReflectionResult:
        """基于最近学习记录生成反思"""
        episodes = self.memory.get_episodes(student_id, limit=20)
        note = generate_reflection(episodes, student_id)
        self.memory.add_reflection(note)

        # 详细分析
        achievements = []
        challenges = []
        next_steps = []

        if episodes:
            # 1) 找强项
            mastery = self.csn.get_all_mastery(student_id)
            for sid, m in sorted(mastery.items(), key=lambda x: -x[1])[:3]:
                if m >= 0.8:
                    node = self.kg.get_node(sid)
                    if node:
                        achievements.append(f"熟练掌握【{node.name}】（{m:.0%}）")

            # 2) 找弱项
            for sid, m in sorted(mastery.items(), key=lambda x: x[1])[:3]:
                if m < 0.5:
                    node = self.kg.get_node(sid)
                    if node:
                        challenges.append(f"【{node.name}】掌握不足（{m:.0%}）")
                        next_steps.append(f"复习【{node.name}】的核心概念")

            # 3) 错题分析
            wrong_count = sum(1 for e in episodes if e.is_correct is False)
            if wrong_count > 0:
                challenges.append(f"近期 {wrong_count} 次答错，需要巩固")

            # 4) 脚手架使用
            high_scaffold = sum(1 for e in episodes if e.scaffolding_level >= 3)
            if high_scaffold > 3:
                next_steps.append("尝试独立完成更多题目，减少对提示的依赖")

        return ReflectionResult(
            summary=note.content,
            achievements=achievements,
            challenges=challenges,
            next_steps=next_steps,
            reflective_note=note,
        )

    def add_episode(self, episode: Episode):
        """记录一次交互"""
        self.memory.add_episode(episode)
