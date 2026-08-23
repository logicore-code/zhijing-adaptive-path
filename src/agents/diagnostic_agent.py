"""
Diagnostic Agent (学情诊断智能体)
===================================

职责：
- 通过对话 + 微测试动态推断学生认知状态
- 调用 CSN（CognitiveStateNetwork）持续更新
- 输出：学生画像（每个知识点的掌握度 + 置信度 + 不确定性）
- 决定是否需要"自适应测试"（信息熵最大的题）
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.cognitive.student_model import CognitiveStateNetwork
from src.knowledge.knowledge_graph import KnowledgeGraph


@dataclass
class DiagnosticReport:
    """诊断报告"""
    student_id: str
    overall_ability: float
    strong_skills: List[Tuple[str, float]]  # 强项
    weak_skills: List[Tuple[str, float]]    # 弱项
    uncertain_skills: List[Tuple[str, float]]  # 不确定的（需要更多诊断）
    recommended_diagnostic_skill: Optional[str]  # 推荐下一道诊断题对应的知识点
    summary: str


class DiagnosticAgent:
    """学情诊断智能体"""
    def __init__(self, csn: CognitiveStateNetwork, knowledge_graph: KnowledgeGraph):
        self.csn = csn
        self.kg = knowledge_graph

    def update(
        self,
        student_id: str,
        skill_id: str,
        item_id: str,
        is_correct: bool,
        response_time: float = 0.0,
    ) -> Tuple[float, float]:
        return self.csn.update(student_id, skill_id, item_id, is_correct, response_time)

    def report(self, student_id: str) -> DiagnosticReport:
        """生成诊断报告"""
        profile = self.csn.get_or_create_profile(student_id)
        mastery = self.csn.get_all_mastery(student_id)

        # 排序
        sorted_skills = sorted(mastery.items(), key=lambda x: x[1], reverse=True)
        strong = [(s, m) for s, m in sorted_skills if m >= 0.75][:5]
        weak = [(s, m) for s, m in sorted_skills if m < 0.5][:5]
        uncertain = sorted(
            [(s, self.csn.get_confidence(student_id, s)) for s in mastery],
            key=lambda x: x[1]
        )[:5]

        # 推荐下一道诊断题：信息量最大的（最低置信度的）
        all_skills_uncertainty = [
            (s, self.csn.get_confidence(student_id, s))
            for s in self.kg.nodes.keys()
            if s not in mastery or self.csn.get_confidence(student_id, s) < 0.5
        ]
        recommended = min(all_skills_uncertainty, key=lambda x: x[1])[0] if all_skills_uncertainty else None

        summary = self._summarize(profile.overall_ability, len(strong), len(weak), len(uncertain))

        return DiagnosticReport(
            student_id=student_id,
            overall_ability=profile.overall_ability,
            strong_skills=strong,
            weak_skills=weak,
            uncertain_skills=uncertain,
            recommended_diagnostic_skill=recommended,
            summary=summary,
        )

    def _summarize(self, ability: float, n_strong: int, n_weak: int, n_uncertain: int) -> str:
        if ability > 1.0:
            level = "高"
        elif ability > 0:
            level = "中"
        else:
            level = "低"
        return f"学生整体能力：{level}（θ={ability:.2f}）。强项 {n_strong} 项，弱项 {n_weak} 项，不确定 {n_uncertain} 项。"

    def select_diagnostic_item(self, student_id: str, candidate_items: List[str]) -> Optional[str]:
        """
        自适应选题：选择能提供最大信息量的题。
        信息量 ≈ Fisher 信息 ≈ a^2 * p * (1-p)
        """
        if not candidate_items:
            return None

        best_item = None
        best_info = -1.0
        for item_id in candidate_items:
            if item_id not in self.csn.irt.items:
                continue
            item = self.csn.irt.items[item_id]
            theta = self.csn.irt.get_or_create_student(student_id).theta
            p = self.csn.irt.prob_correct(theta, item.a, item.b)
            info = item.a ** 2 * p * (1 - p)
            if info > best_info:
                best_info = info
                best_item = item_id
        return best_item
