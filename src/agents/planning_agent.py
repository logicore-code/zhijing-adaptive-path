"""
Planning Agent (路径规划智能体)
=================================

职责：
- 基于学生当前状态 + 知识图谱，动态生成学习路径
- 使用 Contextual Bandit 在线学习最优策略
- 综合考虑 ZPD（最近发展区）原则
- 输出：下一步推荐 + 理由 + 备选
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.cognitive.student_model import CognitiveStateNetwork
from src.knowledge.knowledge_graph import KnowledgeGraph
from src.planning.contextual_bandit import PathPlanner, PlanningDecision, compute_reward
from src.planning.path_optimizer import plan_learning_path, zpd_score


@dataclass
class PlanResult:
    """规划结果"""
    next_skill: str
    full_path: List[str]  # 接下来 N 步的路径
    decision: PlanningDecision
    expected_mastery_gain: float
    alternative_paths: List[List[str]]


class PlanningAgent:
    """路径规划智能体"""
    def __init__(self, csn: CognitiveStateNetwork, knowledge_graph: KnowledgeGraph):
        self.csn = csn
        self.kg = knowledge_graph
        # 把 KG 中所有节点作为 Bandit 动作
        n_skills = len(knowledge_graph.nodes)
        skill_id_to_idx = {sid: i for i, sid in enumerate(knowledge_graph.nodes.keys())}
        self.planner = PathPlanner(n_skills=n_skills, skill_id_to_idx=skill_id_to_idx)

    def plan(
        self,
        student_id: str,
        target_skill: Optional[str] = None,
        max_steps: int = 5,
    ) -> PlanResult:
        """
        为学生规划学习路径。
        """
        # 1) 获取学生状态
        profile = self.csn.get_or_create_profile(student_id)
        context = self.csn.get_feature_vector(student_id)
        mastery = self.csn.get_all_mastery(student_id)
        mastered = [s for s, m in mastery.items() if m >= 0.85]

        # 2) 确定候选知识点
        candidates = self.kg.candidate_skills(set(mastered), target=target_skill, k=20)
        if not candidates:
            # 没有合适的候选，使用图谱前向
            candidates = [s for s in self.kg.topological_order() if s not in set(mastered)][:5]

        # 3) 用 Bandit 选下一步
        decision = self.planner.plan_next(
            student_context=context,
            available_skills=candidates,
            mastered_skills=mastered,
            target_skill=target_skill,
        )

        # 4) 拼出完整 N 步路径（贪心：每步重新调用，但用 ZPD 过滤）
        full_path = [decision.next_skill]
        remaining = [c for c in candidates if c != decision.next_skill]
        temp_mastered = set(mastered)
        for _ in range(max_steps - 1):
            if not remaining:
                break
            # 选 ZPD 分数最高的（同时考虑未掌握）
            best = None
            best_score = -1
            for c in remaining:
                node = self.kg.get_node(c)
                if not node:
                    continue
                score = zpd_score(profile.overall_ability, node.difficulty)
                if score > best_score:
                    best_score = score
                    best = c
            if best:
                full_path.append(best)
                temp_mastered.add(best)
                remaining.remove(best)

        # 5) 备选路径
        alt_paths = []
        for alt_skill, _ in decision.alternatives[1:3]:
            alt_path = [alt_skill] + [s for s in full_path if s != alt_skill][:max_steps - 1]
            alt_paths.append(alt_path)

        # 6) 预期掌握度提升
        next_node = self.kg.get_node(decision.next_skill)
        current_mastery = self.csn.get_mastery(student_id, decision.next_skill)
        expected = min(1.0, current_mastery + (1.0 - current_mastery) * 0.5)  # 简化估计
        gain = expected - current_mastery

        return PlanResult(
            next_skill=decision.next_skill,
            full_path=full_path,
            decision=decision,
            expected_mastery_gain=gain,
            alternative_paths=alt_paths,
        )

    def feedback(
        self,
        student_id: str,
        skill_id: str,
        mastery_before: float,
        mastery_after: float,
        engagement: float,
        emotion_score: float,
    ):
        """收到反馈，更新 Bandit"""
        context = self.csn.get_feature_vector(student_id)
        reward = compute_reward(mastery_before, mastery_after, engagement, emotion_score)
        self.planner.feedback(skill_id, context, reward)
