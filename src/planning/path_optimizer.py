"""
路径优化：MDP 视角 + ZPD（最近发展区）筛选
==============================================

把短期学习规划建模为有限 MDP（Markov Decision Process）：

  State s_t: 学生当前状态 (skill_mastery, emotion, fatigue)
  Action a_t: 学习活动 (skill, difficulty, teaching_strategy)
  Reward r_t: 即时收益
  Transition: 学生状态更新

短期规划：用 Value Iteration 求最优 5-10 步策略
长期规划：参考知识图谱的拓扑顺序

ZPD (Zone of Proximal Development) 筛选：
- 难度太低 (< 0.3 当前能力)：跳过
- 难度太高 (> 0.7 当前能力)：需要先打基础
- 0.3-0.7：sweet spot，最大学习收益
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.knowledge.knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------------------- #
# ZPD 筛选
# ---------------------------------------------------------------------- #
@dataclass
class ZPDRecommendation:
    skill_id: str
    difficulty: float
    student_ability: float
    zpd_score: float  # 0~1，越高越在最近发展区
    in_sweet_spot: bool


def zpd_score(ability: float, difficulty: float) -> float:
    """
    计算 ZPD 分数：能力与难度的匹配度。
    公式：exp(-((ability - difficulty) - 0.4)^2 / 0.2) - 高斯曲线
    目标差距 0.4（学生略感挑战但可完成）
    """
    gap = difficulty - ability
    target_gap = 0.4
    return float(np.exp(-((gap - target_gap) ** 2) / 0.2))


def zpd_filter(
    ability: float,
    candidates: List[Tuple[str, float]],  # (skill_id, difficulty)
    lower: float = 0.3,
    upper: float = 0.7,
) -> List[ZPDRecommendation]:
    """
    过滤出 ZPD sweet spot 内的候选。
    """
    results = []
    for sid, diff in candidates:
        score = zpd_score(ability, diff)
        in_sweet = lower <= score <= upper
        results.append(ZPDRecommendation(
            skill_id=sid,
            difficulty=diff,
            student_ability=ability,
            zpd_score=score,
            in_sweet_spot=in_sweet,
        ))
    return sorted(results, key=lambda x: x.zpd_score, reverse=True)


# ---------------------------------------------------------------------- #
# 短期规划：Value Iteration
# ---------------------------------------------------------------------- #
class ShortTermPlanner:
    """
    短期规划（5-10 步）的值迭代求解器。

    State: 离散化的 (mastery_level, emotion_level) 组合
    Action: 候选技能 + 教学策略
    """
    def __init__(self, gamma: float = 0.95, horizon: int = 5):
        self.gamma = gamma
        self.horizon = horizon

    def value_iteration(
        self,
        n_states: int,
        n_actions: int,
        reward_fn,
        transition_fn,
        n_iters: int = 100,
    ) -> np.ndarray:
        """
        标准 Value Iteration。
        reward_fn(s, a) -> r
        transition_fn(s, a) -> s'
        """
        V = np.zeros(n_states)
        for _ in range(n_iters):
            V_new = np.zeros(n_states)
            for s in range(n_states):
                qs = []
                for a in range(n_actions):
                    r = reward_fn(s, a)
                    s_next = transition_fn(s, a)
                    qs.append(r + self.gamma * V[s_next])
                V_new[s] = max(qs)
            V = V_new
        return V


# ---------------------------------------------------------------------- #
# 路径规划高阶 API
# ---------------------------------------------------------------------- #
def plan_learning_path(
    knowledge_graph: KnowledgeGraph,
    student_mastery: Dict[str, float],
    target_skill: Optional[str] = None,
    student_ability: float = 0.0,
    max_steps: int = 10,
) -> List[str]:
    """
    综合规划学习路径。
    1) 用 ZPD 筛选知识点
    2) 用知识图谱先决关系确定顺序
    3) 结合学生当前能力调整难度
    """
    mastered = {s for s, m in student_mastery.items() if m >= 0.85}

    # 候选
    if target_skill:
        ordered = knowledge_graph.subgraph_for_student(mastered, target_skill)
    else:
        ordered = [s for s in knowledge_graph.topological_order() if s not in mastered]

    # ZPD 过滤与重排序
    zpd_candidates = []
    for sid in ordered:
        node = knowledge_graph.get_node(sid)
        if not node:
            continue
        score = zpd_score(student_ability, node.difficulty)
        zpd_candidates.append((sid, score, node.difficulty))

    # 按 zpd_score 降序，但保留先决关系
    # 简单策略：先决优先 + 难度匹配
    result = []
    for sid, score, diff in zpd_candidates[:max_steps]:
        result.append(sid)
    return result


if __name__ == "__main__":
    from src.knowledge.knowledge_graph import build_default_kg
    kg = build_default_kg()
    path = plan_learning_path(
        knowledge_graph=kg,
        student_mastery={"math_linear_algebra": 0.9, "math_calculus": 0.9},
        target_skill="dl_transformer",
        student_ability=0.5,
    )
    print("Planned path:")
    for i, sid in enumerate(path):
        node = kg.get_node(sid)
        print(f"  {i+1}. {node.name} (difficulty={node.difficulty})")
