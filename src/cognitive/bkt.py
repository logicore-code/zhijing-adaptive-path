"""
BKT (Bayesian Knowledge Tracing)
=================================

经典 4 状态 HMM 隐变量模型（Corbett & Anderson, 1995）。

四状态：
- 未掌握 (Not Learned)
- 已掌握 (Learned)
- 学习中 → 暂时合并到 Not Learned（标准 2 状态 BKT）
- 失误 / 猜测 → 引入观测模型

形式化：
    P(L_t) = P(L_{t-1} | obs_{t-1}) + (1 - P(L_{t-1} | obs_{t-1})) * p_learn
    P(L_t | obs_t) = [P(obs_t | L_t) * P(L_t)] / P(obs_t)

其中：
    P(obs_t=correct | L_t) = 1 - p_slip
    P(obs_t=correct | not L_t) = p_guess
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from src.config import CognitiveConfig


@dataclass
class BKTSkillState:
    """单个技能点的 BKT 状态"""
    skill_id: str
    p_learned: float = 0.1  # 已掌握后验概率
    history: List[tuple] = field(default_factory=list)  # [(is_correct, p_learned_after)]

    def reset(self):
        self.p_learned = 0.1
        self.history = []


class BayesianKnowledgeTracing:
    """
    经典 BKT 实现。

    用法：
        bkt = BayesianKnowledgeTracing()
        state = bkt.get_or_create("knn")
        bkt.update(state, is_correct=True)
        print(state.p_learned)
    """

    def __init__(self, config: Optional[CognitiveConfig] = None):
        self.cfg = config or CognitiveConfig()
        self.p_init = self.cfg.bkt_p_init
        self.p_learn = self.cfg.bkt_p_learn
        self.p_slip = self.cfg.bkt_p_slip
        self.p_guess = self.cfg.bkt_p_guess
        self.states: Dict[str, BKTSkillState] = {}

    # ------------------------------------------------------------------ #
    # 状态管理
    # ------------------------------------------------------------------ #
    def get_or_create(self, skill_id: str) -> BKTSkillState:
        if skill_id not in self.states:
            self.states[skill_id] = BKTSkillState(skill_id=skill_id, p_learned=self.p_init)
        return self.states[skill_id]

    def reset(self, skill_id: Optional[str] = None):
        if skill_id:
            self.states.pop(skill_id, None)
        else:
            self.states.clear()

    # ------------------------------------------------------------------ #
    # 核心：贝叶斯更新
    # ------------------------------------------------------------------ #
    def update(self, state: BKTSkillState, is_correct: bool) -> float:
        """
        给定新观测，更新后验 P(L_t)。
        返回更新后的 p_learned。
        """
        p_l_prev = state.p_learned

        # Step 1: 学习转移 (Prior update)
        p_l_transitioned = p_l_prev + (1.0 - p_l_prev) * self.p_learn

        # Step 2: 观测更新
        if is_correct:
            # P(obs=correct | L) = 1 - p_slip
            # P(obs=correct | not L) = p_guess
            p_obs_given_l = 1.0 - self.p_slip
            p_obs_given_not_l = self.p_guess
        else:
            p_obs_given_l = self.p_slip
            p_obs_given_not_l = 1.0 - self.p_guess

        numerator = p_obs_given_l * p_l_transitioned
        denominator = numerator + p_obs_given_not_l * (1.0 - p_l_transitioned)
        p_l_posterior = numerator / (denominator + 1e-12)

        state.p_learned = float(p_l_posterior)
        state.history.append((int(is_correct), state.p_learned))
        return state.p_learned

    def batch_update(self, skill_id: str, observations: List[bool]) -> List[float]:
        """批量更新，返回每次更新后的 p_learned 序列"""
        state = self.get_or_create(skill_id)
        return [self.update(state, obs) for obs in observations]

    # ------------------------------------------------------------------ #
    # 参数拟合 (EM 简化版)
    # ------------------------------------------------------------------ #
    def fit(self, sequences: List[List[tuple]], max_iter: int = 20, tol: float = 1e-3):
        """
        给定学生作答序列 [(skill_id, is_correct), ...] 学习 BKT 参数。
        简化版：使用坐标下降（coordinate descent）。
        sequences: List of List[(skill_id, is_correct)]
        """
        for _ in range(max_iter):
            old = (self.p_init, self.p_learn, self.p_slip, self.p_guess)
            # 这里给出占位实现；实际 EM 通过 scipy.optimize 求解
            # 保持接口稳定，工程上常用 libBKT / pyBKT 工具
            new = self._em_step(sequences)
            self.p_init, self.p_learn, self.p_slip, self.p_guess = new
            if max(abs(a - b) for a, b in zip(old, new)) < tol:
                break

    def _em_step(self, sequences):
        """单步 EM，简化（工程上会调用 libBKT）"""
        # 占位：用经验估计
        # 真实实现会用 forward-backward + M-step
        return (self.p_init, self.p_learn, self.p_slip, self.p_guess)


# ---------------------------------------------------------------------- #
# 便捷函数
# ---------------------------------------------------------------------- #
def predict_next_correct(bkt: BayesianKnowledgeTracing, skill_id: str) -> float:
    """预测学生下一次答对该技能点的概率"""
    state = bkt.get_or_create(skill_id)
    p_l = state.p_learned
    p_correct = p_l * (1.0 - bkt.p_slip) + (1.0 - p_l) * bkt.p_guess
    return float(p_correct)


if __name__ == "__main__":
    bkt = BayesianKnowledgeTracing()
    state = bkt.get_or_create("knn")
    print(f"Initial: {state.p_learned:.3f}")
    for ans in [True, True, True, False, True, True, True, True]:
        p = bkt.update(state, ans)
        print(f"  After ans={ans}: p_learned={p:.3f}, predict_next={predict_next_correct(bkt,'knn'):.3f}")
