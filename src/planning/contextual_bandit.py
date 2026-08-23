"""
Contextual Bandit 路径优化（LinUCB）
======================================

把"下一步推荐哪个知识点"建模为 Contextual Bandit 问题。

  Context (上下文):  x ∈ R^d (学生当前状态向量)
  Action (动作):     a ∈ A (候选知识点)
  Reward (奖励):     r = 学习收益

算法：LinUCB（线性置信上界）

  E[r | x, a] = θ_a^T x
  UCB_a = θ_a^T x + α * sqrt(x^T A_a^{-1} x)
  a* = argmax_a UCB_a

其中：
  θ_a: 动作 a 的参数向量
  A_a = X_a^T X_a + λ I    (动作 a 的设计矩阵)
  b_a = X_a^T r_a            (动作 a 的收益累加)
  θ_a = A_a^{-1} b_a

优势：
- 冷启动友好（无需离线训练数据）
- 天然支持探索-利用平衡
- 在线学习，不断优化
- 可解释：每个推荐都有 UCB 分数
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import PlanningConfig, get_config


# ---------------------------------------------------------------------- #
# LinUCB 实现
# ---------------------------------------------------------------------- #
class LinUCB:
    """
    LinUCB with disjoint linear models。

    适合动作集合固定且不太大的场景（几十到几百个知识点）。
    """
    def __init__(self, n_actions: int, d: int, config: Optional[PlanningConfig] = None):
        self.n_actions = n_actions
        self.d = d
        self.cfg = config or PlanningConfig()
        self.alpha = self.cfg.linucb_alpha
        self.lambda_reg = self.cfg.linucb_lambda_reg

        # 每个动作独立维护 A_a 和 b_a
        self.A = [self.lambda_reg * np.eye(d) for _ in range(n_actions)]
        self.b = [np.zeros(d) for _ in range(n_actions)]
        self.theta = [np.zeros(d) for _ in range(n_actions)]
        self.pulled_count = np.zeros(n_actions, dtype=int)

    def select(self, context: np.ndarray, available_actions: Optional[List[int]] = None) -> int:
        """
        选择动作（UCB 最大的）。
        context: 1D 向量
        available_actions: 候选动作索引列表（None = 全部）
        """
        if available_actions is None:
            available_actions = list(range(self.n_actions))

        x = context.reshape(-1, 1)  # (d, 1)
        best_action = available_actions[0]
        best_ucb = -np.inf

        for a in available_actions:
            A_inv = np.linalg.inv(self.A[a])
            theta_a = self.theta[a]
            mean = float(theta_a @ context)
            ucb = mean + self.alpha * np.sqrt(float(x.T @ A_inv @ x))
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = a

        return int(best_action)

    def update(self, action: int, context: np.ndarray, reward: float):
        """更新参数"""
        x = context.reshape(-1, 1)
        self.A[action] += x @ x.T
        self.b[action] += reward * context
        self.pulled_count[action] += 1
        self.theta[action] = np.linalg.solve(self.A[action], self.b[action])

    def get_confidence(self, action: int, context: np.ndarray) -> float:
        """获取某个动作在当前上下文下的置信上界"""
        if self.pulled_count[action] == 0:
            return float('inf')
        A_inv = np.linalg.inv(self.A[action])
        theta_a = self.theta[action]
        return float(theta_a @ context + self.alpha * np.sqrt(context @ A_inv @ context))


# ---------------------------------------------------------------------- #
# 路径规划器
# ---------------------------------------------------------------------- #
@dataclass
class PlanningDecision:
    """规划决策结果（含可解释性）"""
    next_skill: str
    expected_reward: float
    confidence: float
    alternatives: List[Tuple[str, float]]  # 备选 (skill_id, ucb_score)
    reasoning: str


class PathPlanner:
    """
    基于 Contextual Bandit 的路径规划器。
    """
    def __init__(
        self,
        n_skills: int,
        skill_id_to_idx: Dict[str, int],
        config: Optional[PlanningConfig] = None,
    ):
        self.n_skills = n_skills
        self.skill_id_to_idx = skill_id_to_idx
        self.idx_to_skill_id = {v: k for k, v in skill_id_to_idx.items()}
        self.cfg = config or PlanningConfig()
        self.bandit = LinUCB(
            n_actions=n_skills,
            d=self.cfg.linucb_d,
            config=self.cfg,
        )

    def plan_next(
        self,
        student_context: np.ndarray,
        available_skills: List[str],
        mastered_skills: Optional[List[str]] = None,
        target_skill: Optional[str] = None,
        k_top: int = 3,
    ) -> PlanningDecision:
        """
        为学生规划下一步学习的知识点。

        student_context: 学生当前状态向量（来自 CSN）
        available_skills: 候选知识点列表（来自知识图谱）
        mastered_skills: 已掌握知识点（用于过滤）
        target_skill: 最终学习目标（如果有）

        返回：PlanningDecision
        """
        mastered = set(mastered_skills or [])
        # 过滤掉已掌握
        candidates = [s for s in available_skills if s not in mastered]
        if target_skill and target_skill in candidates:
            # 偏向目标
            candidates.append(target_skill)
        if not candidates:
            # 兜底：返回第一个候选
            candidates = available_skills or list(self.skill_id_to_idx.keys())

        candidate_indices = [self.skill_id_to_idx[s] for s in candidates if s in self.skill_id_to_idx]

        if not candidate_indices:
            # 全部候选都未知，强行选第一个
            return PlanningDecision(
                next_skill=candidates[0],
                expected_reward=0.0,
                confidence=0.0,
                alternatives=[],
                reasoning="无可用候选，使用默认策略。",
            )

        # 把 context 截断或 pad 到 bandit 维度
        ctx = self._pad_context(student_context)

        # 选最优
        best_idx = self.bandit.select(ctx, candidate_indices)
        best_skill = self.idx_to_skill_id[best_idx]
        confidence = self.bandit.get_confidence(best_idx, ctx)
        expected = float(self.bandit.theta[best_idx] @ ctx)

        # 备选（按 UCB 排序）
        scored = []
        for a in candidate_indices:
            sid = self.idx_to_skill_id[a]
            c = self.bandit.get_confidence(a, ctx)
            scored.append((sid, c))
        scored.sort(key=lambda x: x[1], reverse=True)
        alternatives = scored[:k_top]

        # 生成可解释性说明
        reasoning = self._explain(best_skill, expected, confidence, target_skill)

        return PlanningDecision(
            next_skill=best_skill,
            expected_reward=expected,
            confidence=confidence,
            alternatives=alternatives,
            reasoning=reasoning,
        )

    def feedback(self, skill: str, context: np.ndarray, reward: float):
        """收到学习反馈，更新 bandit"""
        if skill not in self.skill_id_to_idx:
            return
        idx = self.skill_id_to_idx[skill]
        ctx = self._pad_context(context)
        self.bandit.update(idx, ctx, reward)

    def _pad_context(self, ctx: np.ndarray) -> np.ndarray:
        """把任意长度 context 截断或 zero-pad 到 bandit 维度"""
        if len(ctx) >= self.cfg.linucb_d:
            return ctx[:self.cfg.linucb_d]
        out = np.zeros(self.cfg.linucb_d)
        out[:len(ctx)] = ctx
        return out

    def _explain(self, skill: str, expected: float, confidence: float, target: Optional[str]) -> str:
        target_str = f"这是通往目标【{target}】的路径。" if target and skill == target else ""
        if target and skill == target:
            return f"推荐学习【{skill}】（学习目标），UCB 分数 {confidence:.2f}，预期收益 {expected:.2f}。"
        if confidence > 2.0:
            return f"推荐学习【{skill}】（探索候选，UCB={confidence:.2f}），系统对此知识点的学习价值仍不确定。{target_str}"
        return f"推荐学习【{skill}】（置信度 {confidence:.2f}，预期收益 {expected:.2f}）。综合考虑掌握度、难度、情绪状态后认为这是当前最佳选择。{target_str}"


# ---------------------------------------------------------------------- #
# 奖励函数
# ---------------------------------------------------------------------- #
def compute_reward(
    mastery_before: float,
    mastery_after: float,
    engagement: float,
    emotion_score: float,
    weights: Tuple[float, float, float, float] = (0.5, 0.3, 0.1, 0.1),
) -> float:
    """
    综合奖励函数：
    - mastery_gain: 掌握度提升
    - absolute_mastery: 当前掌握度（鼓励达到高水平）
    - engagement: 参与度
    - emotion: 情绪（积极加分）

    返回 0~1 之间的奖励。
    """
    w_gain, w_abs, w_eng, w_emo = weights
    gain = max(0.0, mastery_after - mastery_before)
    r = w_gain * gain + w_abs * mastery_after + w_eng * engagement + w_emo * emotion_score
    return float(np.clip(r, 0, 1))


if __name__ == "__main__":
    # 简单自检
    planner = PathPlanner(
        n_skills=5,
        skill_id_to_idx={"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
    )
    ctx = np.random.rand(32)
    decision = planner.plan_next(ctx, ["a", "b", "c", "d"], mastered_skills=[])
    print(f"Next: {decision.next_skill}, confidence: {decision.confidence:.3f}")
    print(f"Reasoning: {decision.reasoning}")

    # 模拟反馈
    for _ in range(20):
        ctx = np.random.rand(32)
        d = planner.plan_next(ctx, ["a", "b", "c", "d"], mastered_skills=["a"])
        reward = compute_reward(0.3, 0.6, 0.7, 0.8)
        planner.feedback(d.next_skill, ctx, reward)
    print("After 20 updates:")
    decision = planner.plan_next(ctx, ["a", "b", "c", "d"], mastered_skills=["a"])
    print(f"  Best now: {decision.next_skill}, confidence: {decision.confidence:.3f}")
