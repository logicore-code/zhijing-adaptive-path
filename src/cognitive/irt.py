"""
IRT (Item Response Theory) - 2PL 模型
======================================

2 参数 Logistic 模型：
    P(θ) = 1 / (1 + exp(-a * (θ - b)))

其中：
    θ: 学生能力
    a: 题目区分度
    b: 题目难度

适用场景：估计学生整体能力 θ，并诊断每个知识点的难度 / 区分度。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import CognitiveConfig


@dataclass
class IRTStudent:
    """学生 IRT 状态"""
    student_id: str
    theta: float = 0.0  # 能力（logit 尺度）
    theta_var: float = 1.0  # 能力估计的不确定性


@dataclass
class IRTItem:
    """题目 IRT 参数"""
    item_id: str
    a: float = 1.0  # 区分度
    b: float = 0.0  # 难度
    skill_id: str = ""


class IRT2PL:
    """
    2PL IRT 模型。
    使用 MLE / MAP 估计学生能力 θ，使用 MLE 估计题目参数 a, b。
    """

    def __init__(self, config: Optional[CognitiveConfig] = None):
        self.cfg = config or CognitiveConfig()
        self.students: Dict[str, IRTStudent] = {}
        self.items: Dict[str, IRTItem] = {}

    # ------------------------------------------------------------------ #
    # 概率计算
    # ------------------------------------------------------------------ #
    @staticmethod
    def prob_correct(theta: float, a: float, b: float) -> float:
        """2PL 模型：给定学生能力和题目参数，计算答对概率"""
        z = a * (theta - b)
        # 数值稳定
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def prob_correct_batch(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """批量版本"""
        z = a * (theta - b)
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------ #
    # 学生能力估计
    # ------------------------------------------------------------------ #
    def update_student(self, student: IRTStudent, responses: List[Tuple[str, bool]]):
        """
        给定学生的历史作答，MAP 估计 theta。
        responses: [(item_id, is_correct), ...]
        """
        if not responses:
            return

        # 收集 item 参数
        a_arr, b_arr, y_arr = [], [], []
        for item_id, is_correct in responses:
            if item_id not in self.items:
                continue
            item = self.items[item_id]
            a_arr.append(item.a)
            b_arr.append(item.b)
            y_arr.append(1 if is_correct else 0)

        if not a_arr:
            return

        a_arr = np.array(a_arr)
        b_arr = np.array(b_arr)
        y_arr = np.array(y_arr, dtype=float)

        # 牛顿法求解 MLE（先验 N(0,1) -> MAP）
        theta = student.theta
        for _ in range(50):
            p = self.prob_correct_batch(theta, a_arr, b_arr)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            # 梯度
            grad = np.sum(a_arr * (y_arr - p)) - theta  # MAP 先验
            # Hessian
            hess = -np.sum(a_arr ** 2 * p * (1 - p)) - 1
            step = -grad / hess
            theta_new = theta + step
            if abs(theta_new - theta) < 1e-4:
                theta = theta_new
                break
            theta = theta_new

        # Fisher 信息 -> 方差
        p = self.prob_correct_batch(theta, a_arr, b_arr)
        info = np.sum(a_arr ** 2 * p * (1 - p)) + 1
        student.theta = float(theta)
        student.theta_var = float(1.0 / max(info, 1e-6))

    # ------------------------------------------------------------------ #
    # 题目参数估计
    # ------------------------------------------------------------------ #
    def fit_item(self, item: IRTItem, responses: List[Tuple[str, bool]]):
        """
        给定一道题被多个学生的作答，估计 a, b。
        responses: [(student_id, is_correct), ...]
        """
        if not responses:
            return
        thetas, ys = [], []
        for sid, correct in responses:
            if sid in self.students:
                thetas.append(self.students[sid].theta)
                ys.append(1 if correct else 0)
        if len(thetas) < 3:
            return
        thetas = np.array(thetas)
        ys = np.array(ys, dtype=float)

        a, b = item.a, item.b
        for _ in range(50):
            p = self.prob_correct_batch(thetas, a, b)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            r = ys - p
            # ∂l/∂a = Σ r * (θ - b)
            d_a = np.sum(r * (thetas - b))
            # ∂l/∂b = Σ r * (-a)
            d_b = np.sum(r * (-a))
            # Hessian 近似
            H_aa = -np.sum((thetas - b) ** 2 * p * (1 - p))
            H_ab = -np.sum(-(thetas - b) * a * p * (1 - p)) - np.sum(r)
            H_bb = -np.sum(a ** 2 * p * (1 - p))

            det = H_aa * H_bb - H_ab ** 2
            if abs(det) < 1e-6:
                break
            a_new = a - (H_bb * d_a - H_ab * d_b) / det
            b_new = b - (-H_ab * d_a + H_aa * d_b) / det

            # 边界约束
            a_new = float(np.clip(a_new, self.cfg.irt_discrimination_range[0], self.cfg.irt_discrimination_range[1]))
            b_new = float(np.clip(b_new, self.cfg.irt_difficulty_range[0], self.cfg.irt_difficulty_range[1]))

            if abs(a_new - a) < 1e-4 and abs(b_new - b) < 1e-4:
                a, b = a_new, b_new
                break
            a, b = a_new, b_new

        item.a, item.b = a, b

    # ------------------------------------------------------------------ #
    # 状态管理
    # ------------------------------------------------------------------ #
    def get_or_create_student(self, student_id: str) -> IRTStudent:
        if student_id not in self.students:
            self.students[student_id] = IRTStudent(student_id=student_id)
        return self.students[student_id]

    def add_item(self, item_id: str, a: float, b: float, skill_id: str = ""):
        if item_id not in self.items:
            self.items[item_id] = IRTItem(item_id=item_id, a=a, b=b, skill_id=skill_id)
        else:
            self.items[item_id].a = a
            self.items[item_id].b = b


# ---------------------------------------------------------------------- #
# 便捷函数
# ---------------------------------------------------------------------- #
def ability_level(theta: float) -> str:
    """将能力 logit 值映射为人类可读等级"""
    if theta < -1.5:
        return "入门 (Novice)"
    elif theta < -0.5:
        return "初级 (Beginner)"
    elif theta < 0.5:
        return "中级 (Intermediate)"
    elif theta < 1.5:
        return "高级 (Advanced)"
    else:
        return "专家 (Expert)"


if __name__ == "__main__":
    irt = IRT2PL()
    # 简单测试
    s = irt.get_or_create_student("S001")
    irt.add_item("Q001", a=1.0, b=0.0, skill_id="linear_regression")
    irt.add_item("Q002", a=1.2, b=0.5, skill_id="linear_regression")
    irt.add_item("Q003", a=0.8, b=-0.5, skill_id="linear_regression")

    # 模拟作答
    responses = [("Q001", True), ("Q002", False), ("Q003", True)]
    irt.update_student(s, responses)
    print(f"Student {s.student_id} theta={s.theta:.3f}, level={ability_level(s.theta)}")
    print(f"Predict on Q001: {IRT2PL.prob_correct(s.theta, 1.0, 0.0):.3f}")
