"""
Cognitive State Network (CSN)
================================

三模型融合的学生认知状态估计器。

  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │   DKT    │     │   BKT    │     │   IRT    │
  │ (长程序列)│     │(概率更新)│     │(能力估计)│
  └─────┬────┘     └─────┬────┘     └─────┬────┘
        │  dkt_p        │  bkt_p        │  irt_theta
        │               │               │  item_difficulty
        ▼               ▼               ▼
       ┌───────────────────────────────┐
       │  Weighted Bayesian Fusion     │
       │  + Uncertainty Estimation     │
       └───────────────┬───────────────┘
                       ▼
              Mastery + Confidence

融合策略：
1. 短序列时（n < 5）：以 BKT 为主（冷启动友好）
2. 中等序列：等权融合
3. 长序列：DKT 为主（捕捉长程依赖）
4. IRT 提供整体能力锚点 + 不确定性

不确定性估计：
- 用各模型输出方差衡量 epistemic uncertainty
- 用来判断什么时候需要更多诊断（adaptive testing）
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from src.cognitive.bkt import BayesianKnowledgeTracing, BKTSkillState
from src.cognitive.irt import IRT2PL, IRTStudent, IRTItem
from src.cognitive.dkt import NumPyDKT
from src.config import CognitiveConfig, get_config


# ---------------------------------------------------------------------- #
# 学生画像
# ---------------------------------------------------------------------- #
@dataclass
class StudentProfile:
    """学生画像：综合多模型估计"""
    student_id: str
    skill_mastery: Dict[str, float] = field(default_factory=dict)  # 知识点 -> 掌握度
    skill_confidence: Dict[str, float] = field(default_factory=dict)  # 置信度
    overall_ability: float = 0.0      # IRT 估计的能力
    overall_ability_var: float = 1.0
    learning_style: str = "unknown"  # visual/verbal/active/reflective
    interaction_count: int = 0
    last_active: Optional[float] = None
    # 学习者画像向量（供 Contextual Bandit 使用）
    feature_vector: np.ndarray = field(default=None)  # type: ignore

    def to_dict(self) -> Dict:
        return {
            "student_id": self.student_id,
            "skill_mastery": self.skill_mastery,
            "skill_confidence": self.skill_confidence,
            "overall_ability": self.overall_ability,
            "learning_style": self.learning_style,
            "interaction_count": self.interaction_count,
        }


# ---------------------------------------------------------------------- #
# CSN 主类
# ---------------------------------------------------------------------- #
class CognitiveStateNetwork:
    """
    Cognitive State Network：三模型融合的学生认知状态网络。

    这是智径 AdaptivePath 的核心创新点之一。
    """

    def __init__(
        self,
        num_skills: int,
        skill_id_to_idx: Dict[str, int],
        config: Optional[CognitiveConfig] = None,
    ):
        self.cfg = config or CognitiveConfig()
        self.num_skills = num_skills
        self.skill_id_to_idx = skill_id_to_idx
        self.idx_to_skill_id = {v: k for k, v in skill_id_to_idx.items()}

        # 三大模型
        self.bkt = BayesianKnowledgeTracing(self.cfg)
        self.irt = IRT2PL(self.cfg)
        self.dkt = NumPyDKT(num_skills=num_skills, hidden_size=self.cfg.dkt_hidden_size, config=self.cfg)

        # 学生画像存储
        self.profiles: Dict[str, StudentProfile] = {}

        # 各学生的作答历史
        self.history: Dict[str, List[Tuple[int, bool, str]]] = defaultdict(list)  # (skill_idx, is_correct, item_id)

    # ------------------------------------------------------------------ #
    # 学生画像管理
    # ------------------------------------------------------------------ #
    def get_or_create_profile(self, student_id: str) -> StudentProfile:
        if student_id not in self.profiles:
            self.profiles[student_id] = StudentProfile(student_id=student_id)
        return self.profiles[student_id]

    # ------------------------------------------------------------------ #
    # 单次更新
    # ------------------------------------------------------------------ #
    def update(
        self,
        student_id: str,
        skill_id: str,
        item_id: str,
        is_correct: bool,
        response_time: float = 0.0,
    ) -> Tuple[float, float]:
        """
        接收一次作答，更新学生认知状态。
        返回：(mastery, confidence)
        """
        if skill_id not in self.skill_id_to_idx:
            raise ValueError(f"Unknown skill_id: {skill_id}")
        skill_idx = self.skill_id_to_idx[skill_id]

        # 1) BKT 更新
        bkt_state = self.bkt.get_or_create(skill_id)
        bkt_p = self.bkt.update(bkt_state, is_correct)

        # 2) IRT 更新
        irt_student = self.irt.get_or_create_student(student_id)
        if item_id not in self.irt.items:
            # 自动为新题设置默认参数
            self.irt.add_item(item_id, a=1.0, b=0.0, skill_id=skill_id)
        self.irt.update_student(irt_student, [(item_id, is_correct)])

        # 3) 记录 DKT 序列
        self.history[student_id].append((skill_idx, is_correct, item_id))

        # 4) 融合
        dkt_seq = [(s, c) for s, c, _ in self.history[student_id]]
        dkt_p_vec = self.dkt.predict_mastery(dkt_seq)
        dkt_p = float(dkt_p_vec[skill_idx])

        # 5) IRT 概率
        item = self.irt.items[item_id]
        irt_p = IRT2PL.prob_correct(irt_student.theta, item.a, item.b)

        # 融合权重（根据样本量动态调整）
        n = len(self.history[student_id])
        w_dkt, w_bkt, w_irt = self._adaptive_weights(n)

        mastery = w_dkt * dkt_p + w_bkt * bkt_p + w_irt * irt_p

        # 置信度：基于三个模型的方差（越小越置信）
        ps = np.array([dkt_p, bkt_p, irt_p])
        var = float(np.var(ps))
        confidence = 1.0 / (1.0 + 10.0 * var)  # 方差越大置信度越低
        # 样本量越大置信度越高
        confidence = float(confidence * min(1.0, n / 10.0) + 0.1 * (n > 0))

        # 更新画像
        profile = self.get_or_create_profile(student_id)
        profile.skill_mastery[skill_id] = float(np.clip(mastery, 0, 1))
        profile.skill_confidence[skill_id] = float(np.clip(confidence, 0, 1))
        profile.overall_ability = irt_student.theta
        profile.overall_ability_var = irt_student.theta_var
        profile.interaction_count += 1
        profile.feature_vector = self._build_feature_vector(profile)

        return float(mastery), float(confidence)

    def _adaptive_weights(self, n: int) -> Tuple[float, float, float]:
        """
        根据样本量自适应调整三模型权重。
        n < 5:  BKT 主导（冷启动）
        5 ≤ n < 20: 三模型等权
        n ≥ 20: DKT 主导（长程依赖）
        """
        if n < 5:
            return 0.2, 0.5, 0.3
        elif n < 20:
            return 0.4, 0.3, 0.3
        else:
            return 0.5, 0.2, 0.3

    # ------------------------------------------------------------------ #
    # 批量查询
    # ------------------------------------------------------------------ #
    def get_mastery(self, student_id: str, skill_id: str) -> float:
        profile = self.get_or_create_profile(student_id)
        return profile.skill_mastery.get(skill_id, 0.1)

    def get_confidence(self, student_id: str, skill_id: str) -> float:
        profile = self.get_or_create_profile(student_id)
        return profile.skill_confidence.get(skill_id, 0.1)

    def get_all_mastery(self, student_id: str) -> Dict[str, float]:
        profile = self.get_or_create_profile(student_id)
        return dict(profile.skill_mastery)

    def predict_next_correct(self, student_id: str, skill_id: str, item_id: Optional[str] = None) -> float:
        """预测答对概率（综合三模型）"""
        mastery = self.get_mastery(student_id, skill_id)
        if item_id and item_id in self.irt.items:
            item = self.irt.items[item_id]
            irt_student = self.irt.get_or_create_student(student_id)
            return float(IRT2PL.prob_correct(irt_student.theta, item.a, item.b) * 0.5 + mastery * 0.5)
        return mastery

    # ------------------------------------------------------------------ #
    # 特征向量（供 Contextual Bandit 使用）
    # ------------------------------------------------------------------ #
    def _build_feature_vector(self, profile: StudentProfile) -> np.ndarray:
        """
        构建学生特征向量 (32 维)：
        - 前 16 维：各知识点掌握度
        - 1 维：整体能力 θ
        - 1 维：能力方差
        - 1 维：交互次数
        - 其余：当前学习风格、活跃度等
        """
        vec = np.zeros(32)
        mastery_list = list(profile.skill_mastery.items())
        for i, (skill_id, m) in enumerate(mastery_list[:16]):
            vec[i] = m
        vec[16] = profile.overall_ability
        vec[17] = profile.overall_ability_var
        vec[18] = min(profile.interaction_count / 50.0, 1.0)
        # 学习风格 one-hot
        style_map = {"visual": 19, "verbal": 20, "active": 21, "reflective": 22}
        if profile.learning_style in style_map:
            vec[style_map[profile.learning_style]] = 1.0
        return vec

    def get_feature_vector(self, student_id: str) -> np.ndarray:
        profile = self.get_or_create_profile(student_id)
        if profile.feature_vector is None:
            profile.feature_vector = self._build_feature_vector(profile)
        return profile.feature_vector

    # ------------------------------------------------------------------ #
    # 报告
    # ------------------------------------------------------------------ #
    def explain(self, student_id: str, skill_id: str) -> Dict:
        """
        解释性报告：返回某学生某知识点的三模型估计 + 融合结果。
        用于可解释性 UI。
        """
        bkt_state = self.bkt.get_or_create(skill_id)
        mastery = self.get_mastery(student_id, skill_id)
        confidence = self.get_confidence(student_id, skill_id)
        irt_student = self.irt.get_or_create_student(student_id)
        dkt_seq = [(s, c) for s, c, _ in self.history.get(student_id, [])]
        dkt_vec = self.dkt.predict_mastery(dkt_seq)
        skill_idx = self.skill_id_to_idx[skill_id]
        dkt_p = float(dkt_vec[skill_idx])

        return {
            "skill_id": skill_id,
            "mastery": mastery,
            "confidence": confidence,
            "bkt_p": bkt_state.p_learned,
            "dkt_p": dkt_p,
            "irt_theta": irt_student.theta,
            "irt_var": irt_student.theta_var,
            "n_obs": len(self.history.get(student_id, [])),
            "explanation": self._generate_explanation(mastery, confidence, bkt_state.p_learned, irt_student.theta),
        }

    def _generate_explanation(self, mastery, confidence, bkt_p, irt_theta) -> str:
        if mastery > 0.85:
            level = "已熟练掌握"
            rec = "可以进入下一个进阶知识点"
        elif mastery > 0.6:
            level = "基本掌握，但仍有薄弱环节"
            rec = "建议做几道综合题巩固"
        elif mastery > 0.3:
            level = "部分掌握，需要强化"
            rec = "推荐回顾核心概念并做应用题"
        else:
            level = "尚未掌握"
            rec = "建议从基础概念重新学习，配合范例讲解"
        if confidence < 0.3:
            rec += "（注意：当前估计置信度较低，建议增加诊断题目）"
        return f"学生当前状态：{level}。推荐策略：{rec}。"


# ---------------------------------------------------------------------- #
# 工厂函数
# ---------------------------------------------------------------------- #
def build_csn_from_graph(graph) -> CognitiveStateNetwork:
    """
    从知识图谱构建 CSN。
    graph: KnowledgeGraph 实例
    """
    skill_ids = list(graph.nodes.keys())
    skill_id_to_idx = {sid: i for i, sid in enumerate(skill_ids)}
    return CognitiveStateNetwork(num_skills=len(skill_ids), skill_id_to_idx=skill_id_to_idx)


if __name__ == "__main__":
    # 简单自检
    from src.knowledge.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    csn = build_csn_from_graph(kg)

    # 模拟一次作答
    skill = list(kg.nodes.keys())[0]
    m, c = csn.update("S001", skill, "item_1", True)
    print(f"After 1 obs: skill={skill}, mastery={m:.3f}, confidence={c:.3f}")
    report = csn.explain("S001", skill)
    print(f"Explanation: {report['explanation']}")
