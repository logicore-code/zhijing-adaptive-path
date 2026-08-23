"""
知识图谱
==========

本项目配套构建了一个针对"人工智能专业导论"的精细化知识图谱。

特点：
- 节点 = 知识点（skill/concept）
- 边 = 先决关系 (prerequisite) / 相关 (related) / 易错 (confused_with)
- 每个节点携带：难度、典型错误、相关资源

数据可扩展到任意学科（电子信息、计算机、数学等）。
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path

import networkx as nx

from src.config import KG_DIR


# ---------------------------------------------------------------------- #
# 数据类
# ---------------------------------------------------------------------- #
@dataclass
class KnowledgeNode:
    """知识点节点"""
    id: str
    name: str
    category: str  # 类别：基础理论/机器学习/深度学习/应用/数学基础
    difficulty: float  # 0~1
    est_learning_time: int  # 估计学习时长（分钟）
    bloom_level: str  # Bloom 分类：remember/understand/apply/analyze/evaluate/create
    description: str = ""
    typical_errors: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)


@dataclass
class KnowledgeEdge:
    """知识关系边"""
    source: str
    target: str
    relation: str  # prerequisite / related / confused_with / extends
    weight: float = 1.0


# ---------------------------------------------------------------------- #
# 知识图谱主类
# ---------------------------------------------------------------------- #
class KnowledgeGraph:
    """
    知识图谱：基于 NetworkX 实现。
    支持：
    - 先决关系链查询
    - 拓扑排序生成学习顺序
    - 子图提取
    - JSON 持久化
    """

    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.graph = nx.DiGraph()

    # ------------------------------------------------------------------ #
    # 节点 / 边操作
    # ------------------------------------------------------------------ #
    def add_node(self, node: KnowledgeNode):
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.__dict__)

    def add_edge(self, edge: KnowledgeEdge):
        self.edges.append(edge)
        if edge.relation == "prerequisite":
            self.graph.add_edge(edge.source, edge.target, relation=edge.relation, weight=edge.weight)
        else:
            # 非 prereq 边单独存
            if not self.graph.has_edge(edge.source, edge.target):
                self.graph.add_edge(edge.source, edge.target, relation=edge.relation, weight=edge.weight)

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        return self.nodes.get(node_id)

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #
    def prerequisites(self, skill_id: str) -> List[str]:
        """获取某知识点的所有先决知识点"""
        if skill_id not in self.graph:
            return []
        return list(self.graph.predecessors(skill_id))

    def dependents(self, skill_id: str) -> List[str]:
        """获取以某知识点为先决的所有后继"""
        if skill_id not in self.graph:
            return []
        return list(self.graph.successors(skill_id))

    def related_skills(self, skill_id: str) -> List[str]:
        """相关知识点（related 关系）"""
        related = []
        for e in self.edges:
            if e.relation == "related" and e.source == skill_id:
                related.append(e.target)
            if e.relation == "related" and e.target == skill_id:
                related.append(e.source)
        return related

    def confused_with(self, skill_id: str) -> List[str]:
        """易混淆的知识点"""
        result = []
        for e in self.edges:
            if e.relation == "confused_with":
                if e.source == skill_id:
                    result.append(e.target)
                elif e.target == skill_id:
                    result.append(e.source)
        return result

    def is_prerequisite_of(self, a: str, b: str) -> bool:
        """a 是否是 b 的先决"""
        return self.graph.has_edge(a, b) and self.graph[a][b].get("relation") == "prerequisite"

    def all_prerequisites_recursive(self, skill_id: str) -> Set[str]:
        """递归获取所有先决（传递闭包）"""
        if skill_id not in self.graph:
            return set()
        return set(nx.ancestors(self.graph, skill_id))

    # ------------------------------------------------------------------ #
    # 路径 / 拓扑
    # ------------------------------------------------------------------ #
    def topological_order(self) -> List[str]:
        """拓扑排序（学习顺序建议）"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            # 有环，按 difficulty 排序
            return sorted(self.nodes.keys(), key=lambda x: self.nodes[x].difficulty)

    def learning_levels(self) -> Dict[int, List[str]]:
        """按层次划分（同一层无先决关系，可并行学习）"""
        levels = {}
        for node_id in self.topological_order():
            prereqs = self.prerequisites(node_id)
            level = 0 if not prereqs else max(levels.get(p, 0) for p in prereqs) + 1
            levels.setdefault(level, []).append(node_id)
        return levels

    def shortest_learning_path(self, from_skill: str, to_skill: str) -> List[str]:
        """从一个知识点到另一个知识点的最短学习路径"""
        try:
            return list(nx.shortest_path(self.graph, from_skill, to_skill))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    # ------------------------------------------------------------------ #
    # 子图
    # ------------------------------------------------------------------ #
    def subgraph_for_student(self, mastered_skills: Set[str], target_skill: str) -> List[str]:
        """
        为学生规划子图：基于已掌握和目标，提取需要学习的所有相关知识点。
        """
        all_needed = self.all_prerequisites_recursive(target_skill) | {target_skill}
        to_learn = all_needed - mastered_skills
        # 拓扑排序
        subg = self.graph.subgraph(all_needed)
        try:
            order = list(nx.topological_sort(subg))
        except nx.NetworkXUnfeasible:
            order = list(all_needed)
        return [s for s in order if s in to_learn]

    # ------------------------------------------------------------------ #
    # 推荐候选（供 Bandit 使用）
    # ------------------------------------------------------------------ #
    def candidate_skills(self, mastered: Set[str], target: Optional[str] = None, k: int = 10) -> List[str]:
        """
        候选知识点：
        - 若指定 target：target 及其缺失的先决
        - 否则：当前可学（先决都已掌握） + 难度匹配
        """
        if target:
            return self.subgraph_for_student(mastered, target)[:k]

        # 找所有"先决都已掌握"的节点
        candidates = []
        for nid, node in self.nodes.items():
            if nid in mastered:
                continue
            prereqs = set(self.prerequisites(nid))
            if prereqs.issubset(mastered):
                candidates.append(nid)
        return candidates[:k]

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict:
        return {
            "nodes": [n.__dict__ for n in self.nodes.values()],
            "edges": [e.__dict__ for e in self.edges],
        }

    def save(self, path: Optional[Path] = None):
        path = path or (KG_DIR / "ai_intro_kg.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kg = cls()
        for n in data["nodes"]:
            kg.add_node(KnowledgeNode(**n))
        for e in data["edges"]:
            kg.add_edge(KnowledgeEdge(**e))
        return kg


# ---------------------------------------------------------------------- #
# 内置：人工智能导论知识图谱
# ---------------------------------------------------------------------- #
def build_default_kg() -> KnowledgeGraph:
    """
    构建一个针对"人工智能专业导论"的精细化知识图谱。
    涵盖：数学基础、机器学习、深度学习、自然语言处理、计算机视觉、强化学习、伦理。
    """
    kg = KnowledgeGraph()

    # 节点定义
    nodes = [
        # ===== 数学基础 =====
        KnowledgeNode(
            id="math_linear_algebra",
            name="线性代数基础",
            category="数学基础",
            difficulty=0.3,
            est_learning_time=180,
            bloom_level="understand",
            description="向量、矩阵、特征值、特征向量、矩阵分解",
            key_concepts=["向量空间", "矩阵运算", "特征值分解", "SVD"],
            typical_errors=["矩阵乘法顺序搞反", "特征值与特征向量对应关系混乱"],
        ),
        KnowledgeNode(
            id="math_calculus",
            name="微积分",
            category="数学基础",
            difficulty=0.35,
            est_learning_time=180,
            bloom_level="understand",
            description="导数、偏导、梯度、链式法则、积分",
            key_concepts=["导数", "偏导数", "梯度", "链式法则"],
            typical_errors=["偏导与全导混淆", "链式法则漏项"],
        ),
        KnowledgeNode(
            id="math_probability",
            name="概率论",
            category="数学基础",
            difficulty=0.4,
            est_learning_time=200,
            bloom_level="understand",
            description="随机变量、概率分布、条件概率、贝叶斯定理",
            key_concepts=["随机变量", "条件概率", "贝叶斯定理", "期望方差"],
            typical_errors=["先验后验混淆", "独立与条件独立搞不清"],
        ),
        KnowledgeNode(
            id="math_optimization",
            name="最优化方法",
            category="数学基础",
            difficulty=0.5,
            est_learning_time=160,
            bloom_level="apply",
            description="梯度下降、牛顿法、凸优化、拉格朗日乘子",
            key_concepts=["梯度下降", "随机梯度下降", "凸函数", "约束优化"],
            typical_errors=["学习率过大导致发散", "凸非凸判断错误"],
        ),

        # ===== 机器学习基础 =====
        KnowledgeNode(
            id="ml_concept",
            name="机器学习基本概念",
            category="机器学习",
            difficulty=0.2,
            est_learning_time=90,
            bloom_level="remember",
            description="监督/无监督/强化学习、训练/测试、过拟合",
            key_concepts=["监督学习", "无监督学习", "过拟合", "泛化"],
            typical_errors=["训练集和测试集混用", "过拟合与欠拟合判断错"],
        ),
        KnowledgeNode(
            id="ml_linear_regression",
            name="线性回归",
            category="机器学习",
            difficulty=0.3,
            est_learning_time=120,
            bloom_level="apply",
            description="一元/多元线性回归、损失函数、最小二乘",
            key_concepts=["最小二乘", "MSE", "正规方程", "梯度下降"],
            typical_errors=["忘记加偏置项", "特征未归一化"],
        ),
        KnowledgeNode(
            id="ml_logistic_regression",
            name="逻辑回归",
            category="机器学习",
            difficulty=0.4,
            est_learning_time=120,
            bloom_level="apply",
            description="二分类、sigmoid、交叉熵、梯度",
            key_concepts=["sigmoid", "交叉熵损失", "极大似然估计"],
            typical_errors=["与线性回归混淆", "sigmoid 饱和导致梯度消失"],
        ),
        KnowledgeNode(
            id="ml_decision_tree",
            name="决策树",
            category="机器学习",
            difficulty=0.4,
            est_learning_time=120,
            bloom_level="apply",
            description="信息增益、ID3/C4.5/CART、剪枝",
            key_concepts=["信息熵", "信息增益", "基尼系数", "剪枝"],
            typical_errors=["信息增益 vs 增益率", "过拟合未剪枝"],
        ),
        KnowledgeNode(
            id="ml_svm",
            name="支持向量机",
            category="机器学习",
            difficulty=0.6,
            est_learning_time=180,
            bloom_level="analyze",
            description="最大间隔、核函数、SMO",
            key_concepts=["最大间隔", "核函数", "对偶问题", "支持向量"],
            typical_errors=["核函数选择不当", "软间隔参数 C 调节"],
        ),
        KnowledgeNode(
            id="ml_ensemble",
            name="集成学习",
            category="机器学习",
            difficulty=0.55,
            est_learning_time=150,
            bloom_level="analyze",
            description="Bagging、Boosting、随机森林、XGBoost",
            key_concepts=["Bagging", "Boosting", "随机森林", "梯度提升"],
            typical_errors=["Bagging vs Boosting 区别", "XGBoost 调参"],
        ),

        # ===== 深度学习 =====
        KnowledgeNode(
            id="dl_perceptron",
            name="感知机与神经网络基础",
            category="深度学习",
            difficulty=0.4,
            est_learning_time=120,
            bloom_level="understand",
            description="M-P 神经元、激活函数、前馈网络",
            key_concepts=["激活函数", "前馈网络", "万能逼近定理"],
            typical_errors=["激活函数选择错误", "网络层数与表达能力关系"],
        ),
        KnowledgeNode(
            id="dl_backprop",
            name="反向传播算法",
            category="深度学习",
            difficulty=0.6,
            est_learning_time=180,
            bloom_level="apply",
            description="链式法则、梯度计算、梯度消失/爆炸",
            key_concepts=["链式法则", "梯度计算", "梯度消失", "梯度爆炸"],
            typical_errors=["链式法则求导错误", "梯度爆炸未处理"],
        ),
        KnowledgeNode(
            id="dl_cnn",
            name="卷积神经网络",
            category="深度学习",
            difficulty=0.6,
            est_learning_time=180,
            bloom_level="apply",
            description="卷积、池化、CNN 经典架构",
            key_concepts=["卷积", "池化", "感受野", "经典架构"],
            typical_errors=["卷积核大小选择", "步长与填充设置"],
        ),
        KnowledgeNode(
            id="dl_rnn",
            name="循环神经网络",
            category="深度学习",
            difficulty=0.65,
            est_learning_time=180,
            bloom_level="apply",
            description="RNN、LSTM、GRU、序列建模",
            key_concepts=["RNN", "LSTM", "GRU", "序列到序列"],
            typical_errors=["长序列梯度消失", "忘记 LSTM 的门控机制"],
        ),
        KnowledgeNode(
            id="dl_transformer",
            name="Transformer 与注意力",
            category="深度学习",
            difficulty=0.7,
            est_learning_time=240,
            bloom_level="analyze",
            description="Self-Attention、Multi-Head、位置编码",
            key_concepts=["自注意力", "多头注意力", "位置编码", "残差连接"],
            typical_errors=["Q/K/V 维度理解错误", "位置编码实现细节"],
        ),

        # ===== 应用 =====
        KnowledgeNode(
            id="nlp_basic",
            name="自然语言处理基础",
            category="应用",
            difficulty=0.5,
            est_learning_time=150,
            bloom_level="understand",
            description="分词、词向量、TF-IDF",
            key_concepts=["分词", "词袋模型", "词嵌入", "TF-IDF"],
            typical_errors=["词袋模型忽略词序", "中文分词歧义"],
        ),
        KnowledgeNode(
            id="nlp_pretrain",
            name="预训练语言模型",
            category="应用",
            difficulty=0.7,
            est_learning_time=200,
            bloom_level="analyze",
            description="BERT、GPT、Prompt Engineering",
            key_concepts=["BERT", "GPT", "微调", "Prompt 工程"],
            typical_errors=["BERT vs GPT 区别", "微调与提示学习的差异"],
        ),
        KnowledgeNode(
            id="cv_basic",
            name="计算机视觉基础",
            category="应用",
            difficulty=0.55,
            est_learning_time=150,
            bloom_level="understand",
            description="图像基础、特征提取、传统 CV",
            key_concepts=["图像表示", "边缘检测", "特征描述子"],
            typical_errors=["RGB 通道理解错误", "卷积 vs 互相关"],
        ),

        # ===== 前沿 =====
        KnowledgeNode(
            id="rl_basic",
            name="强化学习基础",
            category="前沿",
            difficulty=0.7,
            est_learning_time=200,
            bloom_level="analyze",
            description="MDP、Q-learning、Policy Gradient",
            key_concepts=["MDP", "Q-learning", "策略梯度", "探索与利用"],
            typical_errors=["奖励塑形", "探索策略选择"],
        ),
        KnowledgeNode(
            id="llm_agent",
            name="大模型智能体（Agent）",
            category="前沿",
            difficulty=0.75,
            est_learning_time=240,
            bloom_level="create",
            description="ReAct、Tool Use、Planning、Memory",
            key_concepts=["ReAct", "工具调用", "规划", "记忆机制"],
            typical_errors=["ReAct vs Chain-of-Thought 区别", "Agent 循环死锁"],
        ),

        # ===== 伦理 =====
        KnowledgeNode(
            id="ethics_ai",
            name="AI 伦理与社会影响",
            category="伦理",
            difficulty=0.2,
            est_learning_time=60,
            bloom_level="evaluate",
            description="公平性、可解释性、隐私、就业影响",
            key_concepts=["算法偏见", "可解释性", "数据隐私", "AI 治理"],
            typical_errors=["技术中立论", "忽视社会影响"],
        ),
    ]
    for n in nodes:
        kg.add_node(n)

    # 边定义：先决关系
    edges = [
        # 数学
        ("math_linear_algebra", "ml_linear_regression", "prerequisite", 1.0),
        ("math_calculus", "ml_linear_regression", "prerequisite", 1.0),
        ("math_probability", "ml_logistic_regression", "prerequisite", 1.0),
        ("math_optimization", "dl_backprop", "prerequisite", 1.0),
        ("math_linear_algebra", "dl_backprop", "prerequisite", 0.8),
        ("math_calculus", "dl_backprop", "prerequisite", 0.9),

        # 机器学习
        ("ml_concept", "ml_linear_regression", "prerequisite", 0.7),
        ("ml_linear_regression", "ml_logistic_regression", "prerequisite", 1.0),
        ("ml_linear_regression", "ml_svm", "prerequisite", 0.5),
        ("ml_logistic_regression", "ml_svm", "prerequisite", 0.5),
        ("ml_decision_tree", "ml_ensemble", "prerequisite", 1.0),
        ("ml_logistic_regression", "ml_ensemble", "prerequisite", 0.5),
        ("ml_svm", "ml_ensemble", "related", 0.5),

        # 深度学习
        ("ml_concept", "dl_perceptron", "prerequisite", 0.6),
        ("dl_perceptron", "dl_backprop", "prerequisite", 1.0),
        ("dl_backprop", "dl_cnn", "prerequisite", 1.0),
        ("dl_backprop", "dl_rnn", "prerequisite", 1.0),
        ("dl_rnn", "dl_transformer", "prerequisite", 0.8),
        ("dl_cnn", "cv_basic", "prerequisite", 1.0),
        ("dl_transformer", "nlp_pretrain", "prerequisite", 1.0),
        ("nlp_basic", "nlp_pretrain", "prerequisite", 0.8),
        ("dl_perceptron", "nlp_basic", "prerequisite", 0.5),
        ("ml_concept", "rl_basic", "prerequisite", 0.7),
        ("math_probability", "rl_basic", "prerequisite", 1.0),

        # Agent
        ("nlp_pretrain", "llm_agent", "prerequisite", 1.0),
        ("dl_transformer", "llm_agent", "prerequisite", 0.8),
        ("rl_basic", "llm_agent", "related", 0.6),

        # 易混淆
        ("ml_linear_regression", "ml_logistic_regression", "confused_with", 1.0),
        ("dl_rnn", "dl_transformer", "confused_with", 0.5),
        ("nlp_pretrain", "llm_agent", "related", 0.7),

        # 伦理贯穿
        ("ethics_ai", "ml_logistic_regression", "related", 0.3),
        ("ethics_ai", "llm_agent", "related", 0.5),
    ]
    for src, tgt, rel, w in edges:
        kg.add_edge(KnowledgeEdge(source=src, target=tgt, relation=rel, weight=w))

    return kg


if __name__ == "__main__":
    kg = build_default_kg()
    print(f"Nodes: {len(kg.nodes)}")
    print(f"Edges: {len(kg.edges)}")
    print("Topological order (first 10):")
    for i, nid in enumerate(kg.topological_order()[:10]):
        node = kg.get_node(nid)
        print(f"  {i+1}. {node.name} (difficulty={node.difficulty})")
    print("\nLearning levels:")
    for lvl, skills in kg.learning_levels().items():
        print(f"  Level {lvl}: {[kg.get_node(s).name for s in skills]}")
    print("\nSave to:", KG_DIR / "ai_intro_kg.json")
    kg.save()
    print("Saved.")
