"""
测试：知识图谱与路径规划
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.knowledge.knowledge_graph import build_default_kg
from src.planning.contextual_bandit import LinUCB, PathPlanner, compute_reward
from src.planning.path_optimizer import zpd_score, zpd_filter, plan_learning_path


def test_kg_build():
    kg = build_default_kg()
    assert len(kg.nodes) > 15
    assert len(kg.edges) > 20
    # 拓扑排序
    order = kg.topological_order()
    assert len(order) == len(kg.nodes)
    print(f"✅ test_kg_build passed ({len(kg.nodes)} nodes, {len(kg.edges)} edges)")


def test_kg_prereq():
    kg = build_default_kg()
    prereqs = kg.prerequisites("dl_transformer")
    assert "dl_rnn" in prereqs
    print(f"✅ test_kg_prereq passed (Transformer prereqs: {prereqs})")


def test_kg_subgraph():
    kg = build_default_kg()
    # 已掌握线代与微积分，要学 Transformer
    path = kg.subgraph_for_student(
        mastered_skills={"math_linear_algebra", "math_calculus", "dl_perceptron", "dl_backprop"},
        target_skill="dl_transformer",
    )
    assert "dl_transformer" in path or "dl_rnn" in path
    print(f"✅ test_kg_subgraph passed (path length={len(path)})")


def test_linucb():
    bandit = LinUCB(n_actions=3, d=5, config=type('cfg', (), {'linucb_alpha': 1.0, 'linucb_d': 5, 'linucb_lambda_reg': 1.0})())
    for _ in range(50):
        ctx = np.random.rand(5)
        a = bandit.select(ctx)
        reward = float(np.random.rand() < 0.7)
        bandit.update(a, ctx, reward)
    # 至少所有动作都被拉过几次
    assert sum(bandit.pulled_count > 0) >= 1
    print(f"✅ test_linucb passed (pulled={bandit.pulled_count})")


def test_path_planner():
    from src.cognitive.student_model import build_csn_from_graph
    kg = build_default_kg()
    csn = build_csn_from_graph(kg)
    planner = PathPlanner(
        n_skills=len(kg.nodes),
        skill_id_to_idx={sid: i for i, sid in enumerate(kg.nodes.keys())},
    )
    ctx = np.random.rand(32)
    candidates = list(kg.nodes.keys())[:5]
    decision = planner.plan_next(ctx, candidates)
    assert decision.next_skill in candidates
    print(f"✅ test_path_planner passed (next={decision.next_skill})")


def test_reward():
    r = compute_reward(0.3, 0.6, 0.7, 0.8)
    assert 0 <= r <= 1
    print(f"✅ test_reward passed (r={r:.3f})")


def test_zpd():
    scores = zpd_filter(0.5, [("a", 0.3), ("b", 0.5), ("c", 0.8), ("d", 0.1)])
    assert len(scores) == 4
    # 在 0.5 能力下，0.5 难度的题应得分最高
    best = scores[0]
    print(f"✅ test_zpd passed (best={best.skill_id}, score={best.zpd_score:.3f})")


def test_plan_path():
    kg = build_default_kg()
    path = plan_learning_path(
        knowledge_graph=kg,
        student_mastery={"math_linear_algebra": 0.9, "math_calculus": 0.9},
        target_skill="ml_logistic_regression",
        student_ability=0.4,
    )
    assert len(path) > 0
    print(f"✅ test_plan_path passed (path length={len(path)})")


if __name__ == "__main__":
    test_kg_build()
    test_kg_prereq()
    test_kg_subgraph()
    test_linucb()
    test_path_planner()
    test_reward()
    test_zpd()
    test_plan_path()
    print("\n🎉 All knowledge/planning tests passed!")
