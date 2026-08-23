"""
测试：认知诊断模块
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.cognitive.bkt import BayesianKnowledgeTracing, predict_next_correct
from src.cognitive.irt import IRT2PL, ability_level
from src.cognitive.dkt import NumPyDKT
from src.cognitive.student_model import CognitiveStateNetwork
from src.knowledge.knowledge_graph import build_default_kg


def test_bkt_basic():
    """测试 BKT 基本更新"""
    bkt = BayesianKnowledgeTracing()
    state = bkt.get_or_create("test_skill")
    # 初始掌握度低
    assert state.p_learned < 0.2
    # 答对几次后应上升
    for _ in range(5):
        bkt.update(state, True)
    assert state.p_learned > 0.5
    print("✅ test_bkt_basic passed")


def test_bkt_predict():
    bkt = BayesianKnowledgeTracing()
    state = bkt.get_or_create("s")
    for _ in range(3):
        bkt.update(state, True)
    p = predict_next_correct(bkt, "s")
    assert 0 < p < 1
    print(f"✅ test_bkt_predict passed (p={p:.3f})")


def test_irt():
    irt = IRT2PL()
    s = irt.get_or_create_student("S1")
    irt.add_item("Q1", a=1.0, b=0.0, skill_id="x")
    irt.add_item("Q2", a=1.5, b=1.0, skill_id="x")
    irt.add_item("Q3", a=0.8, b=-0.5, skill_id="x")
    # 答对简单题、答错难题 -> theta 接近 0
    irt.update_student(s, [("Q1", True), ("Q2", False), ("Q3", True)])
    assert -1.0 < s.theta < 1.0
    print(f"✅ test_irt passed (theta={s.theta:.3f}, level={ability_level(s.theta)})")


def test_dkt():
    dkt = NumPyDKT(num_skills=4, hidden_size=16)
    seq = [(0, True), (1, True), (0, True), (2, False)]
    mastery = dkt.predict_mastery(seq)
    assert mastery.shape == (4,)
    assert all(0 <= m <= 1 for m in mastery)
    print(f"✅ test_dkt passed (mastery={mastery})")


def test_csn():
    kg = build_default_kg()
    csn = CognitiveStateNetwork(
        num_skills=len(kg.nodes),
        skill_id_to_idx={sid: i for i, sid in enumerate(kg.nodes.keys())},
    )
    skill = list(kg.nodes.keys())[0]
    m, c = csn.update("S001", skill, "item_1", True)
    assert 0 < m < 1
    assert 0 < c < 1
    # 再更新几次
    for i in range(5):
        csn.update("S001", skill, f"item_{i+2}", i % 2 == 0)
    # 特征向量
    fv = csn.get_feature_vector("S001")
    assert fv.shape == (32,)
    # 报告
    report = csn.explain("S001", skill)
    assert "explanation" in report
    print(f"✅ test_csn passed (mastery={m:.3f}, conf={c:.3f})")


if __name__ == "__main__":
    test_bkt_basic()
    test_bkt_predict()
    test_irt()
    test_dkt()
    test_csn()
    print("\n🎉 All cognitive tests passed!")
