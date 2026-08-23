"""
测试：智能体协同
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import Orchestrator, SessionMode
from src.dialogue.socratic_engine import SocraticEngine, DialogueStage
from src.emotion.sentiment import EmotionDetector, EmotionState, recommend_pedagogy
from src.memory.long_term_memory import LongTermMemory, Episode, generate_reflection


def test_orchestrator_basic():
    orch = Orchestrator()
    state = orch.start_session("S001", target_skill="llm_agent")
    assert state.plan is not None
    assert state.plan.next_skill is not None
    print(f"✅ test_orchestrator_basic passed (next={state.plan.next_skill})")


def test_orchestrator_chat_loop():
    orch = Orchestrator()
    state = orch.start_session("S001", target_skill="ml_logistic_regression")
    inputs = [
        ("我不懂啊", False, False),
        ("还是不会", False, False),
        ("我懂了！", True, True),
    ]
    for i, (text, is_answer, is_correct) in enumerate(inputs):
        result = orch.handle(
            "S001", text,
            is_answer=is_answer,
            item_id=f"q_{i}",
            is_correct=is_correct,
        )
        assert result.response
        assert result.emotion is not None
    print(f"✅ test_orchestrator_chat_loop passed (turns={state.turn_count})")


def test_socratic_engine():
    eng = SocraticEngine()
    for mastery in [0.2, 0.4, 0.6, 0.8]:
        r = eng.respond(
            student_input="怎么算？",
            current_skill="linear_regression",
            mastery=mastery,
            recent_results=[False, False],
        )
        assert r.text
        assert r.scaffolding_level is not None
    print("✅ test_socratic_engine passed")


def test_emotion_detector():
    detector = EmotionDetector()
    state = EmotionState()
    samples = ["我不懂啊", "这什么意思？", "我会了！", "太难了不想做了"]
    for s in samples:
        state = detector.detect(s, state)
    assert state.frustration > 0
    print(f"✅ test_emotion_detector passed (frustration={state.frustration:.2f})")


def test_memory():
    mem = LongTermMemory()
    ep = Episode(student_id="S001", skill_id="x", user_input="hi", is_correct=True)
    mem.add_episode(ep)
    eps = mem.get_episodes("S001")
    assert len(eps) >= 1
    note = generate_reflection(eps, "S001")
    assert note.content
    print(f"✅ test_memory passed (episodes={len(eps)})")


if __name__ == "__main__":
    test_orchestrator_basic()
    test_orchestrator_chat_loop()
    test_socratic_engine()
    test_emotion_detector()
    test_memory()
    print("\n🎉 All agent tests passed!")
