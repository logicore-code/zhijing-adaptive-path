"""
Orchestrator Agent (主控智能体)
==================================

智径 AdaptivePath 的"大脑"，负责五大智能体协同：

  Orchestrator
    ├── Diagnostic Agent
    ├── Planning Agent
    ├── Teaching Agent (Socratic)
    ├── Reflective Agent
    └── Emotional Agent

交互循环：
  1) 学生输入
  2) Emotional Agent 分析情感 → 是否干预？
  3) Diagnostic Agent 更新 CSN（如果是答题）
  4) Planning Agent 决定下一步（如果是路径点）
  5) Teaching Agent 用苏格拉底式回应
  6) Reflective Agent 记录与反思
  7) 返回给学生

主控器用 LangGraph 风格的状态机实现。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
import uuid

from src.cognitive.student_model import CognitiveStateNetwork, build_csn_from_graph
from src.knowledge.knowledge_graph import KnowledgeGraph, build_default_kg
from src.memory.long_term_memory import LongTermMemory, Episode
from src.agents.diagnostic_agent import DiagnosticAgent, DiagnosticReport
from src.agents.planning_agent import PlanningAgent, PlanResult
from src.agents.teaching_agent import TeachingAgent, TeachingDecision
from src.agents.reflective_agent import ReflectiveAgent, ReflectionResult
from src.agents.emotional_agent import EmotionalAgent, EmotionalAnalysis
from src.dialogue.socratic_engine import DialogueStage


class SessionMode(str, Enum):
    """学习模式"""
    FREE_DIALOGUE = "free_dialogue"        # 自由对话
    GUIDED_PATH = "guided_path"            # 按规划路径学习
    ASSESSMENT = "assessment"              # 测评模式
    REFLECTION = "reflection"              # 反思总结


@dataclass
class SessionState:
    """会话状态"""
    student_id: str
    mode: SessionMode = SessionMode.GUIDED_PATH
    current_skill: Optional[str] = None
    target_skill: Optional[str] = None
    plan: Optional[PlanResult] = None
    turn_count: int = 0
    last_response: str = ""
    last_decision: Optional[TeachingDecision] = None
    last_emotion: Optional[EmotionalAnalysis] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)


@dataclass
class OrchestratorResult:
    """主控器输出"""
    response: str
    state: SessionState
    next_action: str
    plan: Optional[PlanResult] = None
    emotion: Optional[EmotionalAnalysis] = None
    diagnostic: Optional[DiagnosticReport] = None
    reflection: Optional[ReflectionResult] = None
    metadata: Dict = field(default_factory=dict)


class Orchestrator:
    """
    主控智能体：协调 5 个子智能体完成完整学习闭环。
    """
    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        llm=None,
    ):
        self.kg = knowledge_graph or build_default_kg()
        self.csn = build_csn_from_graph(self.kg)
        self.memory = LongTermMemory()

        # 五大子智能体
        self.diagnostic = DiagnosticAgent(self.csn, self.kg)
        self.planning = PlanningAgent(self.csn, self.kg)
        self.teaching = TeachingAgent(self.csn, self.kg, llm=llm)
        self.reflective = ReflectiveAgent(self.csn, self.kg, self.memory)
        self.emotional = EmotionalAgent()

        # 会话状态
        self.sessions: Dict[str, SessionState] = {}

    # ------------------------------------------------------------------ #
    # 会话管理
    # ------------------------------------------------------------------ #
    def start_session(
        self,
        student_id: str,
        mode: SessionMode = SessionMode.GUIDED_PATH,
        target_skill: Optional[str] = None,
    ) -> SessionState:
        """开启新会话"""
        # 初始规划
        plan = self.planning.plan(student_id, target_skill=target_skill) if target_skill else None
        state = SessionState(
            student_id=student_id,
            mode=mode,
            current_skill=plan.next_skill if plan else None,
            target_skill=target_skill,
            plan=plan,
        )
        self.sessions[student_id] = state
        return state

    def get_session(self, student_id: str) -> Optional[SessionState]:
        return self.sessions.get(student_id)

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def handle(
        self,
        student_id: str,
        user_input: str,
        is_answer: bool = False,
        item_id: Optional[str] = None,
        is_correct: Optional[bool] = None,
    ) -> OrchestratorResult:
        """
        主处理函数：接收学生输入，返回完整结果。
        """
        state = self.sessions.get(student_id)
        if not state:
            state = self.start_session(student_id)

        state.turn_count += 1

        # 1) 情感分析（总是执行）
        emotion = self.emotional.analyze(
            student_id=student_id,
            text=user_input,
            is_correct=is_correct,
        )
        state.last_emotion = emotion

        # 2) 如果是答题，更新 CSN
        if is_answer and is_correct is not None and state.current_skill:
            mastery, conf = self.diagnostic.update(
                student_id=student_id,
                skill_id=state.current_skill,
                item_id=item_id or "unknown",
                is_correct=is_correct,
            )
            # 给 Bandit 反馈
            self.planning.feedback(
                student_id=student_id,
                skill_id=state.current_skill,
                mastery_before=max(0, mastery - 0.1),
                mastery_after=mastery,
                engagement=emotion.state.engagement,
                emotion_score=emotion.state.engagement,
            )
            # 记录到 memory
            self.reflective.add_episode(Episode(
                student_id=student_id,
                skill_id=state.current_skill,
                user_input=user_input,
                agent_response="",  # 后续填充
                is_correct=is_correct,
                emotion_snapshot=emotion.state.to_dict(),
                scaffolding_level=state.last_decision.response.scaffolding_level.value if state.last_decision else 0,
            ))

        # 3) 路径规划
        if state.mode in [SessionMode.GUIDED_PATH, SessionMode.ASSESSMENT]:
            state.plan = self.planning.plan(
                student_id=student_id,
                target_skill=state.target_skill,
            )
            state.current_skill = state.plan.next_skill

        # 4) 教学回应
        if state.current_skill:
            decision = self.teaching.respond(
                student_input=user_input,
                student_id=student_id,
                skill_id=state.current_skill,
                emotion=emotion.state,
                is_correct=is_correct,
            )
            state.last_decision = decision
            state.last_response = decision.response.text

        # 5) 情感干预：追加鼓励或建议
        response_text = state.last_response
        if emotion.should_intervene and emotion.intervention_type:
            intervention_msg = self._format_intervention(emotion)
            response_text = intervention_msg + "\n\n" + response_text

        # 6) 定期反思
        reflection = None
        if state.turn_count > 0 and state.turn_count % 10 == 0:
            reflection = self.reflective.reflect(student_id)

        # 7) 决定下一步
        next_action = state.last_decision.next_action if state.last_decision else "继续对话"

        # 8) 更新 episode 中的 agent_response
        if state.current_skill and self.memory.episodic.get(student_id):
            eps = self.memory.episodic[student_id]
            if eps and not eps[-1].agent_response:
                eps[-1].agent_response = response_text

        return OrchestratorResult(
            response=response_text,
            state=state,
            next_action=next_action,
            plan=state.plan,
            emotion=emotion,
            diagnostic=self.diagnostic.report(student_id),
            reflection=reflection,
        )

    def _format_intervention(self, emotion: EmotionalAnalysis) -> str:
        """格式化情感干预消息"""
        if emotion.intervention_type and "鼓励" in emotion.intervention_type:
            return "💪 不要灰心！我们换个方式再试试。先深呼吸一下～"
        if emotion.intervention_type and "休息" in emotion.intervention_type:
            return "☕ 你已经学了很久，要不要先休息 5 分钟？我们之后再继续。"
        if emotion.intervention_type and "类比" in emotion.intervention_type:
            return "🎨 感觉有点卡？我用一个更形象的例子帮你理解。"
        if emotion.intervention_type and "肯定" in emotion.intervention_type:
            return "🌟 你已经坚持到这里了，这本身就是很大的进步！"
        if emotion.intervention_type and "趣味" in emotion.intervention_type:
            return "🎮 来点有趣的小挑战？换个角度理解一下。"
        return "🌈 我注意到你可能需要一些支持。"

    # ------------------------------------------------------------------ #
    # 报告接口
    # ------------------------------------------------------------------ #
    def get_student_report(self, student_id: str) -> Dict:
        """获取学生综合报告"""
        diag = self.diagnostic.report(student_id)
        plan = self.planning.plan(student_id)
        emotion_state = self.emotional.get_state(student_id)
        return {
            "diagnostic": diag,
            "current_plan": plan,
            "emotion": emotion_state.to_dict(),
            "memory_summary": {
                "episodes": len(self.memory.episodic.get(student_id, [])),
                "reflections": len(self.memory.reflective.get(student_id, [])),
            },
        }

    def end_session(self, student_id: str) -> ReflectionResult:
        """结束会话，生成最终反思"""
        result = self.reflective.reflect(student_id)
        self.memory.consolidate(student_id)
        if student_id in self.sessions:
            del self.sessions[student_id]
        return result


if __name__ == "__main__":
    orch = Orchestrator()
    state = orch.start_session("S001", target_skill="llm_agent")
    print(f"Started session for S001, plan: {[orch.kg.get_node(s).name for s in state.plan.full_path]}")
    for i in range(3):
        result = orch.handle(
            "S001",
            user_input="我不太理解啊",
            is_answer=True,
            item_id=f"item_{i}",
            is_correct=False,
        )
        print(f"\nTurn {i+1}:")
        print(f"  Response: {result.response[:80]}")
        print(f"  Next skill: {result.state.current_skill}")
        print(f"  Emotion: {result.emotion.dominant if result.emotion else 'N/A'}")
