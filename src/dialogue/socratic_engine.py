"""
苏格拉底式对话引擎 (Socratic Dialogue Engine)
================================================

基于 ReAct + CoT 的对话引擎。
永远不直接给答案，而是通过反诘引导学生自己得出结论。

对话阶段：
1. Diagnose    - 了解学生当前理解
2. Probe       - 反诘深入（"为什么？""如果...会怎样？"）
3. Hint        - 关键概念提示
4. Confirm     - 确认学生理解
5. Reflect     - 反思元认知

ReAct 模式：
  Thought -> Action -> Observation -> Thought -> ...
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import json

from src.dialogue.scaffolding import ScaffoldingEngine, ScaffoldingDecision, ScaffoldingLevel
from src.config import DialogueConfig, get_config


class DialogueStage(str, Enum):
    DIAGNOSE = "diagnose"
    PROBE = "probe"
    HINT = "hint"
    CONFIRM = "confirm"
    REFLECT = "reflect"


@dataclass
class DialogueTurn:
    role: str  # "student" or "tutor"
    content: str
    stage: DialogueStage
    scaffolding_level: Optional[ScaffoldingLevel] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class SocraticResponse:
    """苏格拉底式回复"""
    text: str
    stage: DialogueStage
    scaffolding_level: ScaffoldingLevel
    thought: str           # 内部思考（可解释）
    action: str            # 采取的行动
    next_questions: List[str] = field(default_factory=list)
    should_end: bool = False


class SocraticEngine:
    """
    苏格拉底式对话引擎。

    使用规则 + LLM 混合：规则保证"不直接给答案"，LLM 提供自然语言。
    """
    # 苏格拉底反诘模板（按技能分类）
    SOCRATIC_PROBES = [
        "你能用自己的话解释一下这个概念吗？",
        "为什么你认为是这样？",
        "如果换一种情况，结果会不同吗？",
        "这个结论是基于哪些前提？",
        "你能举一个生活中的例子吗？",
        "这个和我们刚才学的有什么联系？",
        "还有没有其他可能的解法？",
    ]

    # 针对性反诘：按知识点（让对话更专业）
    SKILL_SPECIFIC_PROBES = {
        "math_linear_algebra": [
            "矩阵乘法的维度变化规律是什么？为什么？",
            "特征值在几何上代表什么？",
            "如果不要求正交，向量基还成立吗？",
        ],
        "ml_linear_regression": [
            "为什么用 MSE 而不是 MAE？",
            "如果不加偏置项会怎样？",
            "怎么判断线性回归的假设是否成立？",
        ],
        "ml_logistic_regression": [
            "为什么用交叉熵而不是 MSE？",
            "sigmoid 饱和时梯度会怎样？",
            "逻辑回归的输出是概率吗？",
        ],
        "ml_decision_tree": [
            "信息增益和基尼系数的本质区别？",
            "为什么需要剪枝？",
            "决策树对特征缩放敏感吗？",
        ],
        "dl_backprop": [
            "梯度消失的数学本质是什么？",
            "为什么 ReLU 比 sigmoid 更适合深层网络？",
            "链式法则在 BP 中如何应用？",
        ],
        "dl_transformer": [
            "自注意力的计算复杂度是多少？",
            "为什么需要位置编码？",
            "Multi-Head 注意力相比单头的优势？",
        ],
        "llm_agent": [
            "ReAct 与 Chain-of-Thought 的区别？",
            "Agent 的记忆机制为什么重要？",
            "工具调用失败时 Agent 该怎么办？",
        ],
        "nlp_pretrain": [
            "BERT 和 GPT 的训练目标区别？",
            "为什么预训练+微调比直接训练好？",
            "Prompt Engineering 的核心思想？",
        ],
    }

    # 鼓励性开场（当学生回答正确时）
    POSITIVE_FEEDBACK = [
        "回答得很棒！你能进一步解释一下为什么吗？",
        "完全正确！能否推广到更一般的情况？",
        "思路清晰！那么这个方法有什么局限性？",
        "很好！你能用类比让其他人也理解吗？",
    ]

    # 关键概念提取的启发式关键词
    CONCEPT_HINT_TEMPLATES = {
        "linear_regression": "线性回归的目标是找到一条直线，使预测值与真实值的平方误差最小。回忆最小二乘法。",
        "logistic_regression": "逻辑回归用 sigmoid 函数把线性输出映射到 (0,1)，代表概率。",
        "decision_tree": "决策树通过信息增益或基尼系数选择最优特征进行分裂。",
        "backprop": "反向传播是链式法则在神经网络中的系统应用，逐层计算梯度。",
        "transformer": "Transformer 的核心是自注意力机制，让每个位置都能关注到所有其他位置的信息。",
    }

    def __init__(self, config: Optional[DialogueConfig] = None):
        self.cfg = config or DialogueConfig()
        self.scaffolding = ScaffoldingEngine()
        self.history: List[DialogueTurn] = []
        self.probe_count = 0
        self.llm = None  # 可选注入 LLM

    def set_llm(self, llm):
        """注入 LLM（增强自然语言生成）"""
        self.llm = llm

    def respond(
        self,
        student_input: str,
        current_skill: str,
        mastery: float,
        recent_results: List[bool],
        stage: DialogueStage = DialogueStage.PROBE,
        fatigue: float = 0.0,
        student_request_hint: bool = False,
    ) -> SocraticResponse:
        """
        生成苏格拉底式回复。

        核心：永远不直接给答案。
        """
        # 1) 决定脚手架级别
        scaffold = self.scaffolding.decide(
            mastery=mastery,
            recent_results=recent_results,
            student_request_hint=student_request_hint,
            fatigue=fatigue,
        )

        # 2) 决定对话阶段
        next_stage = self._next_stage(stage, recent_results, mastery)

        # 3) 生成回复
        text, thought, action = self._generate_response(
            student_input=student_input,
            current_skill=current_skill,
            mastery=mastery,
            stage=next_stage,
            scaffold=scaffold,
            recent_results=recent_results,
        )

        # 4) 生成后续追问
        next_questions = self._generate_followups(current_skill, next_stage)

        # 5) 记录历史
        self.history.append(DialogueTurn(
            role="tutor",
            content=text,
            stage=next_stage,
            scaffolding_level=scaffold.level,
        ))

        # 6) 反思阶段 -> 结束
        should_end = next_stage == DialogueStage.REFLECT and mastery > 0.7

        return SocraticResponse(
            text=text,
            stage=next_stage,
            scaffolding_level=scaffold.level,
            thought=thought,
            action=action,
            next_questions=next_questions,
            should_end=should_end,
        )

    def _next_stage(
        self,
        current: DialogueStage,
        recent: List[bool],
        mastery: float,
    ) -> DialogueStage:
        """状态机：决定下一阶段"""
        if current == DialogueStage.DIAGNOSE:
            return DialogueStage.PROBE
        if current == DialogueStage.PROBE:
            if len(recent) >= 2 and recent[-1]:
                return DialogueStage.CONFIRM
            if self.probe_count > self.cfg.max_probing_depth:
                return DialogueStage.HINT
            return DialogueStage.PROBE
        if current == DialogueStage.HINT:
            if len(recent) >= 1 and recent[-1]:
                return DialogueStage.CONFIRM
            return DialogueStage.HINT
        if current == DialogueStage.CONFIRM:
            if mastery > 0.7:
                return DialogueStage.REFLECT
            return DialogueStage.PROBE
        if current == DialogueStage.REFLECT:
            return DialogueStage.REFLECT
        return DialogueStage.PROBE

    def _generate_response(
        self,
        student_input: str,
        current_skill: str,
        mastery: float,
        stage: DialogueStage,
        scaffold: ScaffoldingDecision,
        recent_results: List[bool] = None,
    ) -> Tuple[str, str, str]:
        """
        生成回复文本。
        优先使用 LLM；无 LLM 时使用规则模板。
        """
        thought = f"学生当前掌握度={mastery:.2f}，脚手架级别={scaffold.level.name}，对话阶段={stage.value}。"
        recent_results = recent_results or []

        if self.llm is not None:
            try:
                text = self._llm_generate(
                    student_input, current_skill, mastery, stage, scaffold
                )
                action = f"调用 LLM 生成 {stage.value} 阶段回复"
                return text, thought, action
            except Exception as e:
                thought += f" [LLM 调用失败: {e}]"

        # 规则兜底
        if stage == DialogueStage.DIAGNOSE:
            text = f"开始诊断：你对【{current_skill}】有什么了解？试着说说你的理解。"
            action = "规则：诊断阶段开场"
        elif stage == DialogueStage.PROBE:
            self.probe_count += 1
            # 优先用知识点专属反诘
            skill_probes = self.SKILL_SPECIFIC_PROBES.get(current_skill, [])
            if skill_probes and self.probe_count <= len(skill_probes):
                probe = skill_probes[self.probe_count - 1]
            else:
                probe = self.SOCRATIC_PROBES[self.probe_count % len(self.SOCRATIC_PROBES)]
            text = f"🤔 {probe}"
            action = f"规则：反诘阶段（{current_skill} 专属）" if skill_probes else "规则：反诘阶段（通用）"
        elif stage == DialogueStage.HINT:
            # 尝试给出更具体的概念提示
            concept_hint = self.CONCEPT_HINT_TEMPLATES.get(current_skill, "")
            if concept_hint:
                text = f"💡 {concept_hint}\n\n（提示强度：{scaffold.level.name}）"
            else:
                text = scaffold.prompt
            action = f"规则：提示阶段，{scaffold.reasoning}"
        elif stage == DialogueStage.CONFIRM:
            if recent_results and recent_results[-1]:
                idx = self.probe_count % len(self.POSITIVE_FEEDBACK)
                text = f"🌟 {self.POSITIVE_FEEDBACK[idx]}"
            else:
                text = "看起来还不太确定？没关系，再想想。你可以换个角度试试。"
            action = "规则：确认阶段"
        elif stage == DialogueStage.REFLECT:
            text = "📝 回顾这次学习：你用到了什么方法？哪些地方可以改进？如果再遇到类似问题，你会怎么做？"
            action = "规则：反思阶段，培养元认知"
        else:
            text = "请继续。"
            action = "默认回复"

        return text, thought, action

    def _llm_generate(
        self,
        student_input: str,
        current_skill: str,
        mastery: float,
        stage: DialogueStage,
        scaffold: ScaffoldingDecision,
    ) -> str:
        """调用 LLM 生成苏格拉底式回复"""
        prompt = f"""你是一位苏格拉底式 AI 导师，正在教学生学习【{current_skill}】。

当前状态：
- 学生掌握度：{mastery:.0%}
- 对话阶段：{stage.value}
- 脚手架级别：{scaffold.level.name}（{scaffold.reasoning}）

学生说：「{student_input}」

你的任务：用苏格拉底式反诘引导学生思考。
要求：
1. 永远不要直接给出最终答案
2. 用提问、提示、类比引导学生
3. 语言自然、温暖、有耐心
4. 如果学生明显卡住，参考提示语：「{scaffold.prompt}」

请直接给出你的回复（不超过 200 字）："""
        if hasattr(self.llm, "chat"):
            return self.llm.chat(prompt)
        elif hasattr(self.llm, "invoke"):
            return self.llm.invoke(prompt)
        elif callable(self.llm):
            return self.llm(prompt)
        raise ValueError("Unsupported LLM interface")

    def _generate_followups(self, skill: str, stage: DialogueStage) -> List[str]:
        """生成后续追问"""
        if stage == DialogueStage.PROBE:
            return self.SOCRATIC_PROBES[:3]
        elif stage == DialogueStage.HINT:
            return ["回想一下相关概念的定义", "尝试从已知推导未知"]
        elif stage == DialogueStage.REFLECT:
            return ["你学到了什么？", "下次会怎么做？"]
        return []

    def reset(self):
        self.history = []
        self.probe_count = 0
        self.scaffolding.history = []


# ---------------------------------------------------------------------- #
# 反思提示生成器
# ---------------------------------------------------------------------- #
def generate_reflection_prompt(recent_results: List[bool], recent_skills: List[str]) -> str:
    """生成反思提示"""
    if not recent_results:
        return "还没有学习记录。开始你的第一次学习吧！"
    correct_rate = sum(recent_results) / len(recent_results)
    if correct_rate > 0.8:
        return f"最近正确率 {correct_rate:.0%}，表现不错！思考一下：你是怎么做到的？哪些方法可以迁移到其他知识点？"
    elif correct_rate > 0.5:
        return f"正确率 {correct_rate:.0%}，有进步空间。哪些地方感觉卡住了？是概念理解还是应用层面？"
    else:
        return f"正确率 {correct_rate:.0%}，遇到困难了。先放下问题，回忆一下最基础的定义，可能会有新发现。"


if __name__ == "__main__":
    eng = SocraticEngine()
    for i in range(5):
        r = eng.respond(
            student_input="我不知道怎么算",
            current_skill="linear_regression",
            mastery=0.3 + i * 0.15,
            recent_results=[False, False],
        )
        print(f"[Turn {i+1}] stage={r.stage.value}, level={r.scaffolding_level.name}")
        print(f"  → {r.text[:80]}")
        print(f"  thought: {r.thought[:60]}")
