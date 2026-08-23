"""
脚手架教学 (Scaffolding) - 布鲁姆认知层级驱动
================================================

Vygotsky 的最近发展区 (ZPD) 理论 + Bloom 认知分类。

五级脚手架（按"提示强度"递增）：
  Level 0: 元认知提示（"你想想看，第一步应该是什么？"）
  Level 1: 关键概念提示（"提示：与 XXX 概念相关"）
  Level 2: 类比示例（"想象你在 XXX 场景中..."）
  Level 3: 分步分解（"先做 A，再做 B，最后做 C"）
  Level 4: 完整示范（直接演示一遍完整解法）

升级规则：
  学生连续 2 轮答错或直接说"不会" -> 升级
  学生连续 2 轮答对 -> 降级
  保持 ZPD 区间
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List


class ScaffoldingLevel(IntEnum):
    METACOGNITIVE_HINT = 0      # 元认知
    CONCEPT_HINT = 1             # 概念
    ANALOGY_EXAMPLE = 2          # 类比
    STEP_DECOMPOSITION = 3       # 分步
    FULL_DEMO = 4                # 完整示范


@dataclass
class ScaffoldingDecision:
    level: ScaffoldingLevel
    prompt: str
    reasoning: str
    should_give_answer: bool = False


class ScaffoldingEngine:
    """动态脚手架引擎"""
    def __init__(self, max_level: int = 4, threshold: float = 0.15):
        self.max_level = max_level
        self.threshold = threshold  # 答对率阈值，低于此升级
        self.history: List[tuple] = []  # (level, is_correct)

    def decide(
        self,
        mastery: float,
        recent_results: List[bool],  # 最近几轮的对错
        student_request_hint: bool = False,
        fatigue: float = 0.0,
    ) -> ScaffoldingDecision:
        """
        决定当前应该使用哪一级脚手架。

        策略：
        - 掌握度低 + 连续错 -> 高脚手架
        - 掌握度高 + 连续对 -> 低脚手架
        - 学生主动求提示 -> 中等脚手架
        - 疲劳高 -> 降级（避免认知过载）
        """
        # 默认起点
        if mastery < 0.3:
            level = ScaffoldingLevel.FULL_DEMO
            reasoning = "掌握度较低，给完整示范"
        elif mastery < 0.5:
            level = ScaffoldingLevel.STEP_DECOMPOSITION
            reasoning = "部分掌握，分步引导"
        elif mastery < 0.7:
            level = ScaffoldingLevel.ANALOGY_EXAMPLE
            reasoning = "基本概念已具备，用类比拓展"
        elif mastery < 0.85:
            level = ScaffoldingLevel.CONCEPT_HINT
            reasoning = "较高掌握度，提示关键概念即可"
        else:
            level = ScaffoldingLevel.METACOGNITIVE_HINT
            reasoning = "熟练掌握，鼓励自主思考"

        # 行为调整
        if len(recent_results) >= 2:
            if all(not r for r in recent_results[-2:]) and level < ScaffoldingLevel.FULL_DEMO:
                level = ScaffoldingLevel(level + 1)
                reasoning += "；最近连续答错，升级提示"
            elif all(r for r in recent_results[-2:]) and level > ScaffoldingLevel.METACOGNITIVE_HINT:
                level = ScaffoldingLevel(level - 1)
                reasoning += "；最近连续答对，降级提示以培养独立性"

        if student_request_hint and level < ScaffoldingLevel.ANALOGY_EXAMPLE:
            level = ScaffoldingLevel(min(level + 1, ScaffoldingLevel.ANALOGY_EXAMPLE))
            reasoning += "；学生主动求提示"

        if fatigue > 0.6 and level > ScaffoldingLevel.METACOGNITIVE_HINT:
            level = ScaffoldingLevel(level - 1)
            reasoning += "；检测到疲劳，降低认知负荷"

        prompt = self._generate_prompt(level)
        should_give_answer = mastery < self.threshold and level == ScaffoldingLevel.FULL_DEMO

        return ScaffoldingDecision(
            level=level,
            prompt=prompt,
            reasoning=reasoning,
            should_give_answer=should_give_answer,
        )

    def _generate_prompt(self, level: ScaffoldingLevel) -> str:
        prompts = {
            ScaffoldingLevel.METACOGNITIVE_HINT: "先别急着动手。你想想：要解决这个问题，需要先弄清什么？",
            ScaffoldingLevel.CONCEPT_HINT: "提示：这与【{concept}】这个核心概念相关。回忆一下它的定义。",
            ScaffoldingLevel.ANALOGY_EXAMPLE: "想象一个生活化的场景：{analogy}。把这个场景的解法迁移到原题。",
            ScaffoldingLevel.STEP_DECOMPOSITION: "我们把问题拆开：第 1 步...第 2 步...第 3 步...你先完成第一步。",
            ScaffoldingLevel.FULL_DEMO: "好的，我演示一遍完整思路：\n【第一步】...\n【第二步】...\n你理解后试着自己做。",
        }
        return prompts[level]

    def record(self, level: ScaffoldingLevel, is_correct: bool):
        self.history.append((int(level), int(is_correct)))


if __name__ == "__main__":
    eng = ScaffoldingEngine()
    for mastery in [0.2, 0.4, 0.6, 0.8, 0.95]:
        d = eng.decide(mastery, recent_results=[True, False])
        print(f"mastery={mastery}: level={d.level.name}, prompt='{d.prompt[:30]}...'")
