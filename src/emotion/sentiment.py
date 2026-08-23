"""
情感识别与学习状态建模
========================

实时从学生输入推断：
- 困惑度 (Confusion)
- 挫败感 (Frustration)
- 兴趣度 (Engagement)
- 疲劳度 (Fatigue)
- 自信度 (Confidence)

技术路线：
1. 关键词词典法（基础）
2. 标点/重复模式启发式
3. 交互节奏分析（响应时间、错误率）

不使用 LLM 以保证实时性与可解释性。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import re
import time

from src.config import EmotionConfig, get_config


@dataclass
class EmotionState:
    """学生情感状态"""
    confusion: float = 0.0
    frustration: float = 0.0
    engagement: float = 0.5
    fatigue: float = 0.0
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "confusion": self.confusion,
            "frustration": self.frustration,
            "engagement": self.engagement,
            "fatigue": self.fatigue,
            "confidence": self.confidence,
        }

    def dominant(self) -> str:
        """返回主导情感"""
        s = self.to_dict()
        return max(s, key=s.get)


# 关键词词典（中文 + 英文）
CONFUSION_KEYWORDS = {
    "不懂", "不明白", "不懂啊", "不理解", "搞不懂", "啥意思", "什么意思", "怎么理解",
    "困惑", "糊涂", "蒙", "懵", "怎么会", "为什么这样", "?"
}
FRUSTRATION_KEYWORDS = {
    "烦", "烦躁", "不想", "放弃", "难死了", "太难了", "搞不定", "没用", "差劲",
    "完蛋", "气死", "无语", "崩溃", "难啊", "难啊难", "做不出来", "我不会"
}
ENGAGEMENT_KEYWORDS = {
    "有意思", "有趣", "想学", "继续", "再来", "为什么", "怎么样", "怎么", "如何",
    "可以", "试试", "想试试", "为什么", "明白了", "懂了", "原来如此", "原来", "懂了!"
}
FATIGUE_KEYWORDS = {
    "累", "困", "无聊", "不想学了", "明天再说", "休息", "脑子不转了", "头疼"
}
CONFIDENCE_KEYWORDS = {
    "我会", "我会做", "我懂了", "明白了", "懂了", "ok", "OK", "可以", "没问题", "知道", "了解"
}


class EmotionDetector:
    """
    情感检测器：基于规则 + 启发式。
    """
    def __init__(self, config: Optional[EmotionConfig] = None):
        self.cfg = config or EmotionConfig()
        self.confusion_kw = self._compile_patterns(CONFUSION_KEYWORDS)
        self.frustration_kw = self._compile_patterns(FRUSTRATION_KEYWORDS)
        self.engagement_kw = self._compile_patterns(ENGAGEMENT_KEYWORDS)
        self.fatigue_kw = self._compile_patterns(FATIGUE_KEYWORDS)
        self.confidence_kw = self._compile_patterns(CONFIDENCE_KEYWORDS)

    def _compile_patterns(self, kw_set: set) -> List[re.Pattern]:
        return [re.compile(re.escape(kw)) for kw in kw_set]

    def detect(self, text: str, state: Optional[EmotionState] = None) -> EmotionState:
        """从文本检测情感"""
        if state is None:
            state = EmotionState()

        text = text.strip()
        if not text:
            return state

        # 标点/模式特征
        has_question = "?" in text or "？" in text
        has_exclaim = "!" in text or "！" in text
        repeat_count = self._count_repeats(text)
        is_short = len(text) < 5
        is_very_short = len(text) < 2

        # 各维度得分
        conf = self._count_matches(self.confusion_kw, text)
        frust = self._count_matches(self.frustration_kw, text)
        eng = self._count_matches(self.engagement_kw, text)
        fat = self._count_matches(self.fatigue_kw, text)
        conf_score = self._count_matches(self.confidence_kw, text)

        # 困惑：关键词 + 问号 + 重复
        confusion = min(1.0, conf * 0.4 + (0.3 if has_question else 0) + repeat_count * 0.1 + (0.1 if is_short else 0))

        # 挫败：关键词 + 重复 + 极短回答
        frustration = min(1.0, frust * 0.5 + repeat_count * 0.15 + (0.2 if is_very_short else 0) + (0.1 if has_exclaim else 0))

        # 兴趣：关键词 + 长度适中 + 问号（探索性）
        engagement = min(1.0, 0.3 + eng * 0.3 + (0.1 if has_question and len(text) > 10 else 0) + (0.1 if not is_short else 0))

        # 疲劳：关键词 + 长对话累积（外部传入）
        fatigue = min(1.0, fat * 0.5)

        # 自信：置信词
        confidence = min(1.0, 0.3 + conf_score * 0.4 - frustration * 0.2)

        # 用 EMA 平滑更新
        alpha = 0.4  # 新观测权重
        state.confusion = alpha * confusion + (1 - alpha) * state.confusion
        state.frustration = alpha * frustration + (1 - alpha) * state.frustration
        state.engagement = alpha * engagement + (1 - alpha) * state.engagement
        state.fatigue = alpha * fatigue + (1 - alpha) * state.fatigue
        state.confidence = alpha * confidence + (1 - alpha) * state.confidence

        return state

    def _count_matches(self, patterns: List[re.Pattern], text: str) -> int:
        return sum(1 for p in patterns if p.search(text))

    def _count_repeats(self, text: str) -> int:
        """统计重复字符（如 '啊啊啊啊'）"""
        if len(text) < 2:
            return 0
        max_run = 1
        cur = 1
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 1
        return max(0, max_run - 2)  # 2 次以内算正常


# ---------------------------------------------------------------------- #
# 疲劳度追踪（基于交互节奏）
# ---------------------------------------------------------------------- #
class FatigueTracker:
    """基于交互节奏追踪疲劳度"""
    def __init__(self, window: int = 10):
        self.timestamps: deque = deque(maxlen=window * 2)
        self.is_correct: deque = deque(maxlen=window)
        self.input_lengths: deque = deque(maxlen=window)

    def record(self, is_correct: bool, input_length: int):
        self.timestamps.append(time.time())
        self.is_correct.append(is_correct)
        self.input_lengths.append(input_length)

    def fatigue(self) -> float:
        if len(self.timestamps) < 3:
            return 0.0
        # 1) 错误率上升
        if len(self.is_correct) >= 3:
            recent = list(self.is_correct)[-3:]
            err_rate = 1 - sum(recent) / len(recent)
        else:
            err_rate = 0.0
        # 2) 输入变短
        if len(self.input_lengths) >= 3:
            recent_lens = list(self.input_lengths)[-3:]
            avg_len = sum(recent_lens) / len(recent_lens)
            length_fatigue = max(0, 1 - avg_len / 20.0)  # 平均长度 < 20 视为疲劳
        else:
            length_fatigue = 0.0
        # 3) 交互过快（应付式）
        if len(self.timestamps) >= 3:
            ts = list(self.timestamps)[-3:]
            intervals = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
            avg_interval = sum(intervals) / len(intervals) if intervals else 60
            speed_fatigue = max(0, 1 - avg_interval / 30.0)  # 30s 内连续交互视为疲劳
        else:
            speed_fatigue = 0.0

        return min(1.0, err_rate * 0.4 + length_fatigue * 0.3 + speed_fatigue * 0.3)


# ---------------------------------------------------------------------- #
# 统一接口
# ---------------------------------------------------------------------- #
def detect_emotion(
    text: str,
    state: Optional[EmotionState] = None,
    is_correct: Optional[bool] = None,
    input_length: Optional[int] = None,
    fatigue_tracker: Optional[FatigueTracker] = None,
) -> EmotionState:
    """
    综合检测学生情感状态。
    """
    detector = EmotionDetector()
    if state is None:
        state = EmotionState()
    state = detector.detect(text, state)

    # 用交互节奏更新疲劳
    if fatigue_tracker is not None and is_correct is not None and input_length is not None:
        fatigue_tracker.record(is_correct, input_length)
        state.fatigue = 0.5 * state.fatigue + 0.5 * fatigue_tracker.fatigue()

    return state


# ---------------------------------------------------------------------- #
# 情感 -> 教学策略
# ---------------------------------------------------------------------- #
def recommend_pedagogy(emotion: EmotionState) -> str:
    """根据情感状态推荐教学策略"""
    if emotion.frustration > 0.7:
        return "高挫败：切换为鼓励 + 简化任务，给一个 easy win"
    if emotion.confusion > 0.6:
        return "高困惑：暂停追问，用类比或图示重新讲解"
    if emotion.fatigue > 0.6:
        return "高疲劳：休息提示，或切换到轻量任务（小结、游戏化）"
    if emotion.engagement > 0.7 and emotion.confidence > 0.5:
        return "高兴趣+自信：可挑战更高难度"
    if emotion.confidence < 0.3:
        return "低自信：明确肯定已有进步，避免让其感到失败"
    return "正常节奏：按计划推进"


if __name__ == "__main__":
    state = EmotionState()
    tracker = FatigueTracker()
    samples = [
        "我不懂啊", "这什么意思？", "啊啊啊啊", "我会了！", "太难了不想做了",
        "原来如此", "好的", "?", "为什么会这样呢",
    ]
    for s in samples:
        state = detect_emotion(s, state, is_correct=False, input_length=len(s), fatigue_tracker=tracker)
        print(f"  [{s[:20]:<20}] conf={state.confusion:.2f} frust={state.frustration:.2f} "
              f"eng={state.engagement:.2f} fatigue={state.fatigue:.2f} dom={state.dominant()}")
