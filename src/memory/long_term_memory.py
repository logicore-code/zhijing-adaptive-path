"""
长期记忆 (Long-Term Memory)
==============================

分层记忆架构：

  ┌────────────────────────────┐
  │  Episodic (情景记忆)         │  - 每次交互的完整记录
  │  - turn_id, time, content   │
  └─────────────┬──────────────┘
                │
  ┌─────────────▼──────────────┐
  │  Semantic (语义记忆)         │  - 学到的概念、技能掌握
  │  - skill_id -> mastery      │
  └─────────────┬──────────────┘
                │
  ┌─────────────▼──────────────┐
  │  Reflective (反思记忆)       │  - 元认知笔记
  │  - 每次学习后的总结         │
  └────────────────────────────┘

操作：
- add_episode: 添加一次交互
- consolidate: 压缩旧情景到语义
- recall: 基于关键词/相似度检索
- reflect: 生成反思笔记
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
import uuid

from src.config import STUDENT_DIR


@dataclass
class Episode:
    """一次交互的情景记忆"""
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    student_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    skill_id: str = ""
    user_input: str = ""
    agent_response: str = ""
    is_correct: Optional[bool] = None
    emotion_snapshot: Dict = field(default_factory=dict)
    scaffolding_level: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class ReflectiveNote:
    """反思笔记"""
    note_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    student_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    content: str = ""
    tags: List[str] = field(default_factory=list)
    related_skills: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class LongTermMemory:
    """
    长期记忆：支持多学生、跨会话持久化。
    """
    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path or (STUDENT_DIR / "memory.json")
        self.episodic: Dict[str, List[Episode]] = {}  # student_id -> episodes
        self.semantic: Dict[str, Dict] = {}  # student_id -> {skill_id: mastery}
        self.reflective: Dict[str, List[ReflectiveNote]] = {}

        self._load()

    # ------------------------------------------------------------------ #
    # Episode 操作
    # ------------------------------------------------------------------ #
    def add_episode(self, episode: Episode):
        self.episodic.setdefault(episode.student_id, []).append(episode)
        self._maybe_save()

    def get_episodes(self, student_id: str, skill_id: Optional[str] = None, limit: int = 50) -> List[Episode]:
        eps = self.episodic.get(student_id, [])
        if skill_id:
            eps = [e for e in eps if e.skill_id == skill_id]
        return eps[-limit:]

    # ------------------------------------------------------------------ #
    # Semantic 操作
    # ------------------------------------------------------------------ #
    def update_semantic(self, student_id: str, skill_id: str, mastery: float, confidence: float = 1.0):
        if student_id not in self.semantic:
            self.semantic[student_id] = {}
        self.semantic[student_id][skill_id] = {
            "mastery": mastery,
            "confidence": confidence,
            "updated_at": datetime.now().isoformat(),
        }
        self._maybe_save()

    def get_semantic(self, student_id: str) -> Dict:
        return self.semantic.get(student_id, {})

    # ------------------------------------------------------------------ #
    # Reflective 操作
    # ------------------------------------------------------------------ #
    def add_reflection(self, note: ReflectiveNote):
        self.reflective.setdefault(note.student_id, []).append(note)
        self._maybe_save()

    def get_reflections(self, student_id: str, limit: int = 20) -> List[ReflectiveNote]:
        return self.reflective.get(student_id, [])[-limit:]

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def recall_similar(self, student_id: str, query: str, top_k: int = 5) -> List[Episode]:
        """简单关键词匹配检索（工程上可替换为向量检索）"""
        eps = self.episodic.get(student_id, [])
        query_lower = query.lower()
        scored = []
        for e in eps:
            score = 0
            for field in [e.user_input, e.agent_response, e.skill_id]:
                if field and query_lower in field.lower():
                    score += 1
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    # ------------------------------------------------------------------ #
    # 压缩 / 整合
    # ------------------------------------------------------------------ #
    def consolidate(self, student_id: str, max_recent: int = 20):
        """
        压缩：将旧的情景记忆总结到语义记忆中。
        保留最近 max_recent 条情景，更早的合并为统计特征。
        """
        eps = self.episodic.get(student_id, [])
        if len(eps) <= max_recent:
            return

        old_eps = eps[:-max_recent]
        # 统计
        from collections import Counter
        skill_counter = Counter(e.skill_id for e in old_eps)
        correct_counter = Counter(e.is_correct for e in old_eps if e.is_correct is not None)

        for skill_id, count in skill_counter.items():
            correct_for_skill = [e.is_correct for e in old_eps if e.skill_id == skill_id and e.is_correct is not None]
            if correct_for_skill:
                rate = sum(correct_for_skill) / len(correct_for_skill)
                # 更新语义记忆（取平均）
                current = self.semantic.get(student_id, {}).get(skill_id, {})
                if current:
                    new_mastery = (current.get("mastery", 0.5) * 0.5 + rate * 0.5)
                    self.update_semantic(student_id, skill_id, new_mastery)

        # 截断 episodic
        self.episodic[student_id] = eps[-max_recent:]
        self._maybe_save()

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def _load(self):
        if self.persist_path and self.persist_path.exists():
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 反序列化
                for sid, eps in data.get("episodic", {}).items():
                    self.episodic[sid] = [Episode(**e) for e in eps]
                self.semantic = data.get("semantic", {})
                for sid, notes in data.get("reflective", {}).items():
                    self.reflective[sid] = [ReflectiveNote(**n) for n in notes]
            except Exception as e:
                print(f"[Memory] Load failed: {e}")

    def _save(self):
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "episodic": {sid: [e.to_dict() for e in eps] for sid, eps in self.episodic.items()},
                "semantic": self.semantic,
                "reflective": {sid: [n.__dict__ for n in notes] for sid, notes in self.reflective.items()},
            }
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _maybe_save(self):
        # 实际生产中可以延迟写入
        try:
            self._save()
        except Exception as e:
            print(f"[Memory] Save failed: {e}")


# ---------------------------------------------------------------------- #
# 反思生成器
# ---------------------------------------------------------------------- #
def generate_reflection(
    recent_episodes: List[Episode],
    student_id: str,
) -> ReflectiveNote:
    """
    从最近的情景记忆生成反思笔记。
    启发式生成；生产可让 LLM 润色。
    """
    if not recent_episodes:
        return ReflectiveNote(
            student_id=student_id,
            content="暂无学习记录。",
            tags=["empty"],
        )

    # 统计
    correct = sum(1 for e in recent_episodes if e.is_correct)
    total = sum(1 for e in recent_episodes if e.is_correct is not None)
    rate = correct / total if total else 0

    skills = set(e.skill_id for e in recent_episodes)
    avg_scaffold = sum(e.scaffolding_level for e in recent_episodes) / len(recent_episodes)

    content_lines = [
        f"近 {len(recent_episodes)} 次交互分析：",
        f"- 答对率：{rate:.0%}（{correct}/{total}）",
        f"- 涉及知识点：{len(skills)} 个",
        f"- 平均脚手架级别：{avg_scaffold:.1f}/4",
    ]

    if rate > 0.8:
        content_lines.append("- 表现优秀！建议挑战更高难度的内容。")
        tags = ["excellent", "ready_for_advance"]
    elif rate > 0.5:
        content_lines.append("- 表现稳定。识别仍存在的薄弱环节，针对性练习。")
        tags = ["stable", "review_needed"]
    else:
        content_lines.append("- 遇到较多困难。建议回到基础概念，配合图示与类比重新理解。")
        tags = ["struggling", "back_to_basics"]

    return ReflectiveNote(
        student_id=student_id,
        content="\n".join(content_lines),
        tags=tags,
        related_skills=list(skills),
    )


if __name__ == "__main__":
    mem = LongTermMemory()
    print("Memory loaded.")
    print(f"Students: {list(mem.episodic.keys())}")
    # 添加测试 episode
    ep = Episode(
        student_id="S001",
        skill_id="linear_regression",
        user_input="怎么求最小二乘？",
        agent_response="想想误差平方和最小...",
        is_correct=True,
        scaffolding_level=2,
    )
    mem.add_episode(ep)
    mem.update_semantic("S001", "linear_regression", 0.75, 0.8)
    print(f"After add: {mem.get_semantic('S001')}")
