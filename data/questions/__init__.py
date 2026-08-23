"""
题库查询接口
==============
"""
import json
from pathlib import Path
from typing import List, Dict, Optional


_QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
_CACHE: Optional[Dict] = None


def load_questions() -> Dict:
    global _CACHE
    if _CACHE is None:
        with open(_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            _CACHE = json.load(f)
    return _CACHE


def get_all_questions() -> List[Dict]:
    return load_questions().get("questions", [])


def get_questions_for_skill(skill_id: str) -> List[Dict]:
    return [q for q in get_all_questions() if q.get("skill_id") == skill_id]


def get_question_by_id(qid: str) -> Optional[Dict]:
    for q in get_all_questions():
        if q["id"] == qid:
            return q
    return None


def get_random_question(skill_id: Optional[str] = None) -> Optional[Dict]:
    import random
    qs = get_questions_for_skill(skill_id) if skill_id else get_all_questions()
    return random.choice(qs) if qs else None
