"""
智径 AdaptivePath - 主入口
==============================

命令行模式：
  python -m src.main --mode cli --student S001 --target llm_agent

服务器模式：
  python -m src.main --mode serve --port 8000

Web Demo 模式：
  streamlit run demo/app.py
"""
from __future__ import annotations
import argparse
import sys
import os
import json

# 把项目根加入 path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agents import Orchestrator, SessionMode


def run_cli(student_id: str, target_skill: str = None):
    """命令行交互模式"""
    print("\n" + "=" * 70)
    print("  智径 AdaptivePath · 自适应学习伴学智能体")
    print("  Multi-Agent Adaptive Learning Companion")
    print("=" * 70 + "\n")

    orch = Orchestrator()
    state = orch.start_session(student_id, target_skill=target_skill)

    if state.plan:
        print(f"📚 你的学习路径（接下来 5 步）：")
        for i, sid in enumerate(state.plan.full_path, 1):
            node = orch.kg.get_node(sid)
            if node:
                print(f"   {i}. {node.name} (难度: {node.difficulty})")
        print()

    print("💡 提示：")
    print("  - 直接输入文字与导师对话")
    print("  - 输入 'quiz' 进入测验")
    print("  - 输入 'plan' 查看当前规划")
    print("  - 输入 'report' 查看学习报告")
    print("  - 输入 'quit' 退出")
    print()

    while True:
        try:
            user_input = input(f"\n[{student_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            reflection = orch.end_session(student_id)
            print(f"\n📝 学习总结：\n{reflection.summary}\n")
            break

        if user_input.lower() == "plan":
            plan = orch.planning.plan(student_id)
            print("\n📋 当前规划：")
            for i, sid in enumerate(plan.full_path, 1):
                node = orch.kg.get_node(sid)
                if node:
                    print(f"   {i}. {node.name}")
            print(f"💡 理由：{plan.decision.reasoning}")
            continue

        if user_input.lower() == "report":
            report = orch.get_student_report(student_id)
            print("\n📊 学习报告：")
            print(json.dumps(report.get("diagnostic").__dict__ if hasattr(report.get("diagnostic"), "__dict__") else str(report.get("diagnostic")), ensure_ascii=False, indent=2, default=str))
            continue

        if user_input.lower() == "quiz":
            # 出题
            from data.questions.questions import get_random_question  # noqa
            print("(测验模式：简化版)")
            continue

        # 普通对话
        result = orch.handle(student_id, user_input)
        print(f"\n🤖 {result.response}\n")
        if result.emotion:
            print(f"   (情感: {result.emotion.dominant} | 干预: {result.emotion.intervention_type or '无需'})")


def run_serve(port: int = 8000):
    """HTTP API 服务模式（FastAPI）"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("Please install fastapi uvicorn pydantic: pip install fastapi uvicorn pydantic")
        return

    app = FastAPI(title="智径 AdaptivePath API", version="1.0.0")
    orch = Orchestrator()

    class ChatRequest(BaseModel):
        student_id: str
        message: str
        target_skill: str = None
        is_answer: bool = False
        item_id: str = None
        is_correct: bool = None

    @app.post("/chat")
    def chat(req: ChatRequest):
        if req.student_id not in orch.sessions:
            orch.start_session(req.student_id, target_skill=req.target_skill)
        result = orch.handle(
            req.student_id, req.message,
            is_answer=req.is_answer, item_id=req.item_id, is_correct=req.is_correct,
        )
        return {
            "response": result.response,
            "next_action": result.next_action,
            "current_skill": result.state.current_skill,
            "emotion": result.emotion.state.to_dict() if result.emotion else None,
        }

    @app.get("/report/{student_id}")
    def report(student_id: str):
        return orch.get_student_report(student_id)

    @app.get("/plan/{student_id}")
    def plan(student_id: str):
        p = orch.planning.plan(student_id)
        return {
            "next_skill": p.next_skill,
            "full_path": p.full_path,
            "reasoning": p.decision.reasoning,
            "alternatives": p.decision.alternatives,
        }

    print(f"🌐 Server starting on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    parser = argparse.ArgumentParser(description="智径 AdaptivePath - 自适应学习伴学智能体")
    parser.add_argument("--mode", choices=["cli", "serve"], default="cli")
    parser.add_argument("--student", default="S001")
    parser.add_argument("--target", default="llm_agent")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli(args.student, args.target)
    elif args.mode == "serve":
        run_serve(args.port)


if __name__ == "__main__":
    main()
