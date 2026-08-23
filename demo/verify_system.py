"""
完整系统验证脚本
=================
跑一遍主流程，确认所有模块能协同工作。
"""
import sys
sys.path.insert(0, '.')

from src.agents import Orchestrator


def main():
    print("=" * 70)
    print("  智径 AdaptivePath - 完整系统验证")
    print("=" * 70)

    orch = Orchestrator()
    state = orch.start_session('S001', target_skill='llm_agent')
    print('\n[1] 启动会话')
    print(f'   学习目标: llm_agent')
    print(f'   规划路径: {[orch.kg.get_node(s).name for s in state.plan.full_path]}')
    print(f'   当前知识点: {orch.kg.get_node(state.current_skill).name}')

    print('\n[2] 模拟 5 轮对话')
    turns = [
        ('什么是 Agent？', False, None),
        ('q_pretrain_001', True, True),
        ('q_tr_001', True, True),
        ('我不太懂 ReAct', False, None),
        ('q_agent_001', True, True),
    ]
    for i, (text, is_ans, is_cor) in enumerate(turns):
        result = orch.handle('S001', text, is_answer=is_ans, item_id=text, is_correct=is_cor)
        print(f'\n   [Turn {i+1}] 用户: {text}')
        print(f'            导师: {result.response[:100]}')
        if result.emotion:
            print(f'            (情感={result.emotion.dominant} 干预={result.emotion.intervention_type or "无"})')

    print('\n[3] 学习报告')
    report = orch.get_student_report('S001')
    print(f'   整体能力 theta = {report["diagnostic"].overall_ability:.2f}')
    print(f'   交互次数: {report["memory_summary"]["episodes"]}')
    strong = [orch.kg.get_node(s).name for s, m in report['diagnostic'].strong_skills][:3]
    weak = [orch.kg.get_node(s).name for s, m in report['diagnostic'].weak_skills][:3]
    print(f'   强项: {strong}')
    print(f'   弱项: {weak}')

    print('\n[4] 反思总结')
    reflection = orch.reflective.reflect('S001')
    print(f'   {reflection.summary[:300]}')

    print('\n' + '=' * 70)
    print('  ✅ 系统运行成功！')
    print('=' * 70)


if __name__ == "__main__":
    main()
