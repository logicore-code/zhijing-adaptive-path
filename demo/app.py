"""
智径 AdaptivePath · Web Demo
================================

Streamlit 多页应用：
- 主页面：自适应对话
- 学情诊断：可视化掌握度
- 路径规划：可视化推荐
- 学习报告：统计与反思

运行：
    streamlit run demo/app.py
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# 把项目根加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from src.agents import Orchestrator, SessionMode
from src.knowledge.knowledge_graph import build_default_kg
from src.utils.llm import get_llm


# ---------------------------------------------------------------------- #
# 全局配置
# ---------------------------------------------------------------------- #
st.set_page_config(
    page_title="智径 AdaptivePath",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 主题
COLORS = {
    "primary": "#0075FF",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "muted": "#6B7280",
    "bg": "#F9FAFB",
}


# ---------------------------------------------------------------------- #
# Session State 初始化
# ---------------------------------------------------------------------- #
@st.cache_resource
def init_orchestrator():
    """全局单例 orchestrator"""
    llm = get_llm(provider="mock")
    orch = Orchestrator(llm=llm)
    return orch


orch = init_orchestrator()

if "session_started" not in st.session_state:
    st.session_state.session_started = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "student_id" not in st.session_state:
    st.session_state.student_id = "demo_student"


# ---------------------------------------------------------------------- #
# 侧边栏
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.title("🎓 智径 AdaptivePath")
    st.caption("基于多智能体协同的自适应学习伴学系统")
    st.divider()

    page = st.radio(
        "导航",
        ["🏠 主页对话", "📊 学情诊断", "🗺️ 路径规划", "📝 学习报告", "🧠 智能体架构", "ℹ️ 关于系统"],
        index=0,
    )
    st.divider()

    st.subheader("⚙️ 学习设置")
    student_id = st.text_input("学生 ID", value=st.session_state.student_id)
    st.session_state.student_id = student_id

    target_options = {
        "llm_agent": "🤖 大模型智能体（Agent）",
        "dl_transformer": "🧠 Transformer 与注意力",
        "dl_cnn": "👁️ 卷积神经网络 (CNN)",
        "ml_logistic_regression": "📊 逻辑回归",
        "ml_svm": "📈 支持向量机 (SVM)",
        "ml_ensemble": "🌲 集成学习 (XGBoost)",
        "nlp_pretrain": "💬 预训练语言模型 (BERT/GPT)",
        "rl_basic": "🎮 强化学习基础",
    }
    target = st.selectbox(
        "学习目标",
        options=list(target_options.keys()),
        format_func=lambda x: target_options[x],
        index=0,
    )

    if st.button("🚀 开始新会话", type="primary", use_container_width=True):
        if student_id != st.session_state.student_id:
            st.session_state.student_id = student_id
        orch.start_session(student_id, target_skill=target)
        st.session_state.session_started = True
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("🏆 参赛作品 · 科大讯飞 AI 开发者大赛")
    st.caption("© 2026 智径团队")


# ---------------------------------------------------------------------- #
# 主页对话
# ---------------------------------------------------------------------- #
def render_chat_page():
    st.title("🏠 与智径对话")
    st.markdown("""
    欢迎来到 **智径 AdaptivePath**！这是一个**多智能体协同**的自适应学习系统。

    你可以直接和 AI 导师对话（**苏格拉底式**，从不直接给答案），也可以点击"下一题"进行测试。
    """)

    if not st.session_state.session_started:
        st.info("👈 请在左侧设置学习目标，然后点击「开始新会话」")
        return

    # 显示当前规划
    state = orch.sessions.get(student_id)
    if state and state.plan:
        with st.expander("📋 当前学习路径", expanded=False):
            cols = st.columns(min(5, len(state.plan.full_path)))
            for i, sid in enumerate(state.plan.full_path[:5]):
                with cols[i]:
                    node = orch.kg.get_node(sid)
                    if node:
                        mastery = orch.csn.get_mastery(student_id, sid)
                        color = COLORS["success"] if mastery > 0.7 else (COLORS["warning"] if mastery > 0.4 else COLORS["danger"])
                        st.markdown(f"""
                        <div style='text-align:center; padding:10px; border:2px solid {color}; border-radius:8px;'>
                            <b>{i+1}. {node.name[:8]}</b><br>
                            <span style='color:{color}; font-size:20px;'>{mastery:.0%}</span>
                        </div>
                        """, unsafe_allow_html=True)

    # 聊天界面
    st.subheader("💬 智能伴学对话")

    # 显示历史
    for msg in st.session_state.chat_history:
        role = msg["role"]
        avatar = "🧑‍🎓" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.write(msg["content"])
            if msg.get("meta"):
                meta = msg["meta"]
                cols = st.columns(4)
                cols[0].metric("掌握度", f"{meta.get('mastery', 0):.0%}")
                cols[1].metric("情感", meta.get("emotion", "N/A"))
                cols[2].metric("脚手架", meta.get("scaffold", "N/A"))
                cols[3].metric("知识点", meta.get("skill", "")[:6])

    # 输入
    user_input = st.chat_input("说点什么吧……（输入 'quiz' 开始答题）")

    if user_input:
        # 处理答题
        is_quiz_answer = user_input.startswith("/answer")
        if is_quiz_answer:
            # 格式: /answer qid <True/False>
            try:
                parts = user_input.split()
                item_id = parts[1] if len(parts) > 1 else "unknown"
                is_correct = parts[2].lower() in ("true", "t", "1", "对") if len(parts) > 2 else None
                user_input = f"我回答了题目 {item_id}, {'正确' if is_correct else '错误'}"
            except:
                is_correct = None
        else:
            is_correct = None

        # 显示用户输入
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.write(user_input)

        # 处理
        with st.spinner("🤔 AI 导师思考中..."):
            result = orch.handle(
                student_id,
                user_input,
                is_answer=is_quiz_answer,
                is_correct=is_correct,
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result.response,
            "meta": {
                "mastery": orch.csn.get_mastery(student_id, result.state.current_skill) if result.state.current_skill else 0,
                "emotion": result.emotion.dominant if result.emotion else "N/A",
                "scaffold": result.state.last_decision.response.scaffolding_level.name if result.state.last_decision else "N/A",
                "skill": orch.kg.get_node(result.state.current_skill).name if result.state.current_skill and orch.kg.get_node(result.state.current_skill) else "",
            },
        })
        with st.chat_message("assistant", avatar="🤖"):
            st.write(result.response)

    # 答题快捷按钮
    st.divider()
    st.subheader("📝 快速答题测试")
    if st.session_state.session_started and state and state.current_skill:
        # 找当前 skill 的题目
        from data.questions.questions import get_questions_for_skill
        qs = get_questions_for_skill(state.current_skill)
        if qs:
            q = qs[0]
            st.markdown(f"**{q['stem']}**")
            cols = st.columns(len(q["options"]))
            for i, opt in enumerate(q["options"]):
                if cols[i].button(f"{chr(65+i)}. {opt}", key=f"q_{i}"):
                    is_correct = (i == q["answer"])
                    user_input = f"我选了 {chr(65+i)}"
                    is_quiz_answer = True
                    # 处理
                    result = orch.handle(
                        student_id, user_input,
                        is_answer=True, item_id=q["id"], is_correct=is_correct,
                    )
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input,
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.response + (f"\n\n{'✅ 答对了！' if is_correct else f'❌ 答错了。正确答案是 {chr(65+q[\"answer\"])}'}"),
                    })
                    st.rerun()
            if st.button("🔍 看解析"):
                st.info(q.get("explanation", ""))
        else:
            st.info("当前知识点暂无题目")


# ---------------------------------------------------------------------- #
# 学情诊断
# ---------------------------------------------------------------------- #
def render_diagnostic_page():
    st.title("📊 学情诊断")
    st.markdown("基于 **Cognitive State Network (CSN)** 的多模型融合诊断。")

    if not st.session_state.session_started:
        st.info("👈 请先开始一个学习会话")
        return

    diag = orch.diagnostic.report(student_id)

    # 整体指标
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("整体能力 θ", f"{diag.overall_ability:.2f}")
    col2.metric("强项数量", len(diag.strong_skills))
    col3.metric("弱项数量", len(diag.weak_skills))
    col4.metric("待诊断", len(diag.uncertain_skills))

    st.divider()

    # 掌握度雷达图
    st.subheader("🎯 各知识点掌握度")
    mastery = orch.csn.get_all_mastery(student_id)
    if mastery:
        df = pd.DataFrame([
            {"知识点": orch.kg.get_node(sid).name if orch.kg.get_node(sid) else sid, "掌握度": m, "置信度": orch.csn.get_confidence(student_id, sid)}
            for sid, m in mastery.items()
        ])
        if not df.empty:
            fig = px.bar(df, x="知识点", y="掌握度", color="置信度", color_continuous_scale="RdYlGn",
                         range_y=[0, 1], title="知识点掌握度（颜色=置信度）")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # 强项 / 弱项
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💪 强项")
        for sid, m in diag.strong_skills:
            node = orch.kg.get_node(sid)
            st.success(f"**{node.name if node else sid}** - {m:.0%}")
    with col2:
        st.subheader("🎯 弱项")
        for sid, m in diag.weak_skills:
            node = orch.kg.get_node(sid)
            st.error(f"**{node.name if node else sid}** - {m:.0%}")


# ---------------------------------------------------------------------- #
# 路径规划
# ---------------------------------------------------------------------- #
def render_planning_page():
    st.title("🗺️ 学习路径规划")
    st.markdown("""
    **Contextual Bandit (LinUCB)** 驱动的动态路径优化。

    每一次推荐都基于：
    - 学生当前状态向量
    - 知识点先决关系
    - 历史学习奖励反馈
    - ZPD（最近发展区）原则
    """)

    if not st.session_state.session_started:
        st.info("👈 请先开始一个学习会话")
        return

    plan = orch.planning.plan(student_id)

    # 路径可视化
    st.subheader("📍 接下来 5 步")
    if plan.full_path:
        fig = go.Figure()
        for i, sid in enumerate(plan.full_path):
            node = orch.kg.get_node(sid)
            if not node:
                continue
            mastery = orch.csn.get_mastery(student_id, sid)
            color = "#10B981" if mastery > 0.7 else ("#F59E0B" if mastery > 0.4 else "#3B82F6")
            fig.add_trace(go.Scatter(
                x=[i], y=[0], mode="markers+text",
                marker=dict(size=60 + mastery * 40, color=color),
                text=[f"{i+1}"], textfont=dict(size=16, color="white"),
                name=node.name, showlegend=True,
                hovertemplate=f"<b>{node.name}</b><br>掌握度: {mastery:.0%}<br>难度: {node.difficulty:.1f}<extra></extra>",
            ))
        fig.update_layout(
            height=300, xaxis=dict(range=[-0.5, len(plan.full_path) - 0.5]),
            yaxis=dict(visible=False),
            plot_bgcolor="white", title="学习路径节点（颜色: 绿=已掌握 黄=部分 蓝=新）",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 当前推荐理由
    st.subheader("💡 下一步推荐理由")
    st.info(plan.decision.reasoning)

    # 备选
    st.subheader("🔄 备选知识点")
    for sid, ucb in plan.decision.alternatives:
        node = orch.kg.get_node(sid)
        st.markdown(f"- **{node.name if node else sid}** (UCB 分数: {ucb:.2f})")

    # 知识图谱可视化
    st.divider()
    st.subheader("🕸️ 知识图谱先决关系（子图）")
    if plan.full_path:
        # 提取相关节点
        related = set()
        for sid in plan.full_path:
            related.update(orch.kg.all_prerequisites_recursive(sid))
            related.add(sid)
        subg = orch.kg.graph.subgraph(related)
        pos = _hierarchy_pos(subg, plan.full_path[-1])
        edge_x, edge_y = [], []
        for e in subg.edges():
            if e[0] in pos and e[1] in pos:
                x0, y0 = pos[e[0]]
                x1, y1 = pos[e[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
        node_x = [pos[n][0] for n in subg.nodes() if n in pos]
        node_y = [pos[n][1] for n in subg.nodes() if n in pos]
        node_text = [orch.kg.get_node(n).name if orch.kg.get_node(n) else n for n in subg.nodes() if n in pos]
        node_color = [
            "#10B981" if orch.csn.get_mastery(student_id, n) > 0.7
            else ("#F59E0B" if orch.csn.get_mastery(student_id, n) > 0.4 else "#3B82F6")
            for n in subg.nodes() if n in pos
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#9CA3AF"), hoverinfo="none"))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="top center",
            marker=dict(size=20, color=node_color, line=dict(width=2, color="white")),
        ))
        fig.update_layout(height=500, showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)


def _hierarchy_pos(G, root):
    """简单的层次布局"""
    try:
        import networkx as nx
        pos = nx.spring_layout(G, k=1.5, seed=42)
        return pos
    except:
        return {n: (i, 0) for i, n in enumerate(G.nodes())}


# ---------------------------------------------------------------------- #
# 学习报告
# ---------------------------------------------------------------------- #
def render_report_page():
    st.title("📝 学习报告")
    if not st.session_state.session_started:
        st.info("👈 请先开始一个学习会话")
        return

    report = orch.get_student_report(student_id)

    col1, col2, col3 = st.columns(3)
    col1.metric("交互次数", report["memory_summary"]["episodes"])
    col2.metric("反思笔记", report["memory_summary"]["reflections"])
    col3.metric("整体能力", f"{report['diagnostic'].overall_ability:.2f}")

    st.divider()
    st.subheader("🤖 情感状态")
    emo = report["emotion"]
    df = pd.DataFrame([emo])
    fig = px.bar(df, orientation="h", title="当前情感维度", range_x=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("💭 反思总结")
    if st.button("生成反思"):
        result = orch.reflective.reflect(student_id)
        st.markdown(f"**总结**: {result.summary}")
        if result.achievements:
            st.success("**成就**:\n" + "\n".join([f"- {a}" for a in result.achievements]))
        if result.challenges:
            st.error("**挑战**:\n" + "\n".join([f"- {c}" for c in result.challenges]))
        if result.next_steps:
            st.info("**下一步**:\n" + "\n".join([f"- {n}" for n in result.next_steps]))


# ---------------------------------------------------------------------- #
# 智能体架构
# ---------------------------------------------------------------------- #
def render_architecture_page():
    st.title("🧠 智能体架构")
    st.markdown("""
    智径 AdaptivePath 采用 **5 智能体协同架构**：
    """)

    agents = [
        ("🎯 Diagnostic Agent", "学情诊断", "通过对话与微测试持续推断学生认知状态，调用 CSN 三模型融合"),
        ("🗺️ Planning Agent", "路径规划", "Contextual Bandit (LinUCB) + ZPD 原则动态生成学习路径"),
        ("🎓 Teaching Agent", "教学（苏格拉底）", "ReAct + 五级动态脚手架，永远不直接给答案"),
        ("💭 Reflective Agent", "反思", "元认知培养，自动生成学习总结与下次建议"),
        ("❤️ Emotional Agent", "情感", "实时检测困惑/挫败/兴趣/疲劳，反向影响教学节奏"),
    ]
    for emoji_name, name, desc in agents:
        with st.container():
            st.markdown(f"### {emoji_name}")
            st.markdown(f"_{name}_")
            st.markdown(desc)
            st.divider()

    st.subheader("🔬 底层技术")
    st.markdown("""
    - **Cognitive State Network (CSN)**: BKT × DKT × IRT 三模型融合
    - **LinUCB Contextual Bandit**: 在线学习的路径优化
    - **MDP Value Iteration**: 短期规划
    - **ZPD (Vygotsky)**: 最近发展区
    - **Bloom 认知分类**: 教学策略适配
    - **RAG + 向量检索**: 知识增强
    - **ReAct + CoT**: 推理与行动融合
    - **Affective Computing**: 情感计算
    """)


# ---------------------------------------------------------------------- #
# 关于
# ---------------------------------------------------------------------- #
def render_about_page():
    st.title("ℹ️ 关于智径 AdaptivePath")
    st.markdown("""
    ## 项目简介

    **智径 AdaptivePath** 是一个面向 STEM 学科的**多智能体协同自适应学习伴学系统**。
    它把教育认知科学（IRT、BKT、DKT、ZPD、Bloom）与前沿 AI 技术（LLM Agent、CoT/ReAct、Contextual Bandit、Affective Computing）做了**工程级深度融合**。

    ## 核心创新

    1. **Cognitive State Network (CSN)**：三模型融合的学生认知状态网络
    2. **Contextual Bandit 路径优化**：在线学习，无须离线训练数据
    3. **苏格拉底式支架教学**：五级脚手架 + Bloom 认知适配
    4. **情感-认知双通道**：情感信号反向影响教学节奏
    5. **完整可解释性**：每次推荐都有理由

    ## 技术栈

    - Python 3.10+
    - PyTorch / NumPy / SciPy
    - Streamlit (Web Demo)
    - NetworkX (知识图谱)
    - 自研 RAG / 向量检索

    ## 参赛信息

    - **赛事**: 科大讯飞 AI 开发者大赛
    - **赛道**: 自适应学习路径决策与伴学智能体
    - **提交日期**: 2026-09-09

    ## 团队

    - 主负责人：[待填]
    - 核心成员：[待填]
    - 指导教师：[待填]
    """)


# ---------------------------------------------------------------------- #
# 路由
# ---------------------------------------------------------------------- #
if page == "🏠 主页对话":
    render_chat_page()
elif page == "📊 学情诊断":
    render_diagnostic_page()
elif page == "🗺️ 路径规划":
    render_planning_page()
elif page == "📝 学习报告":
    render_report_page()
elif page == "🧠 智能体架构":
    render_architecture_page()
elif page == "ℹ️ 关于系统":
    render_about_page()
