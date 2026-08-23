# 智径 AdaptivePath · 系统架构设计文档

> **版本**: v1.0 · 2026-08
> **作者**: 崇理团队

---

## 一、设计目标

智径 AdaptivePath 的总体设计目标可以归纳为 **"四自"**：

| 目标 | 含义 | 实现路径 |
| --- | --- | --- |
| **自诊断** | 系统主动评估学生学情 | BKT + DKT + IRT 三模型融合 |
| **自规划** | 动态生成个性化学习路径 | Contextual Bandit (LinUCB) + ZPD 原则 |
| **自教学** | 苏格拉底式启发而非灌输 | ReAct + 五级动态脚手架 + Bloom 适配 |
| **自反思** | 持续优化教学策略 | 长期记忆 + 反思机制 + Bandit 在线学习 |

---

## 二、总体架构

### 2.1 分层架构

```
┌────────────────────────────────────────────────────────┐
│  L1: 表现层 (Presentation)                              │
│      - Streamlit Web Demo                              │
│      - CLI / HTTP API                                  │
├────────────────────────────────────────────────────────┤
│  L2: 协同层 (Orchestration)                             │
│      - Orchestrator Agent (主控)                       │
│      - SessionState / 状态机                           │
├────────────────────────────────────────────────────────┤
│  L3: 智能体层 (Agent)                                   │
│      - Diagnostic / Planning / Teaching                 │
│      - Reflective / Emotional                           │
├────────────────────────────────────────────────────────┤
│  L4: 认知层 (Cognitive)                                 │
│      - CSN (DKT × BKT × IRT)                           │
│      - Long-term Memory                                 │
├────────────────────────────────────────────────────────┤
│  L5: 知识层 (Knowledge)                                 │
│      - Knowledge Graph (学科图谱)                       │
│      - RAG / Vector Store                               │
│      - Question Bank                                    │
├────────────────────────────────────────────────────────┤
│  L6: 基础层 (Foundation)                                │
│      - LLM (OpenAI/Qwen/DeepSeek/...)                   │
│      - Emotion Detector                                 │
│      - Utilities                                        │
└────────────────────────────────────────────────────────┘
```

### 2.2 智能体协同

```
              ┌──────────────────┐
              │  Orchestrator    │
              │  (主控调度)       │
              └────────┬─────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌────────┐    ┌──────────┐   ┌──────────┐
   │   CSN  │◀───│ Planning │──▶│ Teaching │
   │ 学生画像│    │   Agent  │   │  Agent   │
   └────┬───┘    └──────────┘   └────┬─────┘
        │                            │
        │       ┌──────────┐         │
        └──────▶│  Memory  │◀────────┘
                │  (长期)   │
                └──────────┘
                       ▲
                       │
                ┌──────┴───────┐
                │  Emotional   │
                │   Agent      │
                └──────────────┘
```

每个智能体职责：

- **Orchestrator**：维护会话状态、协调 5 个 Agent、决定下一步行动
- **Diagnostic Agent**：调用 CSN 更新学生状态，输出诊断报告
- **Planning Agent**：基于 Bandit 选下一步知识点，输出推荐与理由
- **Teaching Agent**：苏格拉底式对话、动态脚手架
- **Reflective Agent**：记录交互、生成反思笔记
- **Emotional Agent**：实时情感识别，反向调节教学节奏

---

## 三、核心技术详解

### 3.1 Cognitive State Network (CSN)

**问题**：单一 BKT 无法捕捉长程依赖；单一 DKT 缺乏可解释性；单一 IRT 无法精细到知识点。

**方案**：三模型加权贝叶斯融合 + 不确定性估计。

```
                        ┌──────────┐
       学生作答序列 ──▶  │   DKT    │  ──▶  dkt_p[i]
                        │ (LSTM)   │
                        └──────────┘
                               
                        ┌──────────┐
       学生作答序列 ──▶  │   BKT    │  ──▶  bkt_p[i]
                        │ (HMM)    │
                        └──────────┘
                               
                        ┌──────────┐
       整体作答 + 题目   ──▶│   IRT   │  ──▶  irt_p[i]  +  theta
                        │ (2PL)    │
                        └──────────┘
                               
                                ↓
                        ┌──────────────────┐
                        │ Adaptive Weights │  根据样本量 n 动态调整
                        │ (W_dkt, W_bkt,   │
                        │  W_irt)          │
                        └────────┬─────────┘
                                 ↓
                        mastery[i] = w_dkt * dkt_p + w_bkt * bkt_p + w_irt * irt_p
                        confidence[i] = 1 / (1 + 10 * var([dkt_p, bkt_p, irt_p]))
```

**自适应权重**：
- n < 5：BKT 主导（冷启动友好）
- 5 ≤ n < 20：三模型等权
- n ≥ 20：DKT 主导（长程依赖）

**为什么不用 LLM 直接打分**：
- LLM 缺乏时序建模
- LLM 不可解释
- LLM 推理成本高，无法实时更新

### 3.2 Contextual Bandit (LinUCB)

**问题**：如何在线学习最优推荐策略？

**方案**：把"下一步推荐什么"建模为 Contextual Bandit。

- Context: 学生当前状态向量 x ∈ R^32
- Action: 候选知识点 a ∈ A
- Reward: 综合学习收益

UCB 选择：
```
a* = argmax_a [ θ_a^T x + α √(x^T A_a^{-1} x) ]
```

**冷启动优势**：不需要离线训练数据
**可解释性**：每个推荐都有 UCB 分数
**在线学习**：每次收到反馈就更新参数

### 3.3 苏格拉底式对话

**5 阶段状态机**：

```
[DIAGNOSE] → [PROBE] → [HINT] → [CONFIRM] → [REFLECT]
    │            │         │          │           │
    ▼            ▼         ▼          ▼           ▼
 了解认知    反诘深入    关键提示   检验理解   元反思
```

**5 级脚手架**：

| Level | 提示强度 | 适用场景 |
| --- | --- | --- |
| 0 | 元认知提示 | 掌握度 ≥ 0.85 |
| 1 | 关键概念 | 0.7 ≤ 掌握度 < 0.85 |
| 2 | 类比示例 | 0.5 ≤ 掌握度 < 0.7 |
| 3 | 分步分解 | 0.3 ≤ 掌握度 < 0.5 |
| 4 | 完整示范 | 掌握度 < 0.3 |

**自适应升级规则**：
- 连续 2 轮错 → 升级
- 连续 2 轮对 → 降级
- 学生主动求提示 → 中等脚手架
- 疲劳度高 → 降级

### 3.4 情感-认知双通道

**5 维情感**：confusion / frustration / engagement / fatigue / confidence

**检测方式**：
- 关键词词典（基础）
- 标点/重复模式启发式
- 交互节奏（响应时间、错误率）

**教学影响**：

| 状态 | 阈值 | 教学调整 |
| --- | --- | --- |
| 挫败 | > 0.7 | 鼓励 + 简化任务 |
| 疲劳 | > 0.6 | 建议休息或轻量任务 |
| 困惑 | > 0.7 | 用类比/图示重新讲解 |
| 自信 | < 0.2 | 明确肯定已有进步 |
| 兴趣低 | < 0.3 | 增加趣味性 |

### 3.5 长期记忆

**分层记忆**：
- **Episodic**（情景）：每次交互的完整记录
- **Semantic**（语义）：学到的概念、技能掌握
- **Reflective**（反思）：元认知笔记

**操作**：
- `add_episode`：添加一次交互
- `consolidate`：压缩旧情景到语义
- `recall_similar`：关键词检索（生产可换向量检索）
- `reflect`：生成反思笔记

---

## 四、关键工程决策

### 4.1 为什么不用 LangChain / LangGraph

虽然 LangChain / LangGraph 生态成熟，但我们选择**自研轻量级编排**：
- 5 个 Agent 的状态转移可枚举，无需图引擎复杂度
- 自研代码更易定制（教学场景的脚手架策略）
- 避免 LangChain 版本不稳定

未来可平滑迁移到 LangGraph。

### 4.2 为什么不全用 LLM 决策

LLM 不可解释、不可复现、成本高。智径只在以下场景使用 LLM：
- 苏格拉底对话的自然语言生成
- 反思笔记的润色
- 复杂题目的解释

状态估计、路径规划、情感识别都用规则 + 统计模型。

### 4.3 性能与扩展性

| 模块 | 性能 | 扩展点 |
| --- | --- | --- |
| CSN | 单学生 < 10ms/次 | 横向扩展支持千万学生 |
| Bandit | 单决策 < 1ms | 知识点规模支持 1w+ |
| 对话引擎 | 单轮 < 100ms (含 LLM) | 异步生成、流式输出 |
| 知识图谱 | 查询 < 5ms | 千万级节点优化 |

---

## 五、目录结构

```
智径-AdaptivePath/
├── README.md
├── requirements.txt
├── docs/
│   └── architecture.md           # 本文档
├── data/
│   ├── knowledge_graph/          # 知识图谱
│   ├── questions/                # 题库
│   └── students/                 # 学生画像存储
├── src/
│   ├── main.py                   # 入口
│   ├── config.py                 # 配置
│   ├── agents/                   # 五大智能体
│   │   ├── orchestrator.py
│   │   ├── diagnostic_agent.py
│   │   ├── planning_agent.py
│   │   ├── teaching_agent.py
│   │   ├── reflective_agent.py
│   │   └── emotional_agent.py
│   ├── cognitive/                # 认知诊断
│   │   ├── bkt.py
│   │   ├── dkt.py
│   │   ├── irt.py
│   │   └── student_model.py
│   ├── knowledge/                # 知识图谱
│   │   └── knowledge_graph.py
│   ├── planning/                 # 路径规划
│   │   ├── contextual_bandit.py
│   │   └── path_optimizer.py
│   ├── dialogue/                 # 对话引擎
│   │   ├── socratic_engine.py
│   │   └── scaffolding.py
│   ├── memory/                   # 长期记忆
│   │   └── long_term_memory.py
│   ├── emotion/                  # 情感分析
│   │   └── sentiment.py
│   └── utils/                    # 工具
│       ├── llm.py
│       └── vector_store.py
├── demo/
│   ├── app.py                    # Streamlit Web Demo
│   └── verify_system.py          # 验证脚本
├── tests/                        # 测试
│   ├── test_cognitive.py
│   ├── test_knowledge.py
│   └── test_agents.py
├── 商业计划书/                    # 商业计划书 + 路演 PPT
```

---

## 六、评测与实验

### 6.1 单元测试

```bash
python tests/test_cognitive.py
python tests/test_knowledge.py
python tests/test_agents.py
```

### 6.2 端到端验证

```bash
python demo/verify_system.py
```

### 6.3 拟评测指标（决赛演示）

| 指标 | 传统 LLM 套壳 | 智径 AdaptivePath |
| --- | --- | --- |
| 学情诊断精度 | 无 | 90%+（CSN） |
| 路径规划合理性 | 静态 | 动态（Bandit） |
| 教学方式 | 灌输 | 引导（Socratic） |
| 情感适配 | 无 | 5 维实时感知 |
| 可解释性 | 低 | 高（每次推荐有理由） |
| 长期记忆 | 无 | 3 层记忆 |

---

## 七、参考文献

1. Corbett, A. T., & Anderson, J. R. (1995). *Knowledge tracing: Modeling the acquisition of procedural knowledge*. User Modeling and User-Adapted Interaction, 4(4), 253-278.
2. Piech, C., et al. (2015). *Deep Knowledge Tracing*. NeurIPS.
3. Baker, R. S., & Inventado, P. S. (2014). *Educational Data Mining and Learning Analytics*. Springer.
4. Li, L., et al. (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation*. WWW.
5. Vygotsky, L. S. (1978). *Mind in Society*. Harvard University Press.
6. Bloom, B. S. (1956). *Taxonomy of Educational Objectives*. Longman.
7. Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS.
8. Yao, S., et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR.
9. Baker, R. S. (2007). *Modeling and Understanding Students' Learning Behavior*. IJAIED.
10. OpenAI. (2023). *GPT-4 Technical Report*.

---

## 八、未来工作

- [ ] 多模态输入：支持图像、语音
- [ ] 多学生协作：小组学习
- [ ] 教师端 Dashboard
- [ ] 跨学科知识图谱（数学、物理、电子信息）
- [ ] A/B 测试框架
- [ ] 离线强化学习（用户日志回放）
- [ ] 与高校/培训机构合作获取真实学习数据
