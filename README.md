# 智径 · AdaptivePath
## 基于多智能体协同与认知诊断融合的自适应学习伴学系统

> **作品名称**：智径 AdaptivePath
> **参赛赛道**：科大讯飞 AI 开发者大赛 · 自适应学习路径决策与伴学智能体
> **技术亮点**：多智能体协同（Multi-Agent Orchestration）+ 认知诊断三模型融合（DKT × BKT × IRT）+ 上下文赌博机（Contextual Bandit）路径优化 + 苏格拉底式支架教学 + 情感计算
> **面向学科**：人工智能专业导论（可扩展到电子信息、计算机、数学等 STEM 学科）

---

## 一、问题洞察：为什么传统做法不够

在"千人一面"向"千人千面"转型的教育大模型浪潮中，绝大多数参赛作品存在以下三类典型问题：

| 常见做法 | 局限 | 智径的做法 |
| --- | --- | --- |
| 单 LLM 套壳：把 ChatGPT 套一个 UI 就叫"自适应" | 没有"学生模型"，只是被动答疑；缺乏诊断与规划能力 | 五智能体协同 + 学生认知状态持续追踪 |
| 静态知识图谱 + 固定推荐顺序 | 不考虑学习者当前掌握度、兴趣、情绪；无法应对"卡壳—突破"的实时变化 | Contextual Bandit 在线学习 + MDP 规划 |
| 直接给答案或长篇讲解 | 违背 ZPD（最近发展区）原则；学生不会迁移 | 苏格拉底式反诘 + 五级动态脚手架（提示→范例→类比→分解→纠错） |

智径把教育认知科学（IRT、BKT、DKT、Vygotsky ZPD、Bloom 分类）与前沿 AI 技术（LLM Agent、CoT/ReAct、Contextual Bandit、Affective Computing）做了**工程级深度融合**，而不是堆叠名词。

---

## 二、系统架构总览

```
                   ┌───────────────────────────────────────────┐
                   │        Orchestrator Agent（主控调度）       │
                   │   · 任务分发 · 智能体协商 · 全局记忆        │
                   └─────┬─────────┬──────────┬─────────┬──────┘
                         │         │          │         │
              ┌──────────▼─┐ ┌─────▼─────┐ ┌──▼─────┐ ┌──▼──────┐
              │ Diagnostic │ │ Planning  │ │Teaching│ │Reflection│
              │   Agent    │ │   Agent   │ │ Agent  │ │  Agent   │
              │  (学情诊断) │ │ (路径规划) │ │(苏格拉底)│ │ (反思)   │
              └──────┬─────┘ └─────┬─────┘ └───┬────┘ └────┬─────┘
                     │             │            │           │
                     └─────┬───────┴──────┬─────┘           │
                           ▼              ▼                 │
                  ┌─────────────────────────────┐           │
                  │   Cognitive State Network   │◀──────────┘
                  │  (DKT + BKT + IRT 融合)     │
                  └─────────────────────────────┘
                                ▲
                                │
                  ┌─────────────┴───────────────┐
                  │  Emotion & Engagement Net   │
                  │  (情感 + 参与度实时评估)     │
                  └─────────────────────────────┘
                                ▲
                                │
                  ┌─────────────┴───────────────┐
                  │   Knowledge Graph + RAG     │
                  │  (学科知识图谱 + 检索增强)   │
                  └─────────────────────────────┘
```

五大智能体：
1. **Diagnostic Agent** —— 通过对话与微测试持续推断学生认知状态
2. **Planning Agent** —— 基于学生状态 + 知识图谱 + Contextual Bandit 动态生成路径
3. **Teaching Agent** —— 苏格拉底式反诘 + 五级脚手架
4. **Reflection Agent** —— 课后反思 + 元认知培养
5. **Emotional Agent** —— 实时情感/参与度建模，反向影响教学节奏

底层支撑：
- **Cognitive State Network (CSN)**：DKT 长程记忆 + BKT 概率推断 + IRT 能力估计三模型融合
- **Knowledge Graph**：基于 AI 学科导论的精细化知识图谱（含先决关系、难度、典型错误）
- **Long-term Memory**：分层记忆（感知/工作/情景/语义）+ 向量化检索

---

## 三、核心创新点

### 创新 1：认知状态网络（CSN）—— 三模型融合
- **DKT（Deep Knowledge Tracing）**：用 LSTM 建模学生跨知识点长程记忆
- **BKT（Bayesian Knowledge Tracing）**：4 状态 HMM 显式建模"已掌握/未掌握/学习/猜测"
- **IRT（Item Response Theory）**：2PL 模型估计题目区分度与学生能力 θ
- 三个模型通过加权贝叶斯融合 + 不确定性估计，给出"学生此刻对每个知识点的真实掌握度 + 置信区间"

### 创新 2：Contextual Bandit 路径优化
- 把"下一步推荐什么"建模为 Contextual Bandit 问题
- 上下文 = 学生当前状态向量 + 知识点特征 + 教学历史
- 动作 = 候选知识点/学习活动
- 奖励 = 短中期学习收益（掌握度提升 + 参与度 + 情绪改善）
- 用 LinUCB 算法在线学习，无需预先训练数据

### 创新 3：苏格拉底式支架教学
- 基于 Bloom 认知分类，动态选择教学策略（记忆/理解/应用/分析/评价/创造）
- 五级脚手架：轻提示 → 关键概念 → 类比 → 分步分解 → 完整示范
- 永远不直接给答案，但会"渐进逼近"

### 创新 4：情感-认知双通道反馈
- 从学生输入文本实时推断困惑/挫败/兴趣/疲劳
- 当检测到负面情绪，自动降低难度、切换教学风格、增加鼓励
- 情感状态进入 CSN 作为额外特征

### 创新 5：可解释性
- 每一次推荐、每一次干预，都给出"为什么这样做"的理由
- 路径可视化、学习轨迹回放

---

## 四、目录结构

```
智径-AdaptivePath/
├── README.md                          # 本文件
├── requirements.txt                   # 依赖
├── LICENSE                            # 许可证
├── src/                               # 源代码
│   ├── main.py                        # 入口
│   ├── config.py                      # 配置
│   ├── agents/                        # 五智能体
│   ├── cognitive/                     # 认知诊断三模型
│   ├── knowledge/                     # 知识图谱
│   ├── planning/                      # 路径规划（Bandit/MDP）
│   ├── dialogue/                      # 苏格拉底对话引擎
│   ├── memory/                        # 长期记忆
│   ├── emotion/                       # 情感分析
│   └── utils/                         # 工具
├── data/                              # 数据
│   ├── knowledge_graph/               # 知识图谱数据
│   ├── questions/                     # 题库
│   └── students/                      # 学生画像示例
├── demo/                              # Web Demo
├── docs/                              # 架构文档
├── tests/                             # 测试
├── 商业计划书/                         # 提交材料
└── 演示视频脚本/                       # 视频脚本
```

---

## 五、快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行核心模块单元测试
python -m pytest tests/ -v

# 3. 启动 Web Demo
streamlit run demo/app.py

# 4. 命令行体验
python src/main.py --mode cli
```

---

## 六、参赛作品提交清单

| 类型 | 文件 | 位置 |
| --- | --- | --- |
| 商业计划书 | 智径-AdaptivePath-商业计划书.pdf | `商业计划书/` |
| 源代码 | 完整项目 | 根目录 |
| 演示 Demo | Streamlit Web App | `demo/app.py` |
| 演示视频脚本 | 3 分钟路演脚本 | `演示视频脚本/` |
| 架构文档 | 系统设计文档 | `docs/architecture.md` |

---

## 七、团队

- 团队名称：崇理团队
- 主负责人：蒋泽宇
- 核心成员：汪枳航、谷昊洋、王一维、孙布赫
- 指导教师：崇理团队导师组
- 学校：高校联合实验室

## 八、致谢

感谢科大讯飞提供本赛事平台。本项目核心教育认知理论参考：
- Vygotsky, L. S. (1978). *Mind in Society*
- Corbett, A. T., & Anderson, J. R. (1995). *Knowledge tracing: Modeling the acquisition of procedural knowledge*
- Piech, C., et al. (2015). *Deep Knowledge Tracing*
- Baker, R. S., et al. (2010). *Better to Use Simple Affect Detection*
- Bloom, B. S. (1956). *Taxonomy of Educational Objectives*
