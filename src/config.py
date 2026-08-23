"""
全局配置
==================

通过环境变量或 .env 文件覆盖默认值。
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KG_DIR = DATA_DIR / "knowledge_graph"
QUESTION_DIR = DATA_DIR / "questions"
STUDENT_DIR = DATA_DIR / "students"
DOCS_DIR = PROJECT_ROOT / "docs"


@dataclass
class LLMConfig:
    """LLM 配置
    优先使用 OpenAI 兼容协议；如使用本地 LLM（Ollama / vLLM / 讯飞星火等），
    只需设置 base_url 与 api_key 即可。
    """
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY", "sk-placeholder"))
    temperature: float = 0.4
    max_tokens: int = 2048
    timeout: int = 60


@dataclass
class CognitiveConfig:
    """认知诊断模型超参数"""
    # BKT
    bkt_p_init: float = 0.1          # 初始掌握概率
    bkt_p_learn: float = 0.3         # 学习转移概率
    bkt_p_slip: float = 0.1          # 失误概率
    bkt_p_guess: float = 0.2         # 猜测概率

    # DKT
    dkt_hidden_size: int = 64
    dkt_num_layers: int = 1
    dkt_learning_rate: float = 1e-3
    dkt_epochs: int = 20
    dkt_batch_size: int = 32

    # IRT
    irt_difficulty_range: tuple = (-3.0, 3.0)
    irt_discrimination_range: tuple = (0.3, 2.5)

    # 融合权重（可被 EM 在线更新）
    fusion_w_dkt: float = 0.4
    fusion_w_bkt: float = 0.3
    fusion_w_irt: float = 0.3


@dataclass
class PlanningConfig:
    """路径规划超参数"""
    # Contextual Bandit (LinUCB)
    linucb_alpha: float = 0.5         # 探索系数
    linucb_d: int = 32                # 上下文维度
    linucb_lambda_reg: float = 1.0    # L2 正则

    # 路径规划
    short_term_horizon: int = 5       # 短期规划步长
    mastery_threshold: float = 0.85   # 掌握阈值
    struggle_threshold: float = 0.4   # 困难阈值
    fatigue_threshold: float = 0.7    # 疲劳阈值


@dataclass
class DialogueConfig:
    """对话引擎配置"""
    # 苏格拉底反诘深度
    max_probing_depth: int = 4
    # 脚手架级别（0-4）
    scaffolding_levels: int = 5
    # 何时直接给答案
    direct_answer_threshold: float = 0.15
    # 反思回合
    reflection_every_n_turns: int = 5


@dataclass
class EmotionConfig:
    """情感识别配置"""
    # 困惑 / 挫败 / 兴趣 / 疲劳 / 专注
    sentiment_lexicon_path: str = str(DATA_DIR / "sentiment_lexicon_cn.json")
    # 困惑阈值
    confusion_threshold: float = 0.6
    # 挫败阈值
    frustration_threshold: float = 0.7
    # 疲劳阈值（基于交互节奏）
    fatigue_turn_window: int = 10


@dataclass
class AppConfig:
    """顶层配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    log_level: str = "INFO"
    debug_mode: bool = False


def get_config() -> AppConfig:
    """获取单例配置（生产环境可接入配置中心）"""
    return AppConfig()


if __name__ == "__main__":
    # 简单自检
    cfg = get_config()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"LLM provider: {cfg.llm.provider}, model: {cfg.llm.model}")
    print(f"BKT init: p_init={cfg.cognitive.bkt_p_init}, p_learn={cfg.cognitive.bkt_p_learn}")
    print(f"LinUCB alpha: {cfg.planning.linucb_alpha}")
