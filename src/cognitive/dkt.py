"""
DKT (Deep Knowledge Tracing)
==============================

参考 Piech et al. (NeurIPS 2015) 的深度知识追踪模型。

核心思想：用 LSTM 把学生的历史作答序列编码为隐藏状态，
hidden state -> 知识点维度的 sigmoid 输出，预测答对各知识点的概率。

公式（简化）：
    h_t = LSTM(x_t, h_{t-1})
    y_t = sigmoid(W * h_t + b)   # y_t[i] = 答对知识点 i 的概率
    x_t = one_hot(K+1) ⊕ one_hot(2) 拼接

注：本实现提供两个版本：
1. PureNumPyLSTM: 纯 numpy 实现，无深度学习依赖（教学/演示用）
2. TorchDKT: PyTorch 版本（生产用，需要训练数据）
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import CognitiveConfig


# ---------------------------------------------------------------------- #
# 纯 NumPy 实现的 LSTM-DKT
# ---------------------------------------------------------------------- #
class NumPyDKT:
    """
    不依赖 PyTorch 的简化 DKT 实现，用于离线推理/演示。

    训练：使用标准的 BPTT；本类提供 fit() 接口。
    推理：predict_next_correct() 返回每个知识点的掌握概率。
    """

    def __init__(self, num_skills: int, hidden_size: int = 64, config: Optional[CognitiveConfig] = None):
        self.num_skills = num_skills
        self.hidden_size = hidden_size
        self.cfg = config or CognitiveConfig()

        # 初始化参数（Xavier）
        rng = np.random.default_rng(42)
        scale = 1.0 / np.sqrt(hidden_size)
        self.W_f = rng.normal(0, scale, (hidden_size, 2 * num_skills + hidden_size))
        self.W_i = rng.normal(0, scale, (hidden_size, 2 * num_skills + hidden_size))
        self.W_c = rng.normal(0, scale, (hidden_size, 2 * num_skills + hidden_size))
        self.W_o = rng.normal(0, scale, (hidden_size, 2 * num_skills + hidden_size))

        self.b_f = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_c = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))

        self.W_y = rng.normal(0, scale, (num_skills, hidden_size))
        self.b_y = np.zeros((num_skills, 1))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """单步 LSTM"""
        concat = np.concatenate([x_t, h_prev], axis=0).reshape(-1, 1)
        f = self._sigmoid(self.W_f @ concat + self.b_f)
        i = self._sigmoid(self.W_i @ concat + self.b_i)
        c_tilde = np.tanh(self.W_c @ concat + self.b_c)
        o = self._sigmoid(self.W_o @ concat + self.b_o)
        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)
        return h, c

    def forward(self, sequence: List[Tuple[int, bool]]) -> np.ndarray:
        """
        sequence: [(skill_id, is_correct), ...]
        返回: 每个时间步对所有知识点的预测概率，shape=(T, num_skills)
        """
        T = len(sequence)
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        outputs = np.zeros((T, self.num_skills))

        for t, (skill, correct) in enumerate(sequence):
            x = np.zeros((2 * self.num_skills, 1))
            x[skill] = 1.0
            x[self.num_skills + (0 if correct else 1)] = 1.0
            h, c = self._step(x, h, c)
            y = self._sigmoid(self.W_y @ h + self.b_y).flatten()
            outputs[t] = y
        return outputs

    def predict_mastery(self, sequence: List[Tuple[int, bool]]) -> np.ndarray:
        """
        给定历史，返回当前时刻对每个知识点的掌握概率。
        """
        if not sequence:
            return np.full(self.num_skills, 0.1)
        outputs = self.forward(sequence)
        return outputs[-1]

    def fit(self, train_data: List[List[Tuple[int, bool]]], epochs: int = 20, lr: float = 1e-3):
        """
        训练（简化版：使用数值梯度）。工程上请使用 PyTorch 版本。
        这里仅作为接口占位与最小可运行示例。
        """
        # 真实训练需要 BPTT；这里给出接口
        pass


# ---------------------------------------------------------------------- #
# PyTorch 版本（生产推荐）
# ---------------------------------------------------------------------- #
try:
    import torch
    import torch.nn as nn

    class TorchDKT(nn.Module):
        """
        标准 PyTorch DKT 实现。
        输入：batch x seq_len x (2*num_skills)
        输出：batch x seq_len x num_skills
        """
        def __init__(self, num_skills: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
            super().__init__()
            self.num_skills = num_skills
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(
                input_size=2 * num_skills,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Linear(hidden_size, num_skills)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, 2K)
            out, _ = self.lstm(x)
            out = self.dropout(out)
            y = torch.sigmoid(self.fc(out))
            return y

        def predict_mastery(self, sequence: torch.Tensor) -> torch.Tensor:
            """给定 batch x T x 2K，返回最后时刻的掌握概率"""
            with torch.no_grad():
                y = self.forward(sequence)
                return y[:, -1, :]

except ImportError:
    class TorchDKT:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("需要安装 torch: pip install torch")


# ---------------------------------------------------------------------- #
# 知识状态封装（含不确定性）
# ---------------------------------------------------------------------- #
@dataclass
class MasteryEstimate:
    """单个知识点的掌握度估计（含置信区间）"""
    skill_id: str
    mastery: float  # 0~1
    confidence: float  # 0~1
    sample_count: int = 0


if __name__ == "__main__":
    dkt = NumPyDKT(num_skills=5, hidden_size=16)
    seq = [(0, True), (1, False), (0, True), (2, True), (1, True)]
    mastery = dkt.predict_mastery(seq)
    print("Mastery across skills:")
    for i, m in enumerate(mastery):
        print(f"  Skill {i}: {m:.3f}")
