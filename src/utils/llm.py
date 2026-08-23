"""
LLM 工具封装
==============

支持多种 LLM provider：
- OpenAI 兼容（GPT-4o / Qwen / DeepSeek / 智谱 / 讯飞星火）
- 本地 Ollama
- 简单的 Mock LLM（用于测试）

接口：
  llm = get_llm(provider="openai", model="gpt-4o-mini")
  response = llm.chat("你是谁？")
"""
from __future__ import annotations
from typing import List, Dict, Optional, Iterator
import os
import json


class BaseLLM:
    """LLM 基类"""
    def chat(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError

    def stream(self, prompt: str, system: Optional[str] = None, **kwargs) -> Iterator[str]:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """测试用 Mock LLM"""
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or [
            "这是一个 Mock 回复。",
            "好的，让我用苏格拉底式反诘帮助你思考。",
            "你能先告诉我你的想法吗？",
        ]
        self.idx = 0

    def chat(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        r = self.responses[self.idx % len(self.responses)]
        self.idx += 1
        return r

    def stream(self, prompt: str, system: Optional[str] = None, **kwargs) -> Iterator[str]:
        for ch in self.chat(prompt, system):
            yield ch


class OpenAILLM(BaseLLM):
    """OpenAI 兼容 LLM（支持 GPT / Qwen / DeepSeek / 智谱 / 讯飞星火）"""
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = OpenAI(**kwargs)
            self.available = True
        except ImportError:
            self.available = False
            self.client = None

    def chat(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        if not self.available:
            return "[OpenAI client not available. Please install openai package or set LLM_API_KEY.]"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM error: {e}]"

    def stream(self, prompt: str, system: Optional[str] = None, **kwargs) -> Iterator[str]:
        if not self.available:
            yield "[OpenAI client not available]"
            return
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"[LLM error: {e}]"


def get_llm(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    **kwargs,
) -> BaseLLM:
    """工厂函数"""
    if provider == "mock":
        return MockLLM()
    elif provider in ("openai", "qwen", "deepseek", "zhipu", "spark"):
        return OpenAILLM(model=model, **kwargs)
    else:
        # 默认 mock
        return MockLLM()


if __name__ == "__main__":
    llm = get_llm(provider="mock")
    print(llm.chat("你好"))
