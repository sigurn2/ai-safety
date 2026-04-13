"""
大语言模型与向量服务的抽象端口（Protocol）。

功能：把「HTTP + OpenAI 兼容 JSON 契约」与上层 RAG/路由解耦，便于单测打桩或替换厂商实现。
输入：由调用方传入 messages、文本列表、可选 model/超时等。
输出：向量列表或解析后的 JSON 对象；不隐式读写业务库。
上下游：被 core.llm_client 中的具体实现满足；由 engine.rag_ingestion 通过可选依赖注入使用。
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol


class LlmBackend(Protocol):
    """
    统一的 LLM 后端窄接口（嵌入 + 结构化对话）。

    演进说明：新增能力时优先加默认方法或新 Protocol 组合，避免破坏现有实现。
    """

    def embed_texts(
        self,
        texts: List[str],
        *,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ) -> List[List[float]]:
        """
        功能：批量文本转向量（OpenAI 兼容 /embeddings）。
        输入：非空字符串列表；model 缺省则用配置默认嵌入模型。
        输出：与输入等长的 float 向量列表；副作用为对外 HTTP。
        """

    def chat_completion_json(
        self,
        messages: List[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        timeout: float = 120.0,
    ) -> Any:
        """
        功能：聊天补全并解析助手回复为 JSON（可处理常见 Markdown 代码块包裹）。
        输入：OpenAI 风格 messages；temperature 建议分类任务偏低。
        输出：json.loads 后的 Python 对象；副作用为对外 HTTP。
        """
