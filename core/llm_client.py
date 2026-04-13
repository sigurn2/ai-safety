"""
OpenAI 兼容 HTTP 客户端（聊天 + 嵌入）。

功能：封装 BASE_URL 下的 /chat/completions 与 /embeddings，供 RAG 检索与路由、以及其它调用方复用。
输入：环境变量或构造 OpenAICompatibleBackend 时传入的 api_key/base_url；请求级可覆盖 model、超时。
输出：纯数据（文本或向量、JSON 对象）；不访问业务 SQLite。
上下游：默认实现满足 core.llm_ports.LlmBackend；engine.rag_ingestion 可注入自定义 Backend 做测试或换厂商。
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import httpx

from core.config import API_KEY as _DEFAULT_KEY
from core.config import BASE_URL as _DEFAULT_BASE
from core.config import EMBEDDING_MODEL as _DEFAULT_EMBED_MODEL
from core.config import LLM_MODEL as _DEFAULT_CHAT_MODEL


class OpenAICompatibleBackend:
    """
    基于 httpx 的 OpenAI 兼容后端（可同时承担嵌入与 JSON 对话）。

    功能：将密钥与 base_url 实例化，避免在业务代码里散落全局环境读取（便于测试注入）。
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_chat_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
    ) -> None:
        """
        功能：绑定一次调用所需的鉴权与端点。
        输入：缺省字段回退到 core.config 中的环境变量解析结果。
        输出：无；副作用：无 IO。
        """
        self._api_key = (api_key if api_key is not None else _DEFAULT_KEY) or ""
        self._base_url = (base_url if base_url is not None else _DEFAULT_BASE).rstrip("/")
        self._default_chat_model = default_chat_model or _DEFAULT_CHAT_MODEL
        self._default_embedding_model = default_embedding_model or _DEFAULT_EMBED_MODEL

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def embed_texts(
        self,
        texts: List[str],
        *,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ) -> List[List[float]]:
        """
        功能：POST /embeddings，按 index 排序对齐返回向量。
        输入：字符串列表；model 默认实例上的 default_embedding_model。
        输出：List[List[float]]；副作用：同步 HTTP。
        上下游：供 engine.rag_ingestion.embedder 调用。
        """
        if not self._api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        m = model or self._default_embedding_model
        url = f"{self._base_url}/embeddings"
        payload = {"model": m, "input": texts}
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, headers=self._headers(), json=payload)
            r.raise_for_status()
            data = r.json()
        out: List[List[float]] = []
        for item in sorted(data.get("data", []), key=lambda x: x.get("index", 0)):
            vec = item.get("embedding")
            if not isinstance(vec, list):
                raise ValueError("embedding API returned unexpected payload")
            out.append([float(x) for x in vec])
        if len(out) != len(texts):
            raise ValueError("embedding count mismatch")
        return out

    def embed_text(self, text: str, *, model: Optional[str] = None, timeout: float = 60.0) -> List[float]:
        """功能：单条文本嵌入。输入：一段文本。输出：一维向量。"""
        return self.embed_texts([text], model=model, timeout=timeout)[0]

    def chat_completion(
        self,
        messages: List[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: float = 120.0,
    ) -> str:
        """
        功能：POST /chat/completions，取第一条 choice 的文本内容。
        输入：OpenAI 风格 messages。
        输出：助手纯文本；副作用：同步 HTTP。
        """
        if not self._api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        m = model or self._default_chat_model
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": m,
            "messages": messages,
            "temperature": temperature,
        }
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, headers=self._headers(), json=payload)
            r.raise_for_status()
            data = r.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"unexpected chat completion payload: {data!r}") from e

    def chat_completion_json(
        self,
        messages: List[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        timeout: float = 120.0,
    ) -> Any:
        """
        功能：聊天补全后将助手回复解析为 JSON（剥离 ``` / ```json 围栏）。
        输入：与 chat_completion 相同。
        输出：Python 对象（通常为 dict）；解析失败抛 JSON 相关异常。
        上下游：供 engine.rag_ingestion.router 做结构化分类。
        """
        raw = self.chat_completion(messages, model=model, temperature=temperature, timeout=timeout).strip()
        if raw.startswith("```"):
            raw = raw[3:].lstrip()
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()
            end = raw.rfind("```")
            if end != -1:
                raw = raw[:end]
        return json.loads(raw.strip())


_default_backend: Optional[OpenAICompatibleBackend] = None


def default_llm_backend() -> OpenAICompatibleBackend:
    """
    功能：进程内懒加载默认后端（读环境变量）。
    输入：无。
    输出：单例 OpenAICompatibleBackend；便于未注入场景下的模块级函数。
    """
    global _default_backend
    if _default_backend is None:
        _default_backend = OpenAICompatibleBackend()
    return _default_backend


def embed_texts(texts: List[str], *, model: Optional[str] = None, timeout: float = 60.0) -> List[List[float]]:
    """功能：使用默认后端的批量嵌入（兼容旧调用点）。"""
    return default_llm_backend().embed_texts(texts, model=model, timeout=timeout)


def embed_text(text: str, *, model: Optional[str] = None, timeout: float = 60.0) -> List[float]:
    """功能：使用默认后端的单条嵌入。"""
    return default_llm_backend().embed_text(text, model=model, timeout=timeout)


def chat_completion(
    messages: List[dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> str:
    """功能：使用默认后端的纯文本补全。"""
    return default_llm_backend().chat_completion(
        messages, model=model, temperature=temperature, timeout=timeout
    )


def chat_completion_json(
    messages: List[dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.1,
    timeout: float = 120.0,
) -> Any:
    """功能：使用默认后端的 JSON 补全。"""
    return default_llm_backend().chat_completion_json(
        messages, model=model, temperature=temperature, timeout=timeout
    )
