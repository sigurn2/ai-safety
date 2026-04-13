"""
基础设施层：配置、数据库、LLM HTTP 端口与默认实现。

功能：为 crawler、engine、UI 提供无业务语义的配置与 IO 能力。
扩展：替换厂商时优先实现 core.llm_ports.LlmBackend 并注入 RAG 流水线。
"""
