"""
业务引擎层：高频 RAG 吸收、（预留）低频反思 Agent。

功能：承载「检索 + 路由」等与治理域强相关的编排；依赖 core 基础设施。
数据流：爬虫抽取 dict → engine.rag_ingestion → UI/core.db 入库。
"""
