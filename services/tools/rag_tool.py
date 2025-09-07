import logging
from typing import Any, Dict, List

from .base import BaseTool, Decision, ToolResult


logger = logging.getLogger(__name__)


class RagTool:
    name = "rag"
    priority = 10

    def __init__(self, embedding_model, vector_store, generate_response_fn):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.generate_response_fn = generate_response_fn
        self.config: Dict[str, Any] = {
            'thresholds': {
                'accept': 0.0  # Siempre candidato como último recurso
            }
        }

    def configure(self, config: Dict[str, Any]) -> None:
        if not config:
            return
        self.config.update(config)

    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision:
        # RAG siempre es posible; score bajo por defecto para que otros ganen si aplican
        return Decision(score=0.4, params={}, reasons=["rag_baseline"])

    def accepts(self, score: float) -> bool:
        accept = float(self.config.get('thresholds', {}).get('accept', 0.0))
        return score >= accept

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        q = query.strip()
        if not q:
            return ToolResult(response="", sources=[], metadata={})
        query_embedding = self.embedding_model.encode([q])[0]
        k = int(self.config.get('k', 5))
        results = self.vector_store.search(query_embedding, k=k)
        sources = [r.get('filename', '') for r in results if r.get('filename')]
        ctx = '\n\n'.join(r.get('text', '') for r in results)
        if not ctx.strip():
            return ToolResult(response="", sources=[], metadata={'relevant_chunks': results})
        text = self.generate_response_fn(query, ctx, sources)
        return ToolResult(response=text, sources=sources, metadata={'relevant_chunks': results})


