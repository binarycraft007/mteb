from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np

from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts
from mteb.requires_package import requires_package
from mteb.overview import get_task

logger = logging.getLogger(__name__)

TASK_TYPES = {
    "STS": "SEMANTIC_SIMILARITY",
    "Summarization": "SEMANTIC_SIMILARITY",
    "BitextMining": "SEMANTIC_SIMILARITY",
    "Classification": "CLASSIFICATION",
    "Clustering": "CLUSTERING",
    "Reranking": "RETRIEVAL_QUERY",
    "Retrieval": "RETRIEVAL_QUERY",
    "InstructionRetrieval": "RETRIEVAL_QUERY",
    "PairClassification": "SEMANTIC_SIMILARITY",
}

class GoogleWrapper:
    def __init__(self, model_name: str, embed_dim: int | None = None, **kwargs) -> None:
        # requires_package(self, "google-cloud-aiplatform", "Google text embedding")
        from vertexai.preview.language_models import TextEmbeddingModel

        self._client = TextEmbeddingModel.from_pretrained(model_name)
        self._model_name = model_name
        self._embed_dim = embed_dim

    def encode(self, sentences: list[str], prompt_name: str | None = None, **kwargs: Any) -> np.ndarray:
        # requires_package(self, "google-cloud-aiplatform", "Google text embedding")
        from vertexai.preview.language_models import TextEmbeddingInput

        kwargs = (
            dict(output_dimensionality=self._embed_dim)
            if self._embed_dim
            else {}
        )

        task_type = None
        if prompt_name is not None:
            meta = get_task(prompt_name).metadata
            task_type = TASK_TYPES.get(meta.type, "")

        inputs = [TextEmbeddingInput(text, task_type) for text in sentences]

        return self._to_numpy(self._client.get_embeddings(inputs, **kwargs))

    def encode_queries(self, queries: list[str], prompt_name: str | None = None, **kwargs: Any) -> np.ndarray:
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list[str]], prompt_name: str | None = None, **kwargs: Any
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus)
        return self.encode(sentences, **kwargs)

    def _to_numpy(self, embeddings) -> np.ndarray:
        return np.array([embedding.values for embedding in embeddings])


text_embedding_004= ModelMeta(
    name="text-embedding-004",
    revision="1",
    release_date="2024-05-14",
    languages=None,  # supported languages not specified
    loader=partial(GoogleWrapper, model_name="text-embedding-004"),
    max_tokens=8191,
    embed_dim=1536,
    open_source=False,
)
text_multilingual_embedding_002= ModelMeta(
    name="text-multilingual-embedding-002",
    revision="1",
    release_date="2024-05-14",
    languages=None,  # supported languages not specified
    loader=partial(GoogleWrapper, model_name="text-multilingual-embedding-002"),
    max_tokens=8191,
    embed_dim=3072,
    open_source=False,
)
