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
        if model_name.startswith("text-embedding"):
            self._init_text_embedding_client(model_name)
        else:
            self._init_multimodelembedding_client(model_name)

        self._model_name = model_name
        self._embed_dim = embed_dim

    def _init_text_embedding_client(self, model_name: str) -> None:
        from vertexai.language_models import TextEmbeddingModel
        self._client = TextEmbeddingModel.from_pretrained(model_name)

    def _init_multimodelembedding_client(self, model_name: str) -> None:
        from vertexai.vision_models import MultiModalEmbeddingModel
        self._client = MultiModalEmbeddingModel.from_pretrained(model_name)

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 250,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if self._model_name.startswith("text-embedding"):
            return self._encode_text_embedding(
                sentences, batch_size=batch_size, prompt_name=prompt_name, **kwargs,
            )
        else:
            return self._encode_multimodelembedding(
                sentences, batch_size=batch_size, prompt_name=prompt_name, **kwargs,
            )

    def _encode_text_embedding(
        self,
        sentences: list[str],
        *,
        batch_size: int = 250,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        from vertexai.language_models import TextEmbeddingInput

        kwargs = (
            dict(output_dimensionality=self._embed_dim)
            if self._embed_dim
            else {}
        )
        kwargs["auto_truncate"] = False

        task_type = None
        if prompt_name is not None:
            meta = get_task(prompt_name).metadata
            task_type = TASK_TYPES.get(meta.type, "")

        embeddings = []
        for batch in range(0, len(sentences), batch_size):
            text_batch = sentences[batch : batch + batch_size]
            inputs = [TextEmbeddingInput(text, task_type) for text in text_batch]
            embeddings_batch = self._client.get_embeddings(inputs, **kwargs)
            embeddings.extend([el.values for el in embeddings_batch])

        return np.array(embeddings)

    def _encode_multimodelembedding(
        self,
        sentences: list[str],
        *,
        batch_size: int = 1,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        dimension = None
        if self._embed_dim is not None:
            if self._embed_dim <= 128:
                dimension = 128
            elif self._embed_dim <= 256:
                dimension = 256
            elif self._embed_dim <= 512:
                dimension = 512
            else:
                dimension = 1408

        embeddings = []
        for text in sentences:
            embeddings_single = self._client.get_embeddings(
                contextual_text=text,
                dimension=dimension,
            )
            embeddings.extend(embeddings_single.textEmbedding)

        return np.array(embeddings)

    def encode_queries(
        self,
        queries: list[str], 
        batch_size: int = 250,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.encode(
            queries, batch_size=batch_size, prompt_name=prompt_name, **kwargs,
        )

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]],
        batch_size: int = 250,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus)
        return self.encode(
            sentences, batch_size=batch_size, prompt_name=prompt_name, **kwargs,
        )

text_embedding_004= ModelMeta(
    name="text-embedding-004",
    revision="1",
    release_date="2024-05-14",
    languages=["eng_Latn"],  # supported languages not specified
    loader=partial(GoogleWrapper, model_name="text-embedding-004"),
    max_tokens=20000,
    embed_dim=256,
    open_source=False,
)
text_multilingual_embedding_002= ModelMeta(
    name="text-multilingual-embedding-002",
    revision="1",
    release_date="2024-05-14",
    languages=["eng_Latn"],  # supported languages not specified
    loader=partial(GoogleWrapper, model_name="text-multilingual-embedding-002"),
    max_tokens=20000,
    embed_dim=256,
    open_source=False,
)
multimodalembedding_001= ModelMeta(
    name="multimodalembedding@001",
    revision="1",
    release_date="2024-02-07",
    languages=["eng_Latn"],  # supported languages not specified
    loader=partial(GoogleWrapper, model_name="multimodalembedding@001"),
    max_tokens=20000,
    embed_dim=256,
    open_source=False,
)
