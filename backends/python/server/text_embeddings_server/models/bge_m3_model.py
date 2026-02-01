"""BGE-M3 model implementation - OPTIMIZED v2.

Key optimization: Direct forward pass using pre-tokenized input_ids.
Bypasses FlagEmbedding's encode() which does decode → re-tokenize.

Performance improvement: ~10x faster for batch processing.
"""

import os
import logging
from collections import defaultdict

os.environ["TQDM_DISABLE"] = "1"

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Type, List, Dict, Any
from opentelemetry import trace

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding
from text_embeddings_server.pb import embed_pb2
from text_embeddings_server.utils.bge_m3_downloader import setup_bge_m3_sparse

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


class BGEM3Model(Model):
    """
    Optimized BGE-M3 model - directly uses pre-tokenized input_ids.
    """

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding is required for BGE-M3. " "pip install FlagEmbedding"
            )

        logger.info(f"[OPTIMIZED v2] Initializing BGE-M3 model from: {model_path}")

        # Download sparse files if needed
        if setup_bge_m3_sparse(str(model_path)):
            logger.info("Sparse embedding files ready")

        use_fp16 = dtype in [torch.float16, torch.bfloat16]

        # Load the model
        self.flag_model = BGEM3FlagModel(
            str(model_path),
            use_fp16=use_fp16,
            device=str(device),
        )

        # Direct access to internal model for optimized forward
        self._model = self.flag_model.model
        self._model.eval()

        # Cache tokenizer special tokens for filtering
        tokenizer = self.flag_model.tokenizer
        self._unused_tokens = set()
        for attr in ["cls_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                self._unused_tokens.add(tid)

        self.pool = pool
        self.hidden_size = 1024
        self.max_input_length = 8192
        self.device = device
        self.dtype = dtype
        self._vocab_size = len(tokenizer)

        logger.info(f"[OPTIMIZED v2] Ready - device={device}, vocab={self._vocab_size}")

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    def _prepare_inputs(self, batch: PaddedBatch) -> Dict[str, torch.Tensor]:
        """Convert PaddedBatch to model inputs - NO decode/re-tokenize!"""
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)

        if hasattr(batch, "token_type_ids") and batch.token_type_ids is not None:
            token_type_ids = batch.token_type_ids.to(self.device)
        else:
            token_type_ids = torch.zeros_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @torch.no_grad()
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        """Generate dense embeddings."""
        inputs = self._prepare_inputs(batch)

        outputs = self._model(
            inputs,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        dense_vecs = outputs["dense_vecs"]
        dense_vecs = F.normalize(dense_vecs, p=2, dim=-1)

        results = []
        for i in range(len(batch)):
            results.append(Embedding(values=dense_vecs[i].cpu().tolist()))
        return results

    @torch.no_grad()
    def embed_sparse_native(
        self, batch: PaddedBatch
    ) -> List[embed_pb2.SparseEmbedding]:
        """
        Generate sparse embeddings - OPTIMIZED.

        Directly uses input_ids, bypassing FlagEmbedding's encode().
        Returns native sparse format (only non-zero values).
        """
        inputs = self._prepare_inputs(batch)

        # Get token weights from model
        outputs = self._model(
            inputs,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
            return_sparse_embedding=False,  # Get raw token weights
        )

        token_weights = outputs["sparse_vecs"]  # [batch, seq_len, 1]
        if token_weights.dim() == 3:
            token_weights = token_weights.squeeze(-1)  # [batch, seq_len]

        # Apply ReLU to ensure non-negative
        token_weights = torch.relu(token_weights)

        input_ids = inputs["input_ids"]
        batch_size = input_ids.size(0)

        results = []
        for i in range(batch_size):
            weights = token_weights[i].cpu().tolist()
            ids = input_ids[i].cpu().tolist()

            # Aggregate: max weight per token_id
            lexical = defaultdict(float)
            for w, tid in zip(weights, ids):
                if w > 0 and tid not in self._unused_tokens:
                    if w > lexical[tid]:
                        lexical[tid] = w

            sparse_values = [
                embed_pb2.SparseValue(index=int(tid), value=float(w))
                for tid, w in lexical.items()
            ]
            results.append(embed_pb2.SparseEmbedding(values=sparse_values))

        return results

    @torch.no_grad()
    def embed_sparse(self, batch: PaddedBatch) -> List[Embedding]:
        """Generate sparse embeddings (dense format for legacy)."""
        inputs = self._prepare_inputs(batch)

        outputs = self._model(
            inputs,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
            return_sparse_embedding=False,
        )

        token_weights = outputs["sparse_vecs"]
        if token_weights.dim() == 3:
            token_weights = token_weights.squeeze(-1)

        token_weights = torch.relu(token_weights)
        input_ids = inputs["input_ids"]
        batch_size = input_ids.size(0)
        vocab_size = self._vocab_size

        results = []
        for i in range(batch_size):
            weights = token_weights[i].cpu().tolist()
            ids = input_ids[i].cpu().tolist()

            sparse_vec = [0.0] * vocab_size
            for w, tid in zip(weights, ids):
                if w > 0 and tid not in self._unused_tokens:
                    tid = int(tid)
                    if 0 <= tid < vocab_size and w > sparse_vec[tid]:
                        sparse_vec[tid] = w

            results.append(Embedding(values=sparse_vec))

        return results

    def predict(self, batch: PaddedBatch) -> List[Any]:
        raise NotImplementedError("BGE-M3 does not support predict")
