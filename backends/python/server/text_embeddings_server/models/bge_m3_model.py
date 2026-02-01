"""BGE-M3 model implementation using FlagEmbedding for native sparse embedding support.

Optimizations:
- Native sparse format support (embed_sparse_native) - avoids 250K dense vector transfer
- Configurable internal batch_size via BGE_M3_BATCH_SIZE env var (default: 32)
- Pre-cached vocab_size to avoid repeated lookups
"""
import os
import sys

# Disable tqdm progress bars to avoid BrokenPipeError in Docker
os.environ["TQDM_DISABLE"] = "1"
import torch

from pathlib import Path
from typing import Type, List, Dict, Any
from opentelemetry import trace

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding
from text_embeddings_server.pb import embed_pb2

tracer = trace.get_tracer(__name__)


class BGEM3Model(Model):
    """
    BGE-M3 model that supports both dense and sparse embeddings.
    
    Uses FlagEmbedding library internally to leverage BGE-M3's native
    sparse embedding capability which differs from SPLADE.
    
    Environment variables:
    - BGE_M3_BATCH_SIZE: Internal batch size for FlagEmbedding (default: 32)
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
                "FlagEmbedding is required for BGE-M3 sparse embeddings. "
                "Please install it with: pip install FlagEmbedding"
            )
        
        # Determine if we should use fp16
        use_fp16 = dtype in [torch.float16, torch.bfloat16]
        
        # Load BGE-M3 using FlagEmbedding
        self.flag_model = BGEM3FlagModel(
            str(model_path),
            use_fp16=use_fp16,
            device=str(device),
        )
        
        # Store pool mode for choosing output type
        self.pool = pool
        
        # BGE-M3 has 1024 hidden size for dense embeddings
        self.hidden_size = 1024
        
        # Get max sequence length from the underlying model
        if hasattr(self.flag_model.model, 'config'):
            config = self.flag_model.model.config
            position_offset = 0
            if hasattr(config, 'pad_token_id') and config.pad_token_id:
                position_offset = config.pad_token_id + 1
            if hasattr(config, 'max_seq_length'):
                self.max_input_length = config.max_seq_length
            elif hasattr(config, 'max_position_embeddings'):
                self.max_input_length = config.max_position_embeddings - position_offset
            else:
                self.max_input_length = 8192
        else:
            self.max_input_length = 8192
        
        self.device = device
        self.dtype = dtype
        self._tokenizer = None
        
        # Optimization: cache vocab_size
        self._vocab_size = len(self.flag_model.tokenizer)
        
        # Optimization: configurable internal batch size for FlagEmbedding
        self._internal_batch_size = int(os.environ.get('BGE_M3_BATCH_SIZE', '32'))

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    def _decode_batch(self, batch: PaddedBatch) -> List[str]:
        """Decode input_ids back to text for FlagEmbedding processing."""
        if self._tokenizer is None:
            self._tokenizer = self.flag_model.tokenizer
        
        texts = []
        for i in range(len(batch)):
            input_ids = batch.input_ids[i]
            attention_mask = batch.attention_mask[i]
            valid_length = int(attention_mask.sum().item())
            
            if valid_length == 0:
                texts.append(" ")
                continue
                
            valid_ids = input_ids[:valid_length].tolist()
            text = self._tokenizer.decode(valid_ids, skip_special_tokens=True)
            
            if not text or not text.strip():
                text = self._tokenizer.decode(valid_ids, skip_special_tokens=False)
            
            if not text or not text.strip():
                text = " "
            
            texts.append(text)
        return texts

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        """Generate dense embeddings for the batch."""
        texts = self._decode_batch(batch)
        
        output = self.flag_model.encode(
            texts,
            batch_size=self._internal_batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            
        )
        
        dense_embeddings = output['dense_vecs']
        
        results = []
        for i in range(len(batch)):
            embedding_values = dense_embeddings[i].tolist()
            results.append(Embedding(values=embedding_values))
        
        return results

    @tracer.start_as_current_span("embed_sparse_native")
    def embed_sparse_native(self, batch: PaddedBatch) -> List[embed_pb2.SparseEmbedding]:
        """
        Generate sparse embeddings in native sparse format.
        
        This method returns SparseEmbedding protobuf objects directly,
        avoiding the need to transfer 250K-length dense vectors.
        Only non-zero values are transmitted, drastically reducing bandwidth.
        """
        texts = self._decode_batch(batch)
        
        # Use FlagEmbedding to get sparse embeddings (lexical weights)
        output = self.flag_model.encode(
            texts,
            batch_size=self._internal_batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
            
        )
        
        # Extract lexical weights - already in sparse dict format
        lexical_weights_list = output['lexical_weights']
        
        # Convert directly to SparseEmbedding protobuf format
        results = []
        for lexical_weights in lexical_weights_list:
            sparse_values = []
            for token_id, weight in lexical_weights.items():
                if isinstance(token_id, str):
                    token_id = int(token_id)
                sparse_values.append(
                    embed_pb2.SparseValue(index=token_id, value=float(weight))
                )
            results.append(embed_pb2.SparseEmbedding(values=sparse_values))
        
        return results

    @tracer.start_as_current_span("embed_sparse")
    def embed_sparse(self, batch: PaddedBatch) -> List[Embedding]:
        """
        Generate sparse embeddings for the batch (legacy dense format).
        
        Returns sparse embeddings where the values represent the full vocabulary
        with most values being zero. The router will convert this to sparse format.
        
        Note: Use embed_sparse_native for better performance.
        """
        texts = self._decode_batch(batch)
        
        output = self.flag_model.encode(
            texts,
            batch_size=self._internal_batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
            
        )
        
        lexical_weights_list = output['lexical_weights']
        vocab_size = self._vocab_size
        
        results = []
        for lexical_weights in lexical_weights_list:
            sparse_vector = [0.0] * vocab_size
            for token_id, weight in lexical_weights.items():
                if isinstance(token_id, str):
                    token_id = int(token_id)
                if 0 <= token_id < vocab_size:
                    sparse_vector[token_id] = float(weight)
            results.append(Embedding(values=sparse_vector))
        
        return results

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Any]:
        """BGE-M3 does not support classification/prediction."""
        raise NotImplementedError("BGE-M3 does not support predict/classification tasks")
