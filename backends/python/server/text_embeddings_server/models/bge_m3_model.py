"""BGE-M3 model implementation using FlagEmbedding for native sparse embedding support."""
import torch

from pathlib import Path
from typing import Type, List, Dict, Any
from opentelemetry import trace

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding

tracer = trace.get_tracer(__name__)


class BGEM3Model(Model):
    """
    BGE-M3 model that supports both dense and sparse embeddings.
    
    Uses FlagEmbedding library internally to leverage BGE-M3's native
    sparse embedding capability which differs from SPLADE.
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
                self.max_input_length = 8192  # BGE-M3 default
        else:
            self.max_input_length = 8192  # BGE-M3 default
        
        # Note: We don't call super().__init__() with a model since we use FlagEmbedding directly
        self.device = device
        self.dtype = dtype
        self._tokenizer = None

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    def _decode_batch(self, batch: PaddedBatch) -> List[str]:
        """Decode input_ids back to text for FlagEmbedding processing."""
        if self._tokenizer is None:
            # Use the tokenizer from FlagEmbedding model
            self._tokenizer = self.flag_model.tokenizer
        
        texts = []
        for i in range(len(batch)):
            input_ids = batch.input_ids[i]
            # Remove padding (attention_mask == 0)
            attention_mask = batch.attention_mask[i]
            valid_length = attention_mask.sum().item()
            valid_ids = input_ids[:valid_length].tolist()
            # Decode to text
            text = self._tokenizer.decode(valid_ids, skip_special_tokens=True)
            texts.append(text)
        return texts

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        """Generate dense embeddings for the batch."""
        texts = self._decode_batch(batch)
        
        # Use FlagEmbedding to get dense embeddings
        output = self.flag_model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        
        # Extract dense embeddings
        dense_embeddings = output['dense_vecs']
        
        # Convert to list of Embedding objects
        results = []
        for i in range(len(batch)):
            embedding_values = dense_embeddings[i].tolist()
            results.append(Embedding(values=embedding_values))
        
        return results

    @tracer.start_as_current_span("embed_sparse")
    def embed_sparse(self, batch: PaddedBatch) -> List[Embedding]:
        """
        Generate sparse embeddings for the batch.
        
        Returns sparse embeddings where the values represent the full vocabulary
        with most values being zero. The router will convert this to sparse format.
        """
        texts = self._decode_batch(batch)
        
        # Use FlagEmbedding to get sparse embeddings (lexical weights)
        output = self.flag_model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        
        # Extract lexical weights (sparse embeddings)
        # lexical_weights is a list of dicts: [{token_id: weight, ...}, ...]
        lexical_weights_list = output['lexical_weights']
        
        # Convert to dense format for the existing embed response format
        # The vocabulary size is determined by the tokenizer
        vocab_size = len(self.flag_model.tokenizer)
        
        results = []
        for lexical_weights in lexical_weights_list:
            # Create a dense vector with zeros
            sparse_vector = [0.0] * vocab_size
            # Fill in the non-zero weights
            for token_id, weight in lexical_weights.items():
                if isinstance(token_id, str):
                    # If token_id is a string, convert to int
                    token_id = int(token_id)
                if 0 <= token_id < vocab_size:
                    sparse_vector[token_id] = float(weight)
            results.append(Embedding(values=sparse_vector))
        
        return results

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Any]:
        """BGE-M3 does not support classification/prediction."""
        raise NotImplementedError("BGE-M3 does not support predict/classification tasks")
