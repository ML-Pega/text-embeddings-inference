import asyncio
import os
import torch
from grpc import aio
from loguru import logger

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import Optional

from text_embeddings_server.models import Model, get_model
from text_embeddings_server.pb import embed_pb2_grpc, embed_pb2
from text_embeddings_server.utils.tracing import UDSOpenTelemetryAioServerInterceptor
from text_embeddings_server.utils.interceptor import ExceptionInterceptor


class EmbeddingService(embed_pb2_grpc.EmbeddingServiceServicer):
    def __init__(self, model: Model):
        self.model = model
        # Force inference mode for the lifetime of EmbeddingService
        self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    async def Health(self, request, context):
        if self.model.device.type == "cuda":
            torch.zeros((2, 2), device="cuda")
        return embed_pb2.HealthResponse()

    async def Embed(self, request, context):
        max_input_length = self.model.max_input_length
        batch = self.model.batch_type.from_pb(
            request, self.model.device, max_input_length
        )

        # Check if model has embed_sparse method and should use it
        # This is for BGE-M3 sparse pooling mode
        if hasattr(self.model, 'embed_sparse') and hasattr(self.model, 'pool') and self.model.pool == 'bge-m3-sparse':
            embeddings = self.model.embed_sparse(batch)
        else:
            embeddings = self.model.embed(batch)

        return embed_pb2.EmbedResponse(embeddings=embeddings)

    async def EmbedSparse(self, request, context):
        """Separate gRPC method for sparse embeddings - returns sparse format directly"""
        max_input_length = self.model.max_input_length
        batch = self.model.batch_type.from_pb(
            request, self.model.device, max_input_length
        )

        # Call embed_sparse_native if available (returns sparse format directly)
        if hasattr(self.model, 'embed_sparse_native'):
            sparse_embeddings = self.model.embed_sparse_native(batch)
            return embed_pb2.EmbedSparseResponse(embeddings=sparse_embeddings)
        elif hasattr(self.model, 'embed_sparse'):
            # Fallback to dense format if native sparse not available
            embeddings = self.model.embed_sparse(batch)
            # Convert dense to sparse format
            sparse_embeddings = []
            for emb in embeddings:
                sparse_values = []
                for idx, val in enumerate(emb.values):
                    if val != 0.0:
                        sparse_values.append(embed_pb2.SparseValue(index=idx, value=val))
                sparse_embeddings.append(embed_pb2.SparseEmbedding(values=sparse_values))
            return embed_pb2.EmbedSparseResponse(embeddings=sparse_embeddings)
        else:
            # Fallback for models without embed_sparse
            embeddings = self.model.embed(batch)
            # Convert to sparse (all values as sparse)
            sparse_embeddings = []
            for emb in embeddings:
                sparse_values = [
                    embed_pb2.SparseValue(index=idx, value=val) 
                    for idx, val in enumerate(emb.values) if val != 0.0
                ]
                sparse_embeddings.append(embed_pb2.SparseEmbedding(values=sparse_values))
            return embed_pb2.EmbedSparseResponse(embeddings=sparse_embeddings)

    async def Predict(self, request, context):
        max_input_length = self.model.max_input_length
        batch = self.model.batch_type.from_pb(
            request, self.model.device, max_input_length
        )

        scores = self.model.predict(batch)

        return embed_pb2.PredictResponse(scores=scores)


def serve(
    model_path: Path,
    dtype: Optional[str],
    uds_path: Path,
    pool: str,
):
    async def serve_inner(
        model_path: Path,
        dtype: Optional[str] = None,
    ):
        unix_socket = f"unix://{uds_path}"

        try:
            model = get_model(model_path, dtype, pool)
        except Exception:
            logger.exception("Error when initializing model")
            raise

        # Get gRPC message size from environment variable, default to 256 MB
        default_max_size = 256 * 1024 * 1024
        max_grpc_message_size = int(os.environ.get('GRPC_MAX_MESSAGE_SIZE', default_max_size))

        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                UDSOpenTelemetryAioServerInterceptor(),
            ],
            options=[
                ('grpc.max_send_message_length', max_grpc_message_size),
                ('grpc.max_receive_message_length', max_grpc_message_size),
            ]
        )
        embed_pb2_grpc.add_EmbeddingServiceServicer_to_server(
            EmbeddingService(model), server
        )
        SERVICE_NAMES = (
            embed_pb2.DESCRIPTOR.services_by_name["EmbeddingService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(unix_socket)

        await server.start()

        logger.info(f"Server started at {unix_socket}")

        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Signal received. Shutting down")
            await server.stop(0)

    asyncio.run(serve_inner(model_path, dtype))
