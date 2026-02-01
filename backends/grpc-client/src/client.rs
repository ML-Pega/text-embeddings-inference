/// Single shard Client
use crate::pb::embedding::v1::embedding_service_client::EmbeddingServiceClient;
use crate::pb::embedding::v1::*;
use crate::Result;
use grpc_metadata::InjectTelemetryContext;
use tonic::transport::{Channel, Uri};
use tracing::instrument;

/// Default maximum gRPC message size (256 MB)
const DEFAULT_MAX_GRPC_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

/// Get the maximum gRPC message size from environment variable or use default
fn get_max_grpc_message_size() -> usize {
    std::env::var("GRPC_MAX_MESSAGE_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
}

/// Text Embeddings Inference gRPC client
#[derive(Debug, Clone)]
pub struct Client {
    stub: EmbeddingServiceClient<Channel>,
}

impl Client {
    /// Returns a client connected to the given url
    pub async fn connect(uri: Uri) -> Result<Self> {
        let channel = Channel::builder(uri).connect().await?;
        let max_size = get_max_grpc_message_size();

        Ok(Self {
            stub: EmbeddingServiceClient::new(channel)
                .max_decoding_message_size(max_size)
                .max_encoding_message_size(max_size),
        })
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let channel = Channel::from_shared("http://[::]:50051".to_string())
            .unwrap()
            .connect_with_connector(tower::service_fn(move |_: Uri| {
                tokio::net::UnixStream::connect(path.clone())
            }))
            .await?;
        let max_size = get_max_grpc_message_size();

        Ok(Self {
            stub: EmbeddingServiceClient::new(channel)
                .max_decoding_message_size(max_size)
                .max_encoding_message_size(max_size),
        })
    }

    /// Get backend health
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let request = tonic::Request::new(HealthRequest {}).inject_context();
        let response = self.stub.health(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn embed(
        &mut self,
        input_ids: Vec<u32>,
        token_type_ids: Vec<u32>,
        position_ids: Vec<u32>,
        cu_seq_lengths: Vec<u32>,
        max_length: u32,
    ) -> Result<Vec<Embedding>> {
        let request = tonic::Request::new(EmbedRequest {
            input_ids,
            token_type_ids,
            position_ids,
            max_length,
            cu_seq_lengths,
        })
        .inject_context();
        let response = self.stub.embed(request).await?.into_inner();
        Ok(response.embeddings)
    }

    #[instrument(skip_all)]
    pub async fn predict(
        &mut self,
        input_ids: Vec<u32>,
        token_type_ids: Vec<u32>,
        position_ids: Vec<u32>,
        cu_seq_lengths: Vec<u32>,
        max_length: u32,
    ) -> Result<Vec<Score>> {
        let request = tonic::Request::new(EmbedRequest {
            input_ids,
            token_type_ids,
            position_ids,
            max_length,
            cu_seq_lengths,
        })
        .inject_context();
        let response = self.stub.predict(request).await?.into_inner();
        Ok(response.scores)
    }

    /// Embed sparse - returns sparse embeddings in native sparse format
    #[instrument(skip_all)]
    pub async fn embed_sparse(
        &mut self,
        input_ids: Vec<u32>,
        token_type_ids: Vec<u32>,
        position_ids: Vec<u32>,
        cu_seq_lengths: Vec<u32>,
        max_length: u32,
    ) -> Result<Vec<SparseEmbedding>> {
        let request = tonic::Request::new(EmbedRequest {
            input_ids,
            token_type_ids,
            position_ids,
            max_length,
            cu_seq_lengths,
        })
        .inject_context();
        let response = self.stub.embed_sparse(request).await?.into_inner();
        Ok(response.embeddings)
    }
}
