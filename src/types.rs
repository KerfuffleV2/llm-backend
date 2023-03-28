#![allow(unused_imports, unused_variables, dead_code)]
#![allow(non_camel_case_types)]
use std::path::PathBuf;

use llama_rs::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestId(usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelId(usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionId(usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobId(usize);

pub type TokenId = u32;
pub type BiasValue = f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    Loading,
    Ready,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// GGML Llama/Alpaca style model file.
    GGML_Llama {
        /// Context size.
        num_context_tokens: usize,
    },
}

pub mod client {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    pub struct ClientMessage {
        /// Sequence id for client requests. Not used by
        /// the backend itself.
        pub seq: usize,
        /// The client request.
        pub req: ClientRequest,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum ClientRequest {
        /// Load a model by filename.
        LoadModel {
            /// Filename of the model.
            filename: PathBuf,
            /// Type of model to load.
            model_type: ModelType,
        },

        /// Model-specific requests.
        Model {
            /// Model id to interact with.
            model_id: ModelId,
            /// The actual model request.
            req: ClientSessionRequest,
        },

        Session {
            /// Session id to interact with.
            session_id: SessionId,
            /// The actual session request.
            req: ClientSessionRequest,
        },

        Job {
            /// Job id to interact with.
            job_id: JobId,
            /// The actual session request.
            req: ClientJobRequest,
        },
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ClientModelRequest {
        /// Creates a session for the model.
        CreateSession {
            // Some kind of model specific config here.
        },

        /// Resets a model, kills all jobs and sessions.
        Reset,

        /// Unloads a model, kills all jobs and sessions.
        Unload,

        /// Gets information about a model.
        Get,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum ClientSessionRequest {
        /// Convert a string to a list of tokenids.
        Tokenize { string: String },

        /// Convert a list of tokenids to a list of strings.
        Untokenize { tokenids: Vec<TokenId> },

        /// Sets a list of tokenids to feed to the model.
        /// Fails if inference or feed is already running for the model.
        /// Creates a job.
        Inference {
            /// Maximum number of tokens to generate.
            max_tokens: usize,
        },

        /// Sets a list of tokenids to feed to the model.
        /// Fails if inference or feed is already running for the model.
        /// Creates a job.
        Feed {
            /// Whether the job should start start automatically.
            run: bool,

            /// Batch tokens when feeding (may impact other tasks).
            // batch_size: usize,

            /// Tokenids to add.
            tokenids: Vec<TokenId>,
        },

        /// Automatically pauses when when one of the tokenids in
        /// the list is sampled. The token is NOT fed to the model.
        AutoPause {
            /// Tokenids to add.
            tokenids: Vec<TokenId>,
        },

        /// Biases token selection probability based on a list of tokens.
        /// Will overwrite existing bias if already set.
        BiasTokens {
            /// List of tokens to affect as tuple (tokenid,bias)
            bias: Vec<(TokenId, BiasValue)>,
        },

        /// Creates a copy of the current session.
        Fork,

        /// Free the current session. Kills all related jobs.
        Free,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ClientJobRequest {
        /// Pause a job.
        Pause,
        /// Resume or start a job.
        Resume,
        /// Gets information about a job.
        Get,
    }

    pub enum BackendMessage {
        //
    }
}

pub mod backend {
    use super::*;

    //
    #[derive(Debug, Clone, PartialEq)]
    pub struct BackendMessage {
        /// Sequence number matching client request if available.
        pub seq: Option<usize>,
        /// Message or error.
        pub resp: Result<BackendMsg, BackendErrorResponse>,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum BackendMsg {
        ModelLoading {
            model_id: ModelId,
            message: Option<String>,
            available: bool,
        },
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum BackendErrorResponse {
        //
    }
}
