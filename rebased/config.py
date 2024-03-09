from dataclasses import dataclass, field
from .utils import import_from_str

@dataclass
class FunctionConfig:
    name: str
    kwargs: dict = field(default_factory=dict)

    def instantiate(self):
        return partial(import_from_str(self.name), **self.kwargs)


@dataclass
class ModuleConfig:
    name: str
    kwargs: dict = field(default_factory=dict)

    def instantiate(self, **kwargs):
        return import_from_str(self.name)(**kwargs, **self.kwargs)

@dataclass
class ModelConfig:
    sequence_mixer: ModuleConfig = None
    state_mixer: ModuleConfig = None

    affine_norm: bool = True
    pre_norm: bool = True
    use_gamma: bool = True
    use_beta: bool = True
    normalize: bool = True

    d_model: int = 128
    n_layers: int = 2
    num_heads: int = 1
    max_position_embeddings: int = 64
    learnable_word_embeddings: bool = True
    vocab_size: int = 8_192

    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    drop_path: float = 0.0
    layer_norm_epsilon: float = 1e-5
    pad_vocab_size_multiple: int = 1
    gradient_checkpointing: bool = False

    block_type: str = "TransformerBlock"
    log_scores: bool = False
    residuals_in_fp32: bool = True
