import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision.ops import StochasticDepth
from transformers.modeling_outputs import CausalLMOutputWithPast


class TokenEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        learnable: bool = True,
        device="cuda",
        dtype="torch.float32",
    ):
        """
        GPT-2 Learnable Token and Position Embeddings.
        If max_position_embeddings <= 0, there's no position embeddings
        Wwe embed to word_embe_proj_dim dimension then project up to embed_dim
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx
            )
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size,
                word_embed_proj_dim,
                padding_idx=padding_idx,
            )
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False)
        if not learnable:
            self.word_embeddings.weight.requires_grad = False

        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim)

    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(
                    seqlen, dtype=torch.long, device=self.device  # TODO
                )
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


def _init_weights(
    module,
    n_layers,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if "out_proj.weight" in name or "fc2.weight" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers)
                )
            # If using GLU activation for now, we scale the std by 2
            elif "output_linear.0.weight" in name:
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers)
                )


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.pre_norm: bool = config.pre_norm
        kwargs = {}
        if "Hybrid" in config.sequence_mixer.name:
            kwargs = {"layer_idx": layer_idx}
        self.sequence_mixer = config.sequence_mixer.instantiate(
            d_model=config.d_model,
            **kwargs
        )
        self.state_mixer = config.state_mixer.instantiate(
            d_model=config.d_model,
            layer_idx=layer_idx,
        )
        self.dropout1 = nn.Dropout(
            config.embed_dropout if layer_idx == 0 else config.resid_dropout
        )
        self.drop_path1 = StochasticDepth(config.drop_path, mode="row")
        self.norm1 = nn.LayerNorm(config.d_model, elementwise_affine=config.affine_norm)
        self.dropout2 = nn.Dropout(config.resid_dropout)
        self.drop_path2 = StochasticDepth(config.drop_path, mode="row")
        self.norm2 = nn.LayerNorm(config.d_model, elementwise_affine=config.affine_norm)

    def forward(self, hidden_states, residual=None):
        dropped = self.drop_path1(self.dropout1(hidden_states))
        residual = (dropped + residual.float()).to(dropped.dtype) if residual is not None else dropped

        if self.pre_norm:
            hidden_states = self.norm1(
                residual.to(dtype=self.norm1.weight.dtype) if self.norm1.weight is not None else residual)

            hidden_states = self.sequence_mixer(hidden_states)
            dropped = self.drop_path2(self.dropout2(hidden_states))
            residual = (dropped + residual.float()).to(dropped.dtype) if residual is not None else dropped

            hidden_states = self.norm2(
                residual.to(dtype=self.norm2.weight.dtype) if self.norm2.weight is not None else residual)
            hidden_states = self.state_mixer(hidden_states)
            return hidden_states, residual
        else:
            hidden_states = self.sequence_mixer(residual)
            dropped = self.drop_path2(self.dropout2(hidden_states))
            residual = (dropped + residual.float()).to(dropped.dtype) if residual is not None else dropped
            residual = self.norm1(
                residual.to(dtype=self.norm1.weight.dtype) if self.norm1.weight is not None else residual)

            hidden_states = (self.state_mixer(residual) + residual.float()).to(hidden_states.dtype)
            hidden_states = self.norm2(
                hidden_states.to(dtype=self.norm2.weight.dtype) if self.norm2.weight is not None else hidden_states)
            return hidden_states, None


class LMBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = TokenEmbeddings(
            config.d_model,
            config.vocab_size,
            config.max_position_embeddings,
            learnable=config.learnable_word_embeddings,
        )
        if config.block_type == "TransformerBlock":
            block_cls = TransformerBlock
        elif config.block_type == "MambaBlock":
            from aicl.model.zoology.mixers.mamba import MambaBlock

            block_cls = MambaBlock
        self.layers = nn.ModuleList(
            [block_cls(config=config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.drop_f = nn.Dropout(config.resid_dropout)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
            )
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        self.residual_in_fp32 = config.residuals_in_fp32

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(
            input_ids,
            position_ids=position_ids,
        )
        residual = None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, residual = torch.utils.checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    residual
                )
            else:
                hidden_states, residual = layer(hidden_states, residual)
        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        return hidden_states


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )

        self.backbone = LMBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
            )
        )

        # tie weights
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(self, input_ids, labels=None, position_ids=None, state=None):
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        lm_logits = self.lm_head(hidden_states)
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        else:
            lm_loss = None

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits
        )
