from typing import Optional

import torch
from torch import nn

class EmbeddingModel(nn.Module):
    def __init__(
            self,
            embed_tokens,
            layers,
            norm,
            rotary_emb
    ):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = layers
        self.norm = norm
        self.rotary_emb = rotary_emb


    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            # TODO：mask部分
            causal_mask_mapping = {
                "full_attention": None
            }
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

