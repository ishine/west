# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from typing import Optional

import torch
from transformers import AutoModel, GenerationMixin, PreTrainedModel


from .configuration_osum_echat import OSUMEChatConfig


class OSUMEChat(PreTrainedModel, GenerationMixin):
    def __init__(self, config: OSUMEChatConfig, *inputs, **kwargs):
        """
        TODO(Xuelong Geng): Complete the design of OSUMEChat
        """
        super().__init__(config, *inputs, **kwargs)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        talker_features: Optional[torch.FloatTensor] = None,
        talker_features_lengths: Optional[torch.LongTensor] = None,
        talker_offsets: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        """
        TODO(Xuelong Geng): Complete the design of OSUMEChat
        """


    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        """
        TODO(Xuelong Geng): Complete the design of OSUMEChat
        """

    def init_tokenizer(self):
        """
        TODO(Xuelong Geng): Complete the design of OSUMEChat
        """
