# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from transformers import PretrainedConfig


class OSUMEChatConfig(PretrainedConfig):
    model_type = "touch_chat"

    def __init__(
        self,
        **kwargs,
    ):
        # TODO(Xuelong Geng): Complete the design of the configuration
        super().__init__(**kwargs)



__all__ = ["OSUMEChatConfig"]
