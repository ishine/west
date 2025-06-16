# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
from abc import ABC, abstractmethod

import torch
import torchaudio
from transformers.trainer_pt_utils import LabelSmoother


class Extractor(ABC):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        print(self.kwargs)

    @abstractmethod
    def extract(self, item):
        pass


class ExtractorAsrWenet(Extractor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract(self, item):
        tokenizer = self.kwargs.get('tokenizer')
        inference = self.kwargs.get('inference', False)
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        audio, sample_rate = torchaudio.load(item['wav'])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        audio = audio * (1 << 15)
        # mel: (T, 80)
        mel = torchaudio.compliance.kaldi.fbank(audio,
                                                num_mel_bins=80,
                                                frame_length=25,
                                                frame_shift=10,
                                                dither=0.0,
                                                energy_floor=0.0,
                                                sample_frequency=16000)
        # TODO(Binbin Zhang): Refine to instruction + <AUDIO>
        ids_audio = [0] * (mel.size(0) // 8)  # 8 is the final subsampling rate
        tgt_audio = [IGNORE_TOKEN_ID] * len(ids_audio)
        instruction = 'Transcribe the speech'
        content = item['txt']
        t0 = '<|im_start|>system\n' + \
             'You are a helpful assistant<|im_end|>\n' + \
             '<|im_start|>user\n' + instruction + '<|audio_bos|>'
        t1 = '<|audio_eos|><|im_end|>\n' + '<|im_start|>assistant\n'
        ids0 = tokenizer.encode(t0)
        ids1 = tokenizer.encode(t1)
        ids = [tokenizer.bos_token_id] + ids0 + ids_audio + ids1
        tgt = [tokenizer.bos_token_id] + ids0 + tgt_audio + ids1
        if not inference:
            t2 = content + '<|im_end|>\n'
            ids2 = tokenizer.encode(t2)
            ids = ids + ids2 + [tokenizer.eos_token_id]
            tgt = tgt + ids2 + [tokenizer.eos_token_id]
        input_ids = torch.tensor(ids, dtype=torch.int)
        tgt_ids = torch.tensor(tgt, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': tgt_ids,
            'mel': mel,
            'offset': len(ids0),
        }


class ExtractorFactory:

    @staticmethod
    def create(extractor_type: str) -> Extractor:
        classes = {
            'asr_wenet': ExtractorAsrWenet,
        }
        if extractor_type.lower() in classes:
            return classes[extractor_type.lower()]
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
