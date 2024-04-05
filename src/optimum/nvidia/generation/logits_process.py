import torch
from transformers import (
    ForceTokensLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
)
from transformers.generation.logits_process import WhisperNoSpeechDetection


class TrtSuppressTokensLogitsProcessor(SuppressTokensLogitsProcessor):
    def __call__(self, step: int, input_ids: torch.Tensor, scores: torch.Tensor):
        scores = super().__call__(input_ids, scores)
        return scores


class TrtSuppressTokensAtBeginLogitsProcessor(SuppressTokensAtBeginLogitsProcessor):
    def __call__(self, step: int, input_ids: torch.Tensor, scores: torch.Tensor):
        scores = super().__call__(input_ids, scores)
        return scores


class TrtForceTokensLogitsProcessor(ForceTokensLogitsProcessor):
    def __call__(self, step: int, input_ids: torch.Tensor, scores: torch.Tensor):
        scores = super().__call__(input_ids, scores)
        return scores


class TrtWhisperNoSpeechDetection(WhisperNoSpeechDetection):
    def __call__(self, step: int, input_ids: torch.Tensor, scores: torch.Tensor):
        scores = super().__call__(input_ids, scores)
        return scores


LOGITS_PROCESSOR_MAP = {
    SuppressTokensLogitsProcessor: TrtSuppressTokensLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor: TrtSuppressTokensAtBeginLogitsProcessor,
    ForceTokensLogitsProcessor: TrtForceTokensLogitsProcessor,
    WhisperNoSpeechDetection: TrtWhisperNoSpeechDetection,
}
