import warnings
from enum import Enum
from typing import Dict, List, Union

import torch
from transformers import PreTrainedTokenizer, TensorType

from optimum.nvidia import AutoModelForCausalLM, TensorRTForCausalLM

from .base import Pipeline


class ReturnType(Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


class TextGenerationPipeline(Pipeline):
    TARGET_FACTORY = AutoModelForCausalLM

    __slots__ = ("tokenizer", "_runtime", "_bos_token_id", "_eos_token_id", "_pad_token_id")

    def __init__(self, model: TensorRTForCausalLM, tokenizer: PreTrainedTokenizer):
        if tokenizer.eos_token and not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self._runtime = model

        self._bos_token_id = tokenizer.bos_token_id
        self._eos_token_id = tokenizer.eos_token_id
        self._pad_token_id = tokenizer.pad_token_id

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self._forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        prefix=None,
        handle_long_generation=None,
        stop_sequence=None,
        add_special_tokens=False,
        **generate_kwargs,
    ):
        preprocess_params = {"add_special_tokens": add_special_tokens}
        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix, padding=False, add_special_tokens=add_special_tokens, return_tensors=TensorType.PYTORCH
            )
            generate_kwargs["prefix_length"] = prefix_inputs["input_ids"].shape[-1]

        if handle_long_generation is not None:
            if handle_long_generation not in {"hole"}:
                raise ValueError(
                    f"{handle_long_generation} is not a valid value for `handle_long_generation` parameter expected"
                    " [None, 'hole']"
                )
            preprocess_params["handle_long_generation"] = handle_long_generation

        preprocess_params.update(generate_kwargs)
        forward_params = generate_kwargs

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_full_text`")
            if return_tensors is not None:
                raise ValueError("`return_full_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_tensors is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            if len(stop_sequence_ids) > 1:
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim."
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        return preprocess_params, forward_params, postprocess_params

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        prompt_text = model_inputs.pop("prompt_text")
        attention_mask = model_inputs.get("attention_mask", None)

        max_new_tokens = generate_kwargs.pop("max_new_tokens", -1)
        min_length = generate_kwargs.pop("min_length", -1)
        num_beams = generate_kwargs.pop("num_beams", 1)
        temperature = generate_kwargs.pop("temperature", 1.0)
        top_k = generate_kwargs.pop("top_k", 50)
        top_p = generate_kwargs.pop("top_p", 1.0)
        repetition_penalty = generate_kwargs.pop("repetition_penalty", 1.0)
        length_penalty = generate_kwargs.pop("length_penalty", 1.0)
        seed = generate_kwargs.pop("seed", 2017)

        # prefix_length = generate_kwargs.pop("prefix_length", 0)
        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        # if prefix_length > 0:
        #     has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
        #             "generation_config" in generate_kwargs
        #             and generate_kwargs["generation_config"].max_new_tokens is not None
        #     )
        #     if not has_max_new_tokens:
        #         generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self._runtime
        #         generate_kwargs["max_length"] += prefix_length
        #     has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
        #             "generation_config" in generate_kwargs
        #             and generate_kwargs["generation_config"].min_new_tokens is not None
        #     )
        #     if not has_min_new_tokens and "min_length" in generate_kwargs:
        #         generate_kwargs["min_length"] += prefix_length

        # BS x BEAMS x SL
        generated_sequence, lengths = self._runtime.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            seed=seed,
            bos_token_id=self._bos_token_id,
            eos_token_id=self._eos_token_id,
            pad_token_id=self._pad_token_id,
        )

        return {
            "generated_sequence": generated_sequence,
            "lengths": lengths,
            "input_ids": input_ids,
            "prompt_text": prompt_text,
        }

    def preprocess(
        self, prompt_text, prefix="", handle_long_generation=None, add_special_tokens=False, **generate_kwargs
    ) -> Dict[str, torch.Tensor]:
        if isinstance(prompt_text, List):
            text = [prefix + prompt for prompt in prompt_text]
        else:
            text = prefix + prompt_text

        inputs = self.tokenizer(
            text, padding=False, add_special_tokens=add_special_tokens, return_tensors=TensorType.PYTORCH
        )
        inputs["prompt_text"] = prompt_text

        return inputs

    def postprocess(self, model_outputs, return_type=ReturnType.FULL_TEXT, clean_up_tokenization_spaces=True):
        generated_sequence = model_outputs["generated_sequence"]
        # lengths = model_outputs["lengths"]
        # input_ids = model_outputs["input_ids"]
        # prompt_text = model_outputs["prompt_text"]
        generated_sequence = generated_sequence.cpu().numpy().tolist()
        records = []

        if return_type == ReturnType.TENSORS:
            return [{"generated_token_ids": generated for generated in generated_sequence}]

        for sequence in generated_sequence:
            # Decode text
            beam_text = self.tokenizer.batch_decode(
                sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

            record = {"generated_text": beam_text}
            records.append(record)

        return records
