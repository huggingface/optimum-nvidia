from typing import Optional

from huggingface_hub import ModelHubMixin

from tensorrt_llm import ModelConfig, Mapping
from tensorrt_llm.runtime import GenerationSession


class TRTEnginePretrainedModel(ModelHubMixin):
    pass


class TRTEngineForCausalLM(TRTEnginePretrainedModel):
    __slots__ = ("_config", "_mapping", "_session", "_cache", "_stream")

    def __init__(
        self,
        config: ModelConfig,
        mapping: Mapping,
        engine: bytes,
        stream: Optional["torch.cuda.Stream"] = None,
        use_cuda_graph: bool = False,
    ):
        super().__init__()

        self._config = config
        self._mapping = mapping
        self._stream = stream
        self._session = GenerationSession(
            model_config=config,
            engine_buffer=engine,
            mapping=mapping,
            stream=stream,
            cuda_graph_mode=use_cuda_graph
        )