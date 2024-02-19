from optimum.nvidia.runtime import TensorRTCompiledModel


class GemmaForCausalLM(TensorRTCompiledModel):
    __slots__ = ("_runtime", )