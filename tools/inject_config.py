import json
from argparse import ArgumentParser
from glob import iglob

_INJECTION_RULES = {
    "VAR_MAX_BATCH_SIZE": "int:self._config.model_config.max_batch_size",
    "VAR_MAX_BEAM_WIDTH": "int:self._config.model_config.max_beam_width",
    "VAR_GPT_MODEL_PATH": "str:paths.llm.logical",
    "VAR_GPT_MODEL_TYPE": "inflight_fused_batching",
    "VAR_BATCH_SCHEDULER_POLICY": "guaranteed_no_evict",  # For now
    "VAR_ENABLE_STREAMING": "bool:true",
    "VAR_ENABLE_OVERLAP": "bool:true",
    "VAR_KV_CACHE_MEMORY_PERCENT": "float:0.9"
}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config.json for the model")
    parser.add_argument("--model", type=str, help="Path to the model engine(s) folder")
    parser.add_argument("root", type=str, help="Path where is the Triton layout located")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_f:
        config = json.load(config_f)

    for config_file in iglob("**/config.pbtxt"):
        with open(config_file, "r") as pbtxt_f:
            content = pbtxt_f.read()

