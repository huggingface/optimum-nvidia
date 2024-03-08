#!/bin/bash

# We retrieve values from config to inject into the template
MAX_BATCH_SIZE=`cat /repository/engines/config.json | jq -r ".build_config.max_batch_size"`
MAX_BEAM_WIDTH=`cat /repository/engines/config.json | jq -r ".build_config.max_beam_width"`

# Then we inject global variables
sed -i 's/${VAR_TOKENIZER_PATH}/\/repository/g' **/config.pbtxt
sed -i 's/${VAR_MAX_BATCH_SIZE}/'"$MAX_BATCH_SIZE"'/g' **/config.pbtxt
sed -i 's/${VAR_MAX_BEAM_WIDTH}/'"$MAX_BEAM_WIDTH"'/g' **/config.pbtxt

# And finally we inject variables for the LLM model
sed -i 's/${VAR_GPT_MODEL_TYPE}/inflight_fused_batching/g' llm/config.pbtxt
sed -i 's/${VAR_GPT_MODEL_PATH}/\/repository\/engines/g' llm/config.pbtxt
sed -i 's/${VAR_ALLOW_STREAMING}/'"$ARG_ALLOW_STREAMING"'/g' llm/config.pbtxt
sed -i 's/${VAR_BATCH_SCHEDULER_POLICY}/'"$ARG_BATCH_SCHEDULER_POLICY"'/g' llm/config.pbtxt
sed -i 's/${VAR_KV_CACHE_FREE_GPU_MEMORY_FRACTION}/'"$ARG_KV_CACHE_RESERVED_MEMORY_FRACTION"'/g' **/config.pbtxt


python3 launch_triton.py \
  --model_repo /opt/endpoint \
  --tensorrt_llm_model_name text-generation