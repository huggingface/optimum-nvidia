```
/opt/tritonserver/bin/tritonserver --log-verbose=3 --exit-on-error=false --model-repo=/opt/optimum/templates/inference-endpoints/
```

## Serving Inference Templates

### Triton Inference Server

The following variable are supported:
- `VAR_TOKENIZER_DIR` : Path to the folder holding the tokenizer files
- `VAR_GPT_MODEL_PATH`: Path to the folder holding the model engine files
- `VAR_GPT_MODEL_TYPE`: Indicate the kind of inferencing mecanism to leverage. Should be either `inflight_fused_batching` for In-Flight Batching model or `V1` otherwise.
- `VAR_BATCH_SCHEDULER_POLICY`: Indicate the batching policy. Should be either `max_utilization` to greedily pack as many requests as possible in each current in-flight batching iteration or `guaranteed_no_evict` to guarantee that a started request is never paused
- `VAR_ENABLE_STREAMING`: Flag allowing the server to send token while they are generated
- `VAR_ENABLE_OVERLAP`: Flag allowing to partition available requests into 2 'microbatches' that can be run concurrently to hide exposed CPU runtime
- `VAR_MAX_BEAM_WIDTH`: Maximum allowed number of beam to use for sampling (integer greater than 0)
- `VAR_KV_CACHE_MEMORY_PERCENT`: The percentage of maximum GPU memory dedicated to store KV cache (float between 0 and 1)
