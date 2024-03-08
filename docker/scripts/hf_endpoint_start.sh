# First we map the modeling
ln -s /repository/engines/*.engine /opt/endpoint/llm/
ln -s /repository/engines/config.json /opt/endpoint/llm/

python3 launch_triton_server.py \
  --model_repo /opt/endpoint \
  --tensorrt_llm_model_name text-generation