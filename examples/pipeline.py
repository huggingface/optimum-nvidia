from argparse import ArgumentParser

from huggingface_hub import login

from optimum.nvidia import pipeline


if __name__ == "__main__":
    parser = ArgumentParser("Hugging Face optimum-nvidia pipeline example")
    parser.add_argument("--token", type=str, required=False, help="Hugging Face Hub authentication token.")
    parser.add_argument("model_id_or_path", type=str, help="Model's id or path for the pipeline")

    args = parser.parse_args()
    if hasattr(args, "token"):
        login(args.token)

    model = pipeline("text-generation", args.model_id_or_path)
    out = model("What is the latest generation of Nvidia's GPUs?", max_new_tokens=128)
    print(out)
