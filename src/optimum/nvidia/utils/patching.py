import inspect

from tensorrt_llm import Builder


class BuilderPatcher:
    # TODO: Remove this once available natively in TensorRT-LLM - `optimize(network)` lowers predictive performance for Whisper.

    def __init__(self):
        self.builder_path = inspect.getfile(Builder)

    def __enter__(self):
        with open(self.builder_path, "r") as file:
            lines = file.readlines()

        for i in range(len(lines)):
            if (
                i > 0
                and lines[i] == "    optimize(network)\n"
                and "# PATCHED" not in lines[i - 1]
            ):
                lines[i] = (
                    "    if getattr(model.config, 'optimize_network', True):\n        # PATCHED\n        optimize(network)\n    else:\n        logger.info('Network optimization disabled during build.')\n"
                )

        with open(self.builder_path, "w") as f:
            f.writelines(lines)

    def __exit__(self, exc_type, exc_value, traceback):
        with open(self.builder_path, "r") as file:
            lines = file.readlines()

        for i in range(len(lines)):
            if lines[i] == "        optimize(network)\n":
                lines[i - 1] = ""
                lines[i - 2] = ""
                lines[i] = "    optimize(network)\n"
                lines[i + 1] = ""
                lines[i + 2] = ""

        with open(self.builder_path, "w") as f:
            f.writelines(lines)
