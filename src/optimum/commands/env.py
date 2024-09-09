import platform
import subprocess

import huggingface_hub
from tensorrt import __version__ as trt_version
from tensorrt_llm import __version__ as trtllm_version
from transformers import __version__ as transformers_version
from transformers.utils import is_torch_available

from optimum.commands import BaseOptimumCLICommand, CommandInfo
from optimum.version import __version__ as optimum_version


class EnvironmentCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(
        name="env", help="Get information about the environment used."
    )

    @staticmethod
    def print_apt_pkgs():
        apt = subprocess.Popen(["apt", "list", "--installed"], stdout=subprocess.PIPE)
        grep = subprocess.Popen(
            ["grep", "cuda"], stdin=apt.stdout, stdout=subprocess.PIPE
        )
        pkgs_list = list(grep.stdout)
        for pkg in pkgs_list:
            print(pkg.decode("utf-8").split("\n")[0])

    def run(self):
        pt_version = "not installed"
        if is_torch_available():
            import torch

            pt_version = torch.__version__

        platform_info = {
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
        }
        info = {
            "`tensorrt` version": trt_version,
            "`tensorrt-llm` version": trtllm_version,
            "`optimum` version": optimum_version,
            "`transformers` version": transformers_version,
            "`huggingface_hub` version": huggingface_hub.__version__,
            "`torch` version": f"{pt_version}",
        }

        print("\nCopy-and-paste the text below in your GitHub issue:\n")
        print("\nPlatform:\n")
        print(self.format_dict(platform_info))
        print("\nPython packages:\n")
        print(self.format_dict(info))
        print("\nCUDA system packages:\n")
        self.print_apt_pkgs()
