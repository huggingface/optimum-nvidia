#  coding=utf-8
#  Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import re
from distutils.core import setup
from setuptools import find_namespace_packages


# Ensure we match the version set in optimum/neuron/version.py
filepath = "optimum/neuron/version.py"
try:
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

INSTALL_REQUIRES = [
    # "transformers >= 4.32.1",
    "fsspec",
    "huggingface_hub >= 0.14.0",
    "numpy >= 1.24.0"
    "optimum >= 1.13.0",
]

TESTS_REQUIRE = [
    "pytest",
    "psutil",
    "parameterized",
    "datasets",
    "safetensors",
]

QUALITY_REQUIRES = [
    "black",
    "ruff",
    "isort",
    "hf_doc_builder @ git+https://github.com/huggingface/doc-builder.git",
]

EXTRAS_REQUIRE = {
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRES,
}

setup(
    name="optimum-nvidia",
    version=__version__,
    description=(
        "Optimum Nvidia is the interface between the Hugging Face Transformers and NVIDIA GPUs. "
        "It provides a set of tools enabling easy model loading, training and "
        "inference on single and multiple GPU cards for different downstream tasks."
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, neural-network, fine-tuning, inference, nvidia, tensorrt, ampere, hopper",
    url="https://huggingface.co/hardware/nvidia",
    author="HuggingFace Inc. Machine Learning Optimization Team",
    author_email="hardware@huggingface.co",
    license="Apache 2.0",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
)