"""Registers the export command for TRTLLM to the Optimum CLI."""

from ...nvidia.utils.import_utils import is_tensorrt_llm_available
from ..export import ExportCommand


if _tensorrt_llm_export_command_was_imported := is_tensorrt_llm_available():
    from ..export.trtllm import TrtLlmExportCommand  # noqa: F811

if _tensorrt_llm_export_command_was_imported:
    REGISTER_COMMANDS = [(TrtLlmExportCommand, ExportCommand)]
else:
    REGISTER_COMMANDS = []
