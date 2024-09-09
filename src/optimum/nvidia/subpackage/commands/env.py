from optimum.commands import BaseOptimumCLICommand, optimum_cli_subcommand
from optimum.commands.env import EnvironmentCommand


@optimum_cli_subcommand(EnvironmentCommand)
class TrtLlmEnvCommand(BaseOptimumCLICommand):
    pass
