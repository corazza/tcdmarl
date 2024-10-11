from typing import Optional

from tcdmarl.Environments.common import CentralizedEnv
from tcdmarl.Environments.generator.multi_agent_generator_env import (
    MultiAgentGeneratorEnv,
)
from tcdmarl.Environments.laboratory.multi_agent_laboratory_env import (
    MultiAgentLaboratoryEnv,
)
from tcdmarl.Environments.routing.multi_agent_routing_env import MultiAgentRoutingEnv
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA
from tcdmarl.tester.tester import Tester


def create_centralized_environment(
    tester: Tester, use_prm: bool, tlcd: Optional[CausalDFA]
) -> CentralizedEnv:
    """
    Helper method for instantiating the right centralized environment for testing based on the experiment name.
    """
    if "routing" in tester.experiment:
        return MultiAgentRoutingEnv(
            tester.rm_test_file, tester.env_settings, tlcd=tlcd
        ).use_prm(use_prm)
    elif "generator" in tester.experiment:
        return MultiAgentGeneratorEnv(
            tester.rm_test_file, tester.env_settings, tlcd=tlcd
        )
    elif "laboratory" in tester.experiment:
        return MultiAgentLaboratoryEnv(
            tester.rm_test_file, tester.env_settings, tlcd=tlcd
        )
    else:
        raise ValueError(
            f"Experiment '{tester.experiment}' not recognized. Please use one of the following: "
            f"['routing', 'generator', 'laboratory']"
        )
