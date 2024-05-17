from typing import Optional

from tcdmarl.Environments.common import CentralizedEnv
from tcdmarl.Environments.routing.multi_agent_routing_env import MultiAgentRoutingEnv
from tcdmarl.Environments.buttons.multi_agent_buttons_env import MultiAgentButtonsEnv
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA
from tcdmarl.tester.tester import Tester


def create_centralized_environment(
    tester: Tester, use_prm: bool, tlcd: Optional[CausalDFA]
) -> CentralizedEnv:
    """
    Helper method for instantiating the right centralized environment for testing based on the experiment name.
    """
    # print('tester.experiment', tester.experiment)
    if tester.experiment == "routing" or tester.experiment == "centralized_routing":
        return MultiAgentRoutingEnv(
            tester.rm_test_file, tester.env_settings, tlcd=tlcd
        ).use_prm(use_prm)
    
    elif tester.experiment == "buttons" or tester.experiment == "centralized_buttons":
        return MultiAgentButtonsEnv(
            tester.rm_test_file, tester.env_settings, tlcd=tlcd
        ).use_prm(use_prm)
        
    else:
        raise ValueError(
            'experiment should be one of: "routing", "centralized_routing"'
        )
