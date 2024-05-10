from tcdmarl.Environments.common import CentralizedEnv
from tcdmarl.Environments.routing.multi_agent_routing_env import MultiAgentRoutingEnv
from tcdmarl.tester.tester import Tester


def create_centralized_environment(tester: Tester) -> CentralizedEnv:
    """
    Helper method for instantiating the right centralized environment for testing based on the experiment name.
    """
    if tester.experiment == "routing" or tester.experiment == "centralized_routing":
        return MultiAgentRoutingEnv(tester.rm_test_file, tester.env_settings).use_prm(
            tester.use_prm
        )
    else:
        raise ValueError(
            'experiment should be one of: "routing", "centralized_routing"'
        )
