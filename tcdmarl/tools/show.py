"""
Run the experiment specified by the user.
"""

import click

from tcdmarl.config import ExperimentConfig
from tcdmarl.environment_configs.generator_config import generator_config
from tcdmarl.environment_configs.laboratory_config import laboratory_config
from tcdmarl.experiments.common import create_centralized_environment
from tcdmarl.tester.tester import Tester


@click.command()
@click.option(
    "--environment",
    required=True,
    help="MDP to show (generator, laboratory).",
)
def main(environment: str):
    """
    Run the experiment specified by the user.
    """
    if environment == "generator":
        tester: Tester = generator_config(
            num_times=1,
            config=ExperimentConfig(
                environment_name="generator",
                centralized=True,
                use_tlcd=False,
                episode_length=100,
                num_episodes=100,
            ),
        )
    elif environment == "laboratory":
        tester: Tester = laboratory_config(
            num_times=1,
            config=ExperimentConfig(
                environment_name="laboratory",
                centralized=True,
                use_tlcd=False,
                episode_length=100,
                num_episodes=100,
            ),
        )
    else:
        raise ValueError(f"Experiment '{environment}' not recognized.")

    testing_env = create_centralized_environment(
        tester, use_prm=False, tlcd=tester.tlcd
    )
    testing_env.show_graphic(testing_env.get_initial_team_state())


if __name__ == "__main__":
    main()
