"""
Run the experiment specified by the user.
"""

import pickle
from datetime import datetime

import click

from tcdmarl.consts import NUM_SEPARATE_TRIALS
from tcdmarl.experiments.common import create_centralized_environment
from tcdmarl.experiments.dqprm import run_multi_agent_experiment
from tcdmarl.experiments.run_centralized_coordination_experiment import (
    run_centralized_experiment,
)
from tcdmarl.path_consts import RESULTS_DIR
from tcdmarl.routing_config import routing_config
from tcdmarl.tester.tester import Tester


def save_results(tester: Tester):
    """
    Save the results of the experiment to a file.
    """
    experiment_data_path = RESULTS_DIR / tester.experiment
    experiment_data_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    date_time_str: str = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{date_time_str}.p"

    save_file_path = experiment_data_path / filename

    with save_file_path.open("wb") as save_file:
        pickle.dump(tester, save_file)


def run_experiment(experiment: str, use_tlcd: bool, just_display_mdp: bool):
    """
    Run the experiment specified by the user.
    """
    assert experiment in ["routing", "centralized_routing"]

    if experiment == "routing":
        # Get test object from config script
        tester = routing_config(
            num_times=NUM_SEPARATE_TRIALS,
            use_tlcd=use_tlcd,
        )
        if not just_display_mdp:
            run_multi_agent_experiment(
                tester=tester,
                num_agents=tester.num_agents,
                num_times=NUM_SEPARATE_TRIALS,
                show_print=True,
            )
        else:
            testing_env = create_centralized_environment(
                tester, use_prm=False, tlcd=tester.tlcd
            )
            testing_env.show_graphic(testing_env.get_initial_team_state())
    else:
        assert experiment == "centralized_routing"
        # Get test object from config script
        tester = routing_config(
            num_times=NUM_SEPARATE_TRIALS,
            use_tlcd=use_tlcd,
        )
        run_centralized_experiment(
            tester=tester,
            num_agents=tester.num_agents,
            num_times=tester.num_times,
            show_print=True,
        )

    # Save the results
    save_results(tester)


@click.command()
@click.option(
    "--experiment",
    required=True,
    help='The experiment to run. Options: "routing", "centralized_routing"',
)
@click.option(
    "--tlcd",
    is_flag=True,
    help="Use a TL-CD to expedite RL?",
)
@click.option(
    "--show",
    is_flag=True,
    help="Just display the environment",
)
def main(experiment: str, tlcd: bool, show: bool):
    """
    Run the experiment specified by the user.
    """
    run_experiment(experiment, tlcd, show)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
