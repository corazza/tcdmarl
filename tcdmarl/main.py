"""
Run the experiment specified by the user.
"""

import pickle
from datetime import datetime
from types import NoneType
from typing import List

import click
import numpy as np
from matplotlib import pyplot as plt

from tcdmarl.consts import ALL_EXPERIMENT_NAMES
from tcdmarl.defaults import DEFAULT_NUM_SEPARATE_TRIALS
from tcdmarl.experiments.common import create_centralized_environment
from tcdmarl.experiments.dqprm import run_multi_agent_experiment
from tcdmarl.experiments.run_centralized_coordination_experiment import (
    run_centralized_experiment,
)
from tcdmarl.path_consts import RESULTS_DIR, WORK_DIR
from tcdmarl.routing_config import routing_config
from tcdmarl.tester.tester import Tester


def save_results(experiment: str, use_tlcd: bool, tester: Tester):
    """
    Save the results of the experiment to a file.
    """
    use_tlcd_str = "tlcd" if use_tlcd else "no_tlcd"
    experiment_data_path = RESULTS_DIR / f"{experiment}_{use_tlcd_str}"
    experiment_data_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    date_time_str: str = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{date_time_str}.p"

    save_file_path = experiment_data_path / filename

    with save_file_path.open("wb") as save_file:
        pickle.dump(tester, save_file)
        print(f"Results saved to {save_file_path}")


def load_results(path: str) -> Tester:
    """
    Load the results of an experiment from a file.
    """
    with open(path, "rb") as file:
        tester = pickle.load(file)
    return tester


def plot_multi_agent_results(tester: Tester, _num_agents: int):
    """
    Plot the results stored in tester.results for each of the agents.
    """

    prc_25: List[np.float64] = list()
    prc_50: List[np.float64] = list()
    prc_75: List[np.float64] = list()

    # Buffers for plots
    current_step: List[np.float64] = list()
    current_25: List[np.float64] = list()
    current_50: List[np.float64] = list()
    current_75: List[np.float64] = list()
    steps: List[int] = list()

    plot_dict = tester.results["testing_steps"]

    for step in plot_dict.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict[step]), 25))
            current_50.append(np.percentile(np.array(plot_dict[step]), 50))
            current_75.append(np.percentile(np.array(plot_dict[step]), 75))
            current_step.append(sum(plot_dict[step]) / len(plot_dict[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict[step]), 25))
            current_50.append(np.percentile(np.array(plot_dict[step]), 50))
            current_75.append(np.percentile(np.array(plot_dict[step]), 75))
            current_step.append(sum(plot_dict[step]) / len(plot_dict[step]))

        prc_25.append(sum(current_25) / len(current_25))
        prc_50.append(sum(current_50) / len(current_50))
        prc_75.append(sum(current_75) / len(current_75))
        steps.append(step)

    plt.plot(steps, prc_25, alpha=0)
    plt.plot(steps, prc_50, color="red")
    plt.plot(steps, prc_75, alpha=0)
    plt.grid()
    plt.fill_between(steps, prc_50, prc_25, color="red", alpha=0.25)
    plt.fill_between(steps, prc_50, prc_75, color="red", alpha=0.25)
    plt.ylabel("Testing Steps to Task Completion", fontsize=15)
    plt.xlabel("Training Steps", fontsize=15)
    plt.locator_params(axis="x", nbins=5)

    # Save as image to WORK_DIR/tmp
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = WORK_DIR / "tmp" / f"plot-{date_str}.png"
    # Create dirs if they do not exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")

    plt.show()


def run_experiment(
    experiment: str, use_tlcd: bool, num_trials: int, plot_results: bool
):
    """
    Run the experiment specified by the user.
    """
    assert experiment in ALL_EXPERIMENT_NAMES

    if experiment == "routing":
        # Get test object from config script
        tester = routing_config(
            num_times=num_trials,
            use_tlcd=use_tlcd,
        )
        tester = run_multi_agent_experiment(
            tester=tester,
            num_agents=tester.num_agents,
            num_times=num_trials,
            show_print=True,
        )
    else:
        assert experiment == "centralized_routing"
        # Get test object from config script
        tester = routing_config(
            num_times=num_trials,
            use_tlcd=use_tlcd,
        )
        tester = run_centralized_experiment(
            tester=tester,
            _num_agents=tester.num_agents,
            num_times=tester.num_times,
            show_print=True,
        )

    # Plot the results
    if plot_results:
        plot_multi_agent_results(tester, tester.num_agents)

    # Save the results
    save_results(experiment=experiment, use_tlcd=use_tlcd, tester=tester)


@click.command()
@click.option(
    "--experiment",
    required=False,
    help='The experiment to run. Options: "routing", "centralized_routing"',
)
@click.option(
    "--plot-results",
    required=False,
    help="(filepath) Plot results of an experiment",
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
@click.option(
    "--all-experiments",
    is_flag=True,
    help="run all experiments",
)
@click.option(
    "--num-trials",
    type=int,
    default=DEFAULT_NUM_SEPARATE_TRIALS,
    help="Number of separate trials to run the algorithm for",
)
def main(
    experiment: str | NoneType,
    tlcd: bool,
    show: bool,
    all_experiments: bool,
    num_trials: int,
    plot_results: str | NoneType,
):
    """
    Run the experiment specified by the user.
    """
    if experiment is None:
        assert all_experiments or plot_results is not None
        if plot_results is not None:
            tester = load_results(plot_results)
            plot_multi_agent_results(tester, tester.num_agents)
        else:
            assert all_experiments
            for experiment_name in ALL_EXPERIMENT_NAMES:
                for use_tlcd in [True, False]:
                    print(
                        f'Running experiment: "{experiment_name}" (use_tlcd={use_tlcd})'
                    )
                    run_experiment(
                        experiment_name,
                        use_tlcd=use_tlcd,
                        num_trials=num_trials,
                        plot_results=False,
                    )
    else:
        assert not all_experiments
        assert experiment in ALL_EXPERIMENT_NAMES
        if show:
            tester = routing_config(
                num_times=num_trials,
                use_tlcd=tlcd,
            )
            testing_env = create_centralized_environment(
                tester, use_prm=False, tlcd=tester.tlcd
            )
            testing_env.show_graphic(testing_env.get_initial_team_state())
        else:
            print(f'Running experiment: "{experiment}" (use_tlcd={tlcd})')
            run_experiment(experiment, tlcd, num_trials=num_trials, plot_results=True)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
