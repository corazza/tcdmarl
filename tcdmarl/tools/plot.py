"""
Run the experiment specified by the user.
"""

import csv
import pickle
from datetime import datetime
from types import NoneType
from typing import List

import click
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from tcdmarl.config import ExperimentConfig
from tcdmarl.environment_configs.generator_config import generator_config
from tcdmarl.environment_configs.laboratory_config import laboratory_config
from tcdmarl.environment_configs.routing_config import routing_config
from tcdmarl.experiments.common import create_centralized_environment
from tcdmarl.experiments.dqprm import run_multi_agent_experiment
from tcdmarl.experiments.run_centralized_coordination_experiment import (
    run_centralized_experiment,
)
from tcdmarl.path_consts import RESULTS_DIR
from tcdmarl.tester.tester import Tester
from tcdmarl.utils import experiment_name


def save_results(collection: str, experiment: str, use_tlcd: bool, tester: Tester):
    """
    Save the results of the experiment to a file.
    """
    use_tlcd_str = "tlcd" if use_tlcd else "no_tlcd"
    day_month = datetime.now().strftime("%Y-%m-%d")
    experiment_data_path = (
        RESULTS_DIR / day_month / collection / f"{experiment}_{use_tlcd_str}"
    )
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


def save_to_csv(
    steps: List[int],
    prc_25: List[float],
    prc_50: List[float],
    prc_75: List[float],
    save_file_path: str,
) -> None:
    with open(save_file_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["steps", "prc_25", "prc_50", "prc_75"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, step in enumerate(steps):
            writer.writerow(
                {
                    "steps": step,
                    "prc_25": prc_25[i],
                    "prc_50": prc_50[i],
                    "prc_75": prc_75[i],
                }
            )


def plot_multi_agent_results(
    collection: str,
    config: ExperimentConfig,
    tester: Tester,
    save_plot: bool,
    show_plot: bool,
):
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
            current_step.append(np.float64(sum(plot_dict[step]) / len(plot_dict[step])))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict[step]), 25))
            current_50.append(np.percentile(np.array(plot_dict[step]), 50))
            current_75.append(np.percentile(np.array(plot_dict[step]), 75))
            current_step.append(np.float64(sum(plot_dict[step]) / len(plot_dict[step])))

        prc_25.append(np.float64(sum(current_25) / len(current_25)))
        prc_50.append(np.float64(sum(current_50) / len(current_50)))
        prc_75.append(np.float64(sum(current_75) / len(current_75)))
        steps.append(step)

    plt.plot(steps, prc_25, alpha=0)
    plt.plot(steps, prc_50, color="red")
    plt.plot(steps, prc_75, alpha=0)
    plt.grid()
    plt.fill_between(steps, prc_50, prc_25, color="red", alpha=0.25)
    plt.fill_between(steps, prc_50, prc_75, color="red", alpha=0.25)
    plt.ylabel("Testing Steps to Task Completion", fontsize=15)
    plt.xlabel("Training Steps", fontsize=15)

    # plt.gca().xaxis.set_major_locator(MultipleLocator(base=20000))

    def human_format(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return "%.1f%s" % (num, ["", "K", "M", "B", "T", "P"][magnitude])

    # ...

    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: human_format(x)))
    # plt.locator_params(axis="x", nbins=5)

    # Save as image to results
    if save_plot:
        day_month = datetime.now().strftime("%Y-%m-%d")
        directory = experiment_name(config)
        experiment_data_path = RESULTS_DIR / day_month / collection / directory
        experiment_data_path.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        date_time_str: str = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{date_time_str}.png"

        save_file_path = experiment_data_path / filename

        plt.savefig(save_file_path)
        print(f"Plot saved to {save_file_path}")

        csv_save_file_path = experiment_data_path / f"{date_time_str}.csv"
        save_to_csv(steps, prc_25, prc_50, prc_75, csv_save_file_path)
        print(f"CSV saved to {csv_save_file_path}")

    if show_plot:
        plt.show()

    # Clear the current figure
    plt.clf()


def run_experiment(
    collection: str,
    experiment: str,
    use_tlcd: bool,
    num_trials: int,
    show_plot: bool,
    step_unit_factor: int,
) -> Tester:
    """
    Run the experiment specified by the user.
    """
    assert experiment in ALL_EXPERIMENT_NAMES

    if experiment == "generator":
        # Get test object from config script
        tester = generator_config(
            num_times=num_trials,
            use_tlcd=use_tlcd,
            step_unit_factor=step_unit_factor,
        )
        tester = run_multi_agent_experiment(
            tester=tester,
            num_agents=tester.num_agents,
            num_times=num_trials,
            show_print=True,
        )
    elif experiment == "centralized_generator":
        # Get test object from config script
        tester = generator_config(
            num_times=num_trials,
            use_tlcd=use_tlcd,
            step_unit_factor=step_unit_factor,
        )
        tester = run_centralized_experiment(
            tester=tester,
            _num_agents=tester.num_agents,
            num_times=tester.num_times,
            show_print=True,
        )
    elif experiment == "laboratory":
        # Get test object from config script
        tester = laboratory_config(
            num_times=num_trials,
            use_tlcd=use_tlcd,
            step_unit_factor=step_unit_factor,
        )
        tester = run_multi_agent_experiment(
            tester=tester,
            num_agents=tester.num_agents,
            num_times=num_trials,
            show_print=True,
        )
    elif experiment == "centralized_laboratory":
        # Get test object from config script
        tester = laboratory_config(
            num_times=num_trials,
            use_tlcd=use_tlcd,
            step_unit_factor=step_unit_factor,
        )
        tester = run_centralized_experiment(
            tester=tester,
            _num_agents=tester.num_agents,
            num_times=tester.num_times,
            show_print=True,
        )
    elif experiment == "routing":
        # Get test object from config script
        tester = routing_config(
            num_times=num_trials,
            use_tlcd=use_tlcd,
            step_unit_factor=step_unit_factor,
        )
        tester = run_multi_agent_experiment(
            tester=tester,
            num_agents=tester.num_agents,
            num_times=num_trials,
            show_print=True,
        )
    elif experiment == "centralized_routing":
        # Get test object from config script
        tester = routing_config(
            num_times=num_trials,
            use_tlcd=use_tlcd,
            step_unit_factor=step_unit_factor,
        )
        tester = run_centralized_experiment(
            tester=tester,
            _num_agents=tester.num_agents,
            num_times=tester.num_times,
            show_print=True,
        )
    else:
        raise ValueError(f"Experiment '{experiment}' not recognized.")

    # Save the results
    save_results(
        collection=collection, experiment=experiment, use_tlcd=use_tlcd, tester=tester
    )

    # Plot the results
    plot_multi_agent_results(
        collection=collection,
        environment_name=experiment,
        use_tlcd=use_tlcd,
        tester=tester,
        show_plot=show_plot,
        save_plot=True,
    )

    return tester


@click.command()
@click.option(
    "--config",
    required=True,
    help="Path to the config file to use",
)
@click.option(
    "--collection",
    required=True,
    help="Experiment collection name",
)
@click.option(
    "--plot-results-after-experiment",
    is_flag=True,
    help="When running an individual experiment, plot results afterwards?",
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
def main(
    collection: str,
    experiment: str | NoneType,
    tlcd: bool,
    show: bool,
    all_experiments: bool,
    num_trials: int,
    plot_results: str | NoneType,
    plot_results_after_experiment: bool,
    step_unit_factor: int,
):
    """
    Run the experiment specified by the user.
    """
    if experiment is None:
        assert all_experiments or plot_results is not None
        if plot_results is not None:
            tester = load_results(plot_results)
            plot_multi_agent_results(
                collection="none",
                environment_name=tester.experiment,
                use_tlcd=tester.tlcd is not None,
                tester=tester,
                save_plot=False,
                show_plot=True,
            )
        else:
            assert all_experiments
            for experiment_name in ALL_EXPERIMENT_NAMES:
                for use_tlcd in [True, False]:
                    print(
                        f'Running experiment: "{experiment_name}" (use_tlcd={use_tlcd})'
                    )
                    run_experiment(
                        collection=collection,
                        experiment=experiment_name,
                        use_tlcd=use_tlcd,
                        num_trials=num_trials,
                        # We do not plot after each experiments when running all experiments
                        show_plot=False,
                        step_unit_factor=step_unit_factor,
                    )
    else:
        assert not all_experiments
        assert experiment in ALL_EXPERIMENT_NAMES
        if show:
            if experiment == "routing":
                tester = routing_config(
                    num_times=num_trials,
                    use_tlcd=tlcd,
                    step_unit_factor=step_unit_factor,
                )
                testing_env = create_centralized_environment(
                    tester, use_prm=False, tlcd=tester.tlcd
                )
                testing_env.show_graphic(testing_env.get_initial_team_state())
            elif experiment == "generator":
                tester = generator_config(
                    num_times=num_trials,
                    use_tlcd=tlcd,
                    step_unit_factor=step_unit_factor,
                )
                testing_env = create_centralized_environment(
                    tester, use_prm=False, tlcd=tester.tlcd
                )
                testing_env.show_graphic(testing_env.get_initial_team_state())
            elif experiment == "laboratory":
                tester = laboratory_config(
                    num_times=num_trials,
                    use_tlcd=tlcd,
                    step_unit_factor=step_unit_factor,
                )
                testing_env = create_centralized_environment(
                    tester, use_prm=False, tlcd=tester.tlcd
                )
                testing_env.show_graphic(testing_env.get_initial_team_state())
            else:
                raise ValueError(f"Experiment '{experiment}' not recognized.")
        else:
            print(f'Running experiment: "{experiment}" (use_tlcd={tlcd})')
            run_experiment(
                collection=collection,
                experiment=experiment,
                use_tlcd=tlcd,
                num_trials=num_trials,
                show_plot=plot_results_after_experiment,
                step_unit_factor=step_unit_factor,
            )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
