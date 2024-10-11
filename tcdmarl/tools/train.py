"""
Run the experiment specified by the user.
"""

import pickle
from datetime import datetime
from pathlib import Path

import click

from tcdmarl.config import ExperimentConfig, RunConfig
from tcdmarl.environment_configs.generator_config import generator_config
from tcdmarl.environment_configs.laboratory_config import laboratory_config
from tcdmarl.experiments.dqprm import run_multi_agent_experiment
from tcdmarl.experiments.run_centralized_coordination_experiment import (
    run_centralized_experiment,
)
from tcdmarl.path_consts import RESULTS_DIR, WORK_DIR
from tcdmarl.tester.tester import Tester
from tcdmarl.tools.plot import plot_multi_agent_results
from tcdmarl.utils import experiment_name, load_typed_dict


def save_results(
    collection: str,
    config: ExperimentConfig,
    tester: Tester,
):
    """
    Save the results of the experiment to a file.
    """
    directory = experiment_name(config)
    day_month = datetime.now().strftime("%Y-%m-%d")
    experiment_data_path = RESULTS_DIR / day_month / collection / directory
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


def run_experiment(
    collection: str,
    config: ExperimentConfig,
    num_trials: int,
    plot_afterwards: bool,
) -> Tester:
    """
    Run the experiment specified by the user.
    """
    environment_name: str = config["environment_name"]
    centralized: bool = config["centralized"]
    use_tlcd: bool = config["use_tlcd"]

    print(
        f'Running experiment: "{environment_name}" (centralized={centralized}, use_tlcd={use_tlcd})'
    )

    # Get test object from config script
    if environment_name == "generator":
        tester: Tester = generator_config(
            num_times=num_trials,
            config=config,
        )
    elif environment_name == "laboratory":
        tester: Tester = laboratory_config(
            num_times=num_trials,
            config=config,
        )
    else:
        raise ValueError(f"Environment '{environment_name}' not recognized.")

    if centralized:
        tester = run_centralized_experiment(
            tester=tester,
            _num_agents=tester.num_agents,
            num_times=tester.num_times,
            show_print=True,
        )
    else:
        tester = run_multi_agent_experiment(
            tester=tester,
            num_agents=tester.num_agents,
            num_times=num_trials,
            show_print=True,
        )

    # Save the results
    save_results(
        collection=collection,
        config=config,
        tester=tester,
    )
    plot_multi_agent_results(
        collection=collection,
        config=config,
        tester=tester,
        show_plot=plot_afterwards,
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
    "--all-experiments",
    is_flag=True,
    help="run all experiments from the config",
)
@click.option(
    "--experiment-index",
    type=int,
    help="the index of the experiment to run (within the config file)",
)
def main(
    config: str,
    collection: str,
    all_experiments: bool,
    experiment_index: int,
):
    """
    Run the experiment specified by the user.
    """
    # remove all *.p files in WORK_DIR
    for file in WORK_DIR.glob("*.p"):
        file.unlink()

    assert not (all_experiments and experiment_index is not None)
    plot_afterwards = not all_experiments
    config_path: Path = Path(config)
    run_config: RunConfig = load_typed_dict(RunConfig, config_path)

    experiments_to_run: list[ExperimentConfig] = (
        run_config["experiments"]
        if all_experiments
        else [run_config["experiments"][experiment_index]]
    )

    for experiment_config in experiments_to_run:
        run_experiment(
            collection=collection,
            config=experiment_config,
            num_trials=run_config["num_trials"],
            plot_afterwards=plot_afterwards,
        )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
