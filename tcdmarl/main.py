import os
import pickle
from datetime import datetime

import click

from tcdmarl.consts import NUM_SEPARATE_TRIALS
from tcdmarl.experiments.dqprm import run_multi_agent_experiment
from tcdmarl.experiments.run_centralized_coordination_experiment import (
    run_centralized_experiment,
)
from tcdmarl.routing_config import routing_config


@click.command()
@click.option(
    "--experiment",
    required=True,
    help='The experiment to run. Options: "routing", "centralized_routing"',
)
def main(experiment: str):
    assert experiment in ["routing", "centralized_routing"]

    if experiment == "routing":
        tester = routing_config(
            num_times=NUM_SEPARATE_TRIALS
        )  # Get test object from config script
        run_multi_agent_experiment(
            tester,
            num_agents=tester.num_agents,
            num_times=NUM_SEPARATE_TRIALS,
            show_print=True,
        )
    else:
        assert experiment == "centralized_routing"
        tester = routing_config(num_times=NUM_SEPARATE_TRIALS)
        run_centralized_experiment(
            tester,
            experiment=experiment,
            num_agents=tester.num_agents,
            num_times=tester.num_times,
            show_print=True,
        )

    # Save the results
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    data_path = os.path.join(parent_dir, "data")
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    experiment_data_path = os.path.join(data_path, experiment)

    if not os.path.isdir(experiment_data_path):
        os.mkdir(experiment_data_path)

    now = datetime.now()
    save_file_str = r"\{}_".format(now.strftime("%Y-%m-%d_%H-%M-%S"))
    save_file_str = save_file_str + experiment + ".p"
    save_file = open(experiment_data_path + save_file_str, "wb")
    pickle.dump(tester, save_file)


if __name__ == "__main__":
    main()
