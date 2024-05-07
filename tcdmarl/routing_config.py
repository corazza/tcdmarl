from pathlib import Path
from typing import Any, Dict, List

from tcdmarl.path_consts import RM_DIR
from tcdmarl.tester.learning_params import LearningParameters
from tcdmarl.tester.tester import Tester
from tcdmarl.tester.tester_params import TestingParameters


def routing_config(num_times: int) -> Tester:
    """
    Function setting the experiment parameters and environment.

    Output
    ------
    Tester : tester object
        Object containing the information necessary to run this experiment.
    """
    joint_rm_file: Path = RM_DIR / "routing" / "joint_rm.txt"

    local_rm_files: List[Path] = []
    for i in range(num_agents):
        local_rm_files.append(RM_DIR / "routing" / f"proj_{i+1}.txt")

    # TODO document, what is this?
    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 1 * step_unit
    testing_params.num_steps = step_unit

    # configuration of learning params
    # TODO fix this mess
    learning_params = LearningParameters()
    learning_params.gamma = 0.9  # 0.9
    learning_params.alpha = 0.8
    learning_params.T = 50
    learning_params.initial_epsilon = 0.0  # Set epsilon to zero to turn off epsilon-greedy exploration (only using boltzmann)
    learning_params.max_timesteps_per_task = testing_params.num_steps

    total_steps = 200 * step_unit  # 100 * step_unit
    min_steps = 1

    # Set the environment settings for the experiment
    # TODO fix this mess
    env_settings: Dict[str, Any] = dict()
    env_settings["Nr"] = 13
    env_settings["Nc"] = 13
    env_settings["initial_states"] = [11, 1]
    env_settings["walls"] = [
        (0, 2),
        (1, 2),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 8),
        (2, 9),
        (2, 10),
        (2, 11),
        (2, 12),
        (3, 5),
        (4, 5),
        (5, 5),
        (6, 5),
        (3, 9),
        (4, 9),
        (5, 9),
        (6, 9),
        (6, 2),
        (6, 3),
        (6, 4),
        (7, 2),
        (8, 2),
        (9, 2),
        (10, 2),
        (11, 2),
        (12, 2),
        (4, 7),
        (5, 7),
        (6, 7),
        (7, 7),
        (8, 7),
        (8, 8),
        (8, 9),
        (8, 10),
        (8, 11),
        (8, 12),
    ]

    env_settings["oneway"] = [(5, 6), (5, 8), (7, 8)]
    # env_settings['goal_location'] = (12,7)
    env_settings["goal_location"] = (7, 6)
    env_settings["B1"] = (1, 3)
    # env_settings['B2'] = (12,1)
    env_settings["B2"] = (1, 1)
    env_settings["K1"] = (6, 8)
    env_settings["D"] = (7, 11)
    env_settings["K2"] = (6, 6)
    env_settings["yellow_tiles"] = [(0, 5), (1, 5)]
    env_settings["green_tiles"] = [(3, 6)]
    env_settings["orange_tiles"] = [(3, 8)]
    env_settings["blue_tiles"] = [
        (11, 6),
        (11, 7),
        (11, 8),
        (12, 6),
        (12, 8),
    ]  # Key - BLUE tiles

    env_settings["p"] = 0.98

    return Tester(
        learning_params=learning_params,
        testing_params=testing_params,
        num_agents=num_agents,
        min_steps=min_steps,
        total_steps=total_steps,
        step_unit=step_unit,
        num_times=num_times,
        rm_test_file=joint_rm_file,
        rm_learning_file_list=local_rm_files,
        env_settings=env_settings,
        experiment="buttons",
    )
