from pathlib import Path
from typing import Any, List

from tcdmarl.config import ExperimentConfig
from tcdmarl.consts import FINAL_EPISLON, INITIAL_EPISLON
from tcdmarl.Environments.common import Actions
from tcdmarl.Environments.laboratory.causal_dfa_block import dfa_paper_specimen
from tcdmarl.path_consts import RM_DIR
from tcdmarl.tester.learning_params import LearningParameters
from tcdmarl.tester.tester import Tester
from tcdmarl.tester.tester_params import TestingParameters


def laboratory_config(num_times: int, config: ExperimentConfig) -> Tester:
    """
    Function setting the experiment parameters and environment.

    Output
    ------
    Tester : tester object
        Object containing the information necessary to run this experiment.
    """
    num_agents: int = 2
    joint_rm_file: Path = RM_DIR / "laboratory" / "joint_rm.txt"

    local_rm_files: List[Path] = []
    for i in range(num_agents):
        local_rm_files.append(RM_DIR / "laboratory" / f"proj_{i+1}.txt")

    # episode length: num_steps=step_unit, max_timesteps_per_task=num_steps
    step_unit = config["episode_length"]

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 1 * step_unit
    testing_params.num_steps = step_unit

    # configuration of learning params
    # Set epsilon to zero to turn off epsilon-greedy exploration (only using boltzmann)
    learning_params = LearningParameters(
        gamma=0.9,
        alpha=0.8,
        t_param=50,
        initial_epsilon=INITIAL_EPISLON,
        final_epsilon=FINAL_EPISLON,
        max_timesteps_per_task=testing_params.num_steps,
    )

    # for testing
    # total_steps = 3 * step_unit
    # enough to converge for decentralized
    # total_steps = 100 * step_unit
    # enough to converge for centralized
    # total_steps = 500 * step_unit
    # from argument
    step_unit_factor = config["num_episodes"]
    total_steps = step_unit_factor * step_unit
    min_steps = 1

    # Set the environment settings for the experiment
    # TODO fix this mess (Dict[str, Any])
    env_settings: dict[str, Any] = dict()
    env_settings["Nr"] = 11
    env_settings["Nc"] = 15

    env_settings["initial_states"] = [4 * 15 + 2, 6 * 15 + 2]
    env_settings["walls"] = [
        (6, 5),
        (6, 6),
        (6, 7),
        (6, 8),
        (6, 9),
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8),
        (4, 9),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (7, 5),
        (8, 5),
        (9, 5),
        (10, 5),
        (0, 9),
        (1, 9),
        (2, 9),
        (3, 9),
        (7, 9),
        (8, 9),
        (9, 9),
        (10, 9),
    ]

    env_settings["forcemove"] = {
        Actions.RIGHT: [(5, 5), (5, 6), (5, 7), (5, 8), (5, 9)]
    }
    env_settings["C"] = (5, 5)
    env_settings["AB"] = (5, 9)
    env_settings["D"] = (9, 11)
    env_settings["E"] = (1, 13)
    env_settings["F"] = (5, 12)
    env_settings["yellow_tiles"] = []
    env_settings["sinks"] = [env_settings["F"]]

    env_settings["p"] = 0.98

    if config["use_tlcd"]:
        tlcd = dfa_paper_specimen()
    else:
        tlcd = None

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
        experiment="laboratory",
        use_prm=True,
        tlcd=tlcd,
    )
