from pathlib import Path
from typing import Any, Dict, List

from tcdmarl.path_consts import RM_DIR
from tcdmarl.tcrl.custom_dfas_neurips import dfa_paper_no_goal_after_flowers
from tcdmarl.tester.learning_params import LearningParameters
from tcdmarl.tester.tester import Tester
from tcdmarl.tester.tester_params import TestingParameters


def buttons_config(num_times: int, use_tlcd: bool, step_unit_factor: int) -> Tester:
    """
    Function setting the experiment parameters and environment.

    Output
    ------
    Tester : tester object
        Object containing the information necessary to run this experiment.
    """
    num_agents: int = 3
    joint_rm_file: Path = RM_DIR / "buttons" / "joint_rm.txt"

    local_rm_files: List[Path] = []
    for i in range(num_agents):
        local_rm_files.append(RM_DIR / "buttons" / f"proj_{i+1}.txt")

    # TODO document, what is this?
    step_unit = 1000

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
        initial_epsilon=0.0,
        max_timesteps_per_task=testing_params.num_steps,
    )

    # for testing
    # total_steps = 3 * step_unit
    # enough to converge for decentralized
    # total_steps = 100 * step_unit
    # enough to converge for centralized
    # total_steps = 500 * step_unit
    # from argument
    total_steps = step_unit_factor * step_unit
    min_steps = 1

    # Set the environment settings for the experiment
    # TODO fix this mess (Dict[str, Any])
    env_settings: Dict[str, Any] = dict()
    env_settings['Nr'] = 12
    env_settings['Nc'] = 12
    env_settings['initial_states'] = [0, 3, 11]
    env_settings['walls'] = [(0,2),
                            # (2, 1),
                            # (2, 2),
                            # (2, 3),
                            # (2, 4),
                            # (2, 5),
                            # (2, 6),
                            # (2, 7),
                            # (2, 8),
                            # (2, 9),
                            # (2, 10),
                            # (3, 10),
                            # (4, 10),
                            # (5, 10),
                            # (6, 10),
                            # (7, 10),
                            # (8, 10),
                            # (9, 10),
                            # (10, 10),
                            # (11, 10),
                            # (6, 9),
                            # (6, 8),
                            # (7, 8),
                            # (8, 8),
                            # (8, 7),
                            # (8, 6),
                            # (8, 5),
                            # (8, 4),
                            # (0, 10),
                            # (1, 10),
                            # (2, 10),
                            # (3, 10),
                            # (4, 10)


                             
                             
                              ]
    env_settings['oneway'] = [(4,3), (4,4), (4,5), (4,6), (4,7), (4,9)]
    env_settings['goal_location'] = (11,11)
    env_settings['yellow_button'] = (0,1)
    env_settings['green_button1'] = (6,3)
    env_settings['green_button2'] = (9,7)
    env_settings['red_button'] = (9,11)
    env_settings['yellow_tiles'] = [(2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9)]
    env_settings['green_tiles'] = [(2,11), (3,11)]
    env_settings['red_tiles'] = [(11,7), (11,8)]

    env_settings["p"] = 0.98
    # env_settings["sinks"] = [env_settings["F1"], env_settings["K1"]]
    # if env_settings["enable_f2"]:
    #     env_settings["sinks"].append(env_settings["F2"])

    if use_tlcd:
        tlcd = dfa_paper_no_goal_after_flowers()
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
        experiment="buttons",
        use_prm=True,
        tlcd=tlcd,
    )
