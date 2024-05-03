from tester.tester import Tester
from tester.tester_params import TestingParameters
from tester.learning_params import LearningParameters
import os

def buttons_config(num_times, num_agents):
    """
    Function setting the experiment parameters and environment.

    Output
    ------
    Tester : tester object
        Object containing the information necessary to run this experiment.
    """
    base_file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    joint_rm_file = os.path.join(base_file_path, 'experiments', 'buttons', 'team_buttons_rm.txt')

    local_rm_files = []
    for i in range(num_agents):
        local_rm_string = os.path.join(base_file_path, 'experiments', 'buttons', 'buttons_rm_agent_{}.txt'.format(i+1))
        local_rm_files.append(local_rm_string)

    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 1*step_unit 
    testing_params.num_steps = step_unit

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9 # 0.9
    learning_params.alpha = 0.8
    learning_params.T = 50
    learning_params.initial_epsilon = 0.0 # Set epsilon to zero to turn off epsilon-greedy exploration (only using boltzmann)
    learning_params.max_timesteps_per_task = testing_params.num_steps

    tester = Tester(learning_params, testing_params)
    tester.step_unit = step_unit
    tester.total_steps = 200 * step_unit # 100 * step_unit
    tester.min_steps = 1

    tester.num_times = num_times
    tester.num_agents = num_agents

    tester.rm_test_file = joint_rm_file
    tester.rm_learning_file_list = local_rm_files

    # Set the environment settings for the experiment
    env_settings = dict()
    env_settings['Nr'] = 13
    env_settings['Nc'] = 13
    env_settings['initial_states'] = [11, 1]
    env_settings['walls'] = [(0,2), (1,2), (2,2), (2,3), (2,4), (2,5), (2,6), (2,8), (2,9), (2,10), (2,11), (2,12),
                                (3,5), (4,5), (5,5), (6,5), (3,9), (4,9), (5,9), (6,9),
                                (6,2), (6,3), (6,4),
                                (7,2), (8,2), (9,2), (10,2), (11,2), (12,2),
                                (4,7), (5,7), (6,7), (7,7), (8,7), (8,8), (8,9), (8,10), (8,11), (8,12)]
    env_settings['oneway'] = [(5,6), (5,8), (7,8)]
    #env_settings['goal_location'] = (12,7)
    env_settings['goal_location'] = (7,6)
    env_settings['B1'] = (1,3)
    # env_settings['B2'] = (12,1)
    env_settings['B2'] = (1,1)
    env_settings['K1'] = (6,8)
    env_settings['D'] = (7,11)
    env_settings['K2'] = (6,6)
    env_settings['yellow_tiles'] = [(0,5), (1,5)]
    env_settings['green_tiles'] = [(3,6)]
    env_settings['orange_tiles'] = [(3,8)]
    env_settings['blue_tiles'] = [(11,6), (11,7), (11,8), (12,6), (12,8)] #Key - BLUE tiles

    env_settings['p'] = 0.98

    tester.env_settings = env_settings

    tester.experiment = 'buttons'

    return tester