import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import int32
from numpy.typing import NDArray

from tcdmarl.Agent.centralized_agent import CentralizedAgent
from tcdmarl.experiments.common import create_centralized_environment
from tcdmarl.tester.learning_params import LearningParameters
from tcdmarl.tester.tester import Tester
from tcdmarl.tester.tester_params import TestingParameters


def run_qlearning_task(
    epsilon: float,
    tester: Tester,
    centralized_agent: CentralizedAgent,
    show_print: bool = False,
):
    """
    This code runs one q-learning episode. q-functions, and accumulated reward values of agents
    are updated accordingly. If the appropriate number of steps have elapsed, this function will
    additionally run a test episode.

    Parameters
    ----------
    epsilon : float
        Numerical value in (0,1) representing likelihood of choosing a random action.
    tester : Tester object
        Object containing necessary information for current experiment.
    centralized_agent : CentralizedAgent object
        Centralized agent object representing the entire team of agents.
    show_print : bool
        Optional flag indicating whether or not to print output statements to terminal.
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params

    centralized_agent.reset_state()
    centralized_agent.initialize_reward_machine()

    num_steps = learning_params.max_timesteps_per_task

    env = create_centralized_environment(
        tester, use_prm=tester.use_prm, tlcd=tester.tlcd
    )

    for t in range(num_steps):
        # Update step count
        tester.add_step()

        # Perform a q-learning step.
        if not (centralized_agent.is_task_complete):
            current_u = centralized_agent.u
            s, a = centralized_agent.get_next_action(epsilon, learning_params)
            r, l, s_new = env.environment_step(s, a)
            # a = np.copy(env.last_action) # due to MDP slip
            centralized_agent.update_agent(s_new, a, r, l, learning_params)

            for s_agent in s_new:
                (row, col) = env.get_map().get_state_description(s_agent)
                if (row, col) in env.get_map().sinks:
                    tester.add_training_stuck_step()

            for u in centralized_agent.all_states:
                if not (u == current_u) and not (
                    u in centralized_agent.terminal_states
                ):
                    l = env.get_mdp_label(s, s_new, u)
                    r = 0
                    u_temp = u
                    u2 = u
                    for e in l:
                        # Get the new reward machine state and the reward of this step
                        u2 = centralized_agent.get_next_state(u_temp, e)
                        r = r + centralized_agent.get_reward(u_temp, u2)
                        # Update the reward machine state
                        u_temp = u2
                    centralized_agent.update_q_function(
                        s, s_new, u, u2, a, r, learning_params
                    )

        # If enough steps have elapsed, test and save the performance of the agents.
        if (
            testing_params.test
            and tester.get_current_step() % testing_params.test_freq == 0
        ):
            t_init = time.time()
            step = tester.get_current_step()

            # Need to create a copy of the agent for testing. If we pass the agent directly
            # mid-episode to the test function, the test will reset the world-state and reward machine
            # state before the training episode has been completed.
            centralized_agent_copy = CentralizedAgent(
                centralized_agent.rm_file,
                centralized_agent.s_i,
                centralized_agent.num_states,
                centralized_agent.actions,
                tlcd=tester.tlcd,
            ).use_prm(tester.use_prm)
            # Pass the q function directly. Note that the q-function will be updated during testing.
            # JAN: it does not seem to be the case that the Q-function will be updated during testing, because we use update_q_function=False in the testing function
            centralized_agent_copy.q = centralized_agent.q
            # if tester.use_prm:
            #     centralized_agent_copy.prm.terminal_states = (
            #         centralized_agent_copy.prm.original_terminal_states
            #     )
            #     centralized_agent_copy.terminal_states = set(
            #         centralized_agent_copy.prm.original_terminal_states
            #     )

            # Run a test of the performance of the agents
            testing_reward, trajectory, testing_steps = run_centralized_qlearning_test(
                centralized_agent_copy,
                tester,
                learning_params,
                testing_params,
                show_print=show_print,
            )
            # if we failed, we record it as full episode
            assert not (testing_steps < testing_params.num_steps and testing_reward < 1)

            if 0 not in tester.results.keys():
                tester.results[0] = {}
            if step not in tester.results[0]:
                tester.results[0][step] = []
            tester.results[0][step].append(testing_reward)

            # Save the testing trace
            if "trajectories" not in tester.results.keys():
                tester.results["trajectories"] = {}
            if step not in tester.results["trajectories"]:
                tester.results["trajectories"][step] = []
            tester.results["trajectories"][step].append(trajectory)

            # Save how many steps it took to complete the task
            if "testing_steps" not in tester.results.keys():
                tester.results["testing_steps"] = {}
            if step not in tester.results["testing_steps"]:
                tester.results["testing_steps"][step] = []
            tester.results["testing_steps"][step].append(testing_steps)

            if len(tester.steps) == 0 or tester.steps[-1] < step:
                tester.steps.append(step)

        # If the agents has completed its task, reset it to its initial state.
        if centralized_agent.is_task_complete:
            centralized_agent.reset_state()
            centralized_agent.initialize_reward_machine()

            # Make sure we've run at least the minimum number of training steps before breaking the loop
            if tester.stop_task(t):
                break

        # checking the steps time-out
        if tester.stop_learning():
            break


def run_centralized_qlearning_test(
    centralized_agent: CentralizedAgent,
    tester: Tester,
    learning_params: LearningParameters,
    testing_params: TestingParameters,
    show_print: bool = True,
) -> Tuple[int, List[Tuple[NDArray[int32], NDArray[int32], int]], int]:
    """
    Run a test of the q-learning with reward machine method with the current q-function.

    Parameters
    ----------
    centralized_agent : CentralizedAgent object
        Centralized agent object representing the entire team of agents.
    learning_params : LearningParameters object
        Object storing parameters to be used in learning.
    Testing_params : TestingParameters object
        Object storing parameters to be used in testing.

    Ouputs
    ------
    testing_reard : float
        Reward achieved by agent during this test episode.
    trajectory : list
        List of dictionaries containing information on current step of test.
    step : int
        Number of testing steps required to complete the task.
    """
    testing_env = create_centralized_environment(
        tester, use_prm=tester.use_prm, tlcd=tester.tlcd
    )

    centralized_agent.reset_state()
    centralized_agent.initialize_reward_machine()

    testing_reward: int = 0
    trajectory: List[Tuple[NDArray[int32], NDArray[int32], int]] = []
    step: int = 0
    stuck_counter = 0

    failed = False

    # Starting interaction with the environment
    for _t in range(testing_params.num_steps):
        step = step + 1

        # Perform a step
        s, a = centralized_agent.get_next_action(-1.0, learning_params)
        r, l, s_team_next = testing_env.environment_step(s, a)

        for s_agent in s_team_next:
            (row, col) = testing_env.get_map().get_state_description(s_agent)
            if (row, col) in testing_env.get_map().sinks:
                stuck_counter += 1

        # trajectory.append({'s' : np.array(s_team, dtype=int), 'a' : np.array(a_team, dtype=int), 'u': int(testing_env.u)})

        testing_reward = testing_reward + r
        # a = np.copy(testing_env.last_action)
        centralized_agent.update_agent(
            s_team_next, a, r, l, learning_params, update_q_function=False
        )
        if centralized_agent.is_task_complete:
            if centralized_agent.is_task_failed:
                failed = True
            break

    if show_print:
        print(
            f"Reward of {testing_reward} achieved in {step} steps. Current step: {tester.current_step} of {tester.total_steps} (stuck for {stuck_counter} steps, stuck in training for {tester.get_training_stuck_counter()/tester.current_step:.4f} steps)"
        )

    if failed:
        step = testing_params.num_steps

    return testing_reward, trajectory, step


def run_centralized_experiment(
    tester: Tester,
    _num_agents: int,
    num_times: int,
    show_print: bool = False,
) -> Tester:
    """
    Run the entire q-learning with reward machines experiment a number of times specified by num_times.

    Inputs
    ------
    tester : Tester object
        Test object holding true reward machine and all information relating
        to the particular tasks, world, learning parameters, and experimental results.
    num_agents : int
        Number of agents in this experiment.
    num_times : int
        Number of times to run the entire experiment (restarting training from scratch).
    show_print : bool
        Flag indicating whether or not to output text to the terminal.
    """

    learning_params = tester.learning_params

    for t in range(num_times):
        # Reseting default step values
        tester.restart()

        rm_test_file = tester.rm_test_file

        # This instance is only used for extracting environment meta-info
        testing_env = create_centralized_environment(tester, use_prm=False, tlcd=None)

        s_i: NDArray[int32] = testing_env.get_initial_team_state()
        actions: NDArray[int32] = testing_env.get_team_action_array()

        centralized_agent = CentralizedAgent(
            rm_test_file,
            s_i,
            testing_env.get_map().get_num_states(),
            actions,
            tlcd=tester.tlcd,
        ).use_prm(tester.use_prm)

        num_episodes = 0

        # Task loop
        epsilon = learning_params.initial_epsilon

        while not tester.stop_learning():
            num_episodes += 1

            # epsilon = epsilon*0.99

            run_qlearning_task(
                epsilon,
                tester,
                centralized_agent,
                show_print=show_print,
            )

        # Backing up the results
        print("Finished iteration ", t)

    return tester
