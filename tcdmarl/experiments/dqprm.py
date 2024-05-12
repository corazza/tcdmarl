"""
Decentralized Q-learning with Projected Reward Machines (DQPRM) experiment.
"""

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from tcdmarl.Agent.agent import Agent
from tcdmarl.Environments.routing.routing_env import RoutingEnv
from tcdmarl.experiments.common import create_centralized_environment
from tcdmarl.tester.learning_params import LearningParameters
from tcdmarl.tester.tester import Tester
from tcdmarl.tester.tester_params import TestingParameters


def run_qlearning_task(
    epsilon: float, tester: Tester, agent_list: List[Agent], show_print: bool = True
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
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
    show_print : bool
        Optional flag indicating whether or not to print output statements to terminal.
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params

    num_agents = len(agent_list)

    for i in range(num_agents):
        agent_list[i].reset_state()
        agent_list[i].initialize_reward_machine()

    num_steps = learning_params.max_timesteps_per_task

    # Load the appropriate environments for training
    if tester.experiment == "routing":
        training_environments: List[RoutingEnv] = []
        for i in range(num_agents):
            training_environments.append(
                RoutingEnv(
                    agent_list[i].rm_file, i, tester.env_settings, tester.tlcd
                ).use_prm(tester.use_prm)
            )
    else:
        raise ValueError('experiment should be one of: "routing"')

    for t in range(num_steps):
        # Update step count
        tester.add_step()

        for i in range(num_agents):
            # Perform a q-learning step.
            if not (agent_list[i].is_task_complete):
                current_u = agent_list[i].u
                s, a = agent_list[i].get_next_action(epsilon, learning_params)
                r, l, s_new = training_environments[i].environment_step(s, a)
                # a = training_environments[i].get_last_action() # due to MDP slip
                agent_list[i].update_agent(s_new, a, r, l, learning_params)

                for u in agent_list[i].all_states:
                    if not (u == current_u) and not (
                        u in agent_list[i].terminal_states
                    ):
                        l = training_environments[i].get_mdp_label(s, s_new, u)
                        # print('l', l)
                        r = 0
                        u_temp = u
                        u2 = u
                        for e in l:
                            # Get the new reward machine state and the reward of this step
                            u2 = agent_list[i].get_next_state(u_temp, e)
                            r = r + agent_list[i].get_reward(u_temp, u2)
                            # Update the reward machine state
                            u_temp = u2
                        agent_list[i].update_q_function(
                            s, s_new, u, u2, a, r, learning_params
                        )

        # If enough steps have elapsed, test and save the performance of the agents.
        if (
            testing_params.test
            and tester.get_current_step() % testing_params.test_freq == 0
        ):
            step = tester.get_current_step()

            agent_list_copy: list[Agent] = []

            # Need to create a copy of the agent for testing. If we pass the agent directly
            # mid-episode to the test function, the test will reset the world-state and reward machine
            # state before the training episode has been completed.
            for i in range(num_agents):
                rm_file = agent_list[i].rm_file
                s_i = agent_list[i].s_i
                actions = agent_list[i].actions
                agent_id = agent_list[i].agent_id
                num_states = agent_list[i].num_states
                agent_copy = Agent(
                    rm_file, s_i, num_states, actions, agent_id, tlcd=tester.tlcd
                ).use_prm(tester.use_prm)
                # Pass only the q-function by reference so that the testing updates the original agent's q-function.
                agent_copy.q = agent_list[i].q
                if tester.use_prm:
                    agent_copy.prm.terminal_states = (
                        agent_copy.prm.original_terminal_states
                    )
                    agent_copy.terminal_states = set(
                        agent_copy.prm.original_terminal_states
                    )

                agent_list_copy.append(agent_copy)

            # Run a test of the performance of the agents
            testing_reward, trajectory, testing_steps = run_multi_agent_qlearning_test(
                agent_list_copy,
                tester,
                learning_params,
                testing_params,
                show_print=show_print,
            )

            # Save the testing reward
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

            # Keep track of the steps taken
            if len(tester.steps) == 0 or tester.steps[-1] < step:
                tester.steps.append(step)

        # If the agents has completed its task, reset it to its initial state.
        if all(agent.is_task_complete for agent in agent_list):
            for i in range(num_agents):
                agent_list[i].reset_state()
                agent_list[i].initialize_reward_machine()

            # Make sure we've run at least the minimum number of training steps before breaking the loop
            if tester.stop_task(t):
                break

        # checking the steps time-out
        if tester.stop_learning():
            break


def run_multi_agent_qlearning_test(
    agent_list: List[Agent],
    tester: Tester,
    learning_params: LearningParameters,
    testing_params: TestingParameters,
    show_print: bool = True,
) -> Tuple[int, List[Dict[str, Any]], int]:
    """
    Run a test of the q-learning with reward machine method with the current q-function.

    Parameters
    ----------
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
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
    num_agents = len(agent_list)
    testing_env = create_centralized_environment(
        tester, use_prm=tester.use_prm, tlcd=tester.tlcd
    )

    for i in range(num_agents):
        agent_list[i].reset_state()
        agent_list[i].initialize_reward_machine()

    s_team = np.full(num_agents, -1, dtype=int)
    for i in range(num_agents):
        s_team[i] = agent_list[i].s
    a_team = np.full(num_agents, -1, dtype=int)
    u_team = np.full(num_agents, -1, dtype=int)
    for i in range(num_agents):
        u_team[i] = agent_list[i].u
    testing_reward = 0

    trajectory: List[Dict[str, Any]] = []
    step = 0
    stuck_counter = 0

    # Starting interaction with the environment
    for _t in range(testing_params.num_steps):
        step = step + 1

        # Perform a team step
        for i in range(num_agents):
            s, a = agent_list[i].get_next_action(-1.0, learning_params)
            s_team[i] = s
            a_team[i] = a
            u_team[i] = agent_list[i].u

        # trajectory.append({'s' : np.array(s_team, dtype=int), 'a' : np.array(a_team, dtype=int), 'u_team': np.array(u_team, dtype=int), 'u': int(testing_env.u)})

        r, l, s_team_next = testing_env.environment_step(s_team, a_team)
        for s_agent in s_team_next:
            (row, col) = testing_env.get_map().get_state_description(s_agent)
            if (row, col) in testing_env.get_map().sinks:
                stuck_counter += 1
        # testing_env.show_graphic(s_team_next)

        testing_reward = testing_reward + r

        projected_l_dict: Dict[int, List[Any]] = {}
        for i in range(num_agents):
            # Agent i's projected label is the intersection of l with its local event set
            projected_l_dict[i] = list(set(agent_list[i].local_event_set) & set(l))
            # Check if the event causes a transition from the agent's current RM state
            if not (agent_list[i].is_local_event_available(projected_l_dict[i])):
                projected_l_dict[i] = []

        for i in range(num_agents):
            # Enforce synchronization requirement on shared events
            if projected_l_dict[i]:
                for event in projected_l_dict[i]:
                    # testing_env.show(s_team)
                    for j in range(num_agents):
                        if (event in set(agent_list[j].local_event_set)) and (
                            not (projected_l_dict[j] == projected_l_dict[i])
                        ):
                            projected_l_dict[i] = []

            # update the agent's internal representation
            # a = testing_env.get_last_action(i)
            agent_list[i].update_agent(
                s_team_next[i],
                a_team[i],
                r,
                projected_l_dict[i],
                learning_params,
                update_q_function=False,
            )

        if all(agent.is_task_complete for agent in agent_list):
            break

    if show_print:
        print(
            f"Reward of {testing_reward} achieved in {step} steps. Current step: {tester.current_step} of {tester.total_steps} (stuck for {stuck_counter} steps)"
        )

    return testing_reward, trajectory, step


def run_multi_agent_experiment(
    tester: Tester, num_agents: int, num_times: int, show_print: bool = True
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

    assert num_times > 0
    agent_list: List[Agent] = []

    for t in range(num_times):
        # Reseting default step values
        tester.restart()

        rm_learning_file_list = tester.rm_learning_file_list

        # Verify that the number of local reward machines matches the number of agents in the experiment.
        assert (
            len(tester.rm_learning_file_list) == num_agents
        ), "Number of specified local reward machines must match specified number of agents."

        # This is instance is only used for extracting environment meta-info
        testing_env = create_centralized_environment(tester, use_prm=False, tlcd=None)
        num_states = testing_env.get_map().get_num_states()

        # Create the a list of agents for this experiment
        agent_list = []
        for i in range(num_agents):
            actions = testing_env.get_actions(i)
            s_i = testing_env.get_initial_state(i)
            agent_list.append(
                Agent(
                    rm_learning_file_list[i], s_i, num_states, actions, i, tester.tlcd
                ).use_prm(tester.use_prm)
            )

        num_episodes = 0

        # Task loop
        epsilon = learning_params.initial_epsilon

        while not tester.stop_learning():
            num_episodes += 1

            # epsilon = epsilon*0.99

            run_qlearning_task(epsilon, tester, agent_list, show_print=show_print)

        # Backing up the results
        print("Finished iteration ", t)

    tester.agent_list = agent_list

    return tester
