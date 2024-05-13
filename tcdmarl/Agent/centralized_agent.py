import copy
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
from numpy import int32
from numpy.typing import NDArray

from tcdmarl.reward_machines.sparse_reward_machine import SparseRewardMachine
from tcdmarl.shared_mem import PRM_TLCD_MAP
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA, ProbabilisticRewardMachine
from tcdmarl.tcrl.utils import sparse_rm_to_prm
from tcdmarl.tester.learning_params import LearningParameters


class CentralizedAgent:
    """
    Class meant to represent a centralized learning algorithm for a team of agents.
    The agent maintains a representation of its own q-function and accumulated reward
    which are updated across training episodes.
    The agent also has a representation of its own reward machine, which it uses
    for learning, and of its state in the world/reward machine.

    Note: Users of this class must manually reset the world state and the reward machine
    state when starting a new episode by calling self.initialize_world() and
    self.initialize_reward_machine().
    """

    def __init__(
        self,
        rm_file: Path,
        s_i: NDArray[int32],
        num_states: int,
        actions: NDArray[int32],
        tlcd: Optional[CausalDFA],
    ):
        """
        Initialize agent object.

        Parameters
        ----------
        rm_file : str
            File path pointing to the reward machine this agent is meant to use for learning.
        s_i : (num_agents x 1) numpy integer array
            s_i[j] returns the initial state of the agent indexed by j.
        actions : (num_agents x num_actions) numpy integer array
            actions[j] Returns a vector of actions available to agent j.
        """
        self.rm_file = rm_file
        self.s_i = np.copy(s_i)
        self.s = np.copy(s_i)
        self.actions = np.copy(actions)
        self.num_states = num_states

        self.num_agents = np.shape(s_i)[0]

        self.rm = SparseRewardMachine(self.rm_file)
        self.u = self.rm.get_initial_state()
        self.all_states: List[int] = self.rm.all_states
        self.terminal_states: Set[int] = self.rm.terminal_states

        num_states = self.num_states  # Number of states in the gridworld

        # Create a list of the dimensions of the centralized q-function
        # E.g. for a 2 agent team, q should have dimensions SxSxUxAxA
        q_shape: List[int] = []
        for i in range(self.num_agents):
            q_shape.append(num_states)
        q_shape.append(len(self.rm.all_states))
        for i in range(self.num_agents):
            q_shape.append(len(self.actions[i]))

        self.q = np.zeros(q_shape)
        self.total_local_reward = 0
        self.is_task_complete = 0
        self.is_task_failed = 0

        # PRMs x TL-CDs
        self._saved_rm_path: Path = self.rm_file
        self.tlcd = tlcd
        self._use_prm: bool = False
        self.prm: ProbabilisticRewardMachine

    def get_next_state(self, u: int, e: str) -> int:
        if not self._use_prm:
            return self.rm.get_next_state(u, e)
        else:
            return self.prm.get_next_state(u, e)

    def get_reward(self, u1: int, u2: int) -> int:
        if not self._use_prm:
            return self.rm.get_reward(u1, u2)
        else:
            return self.prm.get_reward(u1, u2)

    def use_prm(self, value: bool) -> "CentralizedAgent":
        if not value:
            return self

        if self.tlcd is not None:
            save_path = f"{self._saved_rm_path}_TLCD"
        else:
            save_path = f"{self._saved_rm_path}_NO_TLCD"

        if not save_path in PRM_TLCD_MAP:
            self.prm = sparse_rm_to_prm(self.rm)
            if self.tlcd is not None:
                self.prm = self.prm.add_tlcd(self.tlcd)
            PRM_TLCD_MAP[save_path] = self.prm
        else:
            self.prm = copy.deepcopy(PRM_TLCD_MAP[save_path])

        self.u = self.prm.get_initial_state()
        self.all_states = list(self.prm.all_states)
        self.terminal_states = set(self.prm.terminal_states)

        num_states = self.num_states  # Number of states in the gridworld

        # Create a list of the dimensions of the centralized q-function
        # E.g. for a 2 agent team, q should have dimensions SxSxUxAxA
        q_shape: List[int] = []
        for i in range(self.num_agents):
            q_shape.append(num_states)
        q_shape.append(len(self.prm.all_states))
        for i in range(self.num_agents):
            q_shape.append(len(self.actions[i]))

        self.q = np.zeros(q_shape)
        self.total_local_reward = 0
        self.is_task_complete = 0
        self.is_task_failed = 0

        self._use_prm = True
        return self

    def reset_state(self):
        """
        Reset the agent to the initial state of the environm ent.
        """
        self.s = np.copy(self.s_i)

    def initialize_reward_machine(self):
        """
        Reset the state of the reward machine to the initial state and reset task status.
        """
        if not self._use_prm:
            self.u = self.rm.get_initial_state()
        else:
            self.u = self.prm.get_initial_state()
        self.is_task_complete = 0
        self.is_task_failed = 0

    def get_next_action(
        self, epsilon: float, learning_params: LearningParameters
    ) -> Tuple[NDArray[int32], NDArray[int32]]:
        """
        Return the action next action selected by the agent.

        Outputs
        -------
        s : numpy integer array
            s[i] represents the state of agent i
        a : numpy integer array
            a[i] represents the action taken by agent i
        """

        t_param = learning_params.t_param

        if random.random() < epsilon:
            # With probability epsilon, randomly select an action for each agent.
            a_selected = np.full(self.num_agents, -1, dtype=int)
            for i in range(self.num_agents):
                a_selected[i] = random.choice(self.actions[i])
        else:
            partial_index = (
                []
            )  # Don't include action indexes. As a result, in pr_sum, we are summing over actions.
            for i in range(self.num_agents):
                partial_index.append(self.s[i])
            partial_index.append(self.u)
            partial_index = tuple(partial_index)

            # Sum over all possible actions for fixed team state and reward machine state.
            pr_sum = np.sum(np.exp(self.q[partial_index] * t_param))

            # pr[i] is an array representing the probability values that agent i will take various actions.
            pr = np.exp(self.q[partial_index] * t_param) / pr_sum

            shp = pr.shape
            pr = pr.flatten()

            pr_select = np.zeros([len(pr) + 1, 1])
            pr_select[0] = 0
            for i in range(len(pr)):
                pr_select[i + 1] = pr_select[i] + pr[i]

            randn = random.random()
            for i in range(len(pr)):
                if randn >= pr_select[i] and randn <= pr_select[i + 1]:
                    a_selected = np.unravel_index(i, shp)
                    a_selected = np.array(a_selected, dtype=int)
                    break

        a = a_selected

        return self.s, a

    def update_agent(
        self,
        s_new: NDArray[int32],
        a: NDArray[int32],
        reward: int,
        label: List[str],
        learning_params: LearningParameters,
        update_q_function: bool = True,
    ):
        """
        Update the agent's state, q-function, and reward machine after
        interacting with the environment.

        Parameters
        ----------
        s_new : int
            Index of the agent's next state.
        a : int
            Action the agent took from the last state.
        reward : float
            Reward the agent achieved during this step.
        label : string
            Label returned by the MDP this step.
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        """

        u_start = self.u
        for e in label:
            if not self._use_prm:
                u2 = self.rm.get_next_state(self.u, e)
            else:
                u2 = self.prm.get_next_state(self.u, e)
            self.u = u2

        self.total_local_reward += reward

        if update_q_function == True:
            self.update_q_function(
                self.s, s_new, u_start, self.u, a, reward, learning_params
            )

        # Moving to the next state
        self.s = s_new

        if not self._use_prm:
            if self.rm.is_terminal_state(self.u):
                # Completed task. Set flag.
                self.is_task_complete = 1
        else:
            if self.prm.is_terminal_state(self.u):
                # Completed task. Set flag.
                self.is_task_complete = 1
                if self.u not in self.prm.original_terminal_states:
                    self.is_task_failed = 1

    def update_q_function(self, s, s_new, u, u_new, a, reward, learning_params):
        """
        Update the q function using the action, states, and reward value.

        Parameters
        ----------
        s : array
            Indeces of the agents' previous state
        s_new : array
            Indeces of the agents' updated state
        u : int
            Index of the agent's previous RM state
        u_new : int
            Index of the agent's updated RM state
        a : array
            Actions the agent took from state s
        reward : float
            Reward the agent achieved during this step
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        """
        alpha = learning_params.alpha
        gamma = learning_params.gamma

        ind = self.get_q_function_index(s, u, a)
        partial_ind = []
        for i in range(self.num_agents):
            partial_ind.append(s_new[i])
        partial_ind.append(u_new)
        partial_ind = tuple(partial_ind)

        # Bellman update
        self.q[ind] = (1 - alpha) * self.q[ind] + alpha * (
            reward + gamma * np.amax(self.q[partial_ind])
        )

    def get_q_function_index(self, s, u, a):
        """
        Get the index to be passed into the q-function to reference
        the team-state action pair associated with (s,a).

        Parameters
        ----------
        s : numpy integer array
            s[i] represents the state of agent i.
        u : int
            Index of the reward machine state.
        a : numpy integer array
            a[i] represents the action of agent i.

        Output
        ------
        ind : tuple
            Tuple to be passed into the q-function to reference the corresponding
            q-value. q[ind]
        """
        ind = []
        for i in range(self.num_agents):
            ind.append(s[i])
        ind.append(u)
        for i in range(self.num_agents):
            ind.append(a[i])

        return tuple(ind)
