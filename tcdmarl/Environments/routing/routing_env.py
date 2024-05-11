from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy import int32
from numpy.typing import NDArray

from tcdmarl.consts import SYNCHRONIZATION_THRESH
from tcdmarl.Environments.common import STR_TO_ACTION, DecentralizedEnv, RoutingMap
from tcdmarl.reward_machines.sparse_reward_machine import SparseRewardMachine
from tcdmarl.routing_config import routing_config
from tcdmarl.shared_mem import PRM_TLCD_MAP
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA, ProbabilisticRewardMachine
from tcdmarl.tcrl.utils import sparse_rm_to_prm


class RoutingEnv(DecentralizedEnv):  # TODO rename to DecentralizedRoutingEnv
    """
    Single-agent version.

    Case study 1: routing environment with two agents and a switch door.
    """

    def __init__(
        self,
        rm_file: Path,
        agent_id: int,
        env_settings: Dict[str, Any],
        tlcd: Optional[CausalDFA],
    ):
        """
        Initialize environment.

        Parameters
        ----------
        rm_file : string
            File path leading to the text file containing the reward machine
            encoding this environment's reward function.
        agent_id : int
            Index {0,1} indicating which agent
        env_settings : dict
            Dictionary of environment settings
        """
        self.agent_id = agent_id
        self.map = RoutingMap(env_settings)

        self.s_i = self.map.initial_states[self.agent_id - 1]

        self.reward_machine = SparseRewardMachine(rm_file)

        self.u = self.reward_machine.get_initial_state()
        self.last_action = -1  # Initialize last action to garbage value

        # PRMs x TL-CDs
        self._saved_rm_path: Path = rm_file
        self.tlcd = tlcd
        self._use_prm: bool = False
        self.prm: ProbabilisticRewardMachine

    def use_prm(self, value: bool) -> "RoutingEnv":
        if not value:
            return self

        if self.tlcd is not None:
            save_path = f"{self._saved_rm_path}_TLCD"
        else:
            save_path = f"{self._saved_rm_path}_NO_TLCD"

        if not save_path in PRM_TLCD_MAP:
            self.prm = sparse_rm_to_prm(self.reward_machine).add_tlcd(self.tlcd)
            PRM_TLCD_MAP[save_path] = self.prm
        else:
            self.prm = PRM_TLCD_MAP[save_path]

        self.u = self.prm.get_initial_state()
        self._use_prm = True
        return self

    def get_old_u(self, u: int) -> int:
        if not self._use_prm:
            return u
        else:
            if len(self.prm.state_map_after_product) != 0:
                return self.prm.state_map_after_product[u]
            else:
                return u

    def environment_step(self, s: int, a: int) -> Tuple[int, List[str], int]:
        """
        Execute action a from state s.

        Parameters
        ----------
        s : int
            Index representing the current environment state.
        a : int
            Index representing the action being taken.

        Outputs
        -------
        r : float
            Reward achieved by taking action a from state s.
        l : list
            List of events occuring at this step.
        s_next : int
            Index of next state.
        """
        routing_state = self.map.compute_state(self.agent_id, self.get_old_u(self.u))
        s_next, last_action = self.map.get_next_state(
            s, a, self.agent_id, routing_state
        )
        self.last_action = last_action

        l = self.get_mdp_label(s, s_next, self.u)
        r: int = 0

        for e in l:
            # Get the new reward machine state and the reward of this step
            if not self._use_prm:
                u2 = self.reward_machine.get_next_state(self.u, e)
                r = r + self.reward_machine.get_reward(self.u, u2)
            else:
                u2 = self.prm.get_next_state(self.u, e)
                r = r + self.prm.get_reward(self.u, u2)

            # Update the reward machine state
            self.u = u2

        return r, l, s_next

    def get_mdp_label(self, _s: int, s_next: int, _u: int) -> List[str]:
        """
        Return the label of the next environment state and current RM state.
        """
        row, col = self.map.get_state_description(s_next)

        l: List[str] = []

        if self.agent_id == 0:
            if (row, col) == self.map.env_settings["B1"]:
                l.append("b1")
            if (row, col) == self.map.env_settings["K1"] or (
                row,
                col,
            ) == self.map.env_settings["K2"]:
                l.append("k")
            if (row, col) == self.map.env_settings["F1"]:
                l.append("f")
            if (row, col) == self.map.env_settings["F2"]:
                l.append("f")
            if (row, col) == self.map.env_settings["goal_location"]:
                l.append("g")

            # Multiagent synchronization
            if np.random.random() <= SYNCHRONIZATION_THRESH:
                l.append("b2")
        else:
            assert self.agent_id == 1
            if (row, col) == self.map.env_settings["B2"]:
                l.append("b2")
            if (row, col) == self.map.env_settings["B3"]:
                l.append("b3")

        return l

    def get_actions(self) -> NDArray[int32]:
        """
        Returns the list with the actions that the agent can perform
        """
        return self.map.actions

    def get_last_action(self):
        """
        Returns agent's last action
        """
        return self.last_action

    def get_initial_state(self):
        """
        Outputs
        -------
        s_i : int
            Index of agent's initial state.
        """
        return self.s_i

    def show(self, s: int):
        """
        Create a visual representation of the current state of the gridworld.

        Parameters
        ----------
        s : int
            Index of the current state
        """
        display = np.zeros((self.map.number_of_rows, self.map.number_of_columns))

        # Display the locations of the walls
        for loc in self.map.env_settings["walls"]:
            display[loc] = -1

        display[self.map.env_settings["B1"]] = 9
        display[self.map.env_settings["B2"]] = 9
        display[self.map.env_settings["K1"]] = 9
        display[self.map.env_settings["K2"]] = 9
        display[self.map.env_settings["F1"]] = 9
        display[self.map.env_settings["F2"]] = 9
        display[self.map.env_settings["goal_location"]] = 9

        for loc in self.map.blue_tiles:
            display[loc] = 8
        for loc in self.map.orange_tiles:
            display[loc] = 8
        for loc in self.map.pink_tiles:
            display[loc] = 8
        for loc in self.map.green_tiles:
            display[loc] = 8
        for loc in self.map.yellow_tiles:
            display[loc] = 8

        # Display the location of the agent in the world
        row, col = self.map.get_state_description(s)
        display[row, col] = self.agent_id

        print(display)


def play():
    agent_id = 0

    tester = routing_config(num_times=0)

    env_settings = tester.env_settings
    env_settings["p"] = 0.99

    game = RoutingEnv(tester.rm_learning_file_list[agent_id], agent_id, env_settings)

    s = game.get_initial_state()

    failed_task_flag = False

    while True:
        # Showing game
        game.show(s)

        # Getting action
        print("\nAction? ", end="")
        a = input()
        print()
        # Executing action
        if a in STR_TO_ACTION:
            r, l, s = game.environment_step(s, STR_TO_ACTION[a])
            # r, l, s, failed_task_flag = game.environment_step(s, str_to_action[a])

            print("---------------------")
            print("Next States: ", s)
            print("Label: ", l)
            print("Reward: ", r)
            print("RM state: ", game.u)
            print("failed task: ", failed_task_flag)
            print("---------------------")

            if game.reward_machine.is_terminal_state(game.u):  # Game Over
                break

        else:
            print("Forbidden action")
    game.show(s)


# This code allow to play a game (for debugging purposes)
if __name__ == "__main__":
    play()
