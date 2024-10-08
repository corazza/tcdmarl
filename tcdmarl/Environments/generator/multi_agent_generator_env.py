import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy import int32
from numpy.typing import NDArray

from tcdmarl.Environments.common import CentralizedEnv
from tcdmarl.Environments.generator.map import GeneratorMap
from tcdmarl.reward_machines.sparse_reward_machine import SparseRewardMachine
from tcdmarl.shared_mem import PRM_TLCD_MAP
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA, ProbabilisticRewardMachine
from tcdmarl.tcrl.utils import sparse_rm_to_prm


class MultiAgentGeneratorEnv(CentralizedEnv):  # TODO rename to CentralizedRoutingEnv
    """
    Multi-agent version.

    Case study 1: TODO
    """

    def __init__(
        self, rm_file: Path, env_settings: Dict[str, Any], tlcd: Optional[CausalDFA]
    ):
        """
        Initialize environment.

        Parameters
        ----------
        rm_file : string
            File path leading to the text file containing the reward machine
            encoding this environment's reward function.
        num_agents : int
            Number of agents in the environment.
        env_settings : dict
            Dictionary of environment settings
        """
        self.num_agents: int = 2
        self.map = GeneratorMap(env_settings)

        # Define Initial states of all agents
        self.s_i: NDArray[int32] = np.full(self.num_agents, -1, dtype=int)
        for i in range(self.num_agents):
            self.s_i[i] = self.map.initial_states[i]

        self.actions = np.full((self.num_agents, len(self.map.actions)), -2, dtype=int)
        for i in range(self.num_agents):
            self.actions[i] = self.map.actions

        self.reward_machine = SparseRewardMachine(rm_file)

        self.u: int = self.reward_machine.get_initial_state()
        self.last_action = np.full(
            self.num_agents, -1, dtype=int
        )  # Initialize last action with garbage values

        # TODO abstract away (into MarlEnv -> CentralizedEnv, DecentralizedEnv)
        # PRMs x TL-CDs
        self._saved_rm_path: Path = rm_file
        self.tlcd = tlcd
        self._use_prm: bool = False
        self.prm: ProbabilisticRewardMachine

    def use_prm(self, value: bool) -> "MultiAgentGeneratorEnv":
        if not value:
            return self

        if self.tlcd is not None:
            save_path = f"{self._saved_rm_path}_TLCD"
        else:
            save_path = f"{self._saved_rm_path}_NO_TLCD"

        if save_path not in PRM_TLCD_MAP:
            self.prm = sparse_rm_to_prm(self.reward_machine)
            if self.tlcd is not None:
                self.prm = self.prm.add_tlcd(self.tlcd, Path(save_path).name)
            PRM_TLCD_MAP[save_path] = self.prm
        else:
            self.prm = copy.deepcopy(PRM_TLCD_MAP[save_path])

        self.u = self.prm.get_initial_state()
        self._use_prm = True
        return self

    def get_map(self) -> GeneratorMap:
        return self.map

    def environment_step(
        self, s: NDArray[int32], a: NDArray[int32]
    ) -> Tuple[int, List[str], NDArray[int32]]:
        """
        Execute collective action a from collective state s. Return the resulting reward,
        mdp label, and next state. Update the last action taken by each agent.

        Parameters
        ----------
        s : numpy integer array
            Array of integers representing the environment states of the various agents.
            s[id] represents the state of the agent indexed by index "id".
        a : numpy integer array
            Array of integers representing the actions selected by the various agents.
            a[id] represents the desired action to be taken by the agent indexed by "id.

        Outputs
        -------
        r : float
            Reward achieved by taking action a from state s.
        l : string
            MDP label emitted this step.
        s_next : numpy integer array
            Array of indeces of next team state.
        """
        s_next = np.full(self.num_agents, -1, dtype=int)

        for i in range(self.num_agents):
            last_action: int
            generator_state = self.map.compute_joint_state(self.get_old_u(self.u))
            s_next[i], last_action = self.map.get_next_state(
                s[i], a[i], i, generator_state
            )
            self.last_action[i] = last_action

        label = self.get_mdp_label(s, s_next, self.u)
        r: int = 0

        for e in label:
            # Get the new reward machine state and the reward of this step

            if not self._use_prm:
                u2 = self.reward_machine.get_next_state(self.u, e)
                rm_out = self.reward_machine.get_reward(self.u, u2)
            else:
                u2 = self.prm.get_next_state(self.u, e)
                rm_out = self.prm.get_reward(self.u, u2)
            r = r + rm_out
            # print(f"{self.u} --- ({e}, {rm_out}) ---> {u2}")
            # Update the reward machine state
            self.u = u2

        # self.show_graphic(s)
        return r, label, s_next

    def get_actions(self, agent_id: int) -> NDArray[int32]:
        """
        Returns the list with the actions that a particular agent can perform.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return np.copy(self.actions[agent_id])

    def get_last_action(self, agent_id: int):
        """
        Returns a particular agent's last action.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return self.last_action[agent_id]

    def get_team_action_array(self) -> NDArray[int32]:
        """
        Returns the available actions of the entire team.

        Outputs
        -------
        actions : (num_agents x num_actions) numpy integer array
        """
        return np.copy(self.actions)

    def get_initial_state(self, agent_id: int) -> int:
        """
        Returns the initial state of a particular agent.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return self.s_i[agent_id]

    def get_initial_team_state(self) -> NDArray[int32]:
        """
        Return the intial state of the collective multi-agent team.

        Outputs
        -------
        s_i : numpy integer array
            Array of initial state indices for the agents in the experiment.
        """
        return np.copy(self.s_i)

    def get_old_u(self, u: int) -> int:
        if not self._use_prm:
            return u
        else:
            if len(self.prm.state_map_after_product) != 0:
                return self.prm.state_map_after_product[u]
            else:
                return u

    ############## DQPRM-RELATED METHODS ########################################
    def get_mdp_label(
        self, _s: NDArray[int32], s_next: NDArray[int32], u: int
    ) -> List[str]:
        """
        Get the mdp label resulting from transitioning from state s to state s_next.

        Parameters
        ----------
        s : numpy integer array
            Array of integers representing the environment states of the various agents.
            s[id] represents the state of the agent indexed by index "id".
        s_next : numpy integer array
            Array of integers representing the next environment states of the various agents.
            s_next[id] represents the next state of the agent indexed by index "id".
        u : int
            Index of the reward machine state

        Outputs
        -------
        l : string
            MDP label resulting from the state transition.
        """

        label: list[str] = []

        agent1 = 0
        agent2 = 1

        row1, col1 = self.map.get_state_description(s_next[agent1])
        row2, col2 = self.map.get_state_description(s_next[agent2])

        if (row1, col1) == self.map.env_settings["A"]:
            label.append("a")

        if (row2, col2) == self.map.env_settings["B"]:
            label.append("b")

        if (row1, col1) == self.map.env_settings["C"]:
            if (row2, col2) not in self.map.yellow_tiles:
                label.append("c")

        return label

    ######################### TROUBLESHOOTING METHODS ################################

    def show(self, s: NDArray[int32]):
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
        display[self.map.env_settings["B3"]] = 9
        display[self.map.env_settings["K1"]] = 9
        display[self.map.env_settings["K2"]] = 9
        display[self.map.env_settings["F1"]] = 9
        display[self.map.env_settings["F2"]] = 9
        display[self.map.env_settings["goal_location"]] = 9

        for loc in self.map.yellow_tiles:
            display[loc] = 8

        # Display the agents
        for i in range(self.num_agents):
            row, col = self.map.get_state_description(s[i])
            display[row, col] = i + 1

        print(display)

    def show_graphic(self, s: NDArray[int32]):
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

        special_cells = {
            "A": self.map.env_settings["A"],
            "B": self.map.env_settings["B"],
            "C": self.map.env_settings["C"],
        }

        for label, loc in special_cells.items():
            display[loc] = 1

        for loc in self.map.yellow_tiles:
            display[loc] = 6

        # Display the agents
        for i in range(self.num_agents):
            row, col = self.map.get_state_description(s[i])
            display[row, col] = 2
            special_cells[f"A{i+1}"] = (row, col)

        # Create a custom color map
        cmap = mcolors.ListedColormap(
            [
                "red",  # -1 walls
                "white",  # 0 empty cells
                "gray",  # 1 special cells
                "cyan",  # 2 agents
                "blue",  # 3 blue_tiles
                "orange",  # 4 orange_tiles
                "green",  # 5 green_tiles
                "yellow",  # 6 yellow_tiles
                "pink",  # 7 pink_tiles
                "purple",  # 8
                "brown",  # 9
                "black",  # 10
            ]
        )
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Use matplotlib to display the gridworld
        plt.imshow(display, cmap=cmap, norm=norm)  # type: ignore

        # Add text labels to the special cells
        for label, loc in special_cells.items():
            plt.text(loc[1], loc[0], label, ha="center", va="center", color="white")  # type: ignore

        # Add coordinates to all other cells
        for row in range(display.shape[0]):
            for col in range(display.shape[1]):
                # Skip cells that already have text
                if (row, col) not in special_cells.values():
                    plt.text(  # type: ignore
                        col,
                        row,
                        f"{row}, {col}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=7,
                    )
        plt.show()  # type: ignore
