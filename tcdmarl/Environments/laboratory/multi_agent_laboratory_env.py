import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy import int32
from numpy.typing import NDArray

from tcdmarl.environment_configs.generator_config import generator_config
from tcdmarl.Environments.common import STR_TO_ACTION, Actions, CentralizedEnv
from tcdmarl.Environments.generator.map import GeneratorMap
from tcdmarl.reward_machines.sparse_reward_machine import SparseRewardMachine
from tcdmarl.shared_mem import PRM_TLCD_MAP
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA, ProbabilisticRewardMachine
from tcdmarl.utils import sparse_rm_to_prm


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
        s : NDArray[int32]
            The current state(s) of the agent(s).
        """
        # Initialize the display grid with zeros
        display = np.zeros(
            (self.map.number_of_rows, self.map.number_of_columns), dtype=int
        )

        # Display the locations of the walls
        for loc in self.map.env_settings["walls"]:
            display[loc] = -1  # Walls are marked as -1

        # Mark special cells (A, B, C)
        special_cells = ["A", "B", "C"]
        for cell in special_cells:
            display[self.map.env_settings[cell]] = 9  # Mark A, B, C as 9 in the grid

        # Mark yellow tiles
        for loc in self.map.yellow_tiles:
            display[loc] = 8  # Yellow tiles are marked as 8

        # Display one-way doors
        one_way_doors = self.map.env_settings["oneway"]
        for _direction, locations in one_way_doors.items():
            for loc in locations:
                display[loc] = (
                    5  # Mark one-way doors as 5 (you can change the number if desired)
                )

        # Display the agents
        for i in range(self.num_agents):
            row, col = self.map.get_state_description(s[i])
            display[row, col] = (
                i + 1
            )  # Agents are marked as 1, 2, etc., based on agent index

        # Print the gridworld state
        print(display)

    def show_graphic(self, s: NDArray[int32]):
        """
        Create a visual representation of the current state of the gridworld.

        Parameters
        ----------
        s : NDArray[int32]
            The current state(s) of the agent(s).
        """
        # Constants for cell values
        WALL = -1
        EMPTY = 0
        SPECIAL = 1
        AGENT = 2
        YELLOW_TILE = 6
        ARROW = 3  # New constant for arrows (background will be white)

        # Initialize the display grid
        display = np.full((self.map.number_of_rows, self.map.number_of_columns), EMPTY)

        # Place walls on the grid
        for loc in self.map.env_settings["walls"]:
            display[loc] = WALL

        # Dictionary to hold labels and their positions
        special_cells = {
            "A": self.map.env_settings["A"],
            "B": self.map.env_settings["B"],
            "C": self.map.env_settings["C"],
        }

        # Mark special cells on the grid
        for label, loc in special_cells.items():
            display[loc] = SPECIAL

        # Mark yellow tiles on the grid
        for loc in self.map.yellow_tiles:
            display[loc] = YELLOW_TILE

        # Place agents on the grid and update special_cells with agent positions
        for i in range(self.num_agents):
            row, col = self.map.get_state_description(s[i])
            display[row, col] = AGENT
            agent_label = f"A{i+1}"  # Short label for agents
            special_cells[agent_label] = (row, col)

        # Display one-way doors as arrows in the grid (background will be white)
        arrow_symbols = {
            Actions.UP: "↑",
            Actions.DOWN: "↓",
            Actions.LEFT: "←",
            Actions.RIGHT: "→",
        }
        arrow_locations = {}
        for direction, locations in self.map.env_settings["oneway"].items():
            arrow_symbol = arrow_symbols.get(direction, "")
            for row, col in locations:
                display[row, col] = ARROW  # Background is white for arrows
                arrow_locations[(row, col)] = arrow_symbol

        # Define colors for each cell type, including arrows
        cell_colors = [
            "red",  # WALL (-1)
            "white",  # EMPTY (0)
            "gray",  # SPECIAL (1)
            "cyan",  # AGENT (2)
            "white",  # ARROW (background is white) (3)
            "yellow",  # YELLOW_TILE (6)
        ]
        cmap = mcolors.ListedColormap(cell_colors)
        norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 6.5], cmap.N)

        # Create a figure and axis
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Create a mesh grid for pcolormesh
        x = np.arange(self.map.number_of_columns + 1)
        y = np.arange(self.map.number_of_rows + 1)
        X, Y = np.meshgrid(x, y)

        # Plot the grid using pcolormesh
        mesh = ax.pcolormesh(
            X, Y, display, cmap=cmap, norm=norm, edgecolors="k", linewidth=0.5
        )

        # Set axis labels and ticks
        ax.set_xticks(np.arange(0.5, self.map.number_of_columns, 1))
        ax.set_yticks(np.arange(0.5, self.map.number_of_rows, 1))
        ax.set_xticklabels(np.arange(self.map.number_of_columns))
        ax.set_yticklabels(np.arange(self.map.number_of_rows))
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.set_xlim(0, self.map.number_of_columns)
        ax.set_ylim(0, self.map.number_of_rows)
        ax.set_aspect("equal")
        ax.invert_yaxis()

        # Add text labels to special cells, agents, and arrows
        for label, (row, col) in special_cells.items():
            ax.text(
                col + 0.5,
                row + 0.5,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=12,
            )
        for (row, col), symbol in arrow_locations.items():
            ax.text(
                col + 0.5,
                row + 0.5,
                symbol,
                ha="center",
                va="center",
                color="blue",  # Arrows are blue
                fontsize=25,
                fontweight="bold",
            )

        # Add coordinates to empty cells
        for row in range(display.shape[0]):
            for col in range(display.shape[1]):
                # Only add coordinates to empty cells
                if (
                    (row, col) not in special_cells.values()
                    and (row, col) not in arrow_locations
                    and display[row, col] == EMPTY
                ):
                    ax.text(
                        col + 0.5,
                        row + 0.5,
                        f"{row},{col}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=7,
                    )

        plt.title("Gridworld State")
        plt.show()


def play():
    tester = generator_config(num_times=0, use_tlcd=False, step_unit_factor=100)

    env_settings = tester.env_settings
    env_settings["p"] = 1.0

    num_agents: int = tester.num_agents

    game = MultiAgentGeneratorEnv(tester.rm_test_file, env_settings, tlcd=None)

    s = game.get_initial_team_state()
    print(s)

    while True:
        # Showing game
        game.show(s)

        # Getting action
        a = np.full(num_agents, -1, dtype=int)

        for i in range(num_agents):
            print("\nAction{}? ".format(i + 1), end="")
            usr_inp = input()
            print()

            if usr_inp not in STR_TO_ACTION:
                print("forbidden action")
                a[i] = STR_TO_ACTION["x"]
            else:
                print(STR_TO_ACTION[usr_inp])
                a[i] = STR_TO_ACTION[usr_inp]

        r, l, s = game.environment_step(s, a)

        print("---------------------")
        print("Next States: ", s)
        print("Label: ", l)
        print("Reward: ", r)
        print("RM state: ", game.u)
        print("---------------------")

        if game.reward_machine.is_terminal_state(game.u):  # Game Over
            break

    game.show(s)


# This code allow to play a game (for debugging purposes)
if __name__ == "__main__":
    play()
