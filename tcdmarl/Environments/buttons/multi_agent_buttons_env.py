import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy import int32
from numpy.typing import NDArray

from tcdmarl.Environments.common import STR_TO_ACTION, CentralizedEnv, ButtonsMap
from tcdmarl.reward_machines.sparse_reward_machine import SparseRewardMachine
from tcdmarl.buttons_config import buttons_config
from tcdmarl.shared_mem import PRM_TLCD_MAP
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA, ProbabilisticRewardMachine
from tcdmarl.tcrl.utils import sparse_rm_to_prm


class MultiAgentButtonsEnv(CentralizedEnv):  # TODO rename to CentralizedButtonsEnv
    """
    Multi-agent version.

    Case study 2: buttons environment with two agents and a switch door.
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
        self.num_agents: int = 3
        self.map = ButtonsMap(env_settings)

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

        # PRMs x TL-CDs
        self._saved_rm_path: Path = rm_file
        self.tlcd = tlcd
        self._use_prm: bool = False
        self.prm: ProbabilisticRewardMachine

    def use_prm(self, value: bool) -> "MultiAgentButtonsEnv":
        if not value:
            return self

        if self.tlcd is not None:
            save_path = f"{self._saved_rm_path}_TLCD"
        else:
            save_path = f"{self._saved_rm_path}_NO_TLCD"

        if not save_path in PRM_TLCD_MAP:
            self.prm = sparse_rm_to_prm(self.reward_machine)
            if self.tlcd is not None:
                self.prm = self.prm.add_tlcd(self.tlcd, Path(save_path).name)
            PRM_TLCD_MAP[save_path] = self.prm
        else:
            self.prm = copy.deepcopy(PRM_TLCD_MAP[save_path])

        self.u = self.prm.get_initial_state()
        self._use_prm = True
        return self

    def get_map(self) -> ButtonsMap:
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
            buttons_state = self.map.compute_joint_state(self.get_old_u(self.u))
            s_next[i], last_action = self.map.get_next_state(
                s[i], a[i], i, buttons_state=buttons_state
            )
            self.last_action[i] = last_action

        l = self.get_mdp_label(s, s_next, self.get_old_u(self.u))
        r: int = 0

        # row1, col1 = self.map.get_state_description(s_next[1])

        for e in l:
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
        return r, l, s_next

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

        l: List[str] = []

        agent1 = 0
        agent2 = 1
        agent3 = 2

        row1, col1 = self.map.get_state_description(s_next[agent1])
        row2, col2 = self.map.get_state_description(s_next[agent2])
        row3, col3 = self.map.get_state_description(s_next[agent3])


        if (row1, col1) == self.map.env_settings["F1"]:
            l.append("f")
        # if (row1, col1) == self.map.env_settings["F2"] and self.map.env_settings[
        #     "enable_f2"
        # ]:
        #     l.append("f")

        if u == 0:
        # Now check if agents are on buttons
            if not ((row2, col2) in self.map.yellow_tiles) and (row1,col1) == self.map.env_settings['yellow_button']:
                l.append('by')
        if u == 1:
            if not ((row3, col3) in self.map.green_tiles) and (row2, col2) == self.map.env_settings['green_button1']:
                # print('L in line 385:', 'hi1')
                l.append('bg1')
            if not ((row3, col3) in self.map.green_tiles) and (row2, col2) == self.map.env_settings['green_button2']:
                l.append('bg2')
        if u == 2:
            if (row2, col2) == self.map.env_settings['red_button']:
                l.append('a2br')
            if (row3, col3) == self.map.env_settings['red_button']:
                l.append('a3br')
        if u == 3:
            if not ((row2, col2) == self.map.env_settings['red_button']):
                l.append('a2lr')
            if (row3, col3) == self.map.env_settings['red_button']:
                l.append('a3br')
        if u == 4:
            if (row2, col2) == self.map.env_settings['red_button']:
                l.append('a2br')
            if not ((row3, col3) == self.map.env_settings['red_button']):
                l.append('a3lr')
        if u == 5:
            if ((row2, col2) == self.map.env_settings['red_button']) and ((row3, col3) == self.map.env_settings['red_button']):
                l.append('br')
            if not ((row2, col2) == self.map.env_settings['red_button']):
                l.append('a2lr')
            if not ((row3, col3) == self.map.env_settings['red_button']):
                l.append('a3lr')
        if u == 6:
            # Check if agent 1 has reached the goal
            if (row1, col1) == self.map.env_settings['goal_location']:
                l.append('g')

        return l

    ################## HRL-RELATED METHODS ######################################
    def get_options_list(self, agent_id: int):
        """
        Return a list of strings representing the possible options for each agent.

        Input
        -----
        agent_id : int
            The id of the agent whose option list is to be returned.

        Output
        ------
        options_list : list
            list of strings representing the options avaialble to the agent.
        """

        agent1 = 0
        agent2 = 1

        options_list: List[str] = []

        if agent_id == agent1:
            options_list.append("w1")
            options_list.append("b1")
            options_list.append("k")
            options_list.append("f")
            options_list.append("g")

        if agent_id == agent2:
            options_list.append("w2")
            options_list.append("b2")

        return options_list

    def get_old_u(self, u: int) -> int:
        if not self._use_prm:
            return u
        else:
            if len(self.prm.state_map_after_product) != 0:
                return self.prm.state_map_after_product[u]
            else:
                return u

    def get_avail_options(self, agent_id: int):
        """
        Given the current metastate, get the available options. Some options are unavailable if
        they are not possible to complete at the current stage of the task. In such circumstances
        we don't want the agents to update the corresponding option q-functions.
        """
        agent1 = 0
        agent2 = 1

        avail_options: List[str] = []
        routing_state = self.map.compute_joint_state(self.get_old_u(self.u))

        if agent_id == agent1:
            avail_options.append("w1")
            avail_options.append("b1")
            if routing_state.b1_pressed or routing_state.b2_pressed:
                avail_options.append("k")  # K1
            if not routing_state.b3_pressed:
                avail_options.append("f")
            else:
                avail_options.append("g")
        if agent_id == agent2:
            avail_options.append("w2")
            avail_options.append("b3")
            if routing_state.b3_pressed:
                avail_options.append("b2")

        return avail_options

    def get_avail_meta_action_indeces(self, agent_id: int) -> List[int]:
        """
        Get a list of the indeces corresponding to the currently available meta-action/option
        """
        avail_options = self.get_avail_options(agent_id)
        all_options_list = self.get_options_list(agent_id)
        avail_meta_action_indeces: List[int] = []
        for option in avail_options:
            avail_meta_action_indeces.append(all_options_list.index(option))
        return avail_meta_action_indeces

    def get_completed_options(self, s: NDArray[int32]) -> List[str]:
        """
        Get a list of strings corresponding to options that are deemed complete in the team state described by s.

        Parameters
        ----------
        s : numpy integer array
            Array of integers representing the environment states of the various agents.
            s[id] represents the state of the agent indexed by index "id".

        Outputs
        -------
        completed_options : list
            list of strings corresponding to the completed options.
        """
        agent1 = 0
        agent2 = 1
        agent3 = 2

        completed_options: List[str] = []

        for i in range(self.num_agents):
            row, col = self.map.get_state_description(s[i])
            

            if i == agent1:
                if (row,col) == self.map.env_settings['yellow_button']:
                    completed_options.append('by')
                if (row, col) == self.map.env_settings['goal_location']:
                    completed_options.append('g')
                if s[i] == self.map.env_settings['initial_states'][i]:
                    completed_options.append('w1')
                if (row, col) == self.map.env_settings["F1"]:
                    completed_options.append("f")
                if (row, col) == self.map.env_settings["F2"] and self.map.env_settings[
                    "enable_f2"
                ]:
                    completed_options.append("f")

            elif i == agent2:
                if (row, col) == self.map.env_settings['green_button1']:
                    completed_options.append('bg1')
                if (row, col) == self.map.env_settings['green_button2']:
                    completed_options.append('bg2')
                if (row, col) == self.map.env_settings['red_button']:
                    completed_options.append('a2br')
                if s[i] == self.map.env_settings['initial_states'][i]:
                    completed_options.append('w2')

            elif i == agent3:
                if (row, col) == self.map.env_settings['red_button']:
                    completed_options.append('a3br')
                if s[i] == self.map.env_settings['initial_states'][i]:
                    completed_options.append('w3')
        
        return completed_options

    def get_meta_state(self, _agent_id: int):
        """
        Return the meta-state that the agent should use for it's meta controller.

        Input
        -----
        s_team : numpy array
            s_team[i] is the state of agent i.
        agent_id : int
            Index of agent whose meta-state is to be returned.

        Output
        ------
        meta_state : int
            Index of the meta-state.
        """
        routing_state = self.map.compute_joint_state(self.get_old_u(self.u))
        # Convert the Truth values of which buttons have been pushed to an int
        meta_state = int(
            f"{int(routing_state.b1_pressed)}{int(routing_state.b2_pressed)}{int(routing_state.b3_pressed)}{int(routing_state.key_collected)}",
            2,
        )

        # meta_state = int('{}{}{}'.format(int(self.orange_button_pushed), int(self.green_button_pushed), int(self.yellow_button_pushed)), 2)

        # # if the task has been failed, return 8
        # if self.u == 6:
        #     meta_state = 8

        # meta_state = self.u

        return meta_state

    def get_num_meta_states(self, _agent_id: int) -> int:
        """
        Return the number of meta states for the agent specified by agent_id.
        """
        # how many different combinations of button presses are there
        # return int(8) # TODO check
        raise NotImplementedError("check this")

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
            "yellow_button": self.map.env_settings["yellow_button"],
            "green_button1": self.map.env_settings["green_button1"],
            "green_button2": self.map.env_settings["green_button2"],
            "red_button": self.map.env_settings["red_button"],
            "G": self.map.env_settings["goal_location"],
            "F1": self.map.env_settings["F1"],
            "F2": self.map.env_settings["F2"],
        }

        for label, loc in special_cells.items():
            display[loc] = 1

        for loc in self.map.red_tiles:
            display[loc] = 4
        for loc in self.map.green_tiles:
            display[loc] = 5
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


def play():
    tester = buttons_config(num_times=0)

    env_settings = tester.env_settings
    env_settings["p"] = 1.0

    num_agents: int = tester.num_agents

    game = MultiAgentButtonsEnv(tester.rm_test_file, env_settings)

    s = game.get_initial_team_state()
    print(s)

    while True:
        # Showing game
        game.show(s)

        # Getting action
        a = np.full(num_agents, -1, dtype=int)

        for i in range(num_agents):
            print("\nAction{}?".format(i + 1), end="")
            usr_inp = input()
            print()

            if not (usr_inp in STR_TO_ACTION):
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
        print("Meta state: ", game.get_meta_state(0))
        print("---------------------")

        if game.reward_machine.is_terminal_state(game.u):  # Game Over
            break
    game.show(s)


# This code allow to play a game (for debugging purposes)
if __name__ == "__main__":
    play()
