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

from enum import Enum
import random

class Actions(Enum):
    up    = 0 # move up
    right = 1 # move right
    down  = 2 # move down
    left  = 3 # move left
    none  = 4 # none 

class MultiAgentButtonsEnv(CentralizedEnv):

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
        self.env_settings = env_settings
        self.num_agents = 3
        self.map = ButtonsMap(env_settings)
        self.p = env_settings["p"]
        self.Nr = self.env_settings["Nr"]
        self.Nc = self.env_settings["Nc"]
        self.yellow_tiles = self.env_settings["yellow_tiles"]
        self.green_tiles = self.env_settings["green_tiles"]
        self.red_tiles = self.env_settings["red_tiles"]
        
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
            self.prm = sparse_rm_to_prm(self.reward_machine).add_tlcd(self.tlcd)
            PRM_TLCD_MAP[save_path] = self.prm
        else:
            self.prm = PRM_TLCD_MAP[save_path]

        self.u = self.prm.get_initial_state()
        self._use_prm = True
        return self
    

    def get_map(self) -> ButtonsMap:
        return self.map

    # def _load_map(self):
    #     """
    #     Initialize the environment.
    #     """
    #     self.Nr = self.env_settings['Nr']
    #     self.Nc = self.env_settings['Nc']

    #     initial_states = self.env_settings['initial_states']

    #     # Define Initial states of all agents
    #     self.s_i = np.full(self.num_agents, -1, dtype=int)
    #     for i in range(self.num_agents):
    #         self.s_i[i] = initial_states[i]
    #     self.objects = {}
    #     self.objects[self.env_settings['goal_location']] = "g" # goal location
    #     self.objects[self.env_settings['yellow_button']] = 'by'
    #     self.objects[self.env_settings['green_button1']] = 'bg1'
    #     self.objects[self.env_settings['green_button2']] = 'bg2'
    #     self.objects[self.env_settings['red_button']] = 'rb'
    #     self.yellow_tiles = self.env_settings['yellow_tiles']
    #     self.green_tiles = self.env_settings['green_tiles']
    #     self.red_tiles = self.env_settings['red_tiles']

    #     self.p = self.env_settings['p']

    #     # Set the available actions of all agents. For now all agents have same action set.
    #     actions = np.array([Actions.up.value, Actions.right.value, Actions.left.value, Actions.down.value, Actions.none.value], dtype=int)
    #     self.actions = np.full((self.num_agents, len(actions)), -2, dtype=int)
    #     for i in range(self.num_agents):
    #         self.actions[i] = actions

    #     self.num_states = self.Nr * self.Nc
        
    #     # Define forbidden transitions corresponding to map edges
    #     self.forbidden_transitions = set()
        
    #     for row in range(self.Nr):
    #         self.forbidden_transitions.add((row, 0, Actions.left)) # If in left-most column, can't move left.
    #         self.forbidden_transitions.add((row, self.Nc - 1, Actions.right)) # If in right-most column, can't move right.
    #     for col in range(self.Nc):
    #         self.forbidden_transitions.add((0, col, Actions.up)) # If in top row, can't move up
    #         self.forbidden_transitions.add((self.Nr - 1, col, Actions.down)) # If in bottom row, can't move down

    #     wall_locations = self.env_settings['walls']
    #     # Restrict agent from having the option of moving "into" a wall
    #     for i in range(len(wall_locations)):
    #         (row, col) = wall_locations[i]
    #         self.forbidden_transitions.add((row, col + 1, Actions.left))
    #         self.forbidden_transitions.add((row, col-1, Actions.right))
    #         self.forbidden_transitions.add((row+1, col, Actions.up))
    #         self.forbidden_transitions.add((row-1, col, Actions.down))

    #     #one_way_door_location = self.env_settings['oneway']
    #     self.forbidden_transitions.add((4,3, Actions.up))
    #     self.forbidden_transitions.add((4,4, Actions.up))
    #     self.forbidden_transitions.add((4,5, Actions.up))
    #     self.forbidden_transitions.add((4,6, Actions.up))
    #     self.forbidden_transitions.add((4,7, Actions.up))
    #     self.forbidden_transitions.add((4,9, Actions.up))

    def environment_step(self, s, a):
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

        l = self.get_mdp_label(s, s_next, self.u)
        r = 0

        for e in l:
            # Get the new reward machine state and the reward of this step
            u2 = self.reward_machine.get_next_state(self.u, e)
            # if e=='bg1':
            #     r = -100
            r = r + self.reward_machine.get_reward(self.u, u2)
            # Update the reward machine state
            self.u = u2

        return r, l, s_next

    def get_next_state(self, s, a, agent_id):
        """
        Get the next state in the environment given action a is taken from state s.
        Update the last action that was truly taken due to MDP slip.

        Parameters
        ----------
        s : int
            Index of the current state.
        a : int
            Action to be taken from state s.

        Outputs
        -------
        s_next : int
            Index of the next state.
        last_action : int
            Last action the agent truly took because of slip probability.
        """
        slip_p = [self.p, (1-self.p)/2, (1-self.p)/2]
        check = random.random()

        row, col = self.get_state_description(s)

        self.forbidden_transitions = set()
        
        for row in range(self.Nr):
            self.forbidden_transitions.add((row, 0, Actions.left)) # If in left-most column, can't move left.
            self.forbidden_transitions.add((row, self.Nc - 1, Actions.right)) # If in right-most column, can't move right.
        for col in range(self.Nc):
            self.forbidden_transitions.add((0, col, Actions.up)) # If in top row, can't move up
            self.forbidden_transitions.add((self.Nr - 1, col, Actions.down)) # If in bottom row, can't move down

        wall_locations = self.env_settings['walls']
        # Restrict agent from having the option of moving "into" a wall
        for i in range(len(wall_locations)):
            (row, col) = wall_locations[i]
            self.forbidden_transitions.add((row, col + 1, Actions.left))
            self.forbidden_transitions.add((row, col-1, Actions.right))
            self.forbidden_transitions.add((row+1, col, Actions.up))
            self.forbidden_transitions.add((row-1, col, Actions.down))

        #one_way_door_location = self.env_settings['oneway']
        self.forbidden_transitions.add((4,3, Actions.up))
        self.forbidden_transitions.add((4,4, Actions.up))
        self.forbidden_transitions.add((4,5, Actions.up))
        self.forbidden_transitions.add((4,6, Actions.up))
        self.forbidden_transitions.add((4,7, Actions.up))
        self.forbidden_transitions.add((4,9, Actions.up))

        # up    = 0
        # right = 1 
        # down  = 2 
        # left  = 3 
        # stay  = 4

        if (check<=slip_p[0]) or (a == Actions.none.value):
            a_ = a

        elif (check>slip_p[0]) & (check<=(slip_p[0]+slip_p[1])):
            if a == 0: 
                a_ = 3
            elif a == 2: 
                a_ = 1
            elif a == 3: 
                a_ = 2
            elif a == 1: 
                a_ = 0

        else:
            if a == 0: 
                a_ = 1
            elif a == 2: 
                a_ = 3
            elif a == 3: 
                a_ = 0
            elif a == 1: 
                a_ = 2

        action_ = Actions(a_)
        if (row, col, action_) not in self.forbidden_transitions:
            if action_ == Actions.up:
                row -= 1
            if action_ == Actions.down:
                row += 1
            if action_ == Actions.left:
                col -= 1
            if action_ == Actions.right:
                col += 1

        s_next = self.get_state_from_description(row, col)

        # If the appropriate button hasn't yet been pressed, don't allow the agent into the colored region
        if agent_id == 0:
            if (self.u == 0) or (self.u == 1) or (self.u == 2) or (self.u == 3) or (self.u == 4) or (self.u == 5):
                if (row, col) in self.red_tiles:
                    s_next = s
        if agent_id == 1:
            if (self.u == 0):
                if (row, col) in self.yellow_tiles:
                    s_next = s
        if agent_id == 2:
            if (self.u == 0) or (self.u == 1):
                if (row, col) in self.green_tiles:
                    s_next = s

        last_action = a_
        return s_next, last_action

    def get_state_from_description(self, row, col):
        """
        Given a (row, column) index description of gridworld location, return
        index of corresponding state.

        Parameters
        ----------
        row : int
            Index corresponding to the row location of the state in the gridworld.
        col : int
            Index corresponding to the column location of the state in the gridworld.
        
        Outputs
        -------
        s : int
            The index of the gridworld state corresponding to location (row, col).
        """
        return self.Nc * row + col

    def get_state_description(self, s):
        """
        Return the row and column indeces of state s in the gridworld.

        Parameters
        ----------
        s : int
            Index of the gridworld state.

        Outputs
        -------
        row : int
            The row index of state s in the gridworld.
        col : int
            The column index of state s in the gridworld.
        """
        row = np.floor_divide(s, self.Nr)
        col = np.mod(s, self.Nc)

        return (row, col)

    def get_actions(self, id):
        """
        Returns the list with the actions that a particular agent can perform.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return np.copy(self.actions[id])

    def get_last_action(self, id):
        """
        Returns a particular agent's last action.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return self.last_action[id]

    def get_team_action_array(self):
        """
        Returns the available actions of the entire team.

        Outputs
        -------
        actions : (num_agents x num_actions) numpy integer array
        """
        return np.copy(self.actions)
    
    def get_initial_state(self, id):
        """
        Returns the initial state of a particular agent.

        Parameters
        ----------
        id : int
            Index of the agent whose initial state is being queried.
        """
        return self.s_i[id]

    def get_initial_team_state(self):
        """
        Return the intial state of the collective multi-agent team.

        Outputs
        -------
        s_i : numpy integer array
            Array of initial state indices for the agents in the experiment.
        """
        return np.copy(self.s_i)

    ############## DQPRM-RELATED METHODS ########################################
    def get_mdp_label(self, s, s_next, u):
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

        l = []

        agent1 = 0
        agent2 = 1
        agent3 = 2

        row1, col1 = self.get_state_description(s_next[agent1])
        row2, col2 = self.get_state_description(s_next[agent2])
        row3, col3 = self.get_state_description(s_next[agent3])

        if u == 0:
        # Now check if agents are on buttons
            if not ((row2, col2) in self.yellow_tiles) and (row1,col1) == self.env_settings['yellow_button']:
                l.append('by')
        if u == 1:
            if not ((row3, col3) in self.green_tiles) and (row2, col2) == self.env_settings['green_button1']:
                # print('L in line 385:', 'hi1')
                l.append('bg1')
            if not ((row3, col3) in self.green_tiles) and (row2, col2) == self.env_settings['green_button2']:
                l.append('bg2')
        if u == 2:
            if (row2, col2) == self.env_settings['red_button']:
                l.append('a2br')
            if (row3, col3) == self.env_settings['red_button']:
                l.append('a3br')
        if u == 3:
            if not ((row2, col2) == self.env_settings['red_button']):
                l.append('a2lr')
            if (row3, col3) == self.env_settings['red_button']:
                l.append('a3br')
        if u == 4:
            if (row2, col2) == self.env_settings['red_button']:
                l.append('a2br')
            if not ((row3, col3) == self.env_settings['red_button']):
                l.append('a3lr')
        if u == 5:
            if ((row2, col2) == self.env_settings['red_button']) and ((row3, col3) == self.env_settings['red_button']):
                l.append('br')
            if not ((row2, col2) == self.env_settings['red_button']):
                l.append('a2lr')
            if not ((row3, col3) == self.env_settings['red_button']):
                l.append('a3lr')
        if u == 6:
            # Check if agent 1 has reached the goal
            if (row1, col1) == self.env_settings['goal_location']:
                l.append('g')

        return l

    ################## HRL-RELATED METHODS ######################################
    def get_options_list(self, agent_id):
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
        agent3 = 2

        options_list = []

        if agent_id == agent1:
            options_list.append('w1')
            options_list.append('by')
            options_list.append('g')

        if agent_id == agent2:
            options_list.append('w2')
            options_list.append('bg1')
            options_list.append('bg2')
            options_list.append('a2br')

        if agent_id == agent3:
            options_list.append('w3')
            options_list.append('a3br')

        return options_list
    

    def get_old_u(self, u: int) -> int:
        if not self._use_prm:
            return u
        else:
            if len(self.prm.state_map_after_product) != 0:
                return self.prm.state_map_after_product[u]
            else:
                return u

    def get_avail_options(self, agent_id):
        """
        Given the current metastate, get the available options. Some options are unavailable if 
        they are not possible to complete at the current stage of the task. In such circumstances
        we don't want the agents to update the corresponding option q-functions.
        """
        agent1 = 0
        agent2 = 1
        agent3 = 2

        avail_options = []

        if agent_id == agent1:
            avail_options.append('w1')
            avail_options.append('by')
            if self.red_button_pushed:
                avail_options.append('g')
        if agent_id == agent2:
            avail_options.append('w2')
            if self.yellow_button_pushed:
                avail_options.append('bg1')
                avail_options.append('bg2')
                avail_options.append('a2br')

        if agent_id == agent3:
            avail_options.append('w3')
            if self.green_button_pushed:
                avail_options.append('a3br')

        return avail_options

    def get_avail_meta_action_indeces(self, agent_id):
        """
        Get a list of the indeces corresponding to the currently available meta-action/option
        """
        avail_options = self.get_avail_options(agent_id)
        all_options_list = self.get_options_list(agent_id)
        avail_meta_action_indeces = []
        for option in avail_options:
            avail_meta_action_indeces.append(all_options_list.index(option))
        return avail_meta_action_indeces

    def get_completed_options(self, s):
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

        completed_options = []

        for i in range(self.num_agents):
            row, col = self.get_state_description(s[i])

            if i == agent1:
                if (row,col) == self.env_settings['yellow_button']:
                    completed_options.append('by')
                if (row, col) == self.env_settings['goal_location']:
                    completed_options.append('g')
                if s[i] == self.env_settings['initial_states'][i]:
                    completed_options.append('w1')

            elif i == agent2:
                if (row, col) == self.env_settings['green_button1']:
                    completed_options.append('bg1')
                if (row, col) == self.env_settings['green_button2']:
                    completed_options.append('bg2')
                if (row, col) == self.env_settings['red_button']:
                    completed_options.append('a2br')
                if s[i] == self.env_settings['initial_states'][i]:
                    completed_options.append('w2')

            elif i == agent3:
                if (row, col) == self.env_settings['red_button']:
                    completed_options.append('a3br')
                if s[i] == self.env_settings['initial_states'][i]:
                    completed_options.append('w3')
        
        return completed_options

    def get_meta_state(self, agent_id):
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
        # Convert the Truth values of which buttons have been pushed to an int
        meta_state = int('{}{}{}'.format(int(self.red_button_pushed), int(self.green_button_pushed), int(self.yellow_button_pushed)), 2)


        # # if the task has been failed, return 8
        # if self.u == 6:
        #     meta_state = 8

        # meta_state = self.u

        return meta_state

    def get_num_meta_states(self, agent_id):
        """
        Return the number of meta states for the agent specified by agent_id.
        """
        return int(8) # how many different combinations of button presses are there
    


    ######################### TROUBLESHOOTING METHODS ################################

    def show(self, s):
        """
        Create a visual representation of the current state of the gridworld.

        Parameters
        ----------
        s : int
            Index of the current state
        """
        display = np.zeros((self.Nr, self.Nc))

        # Display the locations of the walls
        for loc in self.env_settings['walls']:
            display[loc] = -1

        display[self.env_settings['red_button']] = 9
        display[self.env_settings['green_button1']] = 9
        display[self.env_settings['green_button2']] = 9
        display[self.env_settings['yellow_button']] = 9
        display[self.env_settings['goal_location']] = 9

        for loc in self.red_tiles:
            display[loc] = 8
        for loc in self.green_tiles:
            display[loc] = 8
        for loc in self.yellow_tiles:
            display[loc] = 8

        # Display the agents
        for i in range(self.num_agents):
            row, col = self.get_state_description(s[i])
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
            "B1": self.map.env_settings["B1"],
            "B2": self.map.env_settings["B2"],
            "B3": self.map.env_settings["B3"],
            "K1": self.map.env_settings["K1"],
            "F1": self.map.env_settings["F1"],
            "F2": self.map.env_settings["F2"],
            "K2": self.map.env_settings["K2"],
            "G": self.map.env_settings["goal_location"],
        }

        for label, loc in special_cells.items():
            display[loc] = 1

        for loc in self.map.blue_tiles:
            display[loc] = 3
        for loc in self.map.orange_tiles:
            display[loc] = 4
        for loc in self.map.green_tiles:
            display[loc] = 5
        for loc in self.map.yellow_tiles:
            display[loc] = 6
        for loc in self.map.pink_tiles:
            display[loc] = 7

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
    n = 3 # num agents
    base_file_dir = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
    rm_string = os.path.join(base_file_dir, 'experiments', 'buttonsCopy', 'team_buttons_rm.txt')
    
    # Set the environment settings for the experiment
    env_settings = dict()
    env_settings['Nr'] = 12
    env_settings['Nc'] = 12
    env_settings['initial_states'] = [0, 3, 11]
    env_settings['walls'] = [(0,2), (1,2), (2,2), (3,2), (4,2), (5,2), (6,2), (7,2), (8,2), (9,2), (10,2),
                                (10,3), (10,4), (10,5), (10,6), (10,7), (10,8), (10,9), (10,10), (10,11),
                                (4,8), (5,8), (6,8), (7,8), (8,8), (8,7), (8,6), (9,6),
                                (0,10), (1,10), (2,10), (3,10), (4,10) ]
    env_settings['oneway'] = [(4,3), (4,4), (4,5), (4,6), (4,7), (4,9)]
    env_settings['goal_location'] = (11,11)
    env_settings['yellow_button'] = (0,1)
    env_settings['green_button1'] = (6,3)
    env_settings['green_button2'] = (9,7)
    env_settings['red_button'] = (9,11)
    env_settings['yellow_tiles'] = [(2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9)]
    env_settings['green_tiles'] = [(2,11), (3,11)]
    env_settings['red_tiles'] = [(11,7), (11,8)]

    env_settings['p'] = 1.0
    
    game = MultiAgentButtonsEnv(rm_string, n, env_settings)

    # User inputs
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value,"x":Actions.none.value}

    s = game.get_initial_team_state()
    print(s)

    while True:
        # Showing game
        game.show(s)

        # Getting action
        a = np.full(n, -1, dtype=int)
        
        for i in range(n):
            print('\nAction{}?'.format(i+1), end='')
            usr_inp = input()
            print()

            if not(usr_inp in str_to_action):
                print('forbidden action')
                a[i] = str_to_action['x']
            else:
                print(str_to_action[usr_inp])
                a[i] = str_to_action[usr_inp]

        r, l, s = game.environment_step(s, a)
        
        print("---------------------")
        print("Next States: ", s)
        print("Label: ", l)
        print("Reward: ", r)
        print("RM state: ", game.u)
        print('Meta state: ', game.get_meta_state(0))
        print("---------------------")

        if game.reward_machine.is_terminal_state(game.u): # Game Over
                break 
    game.show(s)
    




    
# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()