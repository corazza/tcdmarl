import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from numpy import int32
from numpy.typing import NDArray


class Map:
    def __init__(self, env_settings: Dict[str, Any]):
        """
        Initialize the environment.
        """
        self.env_settings = env_settings
        self.number_of_rows = self.env_settings["Nr"]
        self.number_of_columns = self.env_settings["Nc"]
        self.initial_states: List[int] = self.env_settings["initial_states"]
        self.p = env_settings["p"]
        self.sinks = self.env_settings["sinks"]

        # Set the available actions of all agents. For now all agents have same action set.
        self.actions = np.array(
            [
                Actions.UP.value,
                Actions.RIGHT.value,
                Actions.LEFT.value,
                Actions.DOWN.value,
                Actions.NONE.value,
            ],
            dtype=int,
        )

        self.num_states: int = self.number_of_rows * self.number_of_columns

        # Define forbidden transitions corresponding to map edges
        self.forbidden_transitions: Set[Tuple[int, int, Actions]] = set()

        for row in range(self.number_of_rows):
            # If in left-most column, can't move left.
            self.forbidden_transitions.add((row, 0, Actions.LEFT))
            # If in right-most column, can't move right.
            self.forbidden_transitions.add(
                (row, self.number_of_columns - 1, Actions.RIGHT)
            )
        for col in range(self.number_of_columns):
            # If in top row, can't move up
            self.forbidden_transitions.add((0, col, Actions.UP))
            # If in bottom row, can't move down
            self.forbidden_transitions.add((self.number_of_rows - 1, col, Actions.DOWN))

        # Restrict agent from having the option of moving "into" a wall
        wall_locations = self.env_settings["walls"]
        for _i, (row, col) in enumerate(wall_locations):
            self.forbidden_transitions.add((row, col + 1, Actions.LEFT))
            self.forbidden_transitions.add((row, col - 1, Actions.RIGHT))
            self.forbidden_transitions.add((row + 1, col, Actions.UP))
            self.forbidden_transitions.add((row - 1, col, Actions.DOWN))

        # one_way_door_location = self.env_settings['oneway']
        self.forbidden_transitions.add((5, 6, Actions.UP))
        self.forbidden_transitions.add((5, 8, Actions.UP))
        self.forbidden_transitions.add((7, 8, Actions.UP))

    def get_num_states(self) -> int:
        return self.num_states

    def get_state_from_description(self, row: int, col: int) -> int:
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
        return self.number_of_columns * row + col

    def get_state_description(self, s: int) -> Tuple[int, int]:
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
        row: int = np.floor_divide(s, self.number_of_rows)
        col: int = np.mod(s, self.number_of_columns)

        return (row, col)


class RoutingState:
    def __init__(
        self, key_collected: bool, b1_pressed: bool, b2_pressed: bool, b3_pressed: bool
    ):
        self.key_collected: bool = key_collected
        self.b1_pressed: bool = b1_pressed
        self.b2_pressed: bool = b2_pressed
        self.b3_pressed: bool = b3_pressed

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RoutingState):
            return False
        return (
            self.key_collected == other.key_collected
            and self.b1_pressed == other.b1_pressed
            and self.b2_pressed == other.b2_pressed
            and self.b3_pressed == other.b3_pressed
        )

    def __str__(self) -> str:
        return f"Key: {self.key_collected}, B1: {self.b1_pressed}, B2: {self.b2_pressed}, B3: {self.b3_pressed}"

    def __repr__(self) -> str:
        return self.__str__()


class RoutingMap(Map):
    """
    Case study 1: routing environment with two agents and a switch door.
    """

    def __init__(self, env_settings: Dict[str, Any]):
        super().__init__(env_settings=env_settings)
        self.yellow_tiles = self.env_settings["yellow_tiles"]
        self.green_tiles = self.env_settings["green_tiles"]
        self.orange_tiles = self.env_settings["orange_tiles"]
        self.blue_tiles = self.env_settings["blue_tiles"]
        self.pink_tiles = self.env_settings["pink_tiles"]

    def get_next_state(
        self, s: int, a: int, agent_id: int, routing_state: RoutingState
    ) -> Tuple[int, int]:
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
        slip_p = [self.p, (1 - self.p) / 2, (1 - self.p) / 2]
        check = random.random()  # TODO get rid of random

        row, col = self.get_state_description(s)

        stuck = False
        if (row, col) in self.sinks:
            stuck = True

        a_: int
        if (check <= slip_p[0]) or (a == Actions.NONE.value):
            a_ = a
        elif (check > slip_p[0]) and (check <= (slip_p[0] + slip_p[1])):
            if a == 0:
                a_ = 3
            elif a == 2:
                a_ = 1
            elif a == 3:
                a_ = 2
            else:
                assert a == 1
                a_ = 0
        else:
            if a == 0:
                a_ = 1
            elif a == 2:
                a_ = 3
            elif a == 3:
                a_ = 0
            else:
                assert a == 1
                a_ = 2

        action_ = Actions(a_)
        if (row, col, action_) not in self.forbidden_transitions:
            if action_ == Actions.UP:
                row -= 1
            if action_ == Actions.DOWN:
                row += 1
            if action_ == Actions.LEFT:
                col -= 1
            if action_ == Actions.RIGHT:
                col += 1

        s_next = self.get_state_from_description(row, col)

        # If the appropriate button hasn't yet been pressed, don't allow the agent into the colored region
        if agent_id == 0:
            # Exploding door
            if routing_state.b1_pressed and not routing_state.b2_pressed:
                if (row, col) in self.yellow_tiles:
                    s_next = s

            # Routing door (all 3 buttons need to have been pressed)
            if (
                routing_state.b1_pressed
                and routing_state.b2_pressed
                and routing_state.b3_pressed
            ):
                if (row, col) in self.orange_tiles:
                    s_next = s
            else:
                if (row, col) in self.green_tiles:
                    s_next = s

            # Goal
            if not routing_state.key_collected:
                if (row, col) in self.blue_tiles:
                    s_next = s
        else:
            assert agent_id == 1
            if not routing_state.b3_pressed:
                if (row, col) in self.pink_tiles:
                    s_next = s

        if stuck:
            s_next = s

        last_action = a_
        return s_next, last_action

    def compute_joint_state(self, u: int) -> RoutingState:
        key_collected: bool = False
        b1_pressed: bool = False
        b2_pressed: bool = False
        b3_pressed: bool = False

        if u in [8, 9, 10]:
            key_collected = True
        if u in [1, 4, 6, 7, 8, 9]:
            b1_pressed = True
        if u in [2, 4, 5, 7, 8, 9]:
            b2_pressed = True
        if u in [3, 5, 6, 7, 8, 9]:
            b3_pressed = True

        return RoutingState(
            key_collected=key_collected,
            b1_pressed=b1_pressed,
            b2_pressed=b2_pressed,
            b3_pressed=b3_pressed,
        )

    def compute_state(self, agent_id: int, u: int) -> RoutingState:
        key_collected: bool = False
        b1_pressed: bool = False
        b2_pressed: bool = False
        b3_pressed: bool = False

        if agent_id == 0:
            if u in [1, 3, 4, 5]:
                b1_pressed = True
            if u in [2, 3, 4, 5]:
                b2_pressed = True
                b3_pressed = True
            if u in [4, 5, 6]:
                key_collected = True
        else:
            assert agent_id == 1
            if u in [1, 3]:
                b2_pressed = True
            if u in [2, 3]:
                b3_pressed = True

        return RoutingState(
            key_collected=key_collected,
            b1_pressed=b1_pressed,
            b2_pressed=b2_pressed,
            b3_pressed=b3_pressed,
        )


class DecentralizedEnv(ABC):
    """Base class for decentalized training environments."""

    @abstractmethod
    def environment_step(self, s: int, a: int) -> Tuple[int, List[str], int]:
        pass

    @abstractmethod
    def get_actions(self) -> NDArray[int32]:
        pass

    @abstractmethod
    def get_map(self) -> Map:
        pass

    @abstractmethod
    def get_initial_state(self) -> int:
        pass

    @abstractmethod
    def get_mdp_label(self, _s: int, s_next: int, _u: int) -> List[str]:
        pass


class CentralizedEnv(ABC):
    """Base class for centalized training environments."""

    @abstractmethod
    def environment_step(
        self, s: NDArray[int32], a: NDArray[int32]
    ) -> Tuple[int, List[str], NDArray[int32]]:
        pass

    @abstractmethod
    def show_graphic(self, s: NDArray[int32]) -> None:
        pass

    @abstractmethod
    def get_actions(self, agent_id: int) -> NDArray[int32]:
        pass

    @abstractmethod
    def get_team_action_array(self) -> NDArray[int32]:
        pass

    @abstractmethod
    def get_map(self) -> Map:
        pass

    @abstractmethod
    def get_initial_state(self, agent_id: int) -> int:
        pass

    @abstractmethod
    def get_initial_team_state(self) -> NDArray[int32]:
        pass

    @abstractmethod
    def get_mdp_label(
        self, _s: NDArray[int32], s_next: NDArray[int32], u: int
    ) -> List[str]:
        pass


class Actions(Enum):
    """
    Enum with the actions that the agent can execute
    """

    UP = 0  # move up
    RIGHT = 1  # move right
    DOWN = 2  # move down
    LEFT = 3  # move left
    NONE = 4  # none


# User inputs
STR_TO_ACTION = {
    "w": Actions.UP.value,
    "d": Actions.RIGHT.value,
    "s": Actions.DOWN.value,
    "a": Actions.LEFT.value,
    "x": Actions.NONE.value,
}
