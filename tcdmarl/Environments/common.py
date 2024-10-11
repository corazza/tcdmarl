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

        # Define forced transitions
        self.forced_transitions: dict[tuple[int, int], Actions] = {}

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
        # TODO this is left from Routing hardcoding?
        # self.forbidden_transitions.add((5, 6, Actions.UP))
        # self.forbidden_transitions.add((5, 8, Actions.UP))
        # self.forbidden_transitions.add((7, 8, Actions.UP))

        force_move_locations = self.env_settings.get("forcemove", {})
        for direction, locations in force_move_locations.items():
            for row, col in locations:
                self.forced_transitions[(row, col)] = direction

        one_way_door_locations = self.env_settings.get("oneway", {})
        for direction, locations in one_way_door_locations.items():
            for row, col in locations:
                if direction == Actions.UP:
                    # Only allow UP, block DOWN, LEFT, and RIGHT from this location
                    self.forbidden_transitions.add((row, col, Actions.DOWN))
                    self.forbidden_transitions.add((row, col, Actions.LEFT))
                    self.forbidden_transitions.add((row, col, Actions.RIGHT))
                elif direction == Actions.DOWN:
                    # Only allow DOWN, block UP, LEFT, and RIGHT
                    self.forbidden_transitions.add((row, col, Actions.UP))
                    self.forbidden_transitions.add((row, col, Actions.LEFT))
                    self.forbidden_transitions.add((row, col, Actions.RIGHT))
                elif direction == Actions.LEFT:
                    # Only allow LEFT, block UP, DOWN, and RIGHT
                    self.forbidden_transitions.add((row, col, Actions.UP))
                    self.forbidden_transitions.add((row, col, Actions.DOWN))
                    self.forbidden_transitions.add((row, col, Actions.RIGHT))
                elif direction == Actions.RIGHT:
                    # Only allow RIGHT, block UP, DOWN, and LEFT
                    self.forbidden_transitions.add((row, col, Actions.UP))
                    self.forbidden_transitions.add((row, col, Actions.DOWN))
                    self.forbidden_transitions.add((row, col, Actions.LEFT))

        for (row, col), action in self.forced_transitions.items():
            if action == Actions.UP:
                self.forbidden_transitions.add((row - 1, col, Actions.DOWN))
            elif action == Actions.DOWN:
                self.forbidden_transitions.add((row + 1, col, Actions.UP))
            elif action == Actions.LEFT:
                self.forbidden_transitions.add((row, col - 1, Actions.RIGHT))
            elif action == Actions.RIGHT:
                self.forbidden_transitions.add((row, col + 1, Actions.LEFT))

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
        row: int = np.floor_divide(s, self.number_of_columns)
        col: int = np.mod(s, self.number_of_columns)

        return (row, col)


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
