import random
from typing import Any

from tcdmarl.Environments.common import Actions, Map


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

    def __init__(self, env_settings: dict[str, Any]):
        super().__init__(env_settings=env_settings)
        self.yellow_tiles = self.env_settings["yellow_tiles"]
        self.green_tiles = self.env_settings["green_tiles"]
        self.orange_tiles = self.env_settings["orange_tiles"]
        self.blue_tiles = self.env_settings["blue_tiles"]
        self.pink_tiles = self.env_settings["pink_tiles"]

    def get_next_state(
        self, s: int, a: int, agent_id: int, routing_state: RoutingState
    ) -> tuple[int, int]:
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
