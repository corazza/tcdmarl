import random
from typing import Any

from tcdmarl.Environments.common import Actions, Map


class GeneratorState:
    def __init__(self, button_pressed: bool):
        self.button_pressed: bool = button_pressed

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GeneratorState):
            return False
        return self.button_pressed == other.button_pressed

    def __str__(self) -> str:
        return f"Button pressed: {self.button_pressed}"

    def __repr__(self) -> str:
        return self.__str__()


class GeneratorMap(Map):
    def __init__(self, env_settings: dict[str, Any]):
        super().__init__(env_settings=env_settings)
        self.yellow_tiles = self.env_settings["yellow_tiles"]

    def get_next_state(
        self, s: int, a: int, agent_id: int, generator_state: GeneratorState
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

        if agent_id == 0:
            pass
        else:
            assert agent_id == 1
            # Movable door
            if not generator_state.button_pressed:
                if (row, col) in self.yellow_tiles:
                    s_next = s

        if stuck:
            s_next = s

        last_action = a_
        return s_next, last_action

    def compute_joint_state(self, u: int) -> GeneratorState:
        return GeneratorState(button_pressed=u in [2, 3, 4, 5])

    def compute_state(self, agent_id: int, u: int) -> GeneratorState:
        if agent_id == 0:
            return GeneratorState(button_pressed=u in [1, 3, 5])
        else:
            assert agent_id == 1
            return GeneratorState(button_pressed=u in [1])
