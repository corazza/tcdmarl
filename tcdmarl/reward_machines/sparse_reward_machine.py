from pathlib import Path
from types import NoneType
from typing import Dict, List, Set


class SparseRewardMachine:
    """<U,u0,delta_u,delta_r>"""

    def __init__(self, file: Path | NoneType = None):

        # list of machine states
        self.all_states: List[int] = []

        # set of events
        self.events: Set[str] = set()

        # initial state
        self.u0: int = 0

        # state-transition function
        self.delta_u: Dict[int, Dict[str, int]] = {}

        # reward-transition function
        self.delta_r: Dict[int, Dict[int, int]] = {}

        # set of terminal states (they are automatically detected)
        self.terminal_states: Set[int] = set()

        if file is not None:
            self._load_reward_machine(file)

    def __repr__(self):
        s = "MACHINE:\n"
        s += f"init: {self.u0}\n"
        for trans_init_state in self.delta_u:
            for event in self.delta_u[trans_init_state]:
                trans_end_state = self.delta_u[trans_init_state][event]
                s += "({} ---({},{})--->{})\n".format(
                    trans_init_state,
                    event,
                    self.delta_r[trans_init_state][trans_end_state],
                    trans_end_state,
                )
        return s

    # Public methods -----------------------------------

    def load_rm_from_file(self, file: Path):
        self._load_reward_machine(file)

    def get_initial_state(self) -> int:
        return self.u0

    def get_next_state(self, u1: int, event: str) -> int:
        if u1 in self.delta_u:
            if event in self.delta_u[u1]:
                return self.delta_u[u1][event]
        return u1

    def get_reward(self, u1: int, u2: int) -> int:
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            return self.delta_r[u1][u2]

        # This case occurs when the agent falls from the reward machine
        return 0

    def get_states(self):
        return self.all_states

    def is_terminal_state(self, u1: int):
        return u1 in self.terminal_states

    def get_events(self) -> Set[str]:
        return self.events

    def is_event_available(self, u: int, event: str):
        is_event_available = False
        if u in self.delta_u:
            if event in self.delta_u[u]:
                is_event_available = True
        return is_event_available

    # Private methods -----------------------------------

    def _load_reward_machine(self, file: Path):
        """
        Example:
            0                  # initial state
            (0,0,'r1',0)
            (0,1,'r2',0)
            (0,2,'r',0)
            (1,1,'g1',0)
            (1,2,'g2',1)
            (2,2,'True',0)

            Format: (current state, next state, event, reward)
        """
        # Reading the file
        f = open(file)
        lines = [l.rstrip() for l in f]
        f.close()
        # setting the DFA
        self.u0 = eval(lines[0])
        # adding transitions
        for e in lines[1:-1]:
            self._add_transition(*eval(e))
            self.events.add(
                eval(e)[2]
            )  # By convention, the event is in the spot indexed by 2
        # expand event set by the last line
        for e in eval(lines[-1]):
            self.events.add(e)
        # adding terminal states
        for u1 in self.all_states:
            if self._is_terminal(u1):
                self.terminal_states.add(u1)
            if u1 not in self.delta_u:
                self.delta_u[u1] = {}
            if u1 not in self.delta_r:
                self.delta_r[u1] = {}
        self.all_states = sorted(self.all_states)

    def calculate_reward(self, trace):
        total_reward = 0
        current_state = self.get_initial_state()

        for event in trace:
            next_state = self.get_next_state(current_state, event)
            reward = self.get_reward(current_state, next_state)
            total_reward += reward
            current_state = next_state
        return total_reward

    def _is_terminal(self, u1):
        # Check if reward is given for reaching the state in question
        for u0 in self.delta_r:
            if u1 in self.delta_r[u0]:
                if self.delta_r[u0][u1] == 1:
                    return True
        return False

    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.all_states:
                self.all_states.append(u)

    def _add_transition(self, u1, u2, event, reward):
        # Adding machine state
        self._add_state([u1, u2])
        # Adding state-transition to delta_u
        if u1 not in self.delta_u:
            self.delta_u[u1] = {}
        if event not in self.delta_u[u1]:
            self.delta_u[u1][event] = u2
        else:
            raise Exception("Trying to make rm transition function non-deterministic.")
            # self.delta_u[u1][u2].append(event)
        # Adding reward-transition to delta_r
        if u1 not in self.delta_r:
            self.delta_r[u1] = {}
        self.delta_r[u1][u2] = reward
