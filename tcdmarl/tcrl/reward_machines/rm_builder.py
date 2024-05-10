import IPython
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple
from regex.regex_compiler import CompileStateDFA, DFANode, generate_inputs

from regex.regex_parser import parse
from reward_machines.rm_common import (
    CausalDFA,
    DeterministicRewardMachine,
    ProbabilisticRewardMachine,
    RewardMachine,
)


def transitions_in_formula(
    appears: frozenset[str], formula: str
) -> Iterator[frozenset[str]]:
    ast = parse(formula)
    for input_symbol in generate_inputs(appears):
        if ast.satisfied(input_symbol):
            yield input_symbol


class RMBuilder(ABC):
    def __init__(self, appears: frozenset[str]):
        self.appears = appears
        self.terminal_states: set[int] = set()
        # self.sink = -1

    @abstractmethod
    def build(self) -> RewardMachine:
        pass

    def terminal(self, state: int) -> "RMBuilder":
        self.terminal_states.add(state)
        return self


class DFABuilder:
    def __init__(self, appears: frozenset[str]):
        self.appears = appears
        self.tagged_states: set[int] = set()
        self.transitions: dict[int, dict[frozenset[str], int]] = dict()

    @abstractmethod
    def build(self) -> CausalDFA:
        return CausalDFA(self.appears, self.transitions, self.tagged_states)

    def sink(self, state: int) -> "DFABuilder":
        return self.t(state, ".", state)

    def tag(self, state: int) -> "DFABuilder":
        self.tagged_states.add(state)
        return self

    def transition(
        self, from_state: int, input_symbol: frozenset[str], to_state: int
    ) -> "DFABuilder":
        if from_state not in self.transitions:
            self.transitions[from_state] = dict()
        if input_symbol in self.transitions[from_state]:
            raise ValueError(f"{input_symbol} already in state {from_state}")
        self.transitions[from_state][input_symbol] = to_state
        return self

    def t(self, from_state: int, formula: str, to_state: int) -> "DFABuilder":
        for input_symbol in transitions_in_formula(self.appears, formula):
            self.transition(from_state, input_symbol, to_state)
        return self


class DeterministicRMBuilder(RMBuilder):
    def __init__(self, appears: frozenset[str]):
        super().__init__(appears)
        self.transitions: dict[int, dict[frozenset[str], Tuple[int, int]]] = dict()

    def build(self) -> DeterministicRewardMachine:
        return DeterministicRewardMachine(
            self.transitions, frozenset(self.appears), frozenset(self.terminal_states)
        )

    def transition(
        self, from_state: int, input_symbol: frozenset[str], to_state: int, output: int
    ) -> "DeterministicRMBuilder":
        if from_state in self.terminal_states:
            raise ValueError(f"terminal states can't have outgoing connections")
        if from_state not in self.transitions:
            self.transitions[from_state] = dict()
        if input_symbol in self.transitions[from_state]:
            raise ValueError(
                f"{input_symbol} already in state {from_state} ({(from_state, to_state, input_symbol, output)})"
            )
        self.transitions[from_state][input_symbol] = (to_state, output)
        return self

    def t(
        self, from_state: int, formula: str, to_state: int, output: int
    ) -> "DeterministicRMBuilder":
        for input_symbol in self.transitions_in_formula(formula):
            self.transition(from_state, input_symbol, to_state, output)
        return self


class ProbabilisticRMBuilder(RMBuilder):
    def __init__(self, appears: frozenset[str]):
        super().__init__(appears)
        self.transitions: dict[
            int, dict[frozenset[str], dict[int, Tuple[float, int]]]
        ] = dict()

    def build(self) -> ProbabilisticRewardMachine:
        return ProbabilisticRewardMachine(
            self.transitions, frozenset(self.appears), frozenset(self.terminal_states)
        )

    def transition(
        self,
        from_state: int,
        input_symbol: frozenset[str],
        to_state: int,
        prob: float,
        output: int,
    ) -> "ProbabilisticRMBuilder":
        assert from_state >= 0  # we want to use this as array indices eventually
        assert to_state >= 0
        input_symbol = frozenset(input_symbol)
        if from_state in self.terminal_states:
            raise ValueError(f"terminal states can't have outgoing connections")
        if from_state not in self.transitions:
            self.transitions[from_state] = dict()
        if input_symbol not in self.transitions[from_state]:
            self.transitions[from_state][input_symbol] = dict()
        if to_state in self.transitions[from_state][input_symbol]:
            assert self.transitions[from_state][input_symbol][to_state] == (
                prob,
                output,
            )
        self.transitions[from_state][input_symbol][to_state] = (prob, output)
        return self

    def t(
        self,
        from_state: int,
        formula: str,
        to_state: int,
        prob: float,
        output: int,
    ) -> "DeterministicRMBuilder":
        for input_symbol in transitions_in_formula(self.appears, formula):
            self.transition(from_state, input_symbol, to_state, prob, output)
        return self


def dfa_to_rm(dfa: CompileStateDFA, appears: frozenset[str]) -> RewardMachine:
    """Terminal in DFA: positive reward. Terminal in RM: end simulation."""
    builder: DeterministicRMBuilder = DeterministicRMBuilder(appears)
    to_visit: set[DFANode] = set([dfa.initial])
    visited: set[DFANode] = set()
    while len(to_visit) > 0:
        visiting: DFANode = to_visit.pop()
        visited.add(visiting)
        for transition, dfa_child in visiting.transitions.items():
            if dfa_child in dfa.terminal_states:
                r = 1
            else:
                r = 0
            builder = builder.transition(visiting.id, dfa_child.id, transition, r)
            if dfa_child not in visited:
                to_visit.add(dfa_child)
        if visiting in dfa.terminal_states:
            builder = builder.terminal(visiting.id)
    return builder.build()
