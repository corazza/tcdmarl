import itertools

import numpy as np
from typing_extensions import override

from tcdmarl.tcrl.consts import *
from tcdmarl.tcrl.regex.regex_compiler import (CompileStateNFA, NodeCreator,
                                               generate_inputs, nfa_complement,
                                               nfa_union)
from tcdmarl.tcrl.util import *


def shuffle(xs: list):
    xs = xs.copy()
    np.random.shuffle(xs)
    return xs


def shuffle_pick_n(xs: list, n: int) -> list:
    xs = shuffle(xs)
    return xs[: min(n, len(xs))]


class RENode:
    def __init__(self):
        pass

    def compile(self, _node_creator: NodeCreator) -> CompileStateNFA:
        raise NotImplementedError()

    def appears(self) -> frozenset[str]:
        """Variables that appear locally within the expression. Compilation should
        consider root.appears()"""
        raise NotImplementedError()

    def satisfied(self, vars: frozenset[str]) -> bool:
        """Is the formula satisfied by these variables? Assumes a non-temporal formula."""
        raise NotImplementedError()

    def __eq__(self, b) -> bool:
        raise NotImplementedError()


class RENodeSing(RENode):
    """For expressions like Repeat and Plus, that have a single child"""

    def __init__(self, child: RENode, name: str, con: str):
        super().__init__()
        self.child = child
        self.name = name
        self.con = con

    def appears(self) -> frozenset[str]:
        return self.child.appears()

    def __eq__(self, b) -> bool:
        if not isinstance(b, self.__class__):
            return False
        return self.child == b.child


class RENodeMul(RENode):
    """For expressions like Then, And, and Or, that have multiple children"""

    def __init__(self, exprs: list[RENode], name: str, con: str):
        super().__init__()
        self.exprs: list[RENode] = exprs
        self.name: str = name
        self.con: str = con

    def appears(self) -> frozenset[str]:
        r = set()
        for expr in self.exprs:
            r.update(expr.appears())
        return frozenset(r)

    def _ordered__eq__(self, b) -> bool:
        if not isinstance(b, self.__class__):
            return False
        for e1, e2 in zip(self.exprs, b.exprs):
            if e1 != e2:
                return False
        return True

    def _unordered__eq__(self, b) -> bool:
        if not isinstance(b, self.__class__):
            return False
        if len(self.exprs) != len(b.exprs):
            return False
        for child in self.exprs:
            if child not in b.exprs:
                return False
        return True


def reorder_children(children: list[RENode]) -> list[list[RENode]]:
    indices = list(range(len(children)))
    results: list[list[RENode]] = []
    for perm in itertools.permutations(indices):
        p_children: list[RENode] = [children[indices[i]] for i in perm]
        results.append(p_children)
    return results


def slice_from_back(a: list, i: int) -> list:
    return list(reversed(list(reversed(a))[0:i]))


class Or(RENodeMul):
    def __init__(self, exprs: list[RENode]):
        super().__init__(exprs, "Or", "|")

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        compiled = [e.compile(node_creator) for e in self.exprs]
        return nfa_union(compiled, node_creator)

    def satisfied(self, vars: frozenset[str]) -> bool:
        for expr in self.exprs:
            if expr.satisfied(vars):
                return True
        return False

    def __eq__(self, b) -> bool:
        return self._unordered__eq__(b)


class And(RENodeMul):
    def __init__(self, exprs: list[RENode]):
        super().__init__(exprs, "And", "&")
        self.exprs: list[RENode] = exprs

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        compiled: list[CompileStateNFA] = [e.compile(node_creator) for e in self.exprs]
        complements: list[CompileStateNFA] = [
            nfa_complement(c, node_creator) for c in compiled
        ]
        union: CompileStateNFA = nfa_union(complements, node_creator)
        return nfa_complement(union, node_creator)

    def satisfied(self, vars: frozenset[str]) -> bool:
        for expr in self.exprs:
            if not expr.satisfied(vars):
                return False
        return True

    def __eq__(self, b) -> bool:
        return self._unordered__eq__(b)


class Then(RENodeMul):
    def __init__(self, exprs: list[RENode]):
        super().__init__(exprs, "Then", ">")

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        compiled = [e.compile(node_creator) for e in self.exprs]
        for i in range(len(compiled) - 1):
            for compiled_i_terminal in compiled[i].terminal_states:
                compiled_i_terminal.t(frozenset({"*"}), compiled[i + 1].initial)
        return CompileStateNFA(compiled[0].initial, compiled[-1].terminal_states)

    def __eq__(self, b):
        return self._ordered__eq__(b)


class Repeat(RENodeSing):
    def __init__(self, child: RENode):
        super().__init__(child, "Repeat", "*")

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        child = self.child.compile(node_creator)
        new_initial = node_creator.new_nfa_node()
        new_initial.t(frozenset({"*"}), child.initial)
        for child_terminal in child.terminal_states:
            child_terminal.t(frozenset({"*"}), new_initial)
        return CompileStateNFA(new_initial, {new_initial})


class Plus(RENodeSing):
    def __init__(self, child: RENode):
        super().__init__(child, name="Plus", con="+")

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        child = self.child.compile(node_creator)
        for child_terminal in child.terminal_states:
            child_terminal.t(frozenset({"*"}), child.initial)
        return CompileStateNFA(child.initial, child.terminal_states)


class Multiple(RENodeSing):
    def __init__(self, child: RENode, times: str):
        super().__init__(child, name="Mul", con="{" + f"{times}" + "}")
        self.times = times
        self.times_num = DEFAULT_TIMES if "#" in self.times else int(self.times)
        assert self.times_num > 0

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        children = [self.child.compile(node_creator)]
        for i in range(self.times_num - 1):
            next_child = self.child.compile(node_creator)
            children.append(next_child)
            for child_terminal in children[i].terminal_states:
                child_terminal.t(frozenset({"*"}), children[i + 1].initial)
        return CompileStateNFA(children[0].initial, children[-1].terminal_states)


class Complement(RENodeSing):
    def __init__(self, child: RENode):
        super().__init__(child, "Complement", "~")

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        child = self.child.compile(node_creator)
        return nfa_complement(child, node_creator)

    def satisfied(self, vars: frozenset[str]) -> bool:
        return not self.child.satisfied(vars)


class Matcher(RENode):
    def __init__(self, negated: bool):
        super().__init__()
        self.negated: bool = negated

    def matches(self, input_symbol: frozenset[str], appears: frozenset[str]) -> bool:
        raise NotImplementedError()

    def satisfied(self, vars: frozenset[str]) -> bool:
        match_result: bool = self.matches(vars, {})
        if not self.negated:
            return match_result
        else:
            return not match_result

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        terminal = node_creator.new_nfa_node()
        sink = node_creator.new_nfa_sink()
        initial = node_creator.new_nfa_node()
        for input_symbol in generate_inputs(node_creator.appears):
            terminal.t(input_symbol, sink)
            does_match: bool = self.matches(input_symbol, node_creator.appears)
            if does_match and not self.negated or not does_match and self.negated:
                initial.t(input_symbol, terminal)
            else:
                initial.t(input_symbol, sink)
        return CompileStateNFA(initial, {terminal})

    def __str__(self) -> str:
        return f'{"!" if self.negated else ""}{self.content()}'

    def content(self) -> str:
        raise NotImplementedError()


class Symbol(Matcher):
    def __init__(self, symbol: str, negated: bool):
        super().__init__(negated)
        self.symbol = symbol

    def appears(self) -> frozenset[str]:
        return frozenset({self.symbol})

    @override
    def matches(self, input_symbol: frozenset[str], appears: frozenset[str]) -> bool:
        return self.symbol in input_symbol

    def __eq__(self, b) -> bool:
        if not isinstance(b, Symbol):
            return False
        return self.symbol == b.symbol and self.negated == b.negated

    def content(self) -> str:
        return self.symbol


class Nonempty(Matcher):
    def __init__(self, negated: bool):
        super().__init__(negated)

    def appears(self) -> frozenset[str]:
        return frozenset()

    @override
    def matches(self, input_symbol: frozenset[str], appears: frozenset[str]) -> bool:
        return len(input_symbol) > 0

    def __eq__(self, b) -> bool:
        return isinstance(b, Nonempty) and self.negated == b.negated

    def content(self) -> str:
        return ":"


class Any(Matcher):
    def __init__(self, negated: bool):
        super().__init__(negated)

    def appears(self) -> frozenset[str]:
        return frozenset()

    @override
    def matches(self, input_symbol: frozenset[str], appears: frozenset[str]) -> bool:
        return True

    def __eq__(self, b) -> bool:
        return isinstance(b, Any) and self.negated == b.negated

    def content(self) -> str:
        return "."


class Nonappear(Matcher):
    def __init__(self, negated: bool):
        super().__init__(negated)

    def appears(self) -> frozenset[str]:
        return frozenset()

    @override
    def matches(self, input_symbol: frozenset[str], appears: frozenset[str]) -> bool:
        for var in input_symbol:
            if var in appears:
                return False
        return True

    def __eq__(self, b) -> bool:
        return isinstance(b, Any) and self.negated == b.negated

    def content(self) -> str:
        return "_"
