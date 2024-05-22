from tcdmarl.tcrl.reward_machines.rm_builder import DFABuilder
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA


def dfa_paper_no_goal_after_flowers() -> CausalDFA:
    builder = DFABuilder(appears=frozenset({"f", "g"}))

    builder.t(0, "!f", 0)
    builder.t(0, "f", 1)
    builder.t(1, "!g", 1)
    builder.t(1, "g", 2)

    builder.sink(2)
    builder.tag(2)

    return builder.build()


def dfa_paper_no_br_after_f() -> CausalDFA:
    builder = DFABuilder(appears=frozenset({"f", "br"}))

    builder.t(0, "!f", 0)
    builder.t(0, "f", 1)
    builder.t(1, "!br", 1)
    builder.t(1, "br", 2)

    builder.sink(2)
    builder.tag(2)

    return builder.build()
