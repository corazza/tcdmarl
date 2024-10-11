from tcdmarl.tcrl.reward_machines.rm_builder import DFABuilder
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA


def dfa_paper_buttons_exclusive() -> CausalDFA:
    builder = DFABuilder(appears=frozenset({"d", "e"}))

    builder.t(0, "!d&!e", 0)
    builder.t(0, "d&!e", 1)
    builder.t(0, "e", 2)
    builder.t(1, "!e", 1)
    builder.t(1, "e", 3)
    builder.t(2, "!d", 2)
    builder.t(2, "d", 3)

    builder.sink(3)
    builder.tag(3)

    return builder.build()
