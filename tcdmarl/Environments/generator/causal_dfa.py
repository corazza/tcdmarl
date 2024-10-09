from tcdmarl.tcrl.reward_machines.rm_builder import DFABuilder
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA


def dfa_paper_no_drain_after_unlock() -> CausalDFA:
    builder = DFABuilder(appears=frozenset({"a", "b", "c"}))

    builder.t(0, "!c", 0)
    builder.t(0, "c", 1)
    builder.t(1, "!a", 1)
    builder.t(1, "a", 2)

    builder.sink(2)
    builder.tag(2)

    return builder.build()
