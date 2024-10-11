from tcdmarl.tcrl.reward_machines.rm_builder import DFABuilder
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA


def dfa_paper_specimen() -> CausalDFA:
    builder = DFABuilder(appears=frozenset({"f", "d", "e"}))

    builder.t(0, "!f", 0)
    builder.t(0, "f", 1)
    builder.t(1, "(d|e)~", 1)
    builder.t(1, "d|e", 2)

    builder.sink(2)
    builder.tag(2)

    return builder.build()
