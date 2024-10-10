from tcdmarl.tcrl.reward_machines.rm_builder import DFABuilder
from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA


def dfa_paper_sync_next_sensors() -> CausalDFA:
    builder = DFABuilder(appears=frozenset({"a", "b", "c", "d", "e"}))

    # Transitions
    builder.t(0, "!c", 0)  # Stay in state 0 if 'c' does not appear
    builder.t(0, "c", 1)  # Transition to state 1 on 'c'

    # Happpy
    builder.t(1, "a&!b", 2)  # Transition to state 2 on 'a'
    builder.t(1, "b&!a", 3)  # Transition to state 3 on 'b'

    # Unhappy
    builder.t(1, "(a&!b|b&!a)~", 4)  # Transition to sink state 4 on 'c'

    # Transitions from state 2 (unhappy)
    builder.t(2, "!b", 2)  # Stay in state 2 if 'b' does not appear
    builder.t(2, "b", 4)  # Transition to sink state 4 on 'b'

    # Transitions from state 3 (unhappy)
    builder.t(3, "!a", 3)  # Stay in state 3 if 'a' does not appear
    builder.t(3, "a", 4)  # Transition to sink state 4 on 'a'

    builder.tag(1)
    builder.sink(4)
    builder.tag(4)

    return builder.build()
