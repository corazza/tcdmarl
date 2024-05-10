from reward_machines.rm_builder import DFABuilder
from reward_machines.rm_common import CausalDFA


def dfa_paper_after_drinks_first_flowers() -> CausalDFA:
    builder = DFABuilder(appears=frozenset({"drink", "flowers", "office"}))

    builder.t(0, "!drink", 0)
    builder.t(0, "drink", 1)
    builder.t(1, "!flowers & !office", 1)
    builder.t(1, "flowers & !office", 2)
    builder.t(1, "office", 3)

    builder.sink(2)
    builder.sink(3)
    builder.tag(3)

    return builder.build()


def dfa_paper_no_office_after_flowers() -> CausalDFA:
    builder = DFABuilder(appears=frozenset({"flowers", "office"}))

    builder.t(0, "!flowers", 0)
    builder.t(0, "flowers", 1)
    builder.t(1, "!office", 1)
    builder.t(1, "office", 2)

    builder.sink(2)
    builder.tag(2)

    return builder.build()


def dfa_review_spurious() -> CausalDFA:
    builder = DFABuilder(appears=frozenset({"left", "right", "up", "down"}))

    builder.t(0, "(left | right | up | down)~", 0)
    builder.t(0, "left | right | up | down", 1)

    builder.t(1, "(left | right | up | down)~", 1)
    builder.t(1, "left | right | up | down", 2)

    builder.t(2, "(left | right | up | down)~", 2)
    builder.t(2, "left | right | up | down", 3)

    builder.t(3, "(left | right | up | down)~", 3)
    builder.t(3, "left | right | up | down", 4)

    builder.t(4, "(left | right | up | down)~", 4)
    builder.t(4, "left | right | up | down", 0)

    return builder.build()


def dfa_paper_b_doesnt_follow_a() -> CausalDFA:
    builder = DFABuilder(appears=frozenset({"a", "b"}))

    builder.t(0, "!a", 0)
    builder.t(0, "a", 1)
    builder.t(1, "!b", 1)
    builder.t(1, "b", 2)
    builder.sink(2)
    builder.tag(2)

    return builder.build()


################# OLD


def flowers_disallowed(appears: frozenset[str]) -> CausalDFA:
    builder = DFABuilder(appears=appears)

    builder.t(0, "!flowers", 0)
    builder.t(0, "flowers", 1)
    builder.sink(1)
    builder.tag(1)

    return builder.build()
