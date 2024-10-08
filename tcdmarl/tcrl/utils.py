from typing import List

from tcdmarl.reward_machines.sparse_reward_machine import SparseRewardMachine
from tcdmarl.tcrl.reward_machines.rm_builder import ProbabilisticRMBuilder
from tcdmarl.tcrl.reward_machines.rm_common import ProbabilisticRewardMachine


def sparse_rm_to_prm(sparse_rm: SparseRewardMachine) -> ProbabilisticRewardMachine:
    events: List[str] = [
        event
        for event in sparse_rm.get_events()
        if event != "True" and event != "False"
    ]

    builder = ProbabilisticRMBuilder(frozenset(events))

    initial_state: int = sparse_rm.get_initial_state()
    assert initial_state == 0

    for u1 in sparse_rm.get_states():
        # Is the state terminal?
        if sparse_rm.is_terminal_state(u1):
            builder.terminal(u1)
            continue

        # Record every transition this state cares about
        formulas: List[str] = []
        for event in sparse_rm.delta_u[u1].keys():
            others_events = [e for e in events if e != event]
            not_others_conjunction = " & ".join([f"!{e}" for e in others_events])
            u2 = sparse_rm.get_next_state(u1, event)
            reward = sparse_rm.get_reward(u1, u2)
            formula = f"{event}&{not_others_conjunction}"
            builder.t(u1, formula, u2, prob=1, output=reward)
            formulas.append(formula)

        # In all other cases, the RM should self-loop with output 0
        if len(formulas) == 0:
            # no outgoing transitions
            builder.t(u1, ".", u1, prob=1, output=0)
        else:
            # outgoing transitions corresponding to any of the formulas
            any_transition = " | ".join([f"({f})" for f in formulas])
            otherwise = f"({any_transition})~"
            builder.t(u1, otherwise, u1, prob=1, output=0)

    prm = builder.build()
    return prm
