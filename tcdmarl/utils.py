import json
from pathlib import Path
from typing import List, Type, TypeVar, cast

from tcdmarl.config import ExperimentConfig
from tcdmarl.reward_machines.sparse_reward_machine import SparseRewardMachine
from tcdmarl.tcrl.reward_machines.rm_builder import ProbabilisticRMBuilder
from tcdmarl.tcrl.reward_machines.rm_common import ProbabilisticRewardMachine


def compute_caching_name(saved_rm_path: Path, tlcd_suffix: str = "TLCD") -> str:
    # Extract the parent directory name and the file name
    parent_dir = saved_rm_path.parent.name  # e.g., 'laboratory'
    file_name = saved_rm_path.name  # e.g., 'proj_1.txt'

    # Combine them into the desired format
    caching_str = f"{parent_dir}_{file_name}_{tlcd_suffix}"

    return caching_str


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


T = TypeVar("T")


def load_typed_dict(typed_dict_class: Type[T], file_path: Path) -> T:
    with open(file_path, "r") as json_file:
        return cast(typed_dict_class, json.load(json_file))


def experiment_name(config: ExperimentConfig) -> str:
    use_tlcd_str = "TLCD" if config["use_tlcd"] else "NO_TLCD"
    centralized_str = "CENTRALIZED" if config["centralized"] else "DECENTRALIZED"
    return f"{config['environment_name']}_{centralized_str}_{use_tlcd_str}"
