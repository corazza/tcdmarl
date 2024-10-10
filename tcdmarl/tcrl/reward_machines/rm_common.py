"""
Defines label-based reward machines and probabilistic reward machines.
"""

import copy
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Set, Tuple

import gym
import numpy as np
from gym import spaces
from tqdm import tqdm

from tcdmarl.path_consts import WORK_DIR
from tcdmarl.tcrl.regex.regex_compiler import generate_inputs
from tcdmarl.tcrl.rl_common import RunConfig


class RewardMachine(ABC):
    """
    State set: subset of Nat_0, implicitly set by transition and output functions
    Initial state: 0
    Output alphabet: implicitly set by transition and output functions
    Input alphabet: implicitly set by transition function
    """

    def __init__(
        self,
        appears: frozenset[str],
        nonterminal_states: frozenset[int],
        terminal_states: frozenset[int],
    ):
        super().__init__()
        self.appears = appears
        self.terminal_states = terminal_states
        self.nonterminal_states = nonterminal_states
        self.all_states = self.nonterminal_states.union(self.terminal_states)

        self.do_reward_shaping: bool = False
        self.alternative_rs: bool = False
        self.cheat_rs: bool = False
        self.true_reward: float = 0
        self.rs_potential: list[float] = [0.0] * len(self.all_states)
        self.rs_gamma: float = 0

        # nonterminal_states: set[int] = set()
        # for state in self.transitions:
        #     result.add(state)
        # filtered: frozenset[int] = frozenset(result) - self.terminal_states
        # assert len(filtered) == len(
        #     result
        # )  # terminal states do not have outgoing transitions
        # return frozenset(result)

    @abstractmethod
    def transition(
        self, current_state: int, input_symbol: frozenset[str]
    ) -> Tuple[int, float, bool]:
        raise NotImplementedError()

    def multiple_transitions(
        self,
        current_state: int,
        input_symbols: list[frozenset[str]],
        states: bool = False,
    ) -> list[int]:
        """Used for demos/testing"""
        rs: List[float] = []
        for input_symbol in input_symbols:
            current_state, r, _done = self.transition(current_state, input_symbol)
            if not states:
                rs.append(r)
            else:
                rs.append((current_state, r))
        return rs

    def reward_sum(self, input_symbols: list[frozenset[str]]) -> int:
        rs: list[int] = self.multiple_transitions(0, input_symbols)
        return sum(rs)

    def set_rs_potential(
        self,
        rs_potential: list[float],
        rs_potential_symbols: list[dict[frozenset[str], float]],
        rs_gamma: float,
    ):
        self.rs_potential = rs_potential
        self.rs_potential_symbols = rs_potential_symbols
        self.rs_gamma = rs_gamma
        self.do_reward_shaping = True

    def with_rs(self, config: RunConfig) -> "RewardMachine":
        new_rm = copy.deepcopy(self)
        rs_potential: list[float]
        rs_potential_symbols: list[dict[frozenset[str], float]]
        rs_potential, rs_potential_symbols = get_rs_potential_new(new_rm, config)
        new_rm.set_rs_potential(rs_potential, rs_potential_symbols, config.gamma)
        return new_rm

    def __call__(self, *input_symbols: Iterable[str]) -> list[int]:
        """Just a nicer interface for RewardMachine.multiple_transitions"""
        return self.multiple_transitions(
            0, [frozenset(x) for x in input_symbols], False
        )


class DeterministicRewardMachine(RewardMachine):
    """
    Deterministic reward machine definition.

    State set: subset of Nat_0, implicitly set by transition and output functions
    Initial state: 0
    Output alphabet: implicitly set by transition and output functions
    Input alphabet: implicitly set by transition function
    """

    def __init__(
        self,
        transitions: dict[int, dict[frozenset[str], Tuple[int, int]]],
        appears: frozenset[str],
        terminal_states: frozenset[int],
    ):
        super().__init__(appears, frozenset(transitions.keys()), terminal_states)
        self.transitions = transitions

    # TODO convert all to numpy
    def transition(
        self, current_state: int, input_symbol: frozenset[str]
    ) -> Tuple[int, float, bool]:
        input_symbol = frozenset(self.appears.intersection(input_symbol))
        if current_state not in self.transitions:
            assert current_state in self.terminal_states
            return (current_state, 0, False)
        assert input_symbol in self.transitions[current_state]
        # if input_symbol not in self.transitions[current_state]:
        #     return (current_state, 0)
        next_state: int
        next_reward: float
        next_state, next_reward = self.transitions[current_state][input_symbol]
        self.true_reward = next_reward
        if self.do_reward_shaping:
            next_reward = (
                next_reward
                + self.rs_gamma * self.rs_potential[next_state]
                - self.rs_potential[current_state]
            )
        done: bool = next_state in self.terminal_states
        return next_state, next_reward, done


# TODO all RMs are PRMs, use this in code
class ProbabilisticRewardMachine(RewardMachine):
    """
    Probabilistic reward machine definition.

    State set: subset of Nat_0, implicitly set by transition and output functions
    Initial state: 0
    Output alphabet: implicitly set by transition and output functions
    Input alphabet: implicitly set by transition function

    If a (state, label, state') entry is missing in the probabilistic transition function,
    the transition probability is assumed to be zero and the output is irrelevant.
    """

    def __init__(
        self,
        transitions: dict[int, dict[frozenset[str], dict[int, Tuple[float, float]]]],
        appears: frozenset[str],
        terminal_states: frozenset[int],
    ):
        super().__init__(appears, frozenset(transitions.keys()), terminal_states)
        self.transitions = transitions
        self.state_map_after_product: Dict[int, int] = dict()
        self.original_terminal_states = terminal_states

        # LOOKBACK
        self.last_state: int = 0
        self.last_input_symbol: frozenset[str] = frozenset()

        for state in self.transitions:
            for input_symbol in self.transitions[state]:
                prob_sum: float = 0.0
                for next_state in self.transitions[state][input_symbol]:
                    prob_sum += self.transitions[state][input_symbol][next_state][0]
                assert abs(prob_sum - 1) < 1e-3

    def negate(self) -> "ProbabilisticRewardMachine":
        transitions = copy.deepcopy(self.transitions)
        for state, state_transitions in transitions.items():
            for input_symbols, transition in state_transitions.items():
                for next_state, (probability, output) in transition.items():
                    transition[next_state] = (probability, -output)
        negated = copy.deepcopy(self)
        negated.transitions = transitions
        return negated

    ### BEGIN CYRUS INTERFACE

    def get_reward(self, u1: int, u2: int) -> int:
        if self.is_terminal_state(u1):
            return 0
        for input_symbol in self.transitions[u1]:
            if u2 in self.transitions[u1][input_symbol]:
                return int(self.transitions[u1][input_symbol][u2][1])
        raise ValueError(f"Transition from {u1} to {u2} not found")

    def get_next_state(self, u: int, event: str) -> int:
        if self.is_terminal_state(u):
            return u
        (u2, _r, _d) = self.transition(u, frozenset({event}))
        return u2

    def is_terminal_state(self, u: int) -> bool:
        return u in self.terminal_states

    def get_initial_state(self) -> int:
        return 0

    def is_event_available(self, u: int, event: str):
        is_event_available = False
        if u in self.transitions:
            if frozenset({event}) in self.transitions[u]:
                is_event_available = True
        return is_event_available

    ### END CYRUS INTERFACE

    def transition(
        self, current_state: int, input_symbol: frozenset[str]
    ) -> Tuple[int, int, bool]:
        input_symbol = frozenset(self.appears.intersection(input_symbol))
        if current_state not in self.transitions:
            assert current_state in self.terminal_states
            return (current_state, 0, False)
        assert input_symbol in self.transitions[current_state]
        # if input_symbol not in self.transitions[current_state]:
        #     return (current_state, 0, False)
        probs: list[float] = []
        rewards: list[int] = []
        states: list[int] = []

        for next_state, prob_reward in self.transitions[current_state][
            input_symbol
        ].items():
            current_probs: float = sum(probs)
            probs.append(prob_reward[0] + current_probs)
            rewards.append(prob_reward[1])
            states.append(next_state)

        x: float = np.random.random()

        for i, probs_i in enumerate(probs):
            if x <= probs_i:
                next_state = states[i]
                next_reward = rewards[i]
                # TODO abstract away
                self.true_reward = next_reward
                # if self.do_reward_shaping and current_state == next_state and False:
                if self.do_reward_shaping:
                    if not self.alternative_rs and not self.cheat_rs:
                        next_reward = (
                            next_reward
                            + self.rs_gamma * self.rs_potential[next_state]
                            - self.rs_potential[current_state]
                        )
                    elif not self.cheat_rs:
                        next_reward = (
                            next_reward
                            + self.rs_gamma
                            * self.rs_potential_symbols[next_state][frozenset({})]
                            - self.rs_potential_symbols[current_state][input_symbol]
                        )
                    elif self.cheat_rs:
                        if current_state == next_state:
                            next_reward = (
                                next_reward
                                + self.rs_gamma * self.rs_potential[next_state]
                                - self.rs_potential[current_state]
                            )
                        # next_reward = (
                        #     next_reward
                        #     + self.rs_potential_symbols[current_state][input_symbol]
                        #     - (1.0 / self.rs_gamma)
                        #     * self.rs_potential_symbols[self.last_state][
                        #         self.last_input_symbol
                        #     ]
                        # )

                done: bool = next_state in self.terminal_states
                return next_state, next_reward, done
        assert False

    def add_tlcd(
        self, causal_dfa: "CausalDFA", cache_name: str
    ) -> "ProbabilisticRewardMachine":
        value_iteration_params: RunConfig = RunConfig(
            agent_name="----",
            total_timesteps=int(1e03),
            learning_rate=1e-1,
            gamma=0.9,
            q_init=2,
            epsilon=0.2,
            per_episode_steps=2000,
            print_freq=1000,
            reward_window_size=5000,
            print_actions=False,
            per_episode_steps_demo=20,
            num_demo_episodes=10,
        )

        prm: ProbabilisticRewardMachine = copy.deepcopy(self)
        b: ProbabilisticRewardMachine = prm
        b1: ProbabilisticRewardMachine = copy.deepcopy(self)
        # b2: ProbabilisticRewardMachine = copy.deepcopy(self).negate()

        b = prm_causal_product(b, causal_dfa, scheme="no_effect")
        b1 = prm_causal_product(b1, causal_dfa, scheme="reward_shaping")
        # b2 = prm_causal_product(b2, causal_dfa, scheme="reward_shaping")

        # print("Computing B1...")
        # state_potentials_b1, _state_action_potentials_b1 = get_rs_potential_new(
        #     b1, value_iteration_params
        # )

        # Construct the path to the cache file
        cache_file_path = WORK_DIR / f"{cache_name}.p"

        # Check if the cache file exists
        if cache_file_path.exists():
            print("Loading B1 from cache...")
            # Load the data from the cache file
            with open(cache_file_path, "rb") as cache_file:
                state_potentials_b1, _state_action_potentials_b1 = pickle.load(
                    cache_file
                )
        else:
            # Compute the function
            print("Computing B1...")
            state_potentials_b1, _state_action_potentials_b1 = get_rs_potential_new(
                b1, value_iteration_params
            )

            # Save the result to the cache file
            with open(cache_file_path, "wb") as cache_file:
                pickle.dump(
                    (state_potentials_b1, _state_action_potentials_b1), cache_file
                )

        # print("Computing B2...")
        # state_potentials_b2, _state_action_potentials_b2 = get_rs_potential_new(
        #     b2, value_iteration_params
        # )

        b.original_terminal_states = b.terminal_states

        for rm_state, _potential in enumerate(state_potentials_b1):
            if (
                abs(state_potentials_b1[rm_state])
                < 1e-4
                # We do not need to use B2 in case there are no negative rewards
                # and abs(state_potentials_b2[rm_state]) < 1e-4
            ):
                b.terminal_states = b.terminal_states.union(frozenset({rm_state}))

        return b


class RewardMachineRunner:
    """Stateful runner. Contains a reward machine definition and current state."""

    def __init__(self, reward_machine: RewardMachine):
        self.reward_machine: RewardMachine = reward_machine
        self.current_state: int = 0

    def transition(self, input_symbol: frozenset[str]) -> tuple[int, bool]:
        next_state: int
        reward: int
        done: bool
        next_state, reward, done = self.reward_machine.transition(
            self.current_state, input_symbol
        )
        self.current_state = next_state
        return reward, done


class CausalDFA:
    def __init__(
        self,
        appears: frozenset[str],
        transitions: dict[int, dict[frozenset[str], int]],
        tagged_states: frozenset[int],
    ):
        self.appears = appears
        self.transitions: dict[int, dict[frozenset[str], int]] = transitions
        self.tagged_states: frozenset[int] = tagged_states

    def transition(self, current_state: int, input_symbol: frozenset[str]) -> int:
        input_symbol = frozenset(self.appears.intersection(input_symbol))
        assert input_symbol in self.transitions[current_state]
        return self.transitions[current_state][input_symbol]


def prm_causal_product(
    prm: ProbabilisticRewardMachine, dfa: CausalDFA, scheme: str
) -> ProbabilisticRewardMachine:
    appears: frozenset[str] = prm.appears.union(dfa.appears)
    pair_to_self_state_map: dict[tuple[int, int], int] = dict()
    self_to_pair_state_map: dict[int, tuple[int, int]] = dict()
    nonterminal_states: Set[int] = set()
    terminal_states: Set[int] = set()
    transitions: dict[int, dict[frozenset[str], dict[int, Tuple[float, int]]]] = dict()

    state_counter: int = 0
    for prm_state in prm.all_states:
        for dfa_state in dfa.transitions:
            pair_to_self_state_map[(prm_state, dfa_state)] = state_counter
            self_to_pair_state_map[state_counter] = (prm_state, dfa_state)
            if prm_state in prm.terminal_states:
                terminal_states.add(state_counter)
            else:
                nonterminal_states.add(state_counter)
            state_counter += 1

    for prm_state in prm.transitions:
        for dfa_state in dfa.transitions:
            state = pair_to_self_state_map[(prm_state, dfa_state)]
            transitions[state] = dict()
            for input_symbol in generate_inputs(appears):
                input_symbol_prm = prm.appears.intersection(input_symbol)
                input_symbol_dfa = dfa.appears.intersection(input_symbol)
                transitions[state][input_symbol] = dict()
                next_dfa_state = dfa.transitions[dfa_state][input_symbol_dfa]
                for next_prm_state in prm.transitions[prm_state][input_symbol_prm]:
                    next_state = pair_to_self_state_map[
                        (next_prm_state, next_dfa_state)
                    ]
                    p, r = prm.transitions[prm_state][input_symbol_prm][next_prm_state]
                    if (
                        scheme == "reward_shaping"
                        and next_dfa_state in dfa.tagged_states
                    ):
                        r = -100  # TODO FIXME better
                    elif (
                        scheme == "create_terminals"
                        and next_dfa_state in dfa.tagged_states
                    ):
                        terminal_states.add(next_state)
                    else:
                        # assert scheme == "no_effect"
                        pass
                    transitions[state][input_symbol][next_state] = (p, r)

    state_map_after_product: Dict[int, int] = dict()
    for u, (old_u, _dfa_q) in self_to_pair_state_map.items():
        state_map_after_product[u] = old_u

    result = ProbabilisticRewardMachine(transitions, appears, terminal_states)
    result.state_map_after_product = state_map_after_product

    return result

    # def transition(
    #     self, current_state: int, input_symbol: frozenset[str]
    # ) -> Tuple[int, int, bool]:
    #     current_prm_state, current_dfa_state = self.self_to_pair_state_map[
    #         current_state
    #     ]

    #     next_prm_state, next_prm_reward, prm_done = self.prm.transition(
    #         current_prm_state, input_symbol
    #     )
    #     next_dfa_state = self.dfa.transition(current_dfa_state, input_symbol)

    #     next_state = self.pair_to_self_state_map[(next_prm_state, next_dfa_state)]
    #     next_reward = (
    #         next_prm_reward if next_dfa_state not in self.dfa.tagged_states else -100
    #     )  # TODO lowest

    #     if self.do_reward_shaping:
    #         if not self.alternative_rs:
    #             next_reward = (
    #                 next_reward
    #                 + self.rs_gamma * self.rs_potential[next_state]
    #                 - self.rs_potential[current_state]
    #             )
    #         else:
    #             try:
    #                 next_reward = (
    #                     next_reward
    #                     + self.rs_gamma
    #                     * self.rs_potential_symbols[next_state][frozenset({})]
    #                     - self.rs_potential_symbols[current_state][input_symbol]
    #                 )
    #             except:
    #                 IPython.embed()

    #     return next_state, next_reward, prm_done


# TODO move to rewardshapingenv
def action_map(
    all_actions: list[frozenset[str]],
) -> tuple[dict[frozenset[str], int], dict[int, frozenset[str]]]:
    first: dict[str, int] = {action: i for i, action in enumerate(all_actions)}
    second: dict[int, str] = {i: action for action, i in first.items()}
    return first, second


class RewardShapingEnv(gym.Env):
    def __init__(self, reward_machine: RewardMachine, per_episode_steps: int):
        super().__init__()
        self.reward_machine: RewardMachine = reward_machine
        self.per_episode_steps: int = per_episode_steps
        self.step_counter: int = 0

        self.action_labels: list[frozenset[str]] = list(
            generate_inputs(self.reward_machine.appears)
        )

        self.action_to_id: dict[frozenset[str], int]
        self.id_to_action: dict[int, frozenset[str]]
        self.action_to_id, self.id_to_action = action_map(self.action_labels)

        self.num_actions: int = len(self.action_to_id)

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(
            max(self.reward_machine.all_states) + 1  # highest state index & zero
        )

        self.reset()

    def reset(self):
        self.state: int = 0
        self.step_counter: int = 0
        return self.state

    def step(self, action):
        input_symbol: frozenset[str] = self.id_to_action[action]
        next_state: int
        reward: int
        done: bool
        next_state, reward, done = self.reward_machine.transition(
            self.state, input_symbol
        )
        self.state = next_state
        terminated: bool = done
        truncated: bool = self.step_counter >= self.per_episode_steps

        self.step_counter += 1

        return self.state, reward, terminated, truncated, {}


def get_rs_potential(
    rm: RewardMachine, config: RunConfig
) -> Tuple[list[float], list[dict[str, float]]]:
    env = RewardShapingEnv(rm, config.per_episode_steps)
    Q = rl_agents.reward_shaping_q_learning.learn(
        env=env,
        total_timesteps=config.total_timesteps,
        lr=config.leaning_rate,
        epsilon=config.epsilon,
        gamma=config.gamma,
        q_init=config.q_init,
        print_freq=config.print_freq,
    )
    Q[list(rm.terminal_states)] = 0
    V = -1 * np.max(Q, axis=1)
    rs_potential_symbols: list[dict[str, float]] = []
    for state in range(Q.shape[0]):
        rs_potential_symbols.append(dict())
        for action in range(Q.shape[1]):
            rs_potential_symbols[state][env.id_to_action[action]] = -Q[state, action]
    return V.tolist(), rs_potential_symbols


def get_rs_potential_new(
    rm: ProbabilisticRewardMachine, config: RunConfig
) -> Tuple[list[float], list[dict[str, float]]]:
    assert not rm.alternative_rs and not rm.do_reward_shaping

    action_labels: list[frozenset[str]] = list(generate_inputs(rm.appears))
    action_to_id: dict[frozenset[str], int]
    id_to_action: dict[int, frozenset[str]]
    action_to_id, id_to_action = action_map(action_labels)

    Q = np.full((len(rm.all_states), len(action_to_id)), config.q_init, dtype=float)

    current_iteration: int = 0

    begin_lr: float = 0.5
    end_lr: float = 0.05

    with tqdm(total=config.total_timesteps) as pbar:
        while current_iteration < config.total_timesteps:
            # error_tot: float = 0
            percent: float = current_iteration / config.total_timesteps
            lr = end_lr + (begin_lr - end_lr) * percent
            for rm_state in rm.nonterminal_states:
                for input_code in id_to_action:
                    input_symbol = id_to_action[input_code]
                    estimate: float = 0
                    for next_state in rm.transitions[rm_state][input_symbol]:
                        p, r = rm.transitions[rm_state][input_symbol][next_state]
                        done = next_state in rm.terminal_states
                        if done:
                            value = r
                        else:
                            value = r + config.gamma * np.max(Q[next_state, :])
                        estimate += p * value
                    # next_state, r, done = rm.transition(rm_state, input_symbol)
                    delta = estimate - Q[rm_state][input_code]
                    # error_tot += delta
                    Q[rm_state][input_code] += lr * delta
            current_iteration += 1
            pbar.update()
            # if current_iteration % config.print_freq == 0:
            #     error_tot: float = 0

    Q[list(rm.terminal_states)] = 0
    V = np.max(Q, axis=1)

    rs_potential_symbols: list[dict[str, float]] = []
    for state in range(Q.shape[0]):
        rs_potential_symbols.append(dict())
        for input_code in id_to_action:
            rs_potential_symbols[state][id_to_action[input_code]] = -Q[
                state, input_code
            ]

    return (-V).tolist(), rs_potential_symbols
