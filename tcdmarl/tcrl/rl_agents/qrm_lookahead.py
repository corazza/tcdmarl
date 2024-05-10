"""
https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/rl_agents/qlearning/qlearning.py
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import time
from tqdm import tqdm
import IPython

# import baselines_logger as logger
from consts import *
from rl_common import TrainingResults

# from baselines import logger


def get_qmax(Q, s, actions, q_init):
    if s not in Q:
        Q[s] = dict([(a, q_init) for a in actions])
    return max(Q[s].values())


def get_best_action(Q, s, actions, q_init):
    qmax = get_qmax(Q, s, actions, q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)


def learn(
    env,
    total_timesteps,
    lr,
    epsilon,
    gamma,
    q_init,
    print_freq,
    displayer=None,
    network=None,
    add_message: str = "",
):
    """Train a tabular q-learning model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        This is just a placeholder to be consistent with the openai-baselines interface, but we don't really use state-approximation in tabular q-learning
    lr: float
        learning rate
    total_timesteps: int
        number of env steps to optimizer for
    epsilon: float
        epsilon-greedy exploration
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    gamma: float
        discount factor
    q_init: float
        initial q-value for unseen states
    """

    assert env.reward_machine.alternative_rs

    # Running Q-Learning
    reward_total = 0
    rs_total = 0
    step = 0
    num_episodes = 0
    Q = {}
    actions = list(range(env.action_space.n))

    to_save: list[tuple[float, int, float, float]] = []
    steps_on_left: int = 0

    with tqdm(total=total_timesteps) as pbar:
        while step < total_timesteps:
            s = tuple(env.reset())

            if s not in Q:
                Q[s] = dict([(a, q_init) for a in actions])
            while True:
                # Selecting and executing the action
                a = (
                    random.choice(actions)
                    if random.random() < epsilon
                    else get_best_action(Q, s, actions, q_init)
                )
                sn, r, done, info = env.step(a)
                sn = tuple(sn)

                if sn[1] <= 2:
                    steps_on_left += 1

                experiences = []
                # Adding counterfactual experience (this will alrady include shaped rewards if use_rs=True)
                for _s, _a, _r, _sn, _done in info["crm-experience"]:
                    experiences.append((tuple(_s), _a, _r, tuple(_sn), _done))

                for _s, _a, _r, _sn, _done in experiences:
                    if _s not in Q:
                        Q[_s] = dict([(b, q_init) for b in actions])
                    if _done:
                        _delta = _r - Q[_s][_a]
                    else:
                        _delta = (
                            _r + gamma * get_qmax(Q, _sn, actions, q_init) - Q[_s][_a]
                        )
                    Q[_s][_a] += lr * _delta

                # moving to the next state
                reward_total += info["true_reward"]
                rs_total += r - info["true_reward"]
                step += 1
                if step % print_freq == 0:
                    logs: list[tuple[str, str]] = []
                    # message: str = 'Training for:\n\n"' + add_message + '"\n\n'
                    message: str = ""
                    logs.append(
                        (
                            "Steps",
                            f"{step}/{total_timesteps} ({(step/total_timesteps)*100:.2f}%)",
                        )
                    )
                    logs.append(("Episodes", f"{num_episodes}"))
                    logs.append(("Reward per step", f"{reward_total / print_freq:.5f}"))
                    to_save.append(
                        (
                            time.time(),
                            step,
                            reward_total / print_freq,
                            rs_total / print_freq,
                        )
                    )
                    for log in logs:
                        # logger.record_tabular(log[0], log[1])
                        message = message + f"\n{log[0]}: {log[1]}"
                    # logger.dump_tabular()
                    displayer.display_message(message)  # type: ignore
                    if step > 0:
                        pbar.update(print_freq)
                    reward_total = 0
                    rs_total = 0
                if done:
                    num_episodes += 1
                    break
                s = sn

    times, steps, rewards, rss = zip(*to_save)

    tr = TrainingResults(
        times=times,
        steps=steps,
        reward_totals=rewards,
        rs_totals=rss,
    )

    tr.additional["steps_on_left"] = steps_on_left

    return Q, tr
