"""
https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/rl_agents/qlearning/qlearning.py
"""

import numpy as np
import csv
import random
import time
from tqdm import tqdm
import IPython

# import baselines_logger as logger
from consts import *

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

    # Running Q-Learning
    reward_total = 0
    step = 0
    num_episodes = 0
    states = list(range(env.observation_space.n))
    actions = list(range(env.action_space.n))
    Q = np.full((len(states), len(actions)), q_init, dtype=float)

    with tqdm(total=total_timesteps) as pbar:
        while step < total_timesteps:
            s = env.reset()
            while True:
                # Selecting and executing the action
                a = (
                    random.choice(actions)
                    if random.random() < epsilon
                    else np.argmax(Q[s, :])
                )
                sn, r, rm_done, truncated, info = env.step(a)

                done: bool = rm_done or truncated

                if done:
                    delta = r - Q[s][a]
                else:
                    # gamma_factor = gamma if s != sn else gamma**15
                    gamma_factor = gamma
                    delta = r + gamma_factor * np.max(Q[sn, :]) - Q[s][a]
                Q[s][a] += lr * delta

                # moving to the next state
                reward_total += r
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
                    # to_save.append((time.time(), step, reward_total / print_freq))
                    for log in logs:
                        # logger.record_tabular(log[0], log[1])
                        message = message + f"\n{log[0]}: {log[1]}"
                    # logger.dump_tabular()
                    # displayer.display_message(message)  # type: ignore
                    if step > 0:
                        pbar.update(print_freq)
                    reward_total = 0
                if done:
                    num_episodes += 1
                    break
                s = sn

    # with open("qrm_results.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Wall time", "Step", "Value"])  # write the header
    #     for item in to_save:
    #         writer.writerow(item)  # write the data

    return Q
