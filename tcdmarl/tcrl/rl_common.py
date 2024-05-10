from datetime import datetime
import csv
import os
import IPython
from matplotlib import pyplot as plt
import numpy as np


class TrainingResults:
    def __init__(
        self,
        times: list[float],
        steps: list[int],
        reward_totals: list[float],
        rs_totals: list[float],
    ):
        self.times: list[float] = times
        self.steps: list[int] = steps
        self.reward_totals: list[float] = reward_totals
        self.rs_totals: list[float] = rs_totals
        self.additional = {}


def average_results(results_list: list[TrainingResults]) -> TrainingResults:
    num_results = len(results_list)
    result_length = min(len(r.reward_totals) for r in results_list)

    avg_times = []
    avg_reward_totals = []
    avg_rs_totals = []

    average_additional = {}

    for i in range(result_length):
        avg_time = sum(result.times[i] for result in results_list) / num_results
        try:
            avg_reward_total = (
                sum(result.reward_totals[i] for result in results_list) / num_results
            )
        except:
            IPython.embed()
        avg_rs_total = sum(result.rs_totals[i] for result in results_list) / num_results

        avg_times.append(avg_time)
        avg_reward_totals.append(avg_reward_total)
        avg_rs_totals.append(avg_rs_total)

    for r in results_list:
        for a in r.additional:
            if a not in average_additional:
                average_additional[a] = 0
            average_additional[a] += r.additional[a]

    for a in average_additional:
        average_additional[a] /= len(results_list)

    tr = TrainingResults(
        avg_times,
        results_list[0].steps[:result_length],
        avg_reward_totals,
        avg_rs_totals,
    )

    tr.additional = average_additional
    return tr


# TODO refactor this entire class
class RunConfig:
    def __init__(
        self,
        agent_name: str,
        total_timesteps: int,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        q_init: float,
        print_freq: int,
        reward_window_size: int,
        per_episode_steps: int,
        num_demo_episodes: int,
        print_actions: bool,
        per_episode_steps_demo: int,
    ):
        self.agent_name: str = agent_name
        self.total_timesteps: int = total_timesteps
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.q_init: float = q_init
        self.leaning_rate: float = learning_rate
        self.print_freq: int = print_freq
        self.reward_window_size: int = reward_window_size
        self.per_episode_steps: int = per_episode_steps
        self.print_actions = print_actions
        self.num_demo_episodes: int = num_demo_episodes
        self.per_episode_steps_demo = per_episode_steps_demo


class PrintDisplayer:
    def __init__(self, silence: bool = False):
        self.silence = silence

    def display_message(self, msg: str):
        if not self.silence:
            print(msg)


def save_results_to_dynamic_folder(
    prefix, name, downsampled_steps, downsampled_rewards
):
    # Create a folder based on the current date
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_path = os.path.join("results", current_datetime)

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{prefix}_qrm_results_{name}.csv")

    print(f"Saving results to {file_path}")

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Value"])
        for step, reward_total in zip(downsampled_steps, downsampled_rewards):
            writer.writerow([step, reward_total])


def display_and_save_data(
    prefix: str, results_list: dict[str, TrainingResults], show_rs: bool = False
):
    plt.figure(figsize=(10, 6))

    for name, results in results_list.items():
        window_size = 10

        smoothed_rewards = np.convolve(
            results.reward_totals, np.ones(window_size) / window_size, mode="valid"
        )
        smoothed_rs = np.convolve(
            results.rs_totals, np.ones(window_size) / window_size, mode="valid"
        )

        smoothed_steps = results.steps[window_size - 1 :]

        plt.plot(smoothed_steps, smoothed_rewards, label=name)
        if show_rs:
            plt.plot(smoothed_steps, smoothed_rs, label=name + " (RS)")

        # Down-sample the data by selecting every nth data point
        n = len(smoothed_steps) // 20  # Adjust max_display_points as needed
        downsampled_steps = smoothed_steps[::n]
        downsampled_rewards = smoothed_rewards[::n]

        save_results_to_dynamic_folder(
            prefix, name, downsampled_steps, downsampled_rewards
        )

    # Set the limits of the x and y axes
    plt.xlim(
        0, None
    )  # Set the x-axis lower limit to 0, and let it auto-scale the upper limit
    plt.ylim(
        0, None
    )  # Set the y-axis lower limit to 0, and let it auto-scale the upper limit

    plt.xlabel("Step")
    plt.ylabel("Smoothed Reward Total")
    plt.title("Smoothed Reward Total Over Time")
    plt.legend()
    plt.grid(True)

    plt.show()
