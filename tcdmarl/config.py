# TODO implement this throughout the project

from typing import TypedDict


class ExperimentConfig(TypedDict):
    # Environment name. E.g., "routing"
    environment_name: str

    # Use centralized or decentralized
    centralized: bool

    # Use the TL-CD to expedite learning
    use_tlcd: bool

    # Episode length
    episode_length: int

    # Number of episodes
    num_episodes: int


class RunConfig(TypedDict):
    # Number of separate trials to run the algorithm for
    num_trials: int

    # Run experiments
    experiments: list[ExperimentConfig]
