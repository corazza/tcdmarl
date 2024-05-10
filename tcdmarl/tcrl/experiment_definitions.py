from rl_common import RunConfig
from custom_dfas import (
    dfa_paper_b_doesnt_follow_a,
    dfa_paper_after_drinks_first_flowers,
    dfa_paper_no_office_after_flowers,
    dfa_review_spurious,
)
from custom_maps import (
    map_paper_coffe_drink_office,
    map_paper_four_rooms,
)
from custom_rms import (
    rm_paper_coffe_drink_office,
    rm_paper_a_and_b,
)

# TODO ADD TYPES
EXPERIMENT_COFFEE_SODA = {
    "name": "soda",
    "use_map": map_paper_coffe_drink_office,
    "use_prm": rm_paper_coffe_drink_office,
    "use_dfa": [
        dfa_paper_after_drinks_first_flowers,
        dfa_paper_no_office_after_flowers,
    ],
    "num_experiments": 20,
    "train_config": RunConfig(  # TODO improve this runconfig
        agent_name="qrm",
        total_timesteps=int(2e04),
        learning_rate=2e-1,
        gamma=0.9,
        q_init=2,
        epsilon=0.1,
        per_episode_steps=400,
        print_freq=100,
        reward_window_size=1000,
        print_actions=False,
        per_episode_steps_demo=50,
        num_demo_episodes=10,
    ),
}

EXPERIMENT_COLLECTION = {
    "name": "collection",
    "use_map": map_paper_four_rooms,
    "use_prm": rm_paper_a_and_b,
    "use_dfa": [dfa_paper_b_doesnt_follow_a],
    "num_experiments": 20,
    "train_config": RunConfig(
        agent_name="qrm",
        total_timesteps=int(4e04),
        learning_rate=2e-1,
        gamma=0.9,
        q_init=2,
        epsilon=0.1,
        per_episode_steps=1200,
        print_freq=100,
        reward_window_size=1000,
        print_actions=False,
        per_episode_steps_demo=50,
        num_demo_episodes=10,
    ),
}

# In response to review iKDv (https://openreview.net/forum?id=rUaxAp2O7M&noteId=ZuGL1NcaIR)
EXPERIMENT_COFFEE_SODA_SPURIOUS = {
    "name": "soda_spurious",
    "use_map": map_paper_coffe_drink_office,
    "use_prm": rm_paper_coffe_drink_office,
    "use_dfa": [
        dfa_paper_after_drinks_first_flowers,
        dfa_paper_no_office_after_flowers,
        dfa_review_spurious,
    ],
    "num_experiments": 20,
    "train_config": RunConfig(  # TODO improve this runconfig
        agent_name="qrm",
        total_timesteps=int(2e04),
        learning_rate=2e-1,
        gamma=0.9,
        q_init=2,
        epsilon=0.1,
        per_episode_steps=400,
        print_freq=100,
        reward_window_size=1000,
        print_actions=False,
        per_episode_steps_demo=50,
        num_demo_episodes=10,
    ),
}

MAP_EXPERIMENTS = {
    "coffee_soda": EXPERIMENT_COFFEE_SODA,
    "collection": EXPERIMENT_COLLECTION,
    "coffee_soda_spurious": EXPERIMENT_COFFEE_SODA_SPURIOUS,
}
