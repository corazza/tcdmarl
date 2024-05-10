import sys

from tcdmarl.tcrl.consts import SEED
from tcdmarl.tcrl.experiment_definitions import MAP_EXPERIMENTS
from tcdmarl.tcrl.maps.map_env import MapEnv
from tcdmarl.tcrl.reward_machines.rm_common import (
    CausalDFA,
    ProbabilisticRewardMachine,
    get_rs_potential_new,
    prm_causal_product,
)
from tcdmarl.tcrl.reward_machines.rm_env import RMEnvWrapper
from tcdmarl.tcrl.rl_common import (
    PrintDisplayer,
    RunConfig,
    TrainingResults,
    average_results,
    display_and_save_data,
)
from tcdmarl.tcrl.util import set_all_seeds


def run_experiment(experiment: dict, seed: int):
    set_all_seeds(seed)

    value_iteration_params: RunConfig = RunConfig(
        agent_name="----",
        total_timesteps=int(1e03),
        learning_rate=1e-1,
        gamma=experiment["train_config"].gamma,
        q_init=2,
        epsilon=0.2,
        per_episode_steps=2000,
        print_freq=1000,
        reward_window_size=5000,
        print_actions=False,
        per_episode_steps_demo=20,
        num_demo_episodes=10,
    )

    map_env: MapEnv = MapEnv(
        experiment["use_map"](),
        "Flowers will sink you",
        per_episode_steps=experiment["train_config"].per_episode_steps,
    )

    prm: ProbabilisticRewardMachine = experiment["use_prm"]()
    b: ProbabilisticRewardMachine = prm
    b1: ProbabilisticRewardMachine = experiment["use_prm"]()
    b2: ProbabilisticRewardMachine = experiment["use_prm"]().negate()

    for use_dfa in experiment["use_dfa"]:
        causal_dfa: CausalDFA = use_dfa()
        b = prm_causal_product(b, causal_dfa, scheme="no_effect")
        b1 = prm_causal_product(b1, causal_dfa, scheme="reward_shaping")
        b2 = prm_causal_product(b2, causal_dfa, scheme="reward_shaping")

    print("Computing B1...")
    state_potentials_b1, _state_action_potentials_b1 = get_rs_potential_new(
        b1, value_iteration_params
    )
    print("Computing B2...")
    state_potentials_b2, _state_action_potentials_b2 = get_rs_potential_new(
        b2, value_iteration_params
    )

    # TODO check reward shaping again

    for rm_state, _potential in enumerate(state_potentials_b1):
        if (
            abs(state_potentials_b1[rm_state]) < 1e-4
            and abs(state_potentials_b2[rm_state]) < 1e-4
        ):
            b.terminal_states = b.terminal_states.union(frozenset({rm_state}))

    results_list_wo_causal: list[TrainingResults] = []
    results_list_with_causal: list[TrainingResults] = []

    for i in range(experiment["num_experiments"]):
        print(f'Experiment {i+1}/{experiment["num_experiments"]}')
        print("Training without causal info...")
        _Q_wo_causal, results_wo_causal = train(
            experiment["train_config"],
            RMEnvWrapper(
                map_env,
                prm,
            ),
            PrintDisplayer(silence=True),
        )
        results_list_wo_causal.append(results_wo_causal)

        print("Training with causal info...")
        _Q_with_causal, results_with_causal = train(
            experiment["train_config"],
            RMEnvWrapper(
                map_env,
                b,
            ),
            PrintDisplayer(silence=True),
        )
        results_list_with_causal.append(results_with_causal)

    avg_resulsts_wo_causal = average_results(results_list_wo_causal)
    avg_resulsts_with_causal = average_results(results_list_with_causal)

    display_and_save_data(
        experiment["name"],
        {
            "without_causal": avg_resulsts_wo_causal,
            "including_causal": avg_resulsts_with_causal,
        },
    )


def train(config: RunConfig, env, displayer):
    if config.agent_name == "random":
        raise NotImplementedError()
    elif config.agent_name == "qrm":
        learn = rl_agents.qrm.learn
    elif config.agent_name == "qrml":
        learn = rl_agents.qrm_lookahead.learn
    else:
        raise ValueError(f"no known agent '{config.agent_name}'")

    return learn(
        env=env,
        total_timesteps=config.total_timesteps,
        lr=config.leaning_rate,
        epsilon=config.epsilon,
        gamma=config.gamma,
        q_init=config.q_init,
        print_freq=config.print_freq,
        reward_window_size=config.reward_window_size,
        add_message=env.title,
        displayer=displayer,
    )


def main():
    if len(sys.argv) != 2:
        print("Expected argument: <experiment_name>")
        print(f"one of: {', '.join(MAP_EXPERIMENTS.keys())}")
        sys.exit(1)

    experiment_name: str = sys.argv[1]
    seed: int = SEED  # alternatively, use int(time.time())

    run_experiment(MAP_EXPERIMENTS[experiment_name], seed)


if __name__ == "__main__":
    main()
