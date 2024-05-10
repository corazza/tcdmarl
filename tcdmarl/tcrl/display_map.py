from main import EXPERIMENT_COFFEE_SODA, EXPERIMENT_COLLECTION
from maps.map_env import MapEnv
from maps.maps_common import Map


def display_map(experiment: dict):
    map: Map = experiment["use_map"]()
    env = MapEnv(map, f"test", per_episode_steps=100)
    env.start_render()
    while True:
        env.render(mode="human", show_coords=True)


def main():
    display_map(EXPERIMENT_COFFEE_SODA)


if __name__ == "__main__":
    main()
