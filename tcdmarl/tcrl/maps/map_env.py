import IPython
import gym
from gym import spaces
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from maps.maps_common import *


NEUTRAL_COLOR = (0.1, 0.1, 0.1)


def action_map(all_actions: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    first: dict[str, int] = {action: i for i, action in enumerate(all_actions)}
    second: dict[int, str] = {i: action for action, i in first.items()}
    return first, second


class MapEnv(gym.Env):
    def __init__(self, map: Map, title: str, per_episode_steps: int):
        super().__init__()
        self.map: Map = map
        self.title: str = title
        self.per_episode_steps: int = per_episode_steps
        self.step_counter: int = 0
        self.redraw: bool = True
        self.action_to_id: dict[str, int]
        self.id_to_action: dict[int, str]
        self.action_to_id, self.id_to_action = action_map(
            get_all_labels_from_tag(map.labels, "ACTION")
        )
        self.num_actions: int = len(self.action_to_id)

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0,
            high=max([self.map.size[0], self.map.size[1]]),  # type: ignore
            shape=(2,),
            dtype=np.uint8,
        )  # type: ignore

        self.reset()

    def step(self, action):
        action_id: int = action  # type: ignore
        action_name: str = self.id_to_action[action_id]

        in_sink: bool = (self.state[0], self.state[1]) in self.map.sinks

        horizontal_delta: int = 0
        vertical_delta: int = 0

        if action_name == "left":
            horizontal_delta -= 1
        elif action_name == "right":
            horizontal_delta += 1
        elif action_name == "up":
            vertical_delta -= 1
        elif action_name == "down":
            vertical_delta += 1

        if "force_up" in self.map.content[self.state[0]][self.state[1]]:
            vertical_delta = -1
            horizontal_delta = 0
        elif "force_down" in self.map.content[self.state[0]][self.state[1]]:
            vertical_delta = 1
            horizontal_delta = 0
        if "force_right" in self.map.content[self.state[0]][self.state[1]]:
            vertical_delta = 0
            horizontal_delta = 1
        elif "force_left" in self.map.content[self.state[0]][self.state[1]]:
            vertical_delta = 0
            horizontal_delta = -1

        new_i: int = self.state[0] + vertical_delta
        new_j: int = self.state[1] + horizontal_delta

        new_i = max(0, min(self.map.size[0] - 1, new_i))
        new_j = max(0, min(self.map.size[1] - 1, new_j))

        if "wall" not in self.map.content[new_i][new_j] and not in_sink:
            self.state = [new_i, new_j]

        labels: frozenset[str] = frozenset([action_name]).union(
            frozenset(self.map.content[self.state[0]][self.state[1]])
        )

        self.label_history.append(labels)

        reward: int = 0
        terminated: bool = False
        truncated: bool = (
            self.step_counter >= self.per_episode_steps
        )  # TODO FIXME this should be abstracted into another class

        self.step_counter += 1

        return self.state, reward, terminated, truncated, {}

    def get_events(self) -> frozenset[str]:
        return self.label_history[-1]

    def reset(self):
        self.state: list[int] = list(self.map.get_new_spawn())
        assert "wall" not in self.map.content[self.state[0]][self.state[1]]
        self.label_history: list[frozenset[str]] = []
        self.step_counter: int = 0
        return self.state

    # TODO (?) remove all this text mode stuff
    def pretty_row(self, row: list[frozenset[str]]) -> list[str]:
        return ["."] + [self.cell_text(vars) for vars in row] + ["."]

    def cell_text(self, vars: frozenset[str]) -> str:
        text_map: dict[str, str] = {
            "force_right": "→",
            "force_up": "↑",
            "force_left": "←",
            "force_down": "↓",
        }
        filtered: set[str] = set()
        for var in vars:
            if "AREA" in self.map.labels[var]:
                continue
            if var == "wall":
                continue
            filtered.add(text_map[var] if var in text_map else var)
        return "\n".join(filtered)

    def cell_color(self, vars: frozenset[str]) -> tuple[float, float, float]:
        if "wall" in vars:
            return (221 / 255, 94 / 255, 94 / 255)
        elif "forest" in vars:
            return (90 / 255, 189 / 255, 36 / 255)
        elif "town" in vars:
            return (157 / 255, 161 / 255, 184 / 255)
        elif "factory" in vars:
            return (95 / 255, 77 / 255, 71 / 255)
        elif "field" in vars:
            return (254 / 255, 243 / 255, 92 / 255)
        else:
            return (255 / 255, 255 / 255, 255 / 255)

    def edge_color(self, vars: frozenset[str]) -> tuple[float, float, float]:
        # all_areas: list[str] = get_all_labels_from_tag(self.map.labels, "AREA")
        # for var in vars:
        #     if var in self.title.lower() and var not in all_areas:
        #         return (1.0, 0.2, 0.2)
        return NEUTRAL_COLOR

    def initialize_color_map(self) -> list[list[tuple[float, float, float]]]:
        self.color_map: list[list[tuple[float, float, float]]] = [
            [self.cell_color(self.map.content[i][j]) for j in range(self.map.size[1])]
            for i in range(self.map.size[0])
        ]
        self.color_map[self.state[0]][self.state[1]] = (29 / 255, 249 / 255, 146 / 255)
        return self.color_map

    def initialize_edge_map(self) -> list[list[tuple[float, float, float]]]:
        self.edge_map: list[list[tuple[float, float, float]]] = [
            [self.edge_color(self.map.content[i][j]) for j in range(self.map.size[1])]
            for i in range(self.map.size[0])
        ]
        return self.edge_map

    def initialize_text_map(self, show_coords: bool = False) -> list[list[str]]:
        self.text_map: list[list[str]] = [
            [
                self.cell_text(self.map.content[i][j])
                + (f"\n({i}, {j})" if show_coords else "")
                for j in range(self.map.size[1])
            ]
            for i in range(self.map.size[0])
        ]
        self.text_map[self.state[0]][self.state[1]] = "@"
        return self.text_map

    def start_render(self, pause: float = 0.01):
        """Initializes human-mode rendering"""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.color_map: list[
            list[tuple[float, float, float]]
        ] = self.initialize_color_map()
        self.edge_map: list[
            list[tuple[float, float, float]]
        ] = self.initialize_edge_map()
        self.text_map: list[list[str]] = self.initialize_text_map()

        self.ax.set_xlim(0, self.map.size[1])
        self.ax.set_ylim(0, self.map.size[0])
        self.ax.set_aspect("equal")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        self.fig.canvas.draw()
        if len(self.title) > 0:
            plt.get_current_fig_manager().window.wm_title(self.title)

        plt.pause(pause)

        # self.fig.canvas.mpl_connect('key_press_event', lambda event: plt.close(
        #     self.fig) if event.key == 'escape' else None)

    def render_human(self, show_coords: bool = False):
        self.text_map = self.initialize_text_map(show_coords=show_coords)
        self.color_map = self.initialize_color_map()
        self.edge_map = self.initialize_edge_map()

        if self.redraw:
            self.rects = [
                [
                    patches.Rectangle(
                        [j, i],
                        1,
                        1,
                        facecolor=self.color_map[i][j],
                        edgecolor=self.edge_map[i][j],
                        zorder=0,
                    )
                    for j in range(self.map.size[1])
                ]
                for i in range(self.map.size[0])
            ]
            self.texts = [
                [
                    self.ax.text(
                        j + 0.5,
                        i + 0.5,
                        self.text_map[i][j],
                        ha="center",
                        va="center",
                        zorder=1,
                        size=10,
                    )
                    for j in range(self.map.size[1])
                ]
                for i in range(self.map.size[0])
            ]
            for i in range(self.map.size[0]):
                for j in range(self.map.size[1]):
                    self.ax.add_patch(self.rects[i][j])
            self.redraw = False

        for i in range(self.map.size[0]):
            for j in range(self.map.size[1]):
                self.rects[i][j].set_facecolor(self.color_map[i][j])
                self.rects[i][j].set_edgecolor(self.edge_map[i][j])
                if self.edge_map[i][j] != NEUTRAL_COLOR:
                    self.rects[i][j].set_linewidth(3)
                self.texts[i][j].set_text(self.text_map[i][j])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

    def flash_agent(self, flash_color=(1, 0.5, 1)):
        original_color = self.rects[self.state[0]][self.state[1]].get_facecolor()

        for _ in range(3):
            self.rects[self.state[0]][self.state[1]].set_facecolor(flash_color)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.2)

            self.rects[self.state[0]][self.state[1]].set_facecolor(original_color)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.2)

    def flash_screen(self, flash_color=(1, 1, 1)):
        original_colors = [
            [self.rects[i][j].get_facecolor() for j in range(self.map.size[1])]
            for i in range(self.map.size[0])
        ]
        original_text_colors = [
            [self.texts[i][j].get_color() for j in range(self.map.size[1])]
            for i in range(self.map.size[0])
        ]

        for i in range(self.map.size[0]):
            for j in range(self.map.size[1]):
                self.rects[i][j].set_facecolor(flash_color)
                self.texts[i][j].set_color(flash_color)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.2)

        for i in range(self.map.size[0]):
            for j in range(self.map.size[1]):
                self.rects[i][j].set_facecolor(original_colors[i][j])
                self.texts[i][j].set_color(original_text_colors[i][j])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

    def display_message(self, msg: str):
        self.ax.clear()
        self.ax.text(
            0.5,
            0.5,
            msg,
            horizontalalignment="center",
            fontsize=20,
            color="black",
            verticalalignment="center",
            transform=self.ax.transAxes,
        )
        self.fig.canvas.draw()
        plt.pause(0.1)
        self.redraw = True

    def render_text(self):
        pretty_rows: list[list[str]] = [
            self.pretty_row(row) for row in self.map.content
        ]
        pretty_rows[self.state[0]][self.state[1]] = "@"
        print(" ".join(["."] * (self.map.size + 2)))
        for row in pretty_rows:
            print(" ".join(row))
        print(" ".join(["."] * (self.map.size + 2)))

    def render(self, mode="human", show_coords: bool = False):
        if mode == "human":
            self.render_human(show_coords=show_coords)
        elif mode == "text":
            self.render_text()

    def display_once(self):
        self.start_render()
        self.render(mode="human")

    def close_display(self):
        plt.close()

    def stop_render(self):
        self.close_display()
