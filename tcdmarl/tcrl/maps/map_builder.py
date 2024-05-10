import IPython
from pathlib import Path
from typing import Iterable

import numpy as np

from consts import DEFAULT_LABELS_PATH, DEFAULT_MAP_PATH
from maps.maps_common import *


class MapBuilder:
    def __init__(self, size: tuple[int, int], labels: dict[str, list[str]]):
        self.size: tuple[int, int] = size
        self.labels = labels
        self.content: list[list[set[str]]] = [
            [set() for j in range(size[1])] for i in range(size[0])
        ]
        self.sinks: set[str] = set()
        self.spawn_type: str = "random"
        self.spawn_location: tuple[int, int] = (0, 0)
        self.spawn_locations: list[tuple[int, int]] = []

    @staticmethod
    def from_map(map: Map) -> "MapBuilder":
        result: MapBuilder = MapBuilder(map.size, map.labels)
        result.content = [[set(vars) for vars in row] for row in map.content]
        result.spawn_type = map.spawn_type
        result.spawn_location = map.spawn_location
        result.spawn_locations = map.spawn_locations
        return result

    def add(self, var: str, i: int, j: int):
        self.content[i][j].add(var)

    def make_sink(self, var: str):
        self.sinks.add(var)

    # def add_top_left(self, var: str):
    #     self.add(var, self.size - 1, 0)

    # def add_top_right(self, var: str):
    #     self.add(var, self.size - 1, self.size - 1)

    def add_bottom_left(self, var: str):
        self.add(var, 0, 0)

    # def add_bottom_right(self, var: str):
    #     self.add(var, 0, self.size - 1)

    def remove(self, var: str, i: int, j: int):
        self.content[i][j].remove(var)

    def add_to_spawn_locations(self, loc: tuple[int, int]):
        if loc not in self.spawn_locations:
            self.spawn_locations.append(loc)

    def clear_nonarea(self, i: int, j: int):
        all_areas = get_all_labels_from_tag(self.labels, "AREA")
        to_remove: set[str] = set()
        for var in self.content[i][j]:
            if var not in all_areas:
                to_remove.add(var)
        for var in to_remove:
            self.remove(var, i, j)

    def add_area(self, var: str, ij: tuple[int, int], hw: tuple[int, int]):
        i, j = ij
        for h in range(hw[0]):
            for w in range(hw[1]):
                self.add(var, i + h, j + w)

    def remove_area(self, var: str, ij: tuple[int, int], hw: tuple[int, int]):
        i, j = ij
        for h in range(hw[0]):
            for w in range(hw[1]):
                self.remove(var, i + h, j + w)

    def appears(self) -> frozenset[str]:
        result: set[str] = set()
        for row in self.content:
            for vars in row:
                result.update(vars)
        return frozenset(result)

    def fill_with_vars_that_dont_appear(self, vars: frozenset[str], times: int = 1):
        appears: frozenset[str] = self.appears()
        all_objects: list[str] = get_all_labels_from_tag(self.labels, "OBJECT")
        all_places: list[str] = get_all_labels_from_tag(self.labels, "PLACE")
        to_add: frozenset[str] = frozenset(
            [
                var
                for var in vars
                if var not in appears and (var in all_places or var in all_objects)
            ]
        )
        for var in to_add:
            self.add_to_random_location(var)
        return self

    def free_to_add_at(self, i: int, j: int) -> bool:
        return "wall" not in self.content[i][j]

    def get_random_free_coordinates(self) -> tuple[int, int]:
        i: int = np.random.randint(0, self.size[0])
        j: int = np.random.randint(0, self.size[1])
        while not self.free_to_add_at(i, j):
            i = np.random.randint(0, self.size[0])
            j = np.random.randint(0, self.size[1])
        return i, j

    def add_to_random_location(self, var: str):
        """Mutates the original map."""
        i: int
        j: int
        i, j = self.get_random_free_coordinates()
        self.content[i][j].add(var)

    def add_wall(self, start: tuple[int, int], delta: tuple[int, int]):
        length: int = abs(delta[0]) + abs(delta[1])
        assert length > 0
        assert delta[0] * delta[1] == 0

        horizontal: bool = delta[0] == 0
        forward: bool = delta[0] > 0 or delta[1] > 0

        for k in range(length):
            i, j = start
            if horizontal:
                if forward:
                    j += k
                else:
                    j -= k
            else:
                if forward:
                    i += k
                else:
                    i -= k
            self.clear_nonarea(i, j)
            self.add("wall", i, j)

    def build(self) -> Map:
        sink_states: set[tuple[int, int]] = set()
        for sink in self.sinks:
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    if sink in self.content[i][j]:
                        sink_states.add((i, j))

        return Map(
            [[frozenset(vars) for vars in row] for row in self.content],
            self.size,
            self.labels,
            frozenset(sink_states),
            self.spawn_type,
            self.spawn_location,
            self.spawn_locations,
        )


class MapConfig:
    def __init__(
        self,
        size: int,
        p_object: float,
        p_color_object: float,
        p_place: float,
        p_color: float,
    ):
        self.size: int = size
        self.p_object: float = p_object
        self.p_color_object: float = p_color_object
        self.p_place: float = p_place
        self.p_color: float = p_color


class VarPicker:
    def __init__(self, pick_from: list[str]):
        self.pick_from = pick_from
        np.random.shuffle(self.pick_from)
        self.picker_i: int = 0

    def pick(self) -> str:
        chosen: str = self.pick_from[self.picker_i]
        self.increment()
        return chosen

    def increment(self):
        self.picker_i += 1
        if self.picker_i == len(self.pick_from):
            self.picker_i = 0
            np.random.shuffle(self.pick_from)

    def remove(self, var: str):
        print(var, self.pick_from)
        self.pick_from.remove(var)

    def remove_all(self, vars: Iterable[str]):
        for var in vars:
            self.remove(var)


def example_map_1(config: MapConfig) -> Map:
    terms: dict[str, list[str]] = load_labels(DEFAULT_LABELS_PATH)

    map: MapBuilder = MapBuilder(config.size, terms)

    map.add_area("forest", (0, 0), (int(config.size / 2), int(config.size / 2)))
    map.add_area(
        "field", (0, int(config.size / 2)), (int(config.size / 2), int(config.size / 2))
    )
    map.add_area(
        "town", (int(config.size / 2), 0), (int(config.size / 2), int(config.size / 2))
    )
    map.add_area(
        "factory",
        (int(config.size / 2), int(config.size / 2)),
        (int(config.size / 2), int(config.size / 2)),
    )

    all_objects: list[str] = get_all_labels_from_tag(terms, "OBJECT")
    all_areas: list[str] = get_all_labels_from_tag(terms, "AREA")
    all_places: list[str] = get_all_labels_from_tag(terms, "PLACE")
    all_colors: list[str] = get_all_labels_from_tag(terms, "COLOR")

    object_picker: VarPicker = VarPicker(all_objects)
    area_picker: VarPicker = VarPicker(all_areas)
    place_picker: VarPicker = VarPicker(all_places)
    color_picker: VarPicker = VarPicker(all_colors)

    place_picker.remove_all(area_picker.pick_from)
    place_picker.remove("wall")

    for i in range(config.size):
        for j in range(config.size):
            if np.random.random() < config.p_object:
                to_add: str = object_picker.pick()
                map.add(to_add, i, j)
                if np.random.random() < config.p_color_object:
                    to_add: str = color_picker.pick()
                    map.add(to_add, i, j)
            if np.random.random() < config.p_place:
                to_add: str = place_picker.pick()
                map.add(to_add, i, j)
            if np.random.random() < config.p_color:
                to_add: str = color_picker.pick()
                map.add(to_add, i, j)

    map.add_wall(
        (int(config.size / 2), int(config.size / 4)), (0, int(config.size / 2) + 1)
    )
    map.add_wall(
        (int(config.size / 4), int(config.size / 2)), (int(config.size / 2) + 1, 0)
    )

    return map.build()


def save_map(path: str | Path, map: Map):
    with open(path, "w") as f:
        f.write(map.to_json())


def map_builder(path: str):
    print(path)
    config: MapConfig = MapConfig(
        size=6,
        p_object=0.3,
        p_color_object=0.2,
        p_place=0.3,
        p_color=0.01,
    )
    map: Map = example_map_1(config)
    save_map(path, map)


def main():
    map_builder(DEFAULT_MAP_PATH)


if __name__ == "__main__":
    main()
