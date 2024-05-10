import json
import more_itertools
import numpy as np

from parsing.parser_util import line_iter
from parsing.terms_parser import parse_terms


def load_labels(path: str) -> dict[str, list[str]]:
    """Loads values of the labeling function from a file."""
    lines = more_itertools.peekable(line_iter(path))
    return parse_terms(lines)


def get_all_labels_from_tag(labels: dict[str, list[str]], label_tag: str) -> list[str]:
    applicable: list[str] = []
    for term, tags in labels.items():
        if label_tag in tags:
            applicable.append(term)
    # assert len(applicable) > 0
    return applicable


class FrozensetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, frozenset):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class Map:
    def __init__(
        self,
        content: list[list[frozenset[str]]],
        size: tuple[int, int],
        labels: dict[str, list[str]],
        sinks: frozenset[tuple[int, int]],
        spawn_type: str,
        spawn_location: tuple[int, int],
        spawn_locations: list[tuple[int, int]],
    ):
        assert spawn_type in ["random", "location", "multiple_locations"]
        self.content: list[list[frozenset[str]]] = content
        self.labels: dict[str, list[str]] = labels
        self.sinks: frozenset[tuple[int, int]] = sinks
        self.size: tuple[int, int] = size
        assert self.size[0] == len(self.content)
        assert self.size[1] == len(self.content[0])
        self.spawn_type: str = spawn_type
        self.spawn_location: tuple[int, int] = spawn_location
        self.spawn_locations: list[tuple[int, int]] = spawn_locations

    def to_json(self):
        return json.dumps(self.__dict__, cls=FrozensetEncoder)

    def get_new_spawn(self) -> tuple[int, int]:
        if self.spawn_type == "random":
            new_loc: tuple[int, int] = (
                np.random.randint(self.size[0]),
                np.random.randint(self.size[1]),
            )
            while "wall" in self.content[new_loc[0]][new_loc[1]]:
                new_loc = (
                    np.random.randint(self.size[0]),
                    np.random.randint(self.size[1]),
                )
            # return (6, 1)
            return new_loc

        elif self.spawn_type == "location":
            return self.spawn_location
        elif self.spawn_type == "multiple_locations":
            i: int = np.random.choice(len(self.spawn_locations))
            return self.spawn_locations[i]
        else:
            raise ValueError(f"unknown spawn type '{self.spawn_type}'")

    @staticmethod
    def from_json(code: str) -> "Map":
        json_dict = json.loads(code)
        return Map(**json_dict)

    @staticmethod
    def from_path(path: str) -> "Map":
        with open(path, "r") as f:
            code = f.read()
            return Map.from_json(code)
