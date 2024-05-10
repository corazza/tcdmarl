from maps.map_builder import MapBuilder
from maps.maps_common import Map, load_labels


def map_paper_coffe_drink_office() -> Map:
    size: tuple[int, int] = (5, 5)
    labels: dict[str, list[str]] = load_labels(
        "datasets/rms/coffee_drink_office_flowers_labels.txt"
    )

    builder: MapBuilder = MapBuilder(size=size, labels=labels)
    builder.spawn_type = "multiple_locations"
    builder.add_to_spawn_locations((0, 2))

    builder.add("office", 4, 4)

    builder.add("wall", 1, 1)
    builder.add("wall", 1, 2)
    builder.add("wall", 2, 2)
    builder.add("wall", 3, 2)
    builder.add("drink", 2, 0)
    builder.add("coffee", 2, 4)
    builder.add("flowers", 4, 2)
    builder.add("force_up", 1, 0)

    builder.make_sink("flowers")
    return builder.build()


def map_paper_four_rooms() -> Map:
    size: tuple[int, int] = (7, 7)
    labels: dict[str, list[str]] = load_labels(
        "datasets/rms/coffee_drink_office_flowers_labels.txt"
    )

    builder: MapBuilder = MapBuilder(size=size, labels=labels)
    builder.spawn_type = "multiple_locations"
    builder.add_to_spawn_locations((3, 3))

    builder.add("a", 0, 0)
    builder.add("b", 0, 6)
    builder.add("c", 6, 6)
    builder.add("d", 6, 0)

    builder.make_sink("a")
    return builder.build()


################# OLD


def flowers_sink() -> Map:
    size: tuple[int, int] = (7, 17)
    jdelta: int = 5
    labels: dict[str, list[str]] = load_labels(
        "datasets/rms/coffee_drink_office_flowers_labels.txt"
    )

    builder: MapBuilder = MapBuilder(size=size, labels=labels)
    builder.spawn_type = "multiple_locations"
    for i in range(7):
        builder.add_to_spawn_locations((i, 3 + jdelta))

    builder.add("office", 1, 3 + jdelta)

    builder.add_wall((5, 2 + jdelta), (-5, 0))
    builder.add("drink", 0, 0)
    builder.add("force_left", 0, 2 + jdelta)
    # builder.add("force_right", 6, 2)
    builder.add("flowers", 6, 2 + jdelta)

    builder.add_wall((5, 4 + jdelta), (-5, 0))
    builder.add("coffee", 2, 16)
    # builder.add("force_right", 0, 4)
    # builder.add("force_left", 6, 4)

    builder.add("a", 6, 0 + jdelta)
    # builder.add("b", 1, 0)
    builder.add("flowers", 1, 0 + jdelta)

    builder.add("a", 6, 6 + jdelta)
    # builder.add("b", 1, 6)
    builder.add("flowers", 1, 6 + jdelta)

    builder.make_sink("flowers")

    return builder.build()


def map_coffee_drink_office_2() -> Map:
    size: tuple[int, int] = (7, 19)
    labels: dict[str, list[str]] = load_labels(
        "datasets/rms/coffee_drink_office_flowers_labels.txt"
    )

    builder: MapBuilder = MapBuilder(size=size, labels=labels)
    builder.spawn_type = "multiple_locations"
    for i in range(7):
        builder.add_to_spawn_locations((i, 9))

    builder.add("office", 1, 9)

    builder.add_wall((5, 8), (-5, 0))
    builder.add("drink", 0, 8)
    builder.add("force_left", 0, 8)
    builder.add("flowers", 6, 8)

    builder.add_wall((5, 10), (-5, 0))
    builder.add("coffee", 0, 10)
    builder.add("force_right", 0, 10)

    builder.add("a", 6, 0)
    builder.add("b", 0, 0)

    builder.add("a", 6, 18)
    builder.add("b", 0, 18)

    return builder.build()


def map_coffee_drink_office() -> Map:
    size: tuple[int, int] = (15, 10)
    labels: dict[str, list[str]] = load_labels(
        "datasets/rms/coffee_drink_office_flowers_labels.txt"
    )

    builder: MapBuilder = MapBuilder(size=size, labels=labels)
    builder.spawn_type = "multiple_locations"
    builder.add_to_spawn_locations((4, 4))
    builder.add_to_spawn_locations((4, 5))

    # for i in [6, 7]:
    #     for j in range(8):
    #         builder.add_to_spawn_locations((i, j))

    # for i in range(5):
    #     for j in [3, 4]:
    #         builder.add_to_spawn_locations((i, j))

    builder.add("office", 7, 4)
    builder.add("office", 7, 5)
    # builder.add("base", 7, 7)

    # builder.add("office", 4, 0)
    # builder.add("coffee", 0, 3)
    # builder.add("coffee", 9, 5)

    # builder.add("drink", 9, 5)
    # builder.add("coffee", 0, 3)
    # builder.add("base", 6, 7)
    # builder.add("flowers", 5, 2)

    builder.add("drink", 0, 1)
    builder.add("drink", 2, 1)
    builder.add("drink", 4, 1)
    builder.add("force_left", 2, 3)
    builder.add("force_left", 1, 3)
    builder.add("force_left", 0, 3)
    builder.add_wall((13, 3), (-11, 0))
    builder.add("flowers", 14, 3)

    builder.add("coffee", 0, 8)
    builder.add("coffee", 2, 8)
    builder.add("coffee", 4, 8)
    builder.add("force_right", 2, 6)
    builder.add("force_right", 1, 6)
    builder.add("force_right", 0, 6)
    builder.add_wall((13, 6), (-11, 0))
    # builder.add_wall((4, 4), (-2, 0))
    # builder.add_wall((1, 4), (-2, 0))

    return builder.build()


def map_coffee_drink_office_small() -> Map:
    size: tuple[int, int] = (7, 17)
    jdelta: int = 5
    labels: dict[str, list[str]] = load_labels(
        "datasets/rms/coffee_drink_office_flowers_labels.txt"
    )

    builder: MapBuilder = MapBuilder(size=size, labels=labels)
    builder.spawn_type = "multiple_locations"
    for i in range(7):
        builder.add_to_spawn_locations((i, 3 + jdelta))

    builder.add("office", 1, 3 + jdelta)

    builder.add_wall((5, 2 + jdelta), (-5, 0))
    builder.add("drink", 0, 0)
    builder.add("force_left", 0, 2 + jdelta)
    # builder.add("force_right", 6, 2)
    builder.add("flowers", 6, 2 + jdelta)

    builder.add_wall((5, 4 + jdelta), (-5, 0))
    builder.add("coffee", 2, 16)
    # builder.add("force_right", 0, 4)
    # builder.add("force_left", 6, 4)

    builder.add("a", 6, 0 + jdelta)
    # builder.add("b", 1, 0)
    builder.add("flowers", 1, 0 + jdelta)

    builder.add("a", 6, 6 + jdelta)
    # builder.add("b", 1, 6)
    builder.add("flowers", 1, 6 + jdelta)

    return builder.build()


def map_many_flowers() -> Map:
    size: tuple[int, int] = (7, 7)
    jdelta: int = 0
    labels: dict[str, list[str]] = load_labels(
        "datasets/rms/coffee_drink_office_flowers_labels.txt"
    )

    builder: MapBuilder = MapBuilder(size=size, labels=labels)
    builder.spawn_type = "multiple_locations"
    for i in range(7):
        builder.add_to_spawn_locations((i, 3 + jdelta))

    builder.add("office", 1, 3 + jdelta)

    builder.add_wall((5, 2 + jdelta), (-5, 0))
    builder.add("drink", 0, 0)
    builder.add("force_left", 0, 2 + jdelta)
    builder.add("flowers", 6, 2 + jdelta)

    builder.add_wall((5, 4 + jdelta), (-5, 0))
    builder.add("coffee", 2, 6)

    builder.make_sink("flowers")
    return builder.build()
