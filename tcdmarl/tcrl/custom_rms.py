from reward_machines.rm_builder import ProbabilisticRMBuilder


def rm_paper_coffe_drink_office():
    builder = ProbabilisticRMBuilder({"coffee", "drink", "office"})
    builder.t(0, "!coffee & !drink", 0, prob=1, output=0)

    builder.t(0, "coffee & !drink", 1, prob=0.9, output=0)
    builder.t(0, "coffee & !drink", 2, prob=0.1, output=0)
    builder.t(0, "drink", 3, prob=1, output=0)

    builder.t(1, "office", 4, prob=1, output=1)
    builder.t(1, "!office", 1, prob=1, output=0)

    builder.t(2, "office", 4, prob=1, output=0.1)
    builder.t(2, "!office", 2, prob=1, output=0)

    builder.t(3, "office", 4, prob=1, output=1)
    builder.t(3, "!office", 3, prob=1, output=0)
    builder.terminal(4)
    return builder.build()


def rm_paper_a_and_b():
    builder = ProbabilisticRMBuilder({"a", "b"})
    builder.terminal(3)

    builder.t(0, "!a & !b", 0, prob=1, output=0)
    builder.t(0, "a | b", 0, prob=0.1, output=0)

    builder.t(0, "a & !b", 1, prob=0.9, output=0)
    builder.t(0, "b", 2, prob=0.9, output=0)

    builder.t(1, "b", 3, prob=0.9, output=1)
    builder.t(1, "b", 1, prob=0.1, output=0)
    builder.t(1, "!b", 1, prob=1, output=0)

    builder.t(2, "a", 3, prob=0.9, output=1)
    builder.t(2, "a", 2, prob=0.1, output=1)
    builder.t(2, "!a", 2, prob=1, output=0)

    return builder.build()


################# OLD


def rm_coffe_drink_flowers_office():
    builder = ProbabilisticRMBuilder({"coffee", "drink", "flowers", "office"})
    builder.terminal(5)
    builder.terminal(4)

    builder.t(0, "!coffee & !flowers & !drink", 0, prob=1, output=0)

    builder.t(0, "coffee & !flowers & !drink", 1, prob=0.9, output=0)
    builder.t(0, "coffee & !flowers & !drink", 2, prob=0.1, output=0)
    builder.t(0, "drink & !flowers", 3, prob=1, output=0)

    builder.t(1, "office & !flowers", 4, prob=1, output=1)
    builder.t(1, "!office & !flowers", 1, prob=1, output=0)

    builder.t(2, "office & !flowers", 4, prob=1, output=0.1)
    builder.t(2, "!office & !flowers", 2, prob=1, output=0)

    builder.t(3, "office & !flowers", 4, prob=1, output=1)
    # builder.t(3, "a & !flowers", 4, prob=1, output=0.5)
    builder.t(3, "!office & !flowers", 3, prob=1, output=0)

    for i in [0, 1, 2, 3]:
        builder.t(i, "flowers", 5, prob=1, output=0)
        # if i != 0:  # TODO do this better in the builder
        #     builder.t(i, "!a & !flowers & !office", i, prob=1, output=0)

    # builder.t(5, ".", 5, 1, 0)

    return builder.build()


def rm_icarte(success_prob: float, prod_output: float):
    builder = ProbabilisticRMBuilder({"coffee", "flowers", "office"})
    builder.terminal(2)

    builder.t(0, "!coffee & !flowers", 0, prob=1, output=0)
    builder.t(0, "coffee & !flowers", 1, prob=1, output=0)

    builder.t(1, "!office & !flowers", 1, prob=1, output=0)
    builder.t(1, "office & !flowers", 2, prob=1, output=1)

    for i in [0, 1]:
        builder.t(i, "flowers", 3, prob=1, output=0)

    if prod_output == -1:
        builder.t(3, ".", 3, prob=1, output=0)
        print("here 1")
    else:
        builder.terminal(3)
        print("here 2")
    return builder.build()


def no_causal_flowers():
    builder = ProbabilisticRMBuilder({"coffee", "flowers", "office"})
    builder.terminal(3)

    builder.t(0, "!coffee", 0, prob=1, output=0)

    builder.t(0, "coffee", 1, prob=0.9, output=0)
    builder.t(0, "coffee", 2, prob=0.1, output=0)

    builder.t(1, "office", 3, prob=1, output=1)
    builder.t(1, "!office", 1, prob=1, output=0)

    builder.t(2, "office", 3, prob=1, output=0)
    builder.t(2, "!office", 2, prob=1, output=0)

    return builder.build()


def rm_coffe_drink_flowers_office_ab(success_prob: float, prod_output: float):
    builder = ProbabilisticRMBuilder({"coffee", "drink", "flowers", "office", "a", "b"})

    builder.t(0, "!coffee & !flowers & !drink", 0, prob=1, output=0)

    builder.t(0, "coffee & !flowers & !drink", 1, prob=success_prob, output=0)
    builder.t(0, "coffee & !flowers & !drink", 2, prob=1 - success_prob, output=0)
    builder.t(0, "drink & !flowers", 3, prob=1, output=0)

    builder.t(1, "a & !flowers", 4, prob=1, output=0)
    builder.t(2, "a & !flowers", 5, prob=1, output=0)
    builder.t(3, "a & !flowers", 6, prob=1, output=0)

    for i in [1, 2, 3]:
        builder.t(i, "!a & !flowers", i, prob=1, output=0)

    builder.t(4, "b & !flowers", 7, prob=1, output=0)
    builder.t(5, "b & !flowers", 8, prob=1, output=0)
    builder.t(6, "b & !flowers", 9, prob=1, output=0)

    for i in [4, 5, 6]:
        builder.t(i, "!b & !flowers", i, prob=1, output=0)

    builder.t(7, "office & !flowers", 10, prob=1, output=1)
    builder.t(8, "office & !flowers", 10, prob=1, output=0)
    builder.t(9, "office & !flowers", 10, prob=1, output=prod_output)

    for i in [7, 8, 9]:
        builder.t(i, "!office & !flowers", i, prob=1, output=0)

    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        builder.t(i, "flowers", 11, prob=1, output=0)
        # if i != 0:  # TODO do this better in the builder
        #     builder.t(i, "!flowers & !office", i, prob=1, output=0)

    builder.terminal(10)
    builder.terminal(11)
    return builder.build()


def rm_coffe_drink_flowers_office_base(success_prob: float):
    builder = ProbabilisticRMBuilder({"coffee", "drink", "flowers", "office", "base"})

    builder.t(0, "!coffee & !flowers & !drink", 0, prob=1, output=0)

    builder.t(0, "coffee & !flowers & !drink", 1, prob=success_prob, output=0)
    builder.t(0, "coffee & !flowers & !drink", 2, prob=1 - success_prob, output=0)
    builder.t(0, "drink & !flowers", 3, prob=1, output=0)

    builder.t(1, "office & !flowers", 4, prob=1, output=0)
    builder.t(2, "office & !flowers", 5, prob=1, output=0)
    builder.t(3, "office & !flowers", 6, prob=1, output=0)

    builder.t(4, "base & !flowers", 7, prob=1, output=1)
    builder.t(5, "base & !flowers", 7, prob=1, output=0)
    builder.t(6, "base & !flowers", 7, prob=1, output=1)

    for i in [0, 1, 2, 3, 4, 5, 6]:
        builder.t(i, "flowers", 8, prob=1, output=0)
        builder.t(i, "!flowers", i, prob=1, output=0)

    builder.terminal(7)
    builder.terminal(8)
    return builder.build()
