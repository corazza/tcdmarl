"""
Testing parameters.
"""


class TestingParameters:
    """
    Testing parameters.
    """

    def __init__(self, test: bool = True, test_freq: int = 1000, num_steps: int = 100):
        """Parameters
        -------
        test: bool
            if True, we test current policy during training
        test_freq: int
            test the model every `test_freq` steps.
        num_steps: int
            number of steps during testing
        """
        self.test = test
        self.test_freq = test_freq
        self.num_steps = num_steps
