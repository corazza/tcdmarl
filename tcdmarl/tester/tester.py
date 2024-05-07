from pathlib import Path
from typing import Any, Dict, List

from tcdmarl.tester.learning_params import LearningParameters
from tcdmarl.tester.tester_params import TestingParameters


class Tester:
    def __init__(
        self,
        learning_params: LearningParameters,
        testing_params: TestingParameters,
        num_agents: int,
        min_steps: int,
        total_steps: int,
        step_unit: int,
        num_times: int,
        rm_test_file: Path,
        rm_learning_file_list: List[Path],
        env_settings: Dict[str, Any],
        experiment: str,
    ):
        """
        Parameters
        ----------
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        testing_params : TestingParameters object
            Object storing parameters to be used in testing.
        min_steps : int
        total_steps : int
            Total steps allowed before stopping learning.
        """
        self.learning_params = learning_params
        self.testing_params = testing_params
        self.num_agents = num_agents
        self.step_unit = step_unit
        self.num_times = num_times
        self.rm_test_file = rm_test_file
        self.rm_learning_file_list = rm_learning_file_list
        self.env_settings = env_settings
        self.experiment = experiment

        self.min_steps = min_steps
        self.total_steps = total_steps

        # Keep track of the number of learning/testing steps taken
        self.current_step = 0

        # Store the results here
        self.results = {}
        self.steps = []

    # Methods to keep track of trainint/testing progress
    def restart(self):
        self.current_step = 0

    def add_step(self):
        self.current_step += 1

    def get_current_step(self):
        return self.current_step

    def stop_learning(self):
        return self.total_steps <= self.current_step

    def stop_task(self, step):
        return self.min_steps <= step
