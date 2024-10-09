import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from tcdmarl.tcrl.reward_machines.rm_common import CausalDFA
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
        use_prm: bool,
        tlcd: Optional[CausalDFA],
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
        self.use_prm = use_prm
        self.tlcd = tlcd

        self.min_steps = min_steps
        self.total_steps = total_steps

        # Keep track of the number of learning/testing steps taken
        self.current_step = 0
        self.training_stuck_counter = 0
        self.early_terminations: int = 0

        # Store the results here
        self.results: Dict[Any, Any] = {}
        self.steps: List[Any] = []
        self.agent_list = []

    # Methods to keep track of trainint/testing progress
    def restart(self):
        self.current_step = 0
        self.training_stuck_counter = 0

    def add_step(self):
        self.current_step += 1

    def record_early_termintation(self):
        self.early_terminations += 1

    def get_current_step(self):
        return self.current_step

    def add_training_stuck_step(self):
        self.training_stuck_counter += 1

    def get_training_stuck_counter(self):
        return self.training_stuck_counter

    def stop_learning(self):
        return self.total_steps <= self.current_step

    def stop_task(self, step: int) -> bool:
        return self.min_steps <= step

    def current_epsilon(self) -> float:
        """
        Calculates the current epsilon value using exponential decay based on the proportion of total steps completed.
        This method adjusts epsilon appropriately for different task configurations and total steps.

        Returns
        -------
        float
            The current epsilon value for the epsilon-greedy policy.
        """
        initial_epsilon = (
            self.learning_params.initial_epsilon
        )  # Starting value of epsilon, e.g., 1.0
        min_epsilon = (
            self.learning_params.final_epsilon
        )  # Minimum value of epsilon, e.g., 0.1
        decay_rate = -math.log(min_epsilon / initial_epsilon) / self.total_steps

        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * math.exp(
            -decay_rate * self.current_step
        )
        return epsilon
