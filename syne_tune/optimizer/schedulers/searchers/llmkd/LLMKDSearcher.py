from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
import logging
from collections import defaultdict
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd

from syne_tune.config_space import Domain, FiniteRange, Categorical, Float, Integer

from syne_tune.optimizer.schedulers.searchers.conformal.surrogate.quantile_regression_surrogate import (
    QuantileRegressionSurrogateModel,
)
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges
from syne_tune.util import catchtime
from mooLLM.mooLLM_builder import mooLLMBuilder
from mooLLM.benchmarks.zdt.zdt import ZDT
from mooLLM.benchmarks.benchmark import BENCHMARK

from mooLLM.utils.logger import LOGGING_CONFIG

# logger = logging.getLogger(__name__)
# logging.getLogger().setLevel(logging.INFO)
logging.config.dictConfig(config=LOGGING_CONFIG)
logger = logging.getLogger("mooLLM")



class SyneTuneBenchmark(BENCHMARK):
    """
    This class is a wrapper around the mooLLM benchmark interface.
    It is used to adapt the configuration space and evaluation process
    to the requirements of the mooLLM package.

    This wrapper enables the mooLLM packages to use its internal pipeline 
    with out needing to change the internal pipeline of mooLLM.
    """
    def __init__(self, config_space: Dict, seed: int):
        super().__init__() 
        self.config_space = config_space 
        self.metrics = []
        self.seed = seed
        self.hp_ranges = make_hyperparameter_ranges(config_space=config_space)
        self.random_state = np.random.RandomState(self.seed)
        self.benchmark_name = "SyneTune"
        self.model_name = "gemini-1-5-flash"
        self.range_parameter_keys = []
        
        print(f"self.config_space: {self.config_space}")
        print(f"self.hp_ranges: {self.hp_ranges}")

    def evaluate_point(self, point, **kwargs) -> Dict:
        pass

    def generate_initialization(self, n_points: int, **kwargs) -> List[Dict]:
        random_samples = [ self._sample_random() for _ in range(n_points)]
        print(f"random_samples: {random_samples}")
        return random_samples

    def _sample_random(self, **kwargs) -> Dict:
        random_samples = {
            k: v.sample(random_state=self.random_state) if isinstance(v, Domain) else v
            for k, v in self.config_space.items()
        }
        print(f"random_samples: {random_samples}")
        return random_samples

    def get_few_shot_samples(self, **kwargs) -> List[Tuple[Dict, Dict]]:
        return [({}, {})]

    def get_metrics_ranges(self, **kwargs) -> Dict[str, List[float]]:
        return []

    def is_valid_candidate(self, candidate) -> bool:
        """
        Check if a candidate point is valid according to the configuration space.
        
        :param candidate: Dictionary containing the hyperparameter values
        :return: True if the point is valid, False otherwise
        """
        print(f"Candidate {candidate}")
        print(f"search space {self.config_space}")

        for param_key, param_val in candidate.items():
            constraint = self.config_space[param_key]
            is_valid = constraint.is_valid(param_val)
            if not is_valid:
                return False
        return True

    def is_valid_evaluation(self, evaluation) -> bool:
        """
        Check if an evaluation result is valid.
        
        :param evaluation: Dictionary containing the evaluation metrics
        :return: True if the evaluation is valid, False otherwise
        """

        print(f"evaluation {evaluation}")
        # Assuming evaluation should contain all metrics defined for the benchmark
        # if not isinstance(evaluation, dict):
        #     return False
            
        # for metric in self.metrics:
        #     if metric not in evaluation:
        #         return False
        #     if not isinstance(evaluation[metric], (int, float)):
        #         return False
                
        return True

    def save_progress(self, results: List[Dict], **kwargs):
        """
        We dont save the results as syn_tune does this for us.
        """
        pass



class LLMKDSearcher(SingleObjectiveBaseSearcher):
    def __init__(
        self,
        config_space: Dict,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[Dict]] = None,
        num_init_random_draws: int = 5,
        update_frequency: int = 1,
        max_fit_samples: int = None,
        metric_targets: list = None,
        **surrogate_kwargs,
    ):
        """
        :param config_space: Configuration space for the evaluation function.
        :param random_seed: Seed for initializing random number generators.
        :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
        :param num_init_random_draws: sampled at random until the number of observation exceeds this parameter.
        :param update_frequency: surrogates are only updated every `update_frequency` results, can be used to save
        scheduling time.
        :param max_fit_samples: if the number of observation exceed this parameter, then `max_fit_samples` random samples
        are used to fit the model.
        :param surrogate_kwargs:
        """
        # Initialize SyneTuneBenchmark with the configuration space
        # This is the interface to the mooLLM package.
        self.benchmark = SyneTuneBenchmark(config_space=config_space, seed=random_seed)
        
        super(LLMKDSearcher, self).__init__(
            config_space=config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed
        )

        self.surrogate_kwargs = surrogate_kwargs
        self.num_init_random_draws = num_init_random_draws
        self.update_frequency = update_frequency
        self.trial_results = defaultdict(list)  # list of results for each trials
        self.trial_configs = {}
        self.hp_ranges = make_hyperparameter_ranges(config_space=config_space)
        self.surrogate_model = None
        self.index_last_result_fit = None
        self.new_candidates_sampled = False
        self.sampler = None
        self.max_fit_samples = max_fit_samples
        self.metrics = ["F1"]         

        self.benchmark.metrics =  self.metrics 

        self.random_state = np.random.RandomState(random_seed)
        
        # Convert config space to mooLLM format
        self.mooLLM_boundary_box = self._create_boundary_box(config_space)
        # if True:
        #     1 / 0
        self.response_format = self._create_response_format(config_space)
        self.config_queue = [] # If we get more than one config, we need to store them for syne tune and return them one by one
        
        range_parameter_keys, integer_parameter_keys, float_parameter_keys = self._create_range_parameter_keys(config_space)
        print(f"config space {config_space}")
        self.mooLLM_config = {
            "model": "gemini-1-5-flash",
            # "model": "gpt-4o-mini",
            "method_name": "mooLLM-KD",
            "optimization_method": "SpacePartitioning",
            "space_partitioning_settings": {
                "top_k": 4,
                "partitions_per_trial": 5,
                "use_clustering": False,
                "region_acquisition_strategy": "MOSS",
                "partitioning_strategy": "kdtree", 
                "alpha": 0.7, 
                "scheduler_settings": { # TODO: This has to be flexible at some point.
                    "scheduler": "EPSILON_DECAY_SCHEDULER",
                    "decay_rate": 0.05,
                    "initial_value": 1,
                    "min_value": 0.01,
                },
                "boundary_box": self.mooLLM_boundary_box,
            },
            "n_trials": 1,
            "total_trials": 100,
            "input_cost_per_1000_tokens": 0.000150,
            "output_cost_per_1000_tokens": 0.000600,
            "initial_samples": 0, # This is set to 0, because we are using the initial samples we already get from syne tune
            "candidates_per_request": 7, # TODO: [7, 10]
            "max_candidates_per_trial": 7, # TODO: [7, 10]
            "evaluations_per_request": 5,
            "max_evaluations_per_trial": 5,
            "max_context_configs": 110,
            "max_requests_per_minute": 5000,
            "max_tokens_per_minute": 4000000,
            "benchmark": "SyneTune",
            "benchmark_settings": {},
            "range_parameter_keys": range_parameter_keys,
            "integer_parameter_keys": integer_parameter_keys,
            "float_parameter_keys": float_parameter_keys,
            "warmstarter": "RANDOM_WARMSTARTER", # We actually do not use any warmstarting see initial samples setting
            "candidate_sampler": "mooLLM_SAMPLER",
            "acquisition_function": "FunctionValueACQ",
            "surrogate_model": "mooLLM_SUR_BATCH",
            "shuffle_icl_columns": False,
            "shuffle_icl_rows": True, 
            "use_few_shot_examples": False,
            "context_limit_strategy": "LastN",
            "warmstarting_prompt_template": "./syne_tune/optimizer/schedulers/searchers/llmkd/prompt_templates_kd/warmstarting.txt", # TODO: This will not work if the path is different
            "candidate_sampler_prompt_template": "./syne_tune/optimizer/schedulers/searchers/llmkd/prompt_templates_kd/candidate_sampler.txt", # TODO: This will not work if the path is different
            "surrogate_model_prompt_template": "./syne_tune/optimizer/schedulers/searchers/llmkd/prompt_templates_kd/surrogate_model.txt",    # TODO: This will not work if the path is different   
            "metrics": self.metrics,
            "metrics_targets": [metric_targets],
            "prompt": {
                "problem_description": "",
                "constraints": self._create_constraints(config_space),
                "metrics": self._create_metrics_description(self.metrics, [metric_targets]),
                "icl_examples_template": f"Configuration: $configuration \n{' '.join([f'{m}: ${m},' for m in self.metrics])}",
                "icl_example_template": self.response_format,
                "warmstarting_response_format": self.response_format,
                "candidate_sampler_response_format": self.response_format,
                "surrogate_model_response_format": self._create_metrics_response_format(),
            },
        }

        self.benchmark.range_parameter_keys = self.mooLLM_config.get("range_parameter_keys", [])
        mooLLM_builder = mooLLMBuilder(config=self.mooLLM_config, benchmark=self.benchmark)
        self.mooLLM = mooLLM_builder.build()

        # Fix for points_to_evaluate being empty
        if points_to_evaluate is None:
            self.points_to_evaluate = self.benchmark.generate_initialization(
                n_points=self.num_init_random_draws
            )
    
    def _create_metrics_description(self, metrics: List[str], metrics_targets: List[str]) -> str:
        """
        Create a metrics description string for mooLLM.
        
        :param metrics: List of metrics
        :return: A string describing the metrics
        """
        m_strings = []
        for metric, target in zip(metrics, metrics_targets):
            m_strings.append(f"{metric} (lower is better)")
            # if target == "min": # TODO: For now keep it like this and change it later
            # elif target == "max":
            #     m_strings.append(f"{metric} (higher is better)")
        return ", ".join([m_string for m_string in m_strings])

    def _create_range_parameter_keys(self, config_space: Dict) -> List[str]:
        """
        Create a list of range parameter keys for the configuration space.
        
        :param config_space: The configuration space
        :return: A list of range parameter keys
        """
        integer_parameter_keys = []
        float_parameter_keys = []
        for name, domain in config_space.items():
            if isinstance(domain, Domain):
                if isinstance(domain, Float):
                    float_parameter_keys.append(name)
                elif isinstance(domain, Integer):
                    integer_parameter_keys.append(name)
        range_parameter_keys = integer_parameter_keys + float_parameter_keys
        return range_parameter_keys, integer_parameter_keys, float_parameter_keys



    def suggest(self, **kwargs) -> Optional[Dict[str, Any]]:
        config = self._next_points_to_evaluate()

        if config is None:
            if self.config_queue:
                # If we have more than one config, we need to return them one by one
                config = self.config_queue.pop(0)
            else:
                # If we have no config, we need to sample a new ones
                config, statistics = self.mooLLM.optimize()
                if type(config) is list:
                    # If we get more than one config, we need to store them for syne tune and return them one by one
                    self.config_queue = config[1:]
                    config = config[0]
                print(statistics["observed_fvals"])
        
        return config
    
    def _create_boundary_box(self, config_space: Dict) -> Dict[str, List[float]]:
        """
        Create a boundary box for the configuration space suitable for mooLLM.
        
        :param config_space: The configuration space
        :return: A dictionary mapping parameter names to [min, max] ranges
        """
        boundary_box = {}
        for i, (name, domain) in enumerate(config_space.items()):
            if isinstance(domain, Domain):
                print(f"domain: {type(domain)}")
                if isinstance(domain, FiniteRange):
                    # If categorical [tanh, relu] -> [tanh, relu] or [0.03, 0.05, 0.07] -> [0.03, 0.05, 0.07] 
                    boundary_box[name] = domain.values
                elif isinstance(domain, Categorical):
                    categories = list(domain.categories)
                    boundary_box[name] = categories
                else:
                    lower, upper = domain.lower, domain.upper
                    boundary_box[name] = [float(lower), float(upper)]
                    
            else:
                # This should never happen
                # For fixed values, use a fixed range
                boundary_box[name] = [0.0, 1.0]
        
        return boundary_box
    
    def _create_response_format(self, config_space: Dict) -> str:
        """
        Create a response format template for mooLLM based on the configuration space.
        
        :param config_space: The configuration space
        :return: A JSON template string
        """
        template_parts = ['{']
        for i, name in enumerate(config_space.keys()):
            template_parts.append(f'"{name}": ${name}')
            if i < len(config_space) - 1:
                template_parts.append(', ')
        template_parts.append('}')
        
        return ''.join(template_parts)
    
    def _create_constraints(self, config_space: Dict) -> Dict:
        """
        Create constraints for mooLLM based on the configuration space.
        
        :param config_space: The configuration space
        :return: A dictionary of constraints
        """
        constraints = {}
        for name, domain in config_space.items():
            if isinstance(domain, Domain):
                try:
                    lower, upper = domain.lower, domain.upper
                    constraints[name] = [float(lower), float(upper)]
                except (AttributeError, TypeError):
                    # For categorical domains, list the categories
                    try:
                        categories = list(domain.categories)
                        constraints[name] = categories
                    except:
                        constraints[name] = "unknown"
            else:
                constraints[name] = domain
        return constraints
    
    def _create_metrics_response_format(self) -> str:
        """
        Create a metrics response format for mooLLM.
        
        :return: A JSON template string for metrics
        """
        template_parts = ['{']
        for i, metric in enumerate(self.metrics):
            template_parts.append(f'"{metric}": ?')
            if i < len(self.metrics) - 1:
                template_parts.append(', ')
        template_parts.append('}')
        
        return ''.join(template_parts)

    

    def should_update(self) -> bool:
        enough_observations = self.num_results() >= self.num_init_random_draws
        if enough_observations:
            if self.index_last_result_fit is None:
                return True
            else:
                new_results_seen_since_last_fit = (
                    self.num_results() - self.index_last_result_fit
                )
                return new_results_seen_since_last_fit >= self.update_frequency
        else:
            return False

    def num_results(self) -> int:
        return len(self.trial_results)

    def make_input_target(self):
        configs = [
            self.trial_configs[trial_id] for trial_id in self.trial_results.keys()
        ]
        X = self._configs_to_df(configs)
        # takes the last value of each fidelity for each trial
        z = np.array([trial_values[-1] for trial_values in self.trial_results.values()])
        return X, z

    def fit_model(self):
        X, z = self.make_input_target()
        self.surrogate_model = QuantileRegressionSurrogateModel(
            config_space=self.config_space,
            max_fit_samples=self.max_fit_samples,
            random_state=self.random_state,
            mode="min",
            min_samples_to_conformalize=32,
            valid_fraction=0.1,
            **self.surrogate_kwargs,
        )
        self.surrogate_model.fit(df_features=X, y=z)

    def on_trial_complete(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int = None,
    ):
        print(f"on trial complete metric: {metric}")
        #metric = -metric
        metric = [-m for m in metric]
        self.trial_configs[trial_id] = config
        self.trial_results[trial_id].append(metric)

        metrics_list = self.mooLLM_config["metrics"]
        candidate_eval = {}
        for i, metric_name in enumerate(metrics_list):
            candidate_eval[metric_name] = metric[i]
        self.mooLLM.update_statistics(sel_candidate_point=config, sel_candidate_eval=candidate_eval)

    def on_trial_result(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int = None,
    ):
        self.trial_configs[trial_id] = config
        self.trial_results[trial_id].append(metric)

    def _configs_to_df(self, configs: List[Dict]) -> pd.DataFrame:
        return pd.DataFrame(configs)