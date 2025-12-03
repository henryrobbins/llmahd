# Adapted from HSEvo: https://github.com/datphamvn/HSEvo/blob/main/hsevo.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

from dataclasses import dataclass
import os
from pathlib import Path
import re
import logging
import numpy as np
import json
import tiktoken

from llamda.ga.base import GeneticAlgorithm
from llamda.ga.hsevo.hsevo_prompts import HSEvoPrompts
from llamda.evaluate import Evaluator
from llamda.individual import Individual
from llamda.llm_client.base import BaseClient
from llamda.problem import Problem
from llamda.ga.utils import extract_code_from_generator, population_checkpoint

logger = logging.getLogger("llamda")


@dataclass
class HSEvoConfig:

    # Main GA loop parameters
    max_fe: int = 450  # maximum number of function evaluations
    pop_size: int = 10  # population size for GA
    init_pop_size: int = 30  # initial population size for GA
    mutation_rate: float = 0.5  # mutation rate for GA

    # Harmony search
    hm_size: int = 5
    hmcr: float = 0.7
    par: float = 0.5
    bandwidth: float = 0.2
    max_iter: int = 5


@dataclass
class HSEvoIndividual(Individual):
    tryHS: bool = False


class HSEvo(GeneticAlgorithm[HSEvoConfig, Problem]):
    def __init__(
        self,
        config: HSEvoConfig,
        problem: Problem,
        evaluator: Evaluator,
        llm_client: BaseClient,
        output_dir: str,
    ) -> None:

        super().__init__(
            config=config,
            problem=problem,
            evaluator=evaluator,
            llm_client=llm_client,
            output_dir=output_dir,
        )

        self.output_dir = output_dir
        os.makedirs(f"{self.output_dir}/thinking", exist_ok=True)
        self.temperature = self.llm_client.temperature

        self.mutation_rate = self.config.mutation_rate
        self.iteration = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.elitist = None
        self.best_obj_overall = float("inf")
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.lst_good_reflection = []
        self.lst_bad_reflection = []

        self.evol = HSEvoPrompts(problem=self.problem)

        self.evaluator = evaluator

        self.str_comprehensive_memory = self.problem.external_knowledge
        self.local_sel_hs = None

        self.scientists = [
            "You are an expert in the domain of optimization heuristics.",
            "You are Albert Einstein, relativity theory developer.",
            "You are Isaac Newton, the father of physics.",
            "You are Marie Curie, pioneer in radioactivity.",
            "You are Nikola Tesla, master of electricity.",
            "You are Galileo Galilei, champion of heliocentrism.",
            "You are Stephen Hawking, black hole theorist.",
            "You are Richard Feynman, quantum mechanics genius.",
            "You are Rosalind Franklin, DNA structure revealer.",
            "You are Ada Lovelace, computer programming pioneer.",
        ]

    def _logging_context(self) -> dict:
        return {
            "method": "HSEvo",
            "problem_name": self.problem.name,
            "max_fe": self.config.max_fe,
            "pop_size": self.config.pop_size,
        }

    def cal_usage_LLM(
        self,
        lst_prompt: list[list[dict]],
        lst_completion: list[str],
        encoding_name: str = "cl100k_base",
    ) -> None:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        for i in range(len(lst_prompt)):
            for message in lst_prompt[i]:
                for _, value in message.items():
                    self.prompt_tokens += len(encoding.encode(value))

            self.completion_tokens += len(encoding.encode(lst_completion[i]))

    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        code = extract_code_from_generator(self.problem.seed_func).replace("v1", "v2")
        seed_ind = HSEvoIndividual(name="seed", code=code, tryHS=False)
        self.population = self.evaluator.batch_evaluate(
            [seed_ind], Path(self.output_dir)
        )
        self.seed_ind = self.population[0]

        # If seed function is invalid, stop
        if not self.seed_ind.exec_success:
            raise RuntimeError(
                "Seed function is invalid. "
                f"Please check the stdout file in {os.getcwd()}."
            )

        self.update_iter()

        messages_lst = []

        for i in range(self.config.init_pop_size):
            scientist = self.scientists[i % len(self.scientists)]
            pre_messages = self.evol.init_population(
                long_term_reflection_str=self.long_term_reflection_str,
                scientist=scientist,
            )
            messages = format_messages(pre_messages)
            messages_lst.append(messages)

        responses = self.llm_client.multi_chat_completion(
            messages_lst, 1, self.temperature + 0.3
        )
        self.cal_usage_LLM(messages_lst, responses)

        population = [
            self.response_to_individual(response, f"seed_population_{i}", messages)
            for i, (response, messages) in enumerate(zip(responses, messages_lst))
        ]

        # Run code and evaluate population
        population = self.evaluator.batch_evaluate(population, Path(self.output_dir))

        # Update iteration
        self.population = population
        self.update_iter()

    def response_to_individual(
        self, response: str, name: str, messages: list[str] | None = None
    ) -> HSEvoIndividual:
        """
        Convert response to individual
        """
        # Write response to file
        individual_dir = f"{self.output_dir}/individuals/{name}"

        os.makedirs(individual_dir, exist_ok=True)
        file_name = f"{individual_dir}/response.txt"

        with open(file_name, "w", encoding="utf-8") as file:
            file.writelines(response + "\n")

        if messages is not None:
            with open(f"{individual_dir}/message.json", "w") as file:
                file.write(json.dumps(messages, indent=4))

        code = extract_code_from_generator(response)

        individual = HSEvoIndividual(name=name, code=code)

        return individual

    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        objs = [individual.obj for individual in population]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))

        # update best overall
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx].code
            self.best_code_path_overall = (
                f"{self.output_dir}/{population[best_sample_idx].name}/code.py"
            )

        # update elitist
        if self.elitist is None or best_obj < self.elitist.obj:
            self.elitist = population[best_sample_idx]
            logger.info(
                f"Iteration {self.iteration}: Elitist: {self.elitist.obj}",
                extra={
                    "iteration": self.iteration,
                    "elitist_obj": self.elitist.obj,
                    **self._logging_context(),
                },
            )

        self.iteration += 1

    def random_select(self, population: list[HSEvoIndividual]) -> list[HSEvoIndividual]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.problem.type == "black_box":
            population = [
                individual
                for individual in population
                if individual.exec_success and individual.obj < self.seed_ind.obj
            ]
        else:
            population = [
                individual for individual in population if individual.exec_success
            ]
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.config.pop_size:
            trial += 1
            parents: list[HSEvoIndividual] = np.random.choice(
                np.array(population), size=2, replace=False
            ).tolist()
            # If two parents have the same objective value, consider them as identical;
            # otherwise, add them to the selected population
            if parents[0].obj != parents[1].obj:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def flash_reflection(self, population: list[HSEvoIndividual]) -> None:
        lst_str_method = []
        seen_elements = set()

        sorted_population: list[HSEvoIndividual] = sorted(
            population, key=lambda x: x.obj, reverse=False
        )
        for idx, individual in enumerate(sorted_population):
            suffix = (
                "th"
                if 11 <= idx + 1 <= 13
                else {1: "st", 2: "nd", 3: "rd"}.get((idx + 1) % 10, "th")
            )
            str_idx_method = f"[Heuristics {idx + 1}{suffix}]"
            # str_idx_method = f"[Heuristics {individual['code_path']}]"
            # str_obj = f"* Objective score: {individual['obj']}"
            str_code = individual.code
            temp_str = str_idx_method + "\n" + str_code + "\n"

            if temp_str not in seen_elements:
                seen_elements.add(temp_str)
                lst_str_method.append(temp_str)

        pre_messages = self.evol.flash_reflection(
            lst_str_method=lst_str_method,
        )
        messages = format_messages(pre_messages)

        flash_reflection_res = self.llm_client.multi_chat_completion(
            [messages], 1, self.temperature
        )[0]
        self.cal_usage_LLM([messages], flash_reflection_res)
        analyze_start = flash_reflection_res.find("**Analysis:**") + len(
            "**Analysis:**"
        )
        exp_start = flash_reflection_res.find("**Experience:**")

        analysis_text = flash_reflection_res[analyze_start:exp_start].strip()
        experience_text = flash_reflection_res[
            exp_start + len("**Experience:**") :
        ].strip()

        # Create the JSON structure
        flash_reflection_json = {"analyze": analysis_text, "exp": experience_text}

        # Convert to JSON string
        self.str_flash_memory = flash_reflection_json

        # Write reflections to file
        file_name = (
            f"{self.output_dir}/thinking/iteration{self.iteration}_lst_code_method.txt"
        )
        with open(file_name, "w") as file:
            file.writelines(json.dumps(pre_messages))

        file_name = (
            f"{self.output_dir}/thinking/iteration{self.iteration}_flash_reflection.txt"
        )
        with open(file_name, "w") as file:
            file.writelines(flash_reflection_res)

    def comprehensive_reflection(self) -> None:

        pre_messages = self.evol.comprehensive_reflection(
            lst_good_reflection=self.lst_good_reflection,
            lst_bad_reflection=self.lst_bad_reflection,
            str_flash_memory=self.str_flash_memory,
        )
        messages = format_messages(pre_messages)

        comprehensive_response = self.llm_client.multi_chat_completion(
            [messages], 1, self.temperature
        )[0]
        self.cal_usage_LLM([messages], comprehensive_response)
        self.str_comprehensive_memory = (
            self.problem.external_knowledge + "\n" + comprehensive_response
        )

        file_name = f"{self.output_dir}/thinking/iteration{self.iteration}_comprehensive_reflection_prompt.txt"
        with open(file_name, "w") as file:
            file.writelines(json.dumps(pre_messages))

        file_name = f"{self.output_dir}/thinking/iteration{self.iteration}_comprehensive_reflection.txt"
        with open(file_name, "w") as file:
            file.writelines(self.str_comprehensive_memory)

    def crossover(self, population: list[HSEvoIndividual]) -> list[HSEvoIndividual]:
        messages_lst = []
        num_choice = 0
        for i in range(0, len(population), 2):
            # Select two individuals
            if population[i].obj < population[i + 1].obj:
                parent_1 = population[i]
                parent_2 = population[i + 1]
            else:
                parent_1 = population[i + 1]
                parent_2 = population[i]

            pre_messages = self.evol.crossover(
                parent_1.code,
                parent_2.code,
                scientist=self.scientists[0],
                str_flash_memory=self.str_flash_memory,
                str_comprehensive_memory=self.str_comprehensive_memory,
            )
            messages = format_messages(pre_messages)

            num_choice += 1

            messages_lst.append(messages)

        # Asynchronously generate responses
        response_lst = self.llm_client.multi_chat_completion(
            messages_lst, 1, self.temperature
        )
        self.cal_usage_LLM(messages_lst, response_lst)
        population = [
            self.response_to_individual(
                response, f"iteration{self.iteration}_crossover_{i}", messages
            )
            for i, (response, messages) in enumerate(zip(response_lst, messages_lst))
        ]
        return population

    def mutate(self) -> list[HSEvoIndividual]:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""

        pre_messages = self.evol.mutate(
            scientist=self.scientists[0],
            str_comprehensive_memory=self.str_comprehensive_memory,
            elitist=self.elitist.code,
        )
        messages = format_messages(pre_messages)

        responses = self.llm_client.multi_chat_completion(
            [messages],
            int(self.config.pop_size * self.mutation_rate),
            self.temperature,
        )
        self.cal_usage_LLM([messages], responses)
        population = [
            self.response_to_individual(
                response, f"iteration{self.iteration}_mutation_{i}", messages
            )
            for i, response in enumerate(responses)
        ]

        return population

    def sel_individual_hs(self) -> str:
        candidate_hs = [
            individual for individual in self.population if individual.tryHS is False
        ]
        best_candidate_id = self.find_best_obj(candidate_hs)
        self.local_sel_hs = best_candidate_id
        self.population[best_candidate_id].tryHS = True
        return self.population[best_candidate_id].code

    def initialize_harmony_memory(self, bounds: list[list[int]]) -> np.ndarray:
        problem_size = len(bounds)
        harmony_memory = np.zeros((self.config.hm_size, problem_size))
        for i in range(problem_size):
            lower_bound, upper_bound = bounds[i]
            harmony_memory[:, i] = np.random.uniform(
                lower_bound, upper_bound, self.config.hm_size
            )
        return harmony_memory

    def responses_to_population(
        self, responses: list[str], try_hs_idx: int | None = None
    ) -> list[HSEvoIndividual]:
        """
        Convert responses to population. Applied to the initial population.
        """
        population = []
        for i, response in enumerate(responses):
            name = (
                f"iteration{self.iteration}_response{i}"
                if try_hs_idx is None
                else f"iteration{self.iteration}_hs{try_hs_idx}"
            )
            individual = self.response_to_individual(response, name)
            population.append(individual)
        return population

    def create_population_hs(
        self,
        str_code: str,
        parameter_ranges: list[int],
        harmony_memory: np.ndarray,
        try_hs_idx: int | None = None,
    ) -> list[HSEvoIndividual] | None:
        str_create_pop = []
        for i in range(len(harmony_memory)):
            tmp_str = str_code
            for j in range(len(list(parameter_ranges))):
                tmp_str = tmp_str.replace(
                    ("{" + list(parameter_ranges)[j] + "}"), str(harmony_memory[i][j])
                )
                if tmp_str == str_code:
                    return None
            str_create_pop.append(tmp_str)

        population_hs = self.responses_to_population(str_create_pop, try_hs_idx)
        return self.evaluator.batch_evaluate(population_hs, Path(self.output_dir))

    def find_best_obj(self, population_hs: list[HSEvoIndividual]) -> int:
        objs = [individual.obj for individual in population_hs]
        best_solution_id = np.argmin(np.array(objs))
        return int(best_solution_id)

    def create_new_harmony(
        self, harmony_memory: np.ndarray, bounds: list[list[int]]
    ) -> np.ndarray:
        new_harmony = np.zeros((harmony_memory.shape[1],))
        for i in range(harmony_memory.shape[1]):
            if np.random.rand() < self.config.hmcr:
                new_harmony[i] = harmony_memory[
                    np.random.randint(0, harmony_memory.shape[0]), i
                ]
                if np.random.rand() < self.config.par:
                    adjustment = (
                        np.random.uniform(-1, 1)
                        * (bounds[i][1] - bounds[i][0])
                        * self.config.bandwidth
                    )
                    new_harmony[i] += adjustment
            else:
                new_harmony[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        return new_harmony

    def update_harmony_memory(
        self,
        population_hs: list[HSEvoIndividual],
        harmony_memory: np.ndarray,
        new_harmony: np.ndarray,
        func_block: str,
        parameter_ranges: list[int],
        try_hs_idx: int,
    ) -> tuple[list[HSEvoIndividual], np.ndarray]:
        objs = [individual.obj for individual in population_hs]
        worst_index = np.argmax(np.array(objs))

        new_individual = self.create_population_hs(
            func_block, parameter_ranges, [new_harmony.tolist()], try_hs_idx
        )[0]

        if new_individual.obj < population_hs[worst_index].obj:
            population_hs[worst_index] = new_individual
            harmony_memory[worst_index] = new_harmony
        return population_hs, harmony_memory

    def harmony_search(self) -> HSEvoIndividual | None:
        pre_messages = self.evol.harmony_search(
            sel_individual_hs=self.sel_individual_hs()
        )
        messages = format_messages(pre_messages)

        # Write to file
        file_name = f"{self.output_dir}/thinking/iteration{self.iteration}_hs.txt"
        with open(file_name, "w") as file:
            file.writelines(json.dumps(pre_messages))

        responses = self.llm_client.multi_chat_completion(
            [messages], 1, self.temperature
        )
        self.cal_usage_LLM([messages], [str(responses[0])])

        parameter_ranges, func_block = extract_to_hs(responses[0])
        if parameter_ranges is None or func_block is None:
            return None
        bounds = [value for value in parameter_ranges.values()]

        harmony_memory = self.initialize_harmony_memory(bounds)
        population_hs = self.create_population_hs(
            func_block, parameter_ranges, harmony_memory
        )

        if population_hs is None:
            return None
        elif (
            len(
                [
                    individual
                    for individual in population_hs
                    if individual.exec_success is True
                ]
            )
            == 0
        ):
            self.evaluator.function_evals -= self.config.hm_size
            return None

        for iteration in range(self.config.max_iter):
            new_harmony = self.create_new_harmony(harmony_memory, bounds)
            population_hs, harmony_memory = self.update_harmony_memory(
                population_hs,
                harmony_memory,
                new_harmony,
                func_block,
                parameter_ranges,
                iteration,
            )
        best_obj_id = self.find_best_obj(population_hs)
        population_hs[best_obj_id].tryHS = True
        return population_hs[best_obj_id]

    def save_log_population(
        self, population: list[HSEvoIndividual], logHS: bool = False
    ) -> None:
        objs = [individual.obj for individual in population]
        if logHS is False:
            file_name = f"{self.output_dir}/objs_log_iter{self.iteration}.txt"
            with open(file_name, "w") as file:
                file.writelines("\n".join(map(str, objs)) + "\n")
        else:
            file_name = f"{self.output_dir}/objs_log_iter{self.iteration}_hs.txt"
            with open(file_name, "w") as file:
                file.writelines("\n".join(map(str, objs + [self.local_sel_hs])) + "\n")

    def evolve(self) -> tuple[str, str]:

        logger.info("Starting HSEvo evolution", extra=self._logging_context())

        self.init_population()
        logger.info(
            "Initial population created",
            extra={
                "population_size": len(self.population),
                **self._logging_context(),
            },
        )

        while self.evaluator.function_evals < self.config.max_fe:
            logger.info(
                f"Evolution iteration [{self.evaluator.function_evals}/{self.config.max_fe}]",  # noqa: E501
                extra={
                    "iteration": self.iteration,
                    "function_evals": self.evaluator.function_evals,
                    **self._logging_context(),
                },
            )
            # If all individuals are invalid, stop
            if all([not individual.exec_success for individual in self.population]):
                logger.error(
                    "All individuals are invalid, stopping evolution",
                    extra=self._logging_context(),
                )
                raise RuntimeError("All individuals are invalid. Stopping evolution.")
            # Select
            population_to_select = (
                self.population
                if (self.elitist is None or self.elitist in self.population)
                else [self.elitist] + self.population
            )  # add elitist to population for selection
            selected_population = self.random_select(population_to_select)
            if selected_population is None:
                logger.error("Selection failed", extra=self._logging_context())
                raise RuntimeError("Selection failed. Please check the population.")

            logger.debug(
                f"Selected {len(selected_population)} individuals",
                extra={
                    "n_selected": len(selected_population),
                    **self._logging_context(),
                },
            )

            # Reflection
            self.flash_reflection(selected_population)
            self.comprehensive_reflection()
            curr_code_path = f"{self.output_dir}/{self.elitist.name}/code.py"

            logger.debug("Reflection complete", extra=self._logging_context())

            # Crossover
            crossed_population = self.crossover(selected_population)
            # Evaluate
            self.population = self.evaluator.batch_evaluate(
                crossed_population, Path(self.output_dir)
            )
            # Update
            self.update_iter()

            logger.debug(
                f"Crossover complete, generated {len(crossed_population)} individuals",
                extra={
                    "n_crossover": len(crossed_population),
                    **self._logging_context(),
                },
            )

            # Mutate
            mutated_population = self.mutate()
            # Evaluate
            self.population.extend(
                self.evaluator.batch_evaluate(mutated_population, Path(self.output_dir))
            )
            # Update
            self.update_iter()

            logger.debug(
                f"Mutation complete, generated {len(mutated_population)} individuals",
                extra={
                    "n_mutation": len(mutated_population),
                    **self._logging_context(),
                },
            )

            if curr_code_path != f"{self.output_dir}/{self.elitist.name}/code.py":
                self.lst_good_reflection.append(self.str_flash_memory["exp"])
                logger.debug(
                    "Elite changed, recording good reflection",
                    extra=self._logging_context(),
                )
            else:
                self.lst_bad_reflection.append(self.str_flash_memory["exp"])
                logger.debug(
                    "Elite unchanged, recording bad reflection",
                    extra=self._logging_context(),
                )

            population_checkpoint(
                population=self.population,
                name=f"iteration_{self.iteration}_pre_hs",
                output_dir=self.output_dir,
            )

            # Harmony Search
            try_hs_num = 3
            while try_hs_num:
                logger.debug(
                    f"Attempting harmony search (attempt {4 - try_hs_num}/3)",
                    extra={"attempt": 4 - try_hs_num, **self._logging_context()},
                )
                individual_hs = self.harmony_search()
                if individual_hs is not None:
                    self.population.extend([individual_hs])
                    # self.update_iter()

                    population_checkpoint(
                        population=self.population,
                        name=f"iteration_{self.iteration}_post_hs",
                        output_dir=self.output_dir,
                    )

                    logger.debug(
                        "Harmony search successful", extra=self._logging_context()
                    )
                    break
                else:
                    try_hs_num -= 1
                    logger.debug(
                        "Harmony search failed, retrying", extra=self._logging_context()
                    )
            self.update_iter()

        logger.info(
            "HSEvo evolution completed",
            extra={
                "best_objective": self.best_obj_overall,
                "function_evals": self.evaluator.function_evals,
                **self._logging_context(),
            },
        )

        return self.best_code_overall, self.best_code_path_overall


def extract_to_hs(input_string: str) -> tuple[dict | None, str | None]:
    code_blocks = input_string.split("```python\n")[1:]

    try:
        parameter_ranges_block = (
            "import numpy as np\n" + code_blocks[1].split("```")[0].strip()
        )
        if any(
            keyword in parameter_ranges_block for keyword in ["inf", "np.inf", "None"]
        ):
            return None, None
        exec_globals = {}
        exec(parameter_ranges_block, exec_globals)
        parameter_ranges = exec_globals["parameter_ranges"]
    except:
        return None, None

    function_block = code_blocks[0].split("```")[0].strip()

    paren_count = 0
    in_signature = False
    signature_start_index = None
    signature_end_index = None

    # Loop through the function block to find the start and end of the function signature
    for i, char in enumerate(function_block):
        if char == "d" and function_block[i : i + 3] == "def":
            in_signature = True
            signature_start_index = i
        if in_signature:
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            if char == ":" and paren_count == 0:
                signature_end_index = i
                break

    if signature_start_index is not None and signature_end_index is not None:
        function_signature = function_block[
            signature_start_index : signature_end_index + 1
        ]
        for param in parameter_ranges:
            pattern = rf"(\b{param}\b[^=]*=)[^,)]+"
            replacement = r"\1 {" + param + "}"
            function_signature = re.sub(
                pattern, replacement, function_signature, flags=re.DOTALL
            )
        function_block = (
            function_block[:signature_start_index]
            + function_signature
            + function_block[signature_end_index + 1 :]
        )

    return parameter_ranges, function_block


def format_messages(pre_messages: dict) -> list[dict]:
    messages = [
        {"role": "system", "content": pre_messages["system"]},
        {"role": "user", "content": pre_messages["user"]},
    ]
    return messages
