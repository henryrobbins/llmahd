# Adapted from HSEvo: https://github.com/datphamvn/HSEvo/blob/main/hsevo.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

from importlib.resources import files
import logging

from llamda.problem import Problem
from llamda.utils import file_to_string, filter_code

logger = logging.getLogger("llamda")


class Evolution:

    def __init__(self, problem: Problem) -> None:

        self.problem = problem

        self.hsevo_prompts_dir = files("llamda.prompts.ga.hsevo")

        # Common prompts
        self.system_generator_prompt = file_to_string(
            self.hsevo_prompts_dir / "system_generator.txt"
        )
        self.system_reflector_prompt = file_to_string(
            self.hsevo_prompts_dir / "system_reflector.txt"
        )
        self.user_generator_prompt = file_to_string(
            self.hsevo_prompts_dir / "user_generator.txt"
        )
        self.system_hs_prompt = file_to_string(
            self.hsevo_prompts_dir / "system_harmony_search.txt"
        )
        self.hs_prompt = file_to_string(self.hsevo_prompts_dir / "harmony_search.txt")

    def init_population(self, long_term_reflection_str: str, scientist: str) -> dict:

        seed_prompt = file_to_string(self.hsevo_prompts_dir / "seed.txt").format(
            seed_func=self.problem.seed_func,
            func_name=self.problem.func_name,
        )

        user_generator_prompt_full = self.user_generator_prompt.format(
            seed=scientist,
            func_name=self.problem.func_name,
            description=self.problem.description,
            func_desc=self.problem.func_desc,
        )

        system_generator_prompt_full = self.system_generator_prompt.format(
            seed=scientist
        )

        # Generate responses
        system = system_generator_prompt_full
        user = (
            user_generator_prompt_full
            + "\n"
            + seed_prompt
            + "\n"
            + long_term_reflection_str
        )

        pre_messages = {"system": system, "user": user}

        logger.info(
            "Initial Population Prompt generated",
            extra={"system": system, "user": user},
        )

        return pre_messages

    def flash_reflection(self, lst_str_method: list[str]) -> dict:

        system = self.system_reflector_prompt

        user_flash_reflection_prompt = file_to_string(
            self.hsevo_prompts_dir / "user_flash_reflection.txt"
        )

        user = user_flash_reflection_prompt.format(
            description=self.problem.description,
            lst_method="\n".join(lst_str_method),
            schema_reflection={"analyze": "str", "exp": "str"},
        )

        pre_messages = {"system": system, "user": user}

        logger.info(
            "Flash reflection Prompt generated",
            extra={"system": system, "user": user},
        )

        return pre_messages

    def comprehensive_reflection(
        self,
        lst_good_reflection: list[str],
        lst_bad_reflection: list[str],
        str_flash_memory: dict,
    ) -> dict:
        system = self.system_reflector_prompt

        good_reflection = (
            "\n\n".join(lst_good_reflection) if len(lst_good_reflection) > 0 else "None"
        )
        bad_reflection = (
            "\n\n".join(lst_bad_reflection) if len(lst_bad_reflection) > 0 else "None"
        )

        user_comprehensive_reflection_prompt = file_to_string(
            self.hsevo_prompts_dir / "user_comprehensive_reflection.txt"
        )

        user = user_comprehensive_reflection_prompt.format(
            bad_reflection=bad_reflection,
            good_reflection=good_reflection,
            curr_reflection=str_flash_memory["exp"],
        )

        pre_messages = {"system": system, "user": user}

        logger.info(
            "Comprehensive reflection Prompt generated",
            extra={"system": system, "user": user},
        )

        return pre_messages

    def crossover(
        self,
        parent_1_code: str,
        parent_2_code: str,
        scientist: str,
        str_flash_memory: dict,
        str_comprehensive_memory: str,
    ) -> dict:

        # Crossover
        system = self.system_generator_prompt.format(seed=scientist)
        func_signature_m1 = self.problem.func_signature.format(version=0)
        func_signature_m2 = self.problem.func_signature.format(version=1)
        user_generator_prompt_full = self.user_generator_prompt.format(
            seed=scientist,
            func_name=self.problem.func_name,
            description=self.problem.description,
            func_desc=self.problem.func_desc,
        )

        crossover_prompt = file_to_string(self.hsevo_prompts_dir / "crossover.txt")

        user = crossover_prompt.format(
            user_generator=user_generator_prompt_full,
            func_signature_m1=func_signature_m1,
            func_signature_m2=func_signature_m2,
            code_method1=filter_code(parent_1_code),
            code_method2=filter_code(parent_2_code),
            analyze=str_flash_memory["analyze"],
            exp=str_comprehensive_memory,
            func_name=self.problem.func_name,
        )
        pre_messages = {"system": system, "user": user}

        logger.info(
            "Crossover Prompt generated", extra={"system": system, "user": user}
        )

        return pre_messages

    def mutate(
        self, scientist: str, str_comprehensive_memory: str, elitist: str
    ) -> dict:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        system = self.system_generator_prompt.format(seed=scientist)
        func_signature1 = self.problem.func_signature.format(version=1)
        user_generator_prompt_full = self.user_generator_prompt.format(
            seed=scientist,
            func_name=self.problem.func_name,
            description=self.problem.description,
            func_desc=self.problem.func_desc,
        )

        mutation_prompt = file_to_string(self.hsevo_prompts_dir / "mutation.txt")
        user = mutation_prompt.format(
            user_generator=user_generator_prompt_full,
            reflection=str_comprehensive_memory,
            func_signature1=func_signature1,
            elitist_code=filter_code(elitist),
            func_name=self.problem.func_name,
        )

        pre_messages = {"system": system, "user": user}

        logger.info(
            "Mutation Prompt generated",
            extra={"system": system, "user": user},
        )

        return pre_messages

    def harmony_search(self, sel_individual_hs: str) -> dict:
        system = self.system_hs_prompt
        user = self.hs_prompt.format(code_extract=sel_individual_hs)
        pre_messages = {"system": system, "user": user}

        logger.info(
            "Harmony Search Prompt generated",
            extra={"system": system, "user": user},
        )

        return pre_messages
