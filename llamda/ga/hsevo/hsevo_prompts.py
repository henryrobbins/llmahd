# Adapted from HSEvo: https://github.com/datphamvn/HSEvo/blob/main/hsevo.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import logging

from jinja2 import Environment, PackageLoader, StrictUndefined

from llamda.problem import Problem
from llamda.ga.utils import filter_code

logger = logging.getLogger("llamda")


class HSEvoPrompts:

    def __init__(self, problem: Problem) -> None:

        self.env = Environment(
            loader=PackageLoader("llamda.prompts.ga", "hsevo"),
            undefined=StrictUndefined,
        )

        self.problem = problem

    def init_population(self, long_term_reflection_str: str, scientist: str) -> dict:

        seed_template = self.env.get_template("seed.j2")
        seed_prompt = seed_template.render(
            seed_func=self.problem.seed_func,
            func_name=self.problem.func_name,
        )

        user_generator_template = self.env.get_template("user_generator.j2")
        user_generator_prompt_full = user_generator_template.render(
            seed=scientist,
            func_name=self.problem.func_name,
            description=self.problem.description,
            func_desc=self.problem.func_desc,
        )

        system_generator_template = self.env.get_template("system_generator.j2")
        system_generator_prompt_full = system_generator_template.render(seed=scientist)

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

        logger.debug(
            "Initial Population Prompt generated",
            extra={"system": system, "user": user},
        )

        return pre_messages

    def flash_reflection(self, lst_str_method: list[str]) -> dict:

        system_template = self.env.get_template("system_reflector.j2")
        system = system_template.render()

        user_flash_reflection_template = self.env.get_template(
            "user_flash_reflection.j2"
        )
        user = user_flash_reflection_template.render(
            description=self.problem.description,
            lst_method="\n".join(lst_str_method),
            schema_reflection={"analyze": "str", "exp": "str"},
        )

        pre_messages = {"system": system, "user": user}

        logger.debug(
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
        system_template = self.env.get_template("system_reflector.j2")
        system = system_template.render()

        good_reflection = (
            "\n\n".join(lst_good_reflection) if len(lst_good_reflection) > 0 else "None"
        )
        bad_reflection = (
            "\n\n".join(lst_bad_reflection) if len(lst_bad_reflection) > 0 else "None"
        )

        user_comprehensive_reflection_template = self.env.get_template(
            "user_comprehensive_reflection.j2"
        )
        user = user_comprehensive_reflection_template.render(
            bad_reflection=bad_reflection,
            good_reflection=good_reflection,
            curr_reflection=str_flash_memory["exp"],
        )

        pre_messages = {"system": system, "user": user}

        logger.debug(
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
        system_generator_template = self.env.get_template("system_generator.j2")
        system = system_generator_template.render(seed=scientist)

        func_signature_m1 = self.problem.func_signature.format(version=0)
        func_signature_m2 = self.problem.func_signature.format(version=1)

        user_generator_template = self.env.get_template("user_generator.j2")
        user_generator_prompt_full = user_generator_template.render(
            seed=scientist,
            func_name=self.problem.func_name,
            description=self.problem.description,
            func_desc=self.problem.func_desc,
        )

        crossover_template = self.env.get_template("crossover.j2")
        user = crossover_template.render(
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

        logger.debug(
            "Crossover Prompt generated", extra={"system": system, "user": user}
        )

        return pre_messages

    def mutate(
        self, scientist: str, str_comprehensive_memory: str, elitist: str
    ) -> dict:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        system_generator_template = self.env.get_template("system_generator.j2")
        system = system_generator_template.render(seed=scientist)

        func_signature1 = self.problem.func_signature.format(version=1)

        user_generator_template = self.env.get_template("user_generator.j2")
        user_generator_prompt_full = user_generator_template.render(
            seed=scientist,
            func_name=self.problem.func_name,
            description=self.problem.description,
            func_desc=self.problem.func_desc,
        )

        mutation_template = self.env.get_template("mutation.j2")
        user = mutation_template.render(
            user_generator=user_generator_prompt_full,
            reflection=str_comprehensive_memory,
            func_signature1=func_signature1,
            elitist_code=filter_code(elitist),
            func_name=self.problem.func_name,
        )

        pre_messages = {"system": system, "user": user}

        logger.debug(
            "Mutation Prompt generated",
            extra={"system": system, "user": user},
        )

        return pre_messages

    def harmony_search(self, sel_individual_hs: str) -> dict:
        system_template = self.env.get_template("system_harmony_search.j2")
        system = system_template.render()

        hs_template = self.env.get_template("harmony_search.j2")
        user = hs_template.render(code_extract=sel_individual_hs)

        pre_messages = {"system": system, "user": user}

        logger.debug(
            "Harmony Search Prompt generated",
            extra={"system": system, "user": user},
        )

        return pre_messages
