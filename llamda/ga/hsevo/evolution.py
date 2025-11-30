from importlib.resources import files
import logging

from llamda.utils.problem import ProblemPrompts
from llamda.utils.utils import file_to_string, filter_code


class Evolution:

    def __init__(self, prompts: ProblemPrompts) -> None:

        self.prompts = prompts

        self.hsevo_prompts_dir = files('llamda.prompts.ga.hsevo')

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
            seed_func=self.prompts.seed_func,
            func_name=self.prompts.func_name,
        )

        user_generator_prompt_full = self.user_generator_prompt.format(
            seed=scientist,
            func_name=self.prompts.func_name,
            problem_desc=self.prompts.problem_desc,
            func_desc=self.prompts.func_desc,
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

        logging.info(
            "Initial Population Prompt: \nSystem Prompt: \n"
            + system
            + "\nUser Prompt: \n"
            + user
        )

        return pre_messages

    def flash_reflection(self, lst_str_method: list[str]) -> dict:

        system = self.system_reflector_prompt

        user_flash_reflection_prompt = file_to_string(
            self.hsevo_prompts_dir / "user_flash_reflection.txt"
        )

        user = user_flash_reflection_prompt.format(
            problem_desc=self.prompts.problem_desc,
            lst_method="\n".join(lst_str_method),
            schema_reflection={"analyze": "str", "exp": "str"},
        )

        pre_messages = {"system": system, "user": user}

        logging.info(
            "Flash reflection Prompt: \nSystem Prompt: \n"
            + system
            + "\nUser Prompt: \n"
            + user
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

        logging.info(
            "Comprehensive reflection Prompt: \nSystem Prompt: \n"
            + system
            + "\nUser Prompt: \n"
            + user
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
        func_signature_m1 = self.prompts.func_signature.format(version=0)
        func_signature_m2 = self.prompts.func_signature.format(version=1)
        user_generator_prompt_full = self.user_generator_prompt.format(
            seed=scientist,
            func_name=self.prompts.func_name,
            problem_desc=self.prompts.problem_desc,
            func_desc=self.prompts.func_desc,
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
            func_name=self.prompts.func_name,
        )
        pre_messages = {"system": system, "user": user}

        # Print crossover prompt for the first iteration
        logging.info(
            "Crossover Prompt: \nSystem Prompt: \n"
            + system
            + "\nUser Prompt: \n"
            + user
        )

        return pre_messages

    def mutate(
        self, scientist: str, str_comprehensive_memory: str, elitist: dict
    ) -> dict:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        system = self.system_generator_prompt.format(seed=scientist)
        func_signature1 = self.prompts.func_signature.format(version=1)
        user_generator_prompt_full = self.user_generator_prompt.format(
            seed=scientist,
            func_name=self.prompts.func_name,
            problem_desc=self.prompts.problem_desc,
            func_desc=self.prompts.func_desc,
        )

        mutation_prompt = file_to_string(self.hsevo_prompts_dir / "mutation.txt")
        user = mutation_prompt.format(
            user_generator=user_generator_prompt_full,
            reflection=str_comprehensive_memory,
            func_signature1=func_signature1,
            elitist_code=filter_code(elitist["code"]),
            func_name=self.prompts.func_name,
        )

        pre_messages = {"system": system, "user": user}

        logging.info(
            "Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user
        )

        return pre_messages

    def harmony_search(self, sel_individual_hs: str) -> dict:
        system = self.system_hs_prompt
        user = self.hs_prompt.format(code_extract=sel_individual_hs)
        pre_messages = {"system": system, "user": user}

        # Print get hs prompt for the first iteration
        logging.info(
            "Harmony Search Prompt: \nSystem Prompt: \n"
            + system
            + "\nUser Prompt: \n"
            + user
        )

        return pre_messages
