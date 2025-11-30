from importlib.resources import files
import logging

from llamda.utils.individual import Individual
from llamda.utils.llm_client.base import BaseClient
from llamda.utils.problem import ProblemPrompts
from llamda.utils.utils import file_to_string, filter_code


class ReEvoLLMClients:
    def __init__(
        self,
        generator_llm: BaseClient,
        reflector_llm: BaseClient | None = None,
        short_reflector_llm: BaseClient | None = None,
        long_reflector_llm: BaseClient | None = None,
        crossover_llm: BaseClient | None = None,
        mutation_llm: BaseClient | None = None,
    ) -> None:
        self.generator_llm = generator_llm
        self.reflector_llm = reflector_llm or generator_llm
        self.short_reflector_llm = short_reflector_llm or self.reflector_llm
        self.long_reflector_llm = long_reflector_llm or self.reflector_llm
        self.crossover_llm = crossover_llm or generator_llm
        self.mutation_llm = mutation_llm or generator_llm


class Evolution:

    def __init__(
        self,
        init_pop_size: int,
        pop_size: int,
        mutation_rate: float,
        llm_clients: ReEvoLLMClients,
        prompts: ProblemPrompts,
    ) -> None:

        self.init_pop_size = init_pop_size
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        self.llm_clients = llm_clients
        self.prompts = prompts

        self.reevo_prompts_dir = files("llamda.prompts.ga.reevo")

        # Common prompts
        self.system_generator_prompt = file_to_string(
            self.reevo_prompts_dir / "system_generator.txt"
        )
        self.system_reflector_prompt = file_to_string(
            self.reevo_prompts_dir / "system_reflector.txt"
        )
        self.user_generator_prompt = file_to_string(
            self.reevo_prompts_dir / "user_generator.txt"
        ).format(
            func_name=self.prompts.func_name,
            problem_desc=self.prompts.func_desc,
            func_desc=self.prompts.func_desc,
        )

    def seed_population(self, long_term_reflection_str: str) -> list[str]:

        seed_prompt = file_to_string(self.reevo_prompts_dir / "seed.txt").format(
            seed_func=self.prompts.seed_func,
            func_name=self.prompts.func_name,
        )

        system = self.system_generator_prompt
        user = (
            self.user_generator_prompt
            + "\n"
            + seed_prompt
            + "\n"
            + long_term_reflection_str
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        logging.info(
            "Initial Population Prompt: \nSystem Prompt: \n"
            + system
            + "\nUser Prompt: \n"
            + user
        )

        responses = self.llm_clients.generator_llm.multi_chat_completion(
            [messages],
            self.init_pop_size,
            temperature=self.llm_clients.generator_llm.temperature + 0.3,
        )  # Increase the temperature for diverse initial population

        return responses

    def _gen_short_term_reflection_prompt(
        self, ind1: Individual, ind2: Individual
    ) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """

        user_reflector_st_prompt = file_to_string(
            self.reevo_prompts_dir / "user_reflector_st.txt"
        )

        if ind1.obj == ind2.obj:
            print(ind1.code, ind2.code)
            raise ValueError(
                "Two individuals to crossover have the same objective value!"
            )
        # Determine which individual is better or worse
        if ind1.obj < ind2.obj:
            better_ind, worse_ind = ind1, ind2
        else:  # robust in rare cases where two individuals have the same objective
            better_ind, worse_ind = ind2, ind1

        worse_code = filter_code(worse_ind.code)
        better_code = filter_code(better_ind.code)

        system = self.system_reflector_prompt
        user = user_reflector_st_prompt.format(
            func_name=self.prompts.func_name,
            func_desc=self.prompts.func_desc,
            problem_desc=self.prompts.problem_desc,
            worse_code=worse_code,
            better_code=better_code,
        )
        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Print reflection prompt for the first iteration
        logging.info(
            "Short-term Reflection Prompt: \nSystem Prompt: \n"
            + system
            + "\nUser Prompt: \n"
            + user
        )

        return message, worse_code, better_code

    def short_term_reflection(
        self, population: list[Individual]
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i + 1]

            # Short-term reflection
            messages, worse_code, better_code = self._gen_short_term_reflection_prompt(
                parent_1, parent_2
            )
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)

        # Asynchronously generate responses
        response_lst = self.llm_clients.short_reflector_llm.multi_chat_completion(
            messages_lst
        )
        return response_lst, worse_code_lst, better_code_lst

    def long_term_reflection(
        self, short_term_reflections: list[str], long_term_reflection_str: str
    ) -> str:
        """
        Long-term reflection before mutation.
        """

        user_reflector_lt_prompt = file_to_string(
            self.reevo_prompts_dir / "user_reflector_lt.txt"
        )

        system = self.system_reflector_prompt
        user = user_reflector_lt_prompt.format(
            problem_desc=self.prompts.problem_desc,
            prior_reflection=long_term_reflection_str,
            new_reflection="\n".join(short_term_reflections),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        logging.info(
            "Long-term Reflection Prompt: \nSystem Prompt: \n"
            + system
            + "\nUser Prompt: \n"
            + user
        )

        response = self.llm_clients.long_reflector_llm.multi_chat_completion(
            [messages]
        )[0]

        return response

    def crossover(
        self, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]
    ) -> list[str]:

        crossover_prompt = file_to_string(self.reevo_prompts_dir / "crossover.txt")

        reflection_content_lst, worse_code_lst, better_code_lst = (
            short_term_reflection_tuple
        )
        messages_lst = []
        for reflection, worse_code, better_code in zip(
            reflection_content_lst, worse_code_lst, better_code_lst
        ):
            # Crossover
            system = self.system_generator_prompt
            func_signature0 = self.prompts.func_signature.format(version=0)
            func_signature1 = self.prompts.func_signature.format(version=1)
            user = crossover_prompt.format(
                user_generator=self.user_generator_prompt,
                func_signature0=func_signature0,
                func_signature1=func_signature1,
                worse_code=worse_code,
                better_code=better_code,
                reflection=reflection,
                func_name=self.prompts.func_name,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            messages_lst.append(messages)

            # Print crossover prompt for the first iteration
            logging.info(
                "Crossover Prompt: \nSystem Prompt: \n"
                + system
                + "\nUser Prompt: \n"
                + user
            )

        # Asynchronously generate responses
        responses = self.llm_clients.crossover_llm.multi_chat_completion(messages_lst)
        return responses

    def mutate(self, long_term_reflection_str: str, elitist: Individual) -> list[str]:
        """
        Elitist-based mutation. We mutate the best to generate n_pop new individuals.
        """

        mutation_prompt = file_to_string(self.reevo_prompts_dir / "mutation.txt")

        system = self.system_generator_prompt
        func_signature1 = self.prompts.func_signature.format(version=1)
        user = mutation_prompt.format(
            user_generator=self.user_generator_prompt,
            reflection=long_term_reflection_str + self.prompts.external_knowledge,
            func_signature1=func_signature1,
            elitist_code=filter_code(elitist.code),
            func_name=self.prompts.func_name,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        logging.info(
            "Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user
        )

        responses = self.llm_clients.mutation_llm.multi_chat_completion(
            [messages], int(self.pop_size * self.mutation_rate)
        )

        return responses
