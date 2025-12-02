# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/reevo.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

from jinja2 import Environment, PackageLoader, StrictUndefined

from llamda.individual import Individual
from llamda.problem import Problem
from llamda.utils import filter_code


class Evolution:

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.env = Environment(
            loader=PackageLoader("llamda.prompts.ga", "reevo"),
            undefined=StrictUndefined,
        )

    def get_seed_population_messages(self, long_term_reflection_str: str) -> list[dict]:

        seed_template = self.env.get_template("seed.j2")
        seed_prompt = seed_template.render(
            seed_func=self.problem.seed_func,
            func_name=self.problem.func_name,
        )

        system_generator_template = self.env.get_template("system_generator.j2")
        system = system_generator_template.render()

        user_generator_template = self.env.get_template("user_generator.j2")
        user_generator_prompt = user_generator_template.render(
            func_name=self.problem.func_name,
            description=self.problem.description,
            func_desc=self.problem.func_desc,
        )

        user = (
            user_generator_prompt + "\n" + seed_prompt + "\n" + long_term_reflection_str
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        return messages

    def _gen_short_term_reflection_prompt(
        self, ind1: Individual, ind2: Individual
    ) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """

        if ind1.obj == ind2.obj:
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

        system_reflector_template = self.env.get_template("system_reflector.j2")
        system = system_reflector_template.render()

        user_reflector_st_template = self.env.get_template("user_reflector_st.j2")
        user = user_reflector_st_template.render(
            func_name=self.problem.func_name,
            func_desc=self.problem.func_desc,
            description=self.problem.description,
            worse_code=worse_code,
            better_code=better_code,
        )
        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        return message, worse_code, better_code

    def get_short_term_reflection_messages(
        self, population: list[Individual]
    ) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Generate short-term reflection messages for crossovering two individuals.
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

        return messages_lst, worse_code_lst, better_code_lst

    def get_long_term_reflection_messages(
        self, short_term_reflections: list[str], long_term_reflection_str: str
    ) -> list[dict]:
        """
        Generate long-term reflection messages.
        """

        system_reflector_template = self.env.get_template("system_reflector.j2")
        system = system_reflector_template.render()

        user_reflector_lt_template = self.env.get_template("user_reflector_lt.j2")
        user = user_reflector_lt_template.render(
            description=self.problem.description,
            prior_reflection=long_term_reflection_str,
            new_reflection="\n".join(short_term_reflections),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        return messages

    def get_crossover_messages(
        self, short_term_reflection_tuple: tuple[list[str], list[str], list[str]]
    ) -> list[list[dict]]:

        reflection_content_lst, worse_code_lst, better_code_lst = (
            short_term_reflection_tuple
        )
        messages_lst = []
        for reflection, worse_code, better_code in zip(
            reflection_content_lst, worse_code_lst, better_code_lst
        ):
            # Crossover
            system_generator_template = self.env.get_template("system_generator.j2")
            system = system_generator_template.render()

            user_generator_template = self.env.get_template("user_generator.j2")
            user_generator_prompt = user_generator_template.render(
                func_name=self.problem.func_name,
                description=self.problem.description,
                func_desc=self.problem.func_desc,
            )

            func_signature0 = self.problem.func_signature.format(version=0)
            func_signature1 = self.problem.func_signature.format(version=1)

            crossover_template = self.env.get_template("crossover.j2")
            user = crossover_template.render(
                user_generator=user_generator_prompt,
                func_signature0=func_signature0,
                func_signature1=func_signature1,
                worse_code=worse_code,
                better_code=better_code,
                reflection=reflection,
                func_name=self.problem.func_name,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            messages_lst.append(messages)

        return messages_lst

    def get_mutation_messages(
        self, long_term_reflection_str: str, elitist: Individual
    ) -> list[dict]:
        """
        Generate mutation messages for elitist-based mutation.
        """

        system_generator_template = self.env.get_template("system_generator.j2")
        system = system_generator_template.render()

        user_generator_template = self.env.get_template("user_generator.j2")
        user_generator_prompt = user_generator_template.render(
            func_name=self.problem.func_name,
            description=self.problem.description,
            func_desc=self.problem.func_desc,
        )

        func_signature1 = self.problem.func_signature.format(version=1)

        mutation_template = self.env.get_template("mutation.j2")
        user = mutation_template.render(
            user_generator=user_generator_prompt,
            reflection=long_term_reflection_str + self.problem.external_knowledge,
            func_signature1=func_signature1,
            elitist_code=filter_code(elitist.code),
            func_name=self.problem.func_name,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        return messages
