# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/baselines/eoh/original/eoh_evolution.py
# Originally from EoH: https://github.com/FeiLiu36/EoH/blob/main/eoh/src/eoh/methods/eoh/eoh_evolution.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

from enum import StrEnum
from dataclasses import dataclass

from jinja2 import Environment, PackageLoader, StrictUndefined

from llamda.individual import Individual
from llamda.problem import EohProblem


@dataclass
class EOHIndividual(Individual):
    algorithm: str | None = None


class EOHOperator(StrEnum):
    I1 = "i1"
    E1 = "e1"
    E2 = "e2"
    M1 = "m1"
    M2 = "m2"


def quote_and_join(items: list[str]) -> str:
    """Quote each item and join with commas"""
    if len(items) > 1:
        return ", ".join(f"'{s}'" for s in items)
    else:
        return f"'{items[0]}'"


class Evolution:

    def __init__(self, problem: EohProblem) -> None:
        self.env = Environment(
            loader=PackageLoader("llamda.prompts.ga", "eoh"), undefined=StrictUndefined
        )
        self.env.filters["quote_and_join"] = quote_and_join
        self.problem = problem

    def _get_problem_context(self) -> dict:
        """Get the common problem context for all templates."""
        return {
            "description": self.problem.description,
            "func_name": self.problem.func_name,
            "func_inputs": self.problem.func_inputs,
            "func_outputs": self.problem.func_outputs,
            "inout_info": self.problem.inout_info,
            "other_info": self.problem.other_info,
        }

    def i1(self) -> str:
        template = self.env.get_template(f"{EOHOperator.I1.value}.j2")
        return template.render(**self._get_problem_context())

    def e1(self, parents: list[EOHIndividual]) -> str:
        template = self.env.get_template(f"{EOHOperator.E1.value}.j2")
        return template.render(**self._get_problem_context(), indivs=parents)

    def e2(self, parents: list[EOHIndividual]) -> str:
        template = self.env.get_template(f"{EOHOperator.E2.value}.j2")
        return template.render(**self._get_problem_context(), indivs=parents)

    def m1(self, parent: EOHIndividual) -> str:
        template = self.env.get_template(f"{EOHOperator.M1.value}.j2")
        return template.render(
            **self._get_problem_context(),
            algorithm=parent.algorithm,
            code=parent.code,
        )

    def m2(self, parent: EOHIndividual) -> str:
        template = self.env.get_template(f"{EOHOperator.M2.value}.j2")
        return template.render(
            **self._get_problem_context(),
            algorithm=parent.algorithm,
            code=parent.code,
        )
