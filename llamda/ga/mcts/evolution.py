# Adapted from MCTS-AHD: https://github.com/zz1358m/MCTS-AHD-master/blob/main/source/evolution.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

from enum import StrEnum
from dataclasses import dataclass

from jinja2 import Environment, PackageLoader, StrictUndefined

from llamda.individual import Individual
from llamda.problem import EohProblem


@dataclass
class MCTSIndividual(Individual):
    algorithm: str | None = None
    thought: str | None = None


class MCTSOperator(StrEnum):
    I1 = "i1"
    E1 = "e1"
    E2 = "e2"
    M1 = "m1"
    M2 = "m2"
    S1 = "s1"


def quote_and_join(items: list[str]) -> str:
    """Quote each item and join with commas"""
    if len(items) > 1:
        return ", ".join(f"'{s}'" for s in items)
    else:
        return f"'{items[0]}'"


class Evolution:

    def __init__(self, problem: EohProblem):
        self.env = Environment(
            loader=PackageLoader("llamda.prompts.ga", "mcts"), undefined=StrictUndefined
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

    def post(self, code: str) -> str:
        template = self.env.get_template("post.j2")
        return template.render(
            description=self.problem.description,
            func_name=self.problem.func_name,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
            code=code,
        )

    def refine(self, code: str, algorithm: str) -> str:
        template = self.env.get_template("refine.j2")
        return template.render(
            description=self.problem.description,
            func_name=self.problem.func_name,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
            algorithm=algorithm,
            code=code,
        )

    def i1(self) -> str:
        template = self.env.get_template(f"{MCTSOperator.I1.value}.j2")
        return template.render(**self._get_problem_context())

    def e1(self, parents: list[MCTSIndividual]) -> str:
        template = self.env.get_template(f"{MCTSOperator.E1.value}.j2")
        return template.render(**self._get_problem_context(), indivs=parents)

    def e2(self, parents: list[MCTSIndividual]) -> str:
        template = self.env.get_template(f"{MCTSOperator.E2.value}.j2")
        return template.render(**self._get_problem_context(), indivs=parents)

    def m1(self, parent: MCTSIndividual) -> str:
        template = self.env.get_template(f"{MCTSOperator.M1.value}.j2")
        return template.render(
            **self._get_problem_context(),
            algorithm=parent.algorithm,
            code=parent.code,
        )

    def m2(self, parent: MCTSIndividual) -> str:
        template = self.env.get_template(f"{MCTSOperator.M2.value}.j2")
        return template.render(
            **self._get_problem_context(),
            algorithm=parent.algorithm,
            code=parent.code,
        )

    def s1(self, parents: list[MCTSIndividual]) -> str:
        template = self.env.get_template(f"{MCTSOperator.S1.value}.j2")
        return template.render(**self._get_problem_context(), indivs=parents)
