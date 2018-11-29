from typing import List

import pandas as pd
from pandas import DataFrame


class Solution:
    def __init__(self, objectives, variables):
        self.objectives = objectives
        self.variables = variables


class Meta:
    def __init__(
            self,
            objective_maximums: List[float],
            variables_count: int,
            variables: DataFrame,
            objectives: DataFrame,
            all_variables: DataFrame,
            all_objectives: DataFrame
    ):
        self.objective_maximums = objective_maximums
        self.variables_count = variables_count
        self.variables = variables
        self.objectives = objectives
        self.all_variables = all_variables
        self.all_objectives = all_objectives


def read_solutions(directory):
    objectives = pd.read_csv(directory + '/objectives.csv', delimiter='\t', index_col=None)
    variables = pd.read_csv(directory + '/variables.csv', delimiter='\t', index_col=None)
    all_objectives = pd.read_csv(directory + '/all_objectives.csv', delimiter='\t', index_col=None)
    all_variables = pd.read_csv(directory + '/all_variables.csv', delimiter='\t', index_col=None)

    solutions = []
    variables_row_iterator = variables.iterrows()

    for _, objectives_row in objectives.iterrows():
        _, variables_row = next(variables_row_iterator)
        solutions.append(Solution(objectives_row.values, variables_row.values))

    meta = Meta(
        objective_maximums=[objectives[col].max() for col in objectives.columns],
        variables_count=len(variables.columns),
        variables=variables,
        objectives=objectives,
        all_variables=all_variables,
        all_objectives=all_objectives
    )

    return solutions, meta
