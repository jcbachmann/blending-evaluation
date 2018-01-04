import pandas as pd


class Solution:
    def __init__(self, objectives, variables):
        self.objectives = objectives
        self.variables = variables


class Meta:
    def __init__(
            self,
            objective_maximums: [float],
            variables_count: int,
            variables: pd.DataFrame,
            objectives: pd.DataFrame,
            all_variables: pd.DataFrame,
            all_objectives: pd.DataFrame
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
