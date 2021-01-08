from src.variables.variable import Variable, VariableType


class ContinuousVariable(Variable):
    type = VariableType.CONTINUOUS

    def __create_func(self, variable: Variable):
        pass

