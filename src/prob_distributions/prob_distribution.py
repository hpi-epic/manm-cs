class ProbDistribution:

    def get_num_values(self):
        raise NotImplementedError()

    def sample(self, num_observations: int):
        raise NotImplementedError()
