import numpy as np

from branch_and_bound import BranchAndBound


class Model:
    def __init__(self, A, b, c, l, u, vtypes, direction='min'):
        assert direction in ['min', 'max']
        assert all(vt in [0, 1] for vt in vtypes)
        assert direction != 'max', 'not implemented yet!'
        # let cylp handing asserting dimensions match


        # figure out default optimization direction
        # TODO make sure all problems get converted to minimization
        # set c to -c if max
        self.A = A  # matrix
        self.b = b
        self.c = c  # cost function
        self.l = l  # lower bound on x
        self.u = u  # upper bound on x
        self.vtypes = vtypes  # needs to be binary array specifying int or continuous
        self.num_var = A.shape[1]

        self.int_index_array = np.array([bool(flag) for flag in self.vtypes])
        self.int_index = [i for i, flag in enumerate(self.vtypes) if flag]

        self.solution = None
        self.obj = None
        self.status_index = None

        self.status = {-1: "Infeasible", 0: "Optimal"}

        self.epsilon = 0.00001  # have test ensuring this is here and reasonable

        self.branch_and_bound = BranchAndBound(self)

    def solve(self) -> None:
        """Solve the model with Branch and Bound, the only currently supported
        algorithm

        :return:
        """
        self.status_index, self.solution, self.obj = self.branch_and_bound.solve()

    def result(self) -> None:
        """Print the solution to this model

        :return:
        """
        print('Status:', self.status[self.status_index])
        print('Objective:', self.obj)
        print('Solution:', self.solution)
