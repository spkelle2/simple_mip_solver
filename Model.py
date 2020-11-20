import numpy as np
from BranchAndBound import BranchAndBound


class Model():
    def __init__(self, A, b, c, l, u, type):
        self.A = A
        self.b = b
        self.c = c
        self.l = l
        self.u = u
        self.type = type

        self.solution = None
        self.obj = None
        self.status_index = -1

        self.status = {-1: "Not Solved", 0: "Optimal"}

    def solve(self, option):
        bb = BranchAndBound(self, option)
        self.status_index, self.solution, self.obj = bb.solve()

    def result(self):
        print('Status:', self.status[self.status_index])
        print('Objective:', self.obj)
        print('Solution:', self.solution)
