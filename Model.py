import numpy as np

from BranchAndBound import BranchAndBound
from CuttingPlane import CuttingPlane


class Model():
    def __init__(self, A, b_l, b_u, c, l, u, type):
        self.A = A
        self.b_l = b_l
        self.b_u = b_u
        self.c = c
        self.l = l
        self.u = u
        self.type = type
        self.num_var = A.shape[1]

        self.int_index_array = np.array([True if flag == 1 else False for flag in self.type])
        self.int_index = [i for i in range(len(self.c)) if self.type[i] == 1]

        self.scale = np.ones(self.num_var)
        self.scale_matrix = np.diag(self.scale)
        self.shift = self.l.copy()
        self.shift[self.type == 1] = np.floor(self.shift[self.type == 1])

        self.A_ = np.dot(self.A, self.scale_matrix)
        self.b_l_ = self.b_l - np.dot(self.A, self.shift)
        self.b_u_ = self.b_u - np.dot(self.A, self.shift)
        self.c_ = np.dot(self.c, self.scale_matrix)
        self.c_shift = np.dot(self.c, self.shift)
        negative_index = self.scale < 0
        u_1 = (self.u - self.shift) / self.scale
        u_2 = u_1.copy()
        u_1[negative_index] = 0
        u_2[~negative_index] = 0
        l_1 = (self.l - self.shift) / self.scale
        l_2 = l_1.copy()
        l_1[negative_index] = 0
        l_2[~negative_index] = 0
        self.l_ = l_1 + u_2
        self.u_ = u_1 + l_2

        self.solution = None
        self.obj = None
        self.status_index = -1

        self.status = {-1: "Not Solved", 0: "Optimal"}

        self.episilon = 0.00001

    def solve(self, option):
        if option['algorithm'] == 'Cutting Plane':
            cutting = CuttingPlane(self, option)
            self.status_index, shift_solution, shift_obj = cutting.solve()
        elif option['algorithm'] == 'Branch and Bound':
            bb = BranchAndBound(self, option)
            self.status_index, shift_solution, shift_obj = bb.solve()

        self.solution = np.dot(self.scale_matrix, shift_solution) + self.shift
        self.obj = shift_obj + self.c_shift

    def result(self):
        print('Status:', self.status[self.status_index])
        print('Objective:', self.obj)
        print('Solution:', self.solution)
