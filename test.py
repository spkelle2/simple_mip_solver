import numpy as np

from Model import Model

A = np.matrix([[50, 31], [-3, 2, ]])
b_l = np.matrix([-10000, -10000])
b_u = np.matrix([250, 4])
c = np.array([-1, -0.64])
l = np.array([0.5, 0.])
u = np.array([10000, 10000])
type = np.array([1, 1])

A = np.matrix([[3, 2], [-3, 2, ]])
b_l = np.matrix([-10000, -10000])
b_u = np.matrix([6, 0])
c = np.array([0, -1])
l = np.array([0, 0.])
u = np.array([10000, 10000])
type = np.array([1, 1])

option = {'algorithm': "Cutting Plane", "Cutting": "GomoryMIR"}
option = {"Branch": "MostFractional", "NodeSelection": "BestFirst", 'algorithm': "Branch and Bound"}
test_problem = Model(A, b_l, b_u, c, l, u, type)
test_problem.solve(option)
test_problem.result()
