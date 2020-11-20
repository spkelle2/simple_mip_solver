from Model import Model

A = [[50, 31], [-3, 2, ]]
b = [250, 4]
c = [-1, -0.64]
l = [0, 0]
u = [10000, 10000]
type = [1, 1]

option = {"Branch": "MostFractional", "NodeSelection": "BestFirst", 'algorithm': "branch and bound"}
test_problem = Model(A, b, c, l, u, type)
test_problem.solve(option)
test_problem.result()
