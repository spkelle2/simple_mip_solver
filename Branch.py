from math import floor

from cylp.cy.CyClpSimplex import CyClpSimplex

from Node import Node


class Branch_MostFractional():
    def __init__(self):
        pass

    def branch(self, model, node):
        solution = node.get_primalVariableSolution()
        index = -1
        int_value = -1
        abs_value = 0.5
        for item in model.int_index:
            if abs(solution[item] - (floor(solution[item]) + 0.5)) < abs_value:
                abs_value = abs(solution[item] - (floor(solution[item]) + 0.5))
                int_value = floor(solution[item])
                index = item
        node_l_cylp = CyClpSimplex()
        upper = node.cylp.variablesUpper.copy()
        upper[index] = int_value
        node_l_cylp.loadProblem(node.cylp.matrix, node.cylp.variablesLower, upper, node.cylp.objective,
                                node.cylp.constraintsLower, node.cylp.constraintsUpper)

        node_r_cylp = CyClpSimplex()
        lower = node.cylp.variablesLower.copy()
        lower[index] = int_value + 1
        node_r_cylp.loadProblem(node.cylp.matrix, lower, node.cylp.variablesUpper, node.cylp.objective,
                                node.cylp.constraintsLower, node.cylp.constraintsUpper)
        return [Node(model, node_l_cylp), Node(model, node_r_cylp)]


Branchs = {'MostFractional': Branch_MostFractional}
