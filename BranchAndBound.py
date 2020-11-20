from Node import Node
from Branch import Branchs
from NodeSelection import NodeSelections


class BranchAndBound():
    def __init__(self, model, option):
        self.primal_bound = 10000000
        self.dual_bound = -100000000
        self.best_solution = None
        self.nodes_list = []
        self.model = model

        self.branch = Branchs[option['Branch']]()
        self.nodeselection = NodeSelections[option['NodeSelection']]()

        self.variable_type = self.model.type
        self.int_index = [i for i in range(len(self.model.c)) if self.model.type[i] == 1]

        self.episilon = 0.000001

    def solve(self):
        subproblem = Node()
        subproblem.cylp_init(self.model.A, self.model.b, self.model.c, self.model.l, self.model.u)
        self.nodes_list.append(subproblem)

        while True:
            if (len(self.nodes_list)) == 0:
                return 0, self.best_solution, self.primal_bound
            current_node = self.nodeselection.choose(self.nodes_list)
            current_node.primal_solve()
            if current_node.check_feasility() == False or current_node.value() >= self.primal_bound:
                self.nodes_list.remove(current_node)
                continue
            if self.check_int(current_node):
                self.best_solution = current_node.get_primalVariableSolution()
                self.primal_bound = current_node.value()
                self.nodes_list.remove(current_node)
            else:
                nodes = self.branch.branch(current_node, self.int_index)
                self.nodes_list.remove(current_node)
                self.nodes_list.extend(nodes)

    def check_int(self, current_node):
        primal_value = current_node.get_primalVariableSolution()
        print(primal_value)
        for index in self.int_index:
            if abs(round(primal_value[index]) - primal_value[index]) > self.episilon:
                return False
        return True
