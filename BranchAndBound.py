from Branch import Branchs
from Node import Node
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

    def solve(self):
        subproblem = Node(self.model)
        subproblem.cylp_init(self.model.A_, self.model.b_l_, self.model.b_u_, self.model.c_, self.model.l_,
                             self.model.u_)
        self.nodes_list.append(subproblem)

        while True:
            if (len(self.nodes_list)) == 0:
                return 0, self.best_solution, self.primal_bound
            current_node = self.nodeselection.choose(self.nodes_list)
            current_node.primal_solve()
            if current_node.check_feasility() == False or current_node.value() >= self.primal_bound:
                self.nodes_list.remove(current_node)
                continue
            if current_node.check_int():
                self.best_solution = current_node.get_primalVariableSolution()
                self.primal_bound = current_node.value()
                self.nodes_list.remove(current_node)
            else:
                nodes = self.branch.branch(self.model, current_node)
                self.nodes_list.remove(current_node)
                self.nodes_list.extend(nodes)
