# simple_mip_solver

The solver is broken into the following two components. The `BranchAndBound` object
in `branch_and_bound.py` runs the branch and bound algorithm for solving MILP's.
In the package `nodes`, we have a variety of `Node` objects that serve as API's
for the different branch, bound, and search methods that Branch and Bound can
employ. To learn more about how `Node` API's work, see the README's in the `nodes`
package and its subpackages. As the core of this package is the
`branch_and_bound.BranchAndBound` object, let's take a moment to highlight how that
part of the solver works by reviewing its public facing methods and attributes.

## branch_and_bound.BranchAndBound

### Public Attributes

##### solution
The list of the values of each variable in the optimal solution ordered by index

##### status
A string indicating how the branch and bound algorithm terminated

##### objective_value
The objective value of the optimal solution, if it was found

### Public Methods

##### init
The constructor for `BranchAndBound` will always instantiate with the following
three arguments:
* model: a reference to a `coinor.cuppy.milpInstance.MILPInstance` object, which
  contains the objective and constraints for the MILP that we solve.
* Node: a class with methods and attributes matching the expectations in the doc
  string. These methods and attributes form the "Node API" that
  `branch_and_bound.BranchAndBound` will use to branch a node, bound a node, and
  search for the next node to evaluate, as well as to determine if a node should
  be pruned, branched, or be considered optimal.
* node_queue: the data structure to use for sorting nodes.

##### solve
This method runs the branch and bound algorithm and sets the values for the
public attributes. I implemented as follows:

* Create an instance of the `Node` argument in the constructor from the `model`
  reference. Add this first "Node" to our `node_queue`
* **While** the `node_queue` is not empty, do the following:
    * Search for the highest priority node in the `node_queue`
    * **If** this node has a lower bound on its optimal objective that is at least
      the optimal objective of the best known MILP feasible solution (we make
      this comparison because all LP relaxations are converted to minimization)
        * **continue** (which effectively prunes this node)
    * Bound the node.
    * **If** the node's LP relaxation is feasible and has an optimal objective value
      less than the current best known MILP feasible solution
        * **If** the node's optimal solution is MILP feasible:
            * Save the node's optimal solution as the best MILP feasible solution
        * **Else**:
            * Branch on this node, adding both children to the `node_queue`
* **If** we find a MILP feasible solution
    * `status` is set to "optimal"
    * `solution` is set to the solution of the best MILP feasible solution
    * `objective_value` is set to the objective value of the solution of the best
      MILP feasible solution
* **Else**
    * `status` is set to "infeasible"
    
Note, however, that search, branch, and bound were left intentionally vague
as they are methods defined by the class passed to the `Node` argument in the
`branch_and_bound.BranchAndBound` constructor. For specifics on those methods,
read on to the READMEs in the `nodes` subpackage.
