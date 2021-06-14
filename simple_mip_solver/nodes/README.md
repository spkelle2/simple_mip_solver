# nodes

As mentioned in the introductory README, this subpackage is where we define
the different node objects that serve as an API for the methods and attributes
off of which we run branch and bound. The main contribution in this package
is the `BaseNode` class, which is defined in `base_node.py` and serves as an
ancestor for all future Node classes that are defined. In addition, we have
`nodes.py`, where contributors can access existing and define new combinations
of `BaseNode` subclasses found in the subpackages of this package. These subpackages
contain classes that inherit from `BaseNode` and override the `BaseNode.branch`,
`BaseNode.bound`, and `BaseNode.search` methods respectively. Details on how that is
done can be seen in the README's for each subpackage. Let's take a moment to
dive deeper on the classes in this level of the package.

## base_node.BaseNode
`BranchAndBound` using the node `BaseNode` will search Best First, branch Most
Fractional, and bound by solving a given LP relaxation. `BaseNode` has the following
public API.

### Public Attributes

##### lower_bound
The lower bound on the optimal value of this node's LP relaxation. All nodes
are instantiated to have LP relaxations that are minimization problems.

##### objective_value
The objective value of the optimal solution of the node's LP relaxation.

##### solution
The optimal solution of the node's LP relaxation.

##### lp_feasible
Whether or not the LP relaxation finds a feasible solution when solving.

##### mip_feasible
Whether or not the optimal solution to the LP relaxation is a feasible solution
to the MILP.

##### depth
How many nodes deep this node is in the branch and bound tree.

##### search_method
String for how nodes are prioritized. Set to "Best First".

##### branch_method
String for how nodes are branched. Set to "Most Fractional".

### Public Methods

##### init
The constructor for `BaseNode` will always instantiate with the following
three arguments:
* lp: A `cylp.cy.CyClpSimplex` object that represents the LP relaxation this
  node might solve.
* integerIndices: indices of variables that need to all have integer values
  in the optimal solution of the LP relaxation for this node to be feasible
  for the original MILP given to `BranchAndBound`
* lower_bound: lower bound on the objective value of the optimal solution of
  this node's LP relaxation
* b_idx: index of variable that was branched on to create this node
* b_dir: direction the parent node was branched to create this node (either "up"
  or "down")
* b_val: initial value of variable that was branched on to create this node
* depth: how deep in the tree this node is

##### bound
The bound method does nothing more than solving the node's LP relaxation and
updating related attributes. For this reason, this method is designed to just be
a wrapper around a private bounding method (`BaseNode._base_bound`) which can be
used as a base for all other custom bound methods. The function that `BaseNode.bound`
wraps does the following:

* Solve the LP relaxation with Dual Simplex from its given basis (to be defined
  in `BaseNode.branch`)
* Update `BaseNode.objective_value` and `BaseNode.solution` from the attributes
  of the solved LP.
* Determine if the solve was LP and MILP feasible, setting the values of
  `BaseNode.lp_feasible` and `BaseNode.mip_feasible` accordingly.

##### branch
As mentioned above, the `BaseNode.branch` method branches on the most fractional
variable that we expect to be integer. Thus, this method has the following two
parts:
* find the most fractional index
* branch on a given index
We describe both steps as follows.
  
We determine which index is most fractional in the following manner:
* Determine how far each variable that must be integer is in the LP relaxation
  solution from the nearest integer value.
* Return the variable with the furthest distance from an integer, provided that
  distance is greater than some small epsilon to account for numerics. Else return
  `None`
  
Branching on a given index is intended to be a process that is the same for any
custom branching method, so it is provided as a private function
(`BaseNode._base_branch`) that can be reused within any subclass's `branch` method.
It is defined as follows:
* Determine the basis of the solution for the current node's LP relaxation.
* Create a "down" node, which consists of the following:
    * The current node's LP relaxation but with an additional constraint setting
      the upper bound of the branched on variable to the floor of its value in the
      current node.
    * A starting basis that is the same as the optimal basis of the current node
    * A lower bound of the current node's LP relaxation optimal objective value
* Create an "up" node, which consists of the following:
    * The current node's LP relaxation but with an additional constraint setting
      the lower bound of the branched on variable to the ceiling of its value in the
      current node
    * A starting basis that is the same as the optimal basis of the current node
    * A lower bound of the current node's LP relaxation optimal objective value
* Return a dictionary of the children nodes keyed by their direction.  

##### search 
As mentioned in the root README, search is implemented within a node by defining
python's `__eq__` and `__lt__` operators, which are what determine if an object
is equal to or less than, respectively, another object of the same type. In 
the case of `BaseNode` objects, we want them to be prioritized in our Branch
and Bound node queue such that those with the lowest lower bounds on their LP
relaxations are explored first (i.e. searched "best first"). A priority queue pops
an object that is evaluated to be less than or equal to all other objects in the
queue, so to implement "best first", we need the following:
* `__lt__` returns `True` if the current object has a lower bound on its LP relaxation
  less than the lower bound of the LP relaxation of the object it is being compared
  to, else `False`
* `__eq__` returns `True` if the current object has a lower bound on its LP relaxation
  equal to the lower bound of the LP relaxation of the object it is being compared
  to, else `False`
  
## nodes
As mentioned above, contributors can access existing and define new combinations
of `BaseNode` subclasses found in the subpackages of this package in the `nodes`
module. One can go about combining different branch, bound, and search subclasses
into a single class by defining a new class in this module that multiply inherits
from your subclasses of choice. An example of how to do so can be seen with
`nodes.PseudoCostBranchDepthFirstSearchNode`, which combines the subclass for
pseudo cost branching with the subclass for depth first search. One can see
`pass` is provided in the class definition since all methods and attributes are
defined in the respective subclasses.
