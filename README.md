# simple_mip_solver

This package serves as a lightweight mixed integer programming solver to
facilitate easy testing for new branch, bound, and search methods. Existing methods
are collected into the different `Node` objects available for import from
`simple_mip_solver` along with the branch and bound method itself
(named `BranchAndBound`). For example, to use Pseudo-Cost branching with Best First
search in Branch and Bound on the test model `small_branch`, we can do the following:

```python
from simple_mip_solver import PseudoCostBranchNode, BranchAndBound
from test_simple_mip_solver.example_models import small_branch

bb = BranchAndBound(small_branch, Node=PseudoCostBranchNode)
bb.solve()
print(f"objective: {bb.objective_value}")
print(f"solution: {bb.solution}")
```

Note, right now I do not currently have this on pypi, so clone the repo and
put the repo's path in a `.pth` file so the above runs.

If you would like to get a little adventurous and experiment with your own
branch, bound, or search methods, make a new module for each branch, bound,
or search method you write, saving them in the respective directory. For examples,
check out the existing modules in the `simple_mip_solver.nodes.search` and
`simple_mip_solver.nodes.branch` subpackages. When writing your own methods,
please make note of the API that branch and bound in
`simple_mip_solver.branch_and_bound.BranchAndBound` expects of the classes you
create. The recommended way to implement new classes is to subclass an existing
class in this package and overwrite public methods as necessary. If you would like
to combine different custom methods you have written into one class, import
the objects containing your methods to `simple_mip_solver.nodes.nodes` and
create a new subclass that inherits from them all. See that module for an
example.
