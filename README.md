# simple_mip_solver repository

This repository serves as a sandbox to facilitate easy testing for new branch,
bound, and search methods for the Branch and Bound algorithm in mixed integer
linear programming. There are three ways one can play in this sandbox of sorts.
One can use classes already defined to solve `MILPInstance` objects from the
`coinor.cuppy.milpInstance` module. One can also define new `BaseNode` subclasses
that override and extend the currently available branch, bound, and search methods
in this package. Lastly, one can test these new methods against an existing
set of test problems and known solutions to ensure their accuracy and compare
their performance. Each of these play styles will have a section to follow with
further detail.

### Set Up Instructions
Since this repository is intended to be a base for further development, the
suggested means of installation is to clone this repository and add its path
on your machine to your `.pth` file. If you've never set up a `.pth` file before,
see [this link](https://medium.com/@arnaud.bertrand/modifying-python-s-search-path-with-pth-files-2a41a4143574).
After adding the path to this repository on your local to your `.pth` file,
executing any module should run as expected.

### Using Existing Classes
Existing branch, bound, and search methods are collected into the different Node
classes (e.g. `BaseNode` or `PseudoCostBranchNode`) available for import from
`simple_mip_solver` along with the branch and bound class itself (named
`BranchAndBound`). To use a combination of existing methods in Branch and Bound
to solve a given MILP, one can create a `coinor.cuppy.milpInstance.MILPInstance`
object from the MILP and pass it as well as a reference to a Node class of choice
to the `BranchAndBound` constructor. Calling `BranchAndBound.solve()` followed
by its public attributes will solve and show the results of the solve respectively.
For example, one can do the following to use Pseudo-Cost branching with Best First
search in Branch and Bound on the `coinor.cuppy.milpInstance.MILPInstance` object
`small_branch`:

```python
from simple_mip_solver import PseudoCostBranchNode, BranchAndBound
from test_simple_mip_solver.example_models import small_branch

bb = BranchAndBound(model=small_branch, Node=PseudoCostBranchNode)
bb.solve()
print(f"objective: {bb.objective_value}")
print(f"solution: {bb.solution}")
```

For an explanation of the `BranchAndBound` and precreated branch, bound, and search
method classes, check the `README.md`'s and docstrings of the `simple_mip_solver`
package and subpackages.

### Developing New Classes
If you would like to get a little adventurous and experiment with your own
branch, bound, or search methods, make a new module for each method you add,
saving it in the respective subpackage in `simple_mip_solver.nodes`. This module
should contain a single class, which is a subclass of `simple_mip_solver.BaseNode`.
In this class, you can overwrite the parent's methods with your new methods. 
Specifically, you will want to do the following for each method:

* branch 
    * Create a new module in `simple_mip_solver/nodes/branch` with a class that
      inherits `simple_mip_solver.BaseNode`.
    * In this class, name your branch method `branch`. Ensure it includes a `**kwargs` argument
      as a catchall for unneeded arguments that `BranchAndBound` objects will send.
    * Update `simple_mip_solver/branch_and_bound.py` to instantiate and send
      any additional attributes to your branch method.
    * Pass the index you would like to branch on to `BaseNode._base_branch`.
    * Return the dictionary returned from `BaseNode._base_branch` with key-value pairs
      you would like added as attribute and value to the `BranchAndBound` object.
    * Update `simple_mip_solver/nodes/branch/README.md` for any details of the
      method not included in the docstrings.
    * See `simple_mip_solver.nodes.branch.psuedo_cost.branch` for an example.
* bound
    * Create a new module in `simple_mip_solver/nodes/bound` with a class that
      inherits `simple_mip_solver.BaseNode`.
    * In this class, name your bound method `bound`. Ensure it includes a `**kwargs` argument
      as a catchall for unneeded arguments that `BranchAndBound` objects will send.
    * Update `simple_mip_solver/branch_and_bound.py` to instantiate and send 
      any additional attributes to your bound method.
    * Call `BaseNode._base_branch` to solve the LP relaxation and update related
      attributes.
    * Return a dictionary with key-value pairs you would like added as attribute
      and value to the `BranchAndBound` object. Return empty dictionary if none.
    * Update `simple_mip_solver/nodes/bound/README.md` for any details of the
      method not included in the docstrings.  
    * See `simple_mip_solver.nodes.branch.psuedo_cost.bound` for an example
      (I know I said to place classes to overwrite bound in the
      `simple_mip_solver.nodes.bound` subpackage, but I made an exception here
      since `simple_mip_solver.nodes.branch.psuedo_cost.branch` needs the bound
      method overwritten as well).  
* search
    * Create a new module in `simple_mip_solver/nodes/search` with a class that
      inherits `simple_mip_solver.BaseNode`.
    * In this class, define a method `__lt__` that takes another instance of your new class.
      `__lt__` returns `True` if the instance it is in the scope of has priority
      in the Branch and Bound queue over the other instance and `False` otherwise.
    * Additionally, define a method `__eq__` that takes another instance of your new class.
      `__eq__` returns `True` if the instance it is in the scope of has the same
      priority in the Branch and Bound queue as the other instance and `False` otherwise.
    * Update `simple_mip_solver/nodes/search/README.md` for any details of the
      method not included in the docstrings.  
    * See `simple_mip_solver.nodes.search.depth_first.DepthFirstSearchNode` for
      an example.
  
The instances of your new classes if structured as the above will be nodes
compatible with `BranchAndBound` but with the new methods included.

##### Combining Classes for Different Methods into One
If you come up with new classes for multiple methods, you can use python's
multiple inheritance to declare classes that combine multiple different methods,
just be careful of inheritance order for classes that have overlapping name spaces.
To combine custom methods, do the following:
* Define the new combination class in `simple_mip_solver/nodes/nodes.py`
* See `simple_mip_solver.nodes.nodes.PseudoCostBranchDepthFirstSearchNode` for an
  example.
* Import the new combination class in `simple_mip_solver/__init__.py` so
  that it can be imported from the `simple_mip_solver`.


### Testing New Classes 
To ensure each new class works (and continues to work in future releases), add
unit tests for your class in
`test_simple_mip_solver/test_nodes/test_<respective method subpackage>`. In these
unit tests, you will want to test your class against each test instance in
`test_simple_mip_solver/example_models.py`. You can do this as follows:
* Instantiate a `BranchAndBound` instance with your Node class and a test instance
* Solve the `BranchAndBound` instance and record the optimal objective value.
* Run the test instance on another solver and check the resulting optimal
  objective value matches.

For an overview of the suite of already defined test instances, see the README
in the `test_simple_mip_solver` package.
