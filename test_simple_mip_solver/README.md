# test_simple_mip_solver package

The test package contains unit tests for each module in the `simple_mip_solver`
package. `test_simple_mip_solver` is laid out in a manner that mirrors
`simple_mip_solver`, with each test module located in a subpackage analogous to
the original modules subpackage (e.g. since `psuedo_cost.py` is in the
`simple_mip_solver.nodes.branch` subpackage, `test_psuedo_cost.py` is in the 
`test_simple_mip_solver.test_nodes.test_branch` subpackage.)

In addition to unit tests, there is a module (`test_simple_mip_solver.example_models`)
and directory (`test_simple_mip_solver/example_models/`) of predefined MILP's,
which as mentioned in the root README can be used as test instances to check correctness
and measure performance of different branch, bound, and search subclasses. There
is also a module (`test_simple_mip_solver.helpers`) with predefined utilities
that can be borrowed across other modules for faster testing development. I will
save the details of that to the curious reader who can check the doc strings, but
let's take a moment to review some of the available MILP test instances described
above.

## example_models module
I decided which instances made this test set by thinking about what kinds of MILP's
I would need to test that a given Node creates an expected branch and bound
tree. Since branch and bound can for each node do something different based on
whether the node is LP feasible, MILP feasible, or infeasible, I knew
I would at a minimum need three different MILP's to test that we see expected behavior
in each of those scenarios. In addition, we need a way of handling when feasible
solutions are unbounded, bringing the base test set to four. These test models are
as follows:

### no_branch
This model is a small MILP with a root node that has an LP relaxation that when
solved to optimality is integer feasible. I use this problem to ensure for a given
Node class that it has branch and bound update the current optimal solution and
MILP upper bound, but that it does not branch.

### small_branch
This model is a small MILP with a root node that has an LP relaxation that when
solved to optimality is not integer feasible. I use this problem to ensure for a
given Node class that it has branch and bound create two new nodes via branching
while not updating the MILP upper bound or current best solution.

### infeasible
This model is a small MILP with a root node that has an LP relaxation that when
solved is infeasible. I use this problem to ensure for a given Node class that
it has branch and bound do nothing after bounding.

### unbounded
to be implemented


(Note, I can use all of the above three problems to create nodes with monkey-patched
lower bounds so that we can test branch and bound immediately prunes them.
After doing such, we have test MILPs for each Node class to ensure its basic,
proper functionality in branch and bound)

## example_models folder
To be sure of my numerics in and to provide a sufficient level of "wear and tear"
on the branch, bound, and search methods developed, I put together a suite of larger
random MILP's to run in tests as well. (I had originally planned to use small
instances from the MIPLIB to test my solver, but I could not get Gurobi or PuLP
to read any of the `.mps` files I downloaded. For this reason, I choose to
generate a variety of random instances using GRUMPY's `GenerateRandomMIP` method.)

I generated this suite of test problems by creating a model for each combination
of values for the following parameters:
* number of variables: 10 or 20 
* number of constraints: 10 or 20
* density of nonzeros in matrix A: 0.2 or 0.8
* range of objective coefficients: 0 to 10 or 0 to 1000
* range of constraint coeffecients: 0 to 10 or 0 to 1000
* range of values for RHS: 1/6 of sum of row's coefficients or 5/6 of sum of row's
coefficients

I chose this combination of tests for a couple of reasons. First, they were the
largest sizes of tests I could run each within a couple seconds so that they all
finish in a short amount of time. I wanted to run tests with as large of coeffecients
and numbers of variables as I could to give myself the most confidence that I
would not run into numerical errors. Otherwise, I wanted to vary as much as I
could about a MIP of the above given size just to give confidence that my solver
would work on any problem of this size.