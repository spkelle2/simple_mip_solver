# test_simple_mip_solver package

The test package contains unit tests for each module in the `simple_mip_solver`
package. `test_simple_mip_solver` is laid out in a manner that mirrors
`simple_mip_solver`, with each test module located in a subpackage analogous to
the original modules subpackage (e.g. since `psuedo_cost.py` is in the
`simple_mip_solver.nodes.branch` subpackage, `test_psuedo_cost.py` is in the 
`test_simple_mip_solver.test_nodes.test_branch` subpackage.)

In addition to unit tests, there is also a module of predefined MILP's 
(`test_simple_mip_solver.example_models`), which
as mentioned in the root README can be used as test instances to check correctness
and measure performance of different branch, bound, and search subclasses. Let's
take a moment to review some of the available MILP test instances.

## example_models
I decided which instances made my test set by thinking about what kinds of MILP's
I would need to test that a given Node creates an expected branch and bound
tree. Since branch and bound can for each node do something different based on
whether the node is LP feasible, MILP feasible, or neither, I knew I would at a 
minimum need three different MILP's to test that we see expected behavior in each
of those scenarios. In addition, to be sure of my numerics in and to provide a
sufficient level of "wear and tear" on the branch, bound, and search methods
developed, I am developing a suite of larger random MILP's to run in tests as well.
(I had originally planned to use small instances from the MIPLIB to test my solver,
but I could not get Gurobi or PuLP to read any of the `.mps` files I downloaded.
For this reason, I choose to generate a variety of random instances using
GRUMPY's `GenerateRandomMIP` method.) I will describe each test MILP in the coming
subsections.

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

(Note, I can use all of the above three problems to create nodes with monkey-patched
lower bounds so that we can test branch and bound immediately prunes them.
After doing such, we have test MILPs for each Node class to ensure its basic,
proper functionality in branch and bound)

### random
I have not had a chance yet to implement this fully, but the plan is to create
random MILPs of varying rows, columns, coefficients, and density, saving their
`.mps` files for later use. In testing a suite of problems with a variety of
coefficients, I think I can ensure I still run accurately against large coefficients
and constraints that are nearly parallel, both of which are known to complicate
solving MILP's. In addition, the variety of densities and problem sizes ensures
different search, branch, and bound methods all still work as expected on different
problem sizes.
