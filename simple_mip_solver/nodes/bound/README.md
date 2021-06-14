# bound

As mentioned in the parent packages' READMEs, this subpackage is where we define
the different `BaseNode` subclasses to override the bound method used by
the priority node queue in Branch and Bound. It is intended that each module
corresponds to one subclass where the public API is only overwritten for the bound
method, unless otherwise stated.

There are currently no bound specific subclasses. For an example of overriding
the bound method in a subclass, see the `simple_mip_solver.nodes.branch.pseudo_cost`
module, in which both branch and bound were overridden.
