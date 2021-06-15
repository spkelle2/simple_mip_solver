# simple_mip_solver.nodes.branch subpackage

As mentioned in the parent packages' READMEs, this subpackage is where we define
the different `BaseNode` subclasses to override the branch method used by
the priority node queue in Branch and Bound. It is intended that each module
corresponds to one subclass where the public API is only overwritten for the branch
method, unless otherwise stated.

## pseudo_cost.PseudoCostBranchNode
`BranchAndBound` instantiated with a node subclassed from
`depth_first.PseudoCostBranchNode` will use Pseudo Costs as its metric for
deciding which index to branch on after solving an LP relaxation. In addition
to updating `BaseNode`'s branch method, `depth_first.PseudoCostBranchNode` will
also update `BaseNode`'s bound method for reasons we will explore shortly.
Thus, `depth_first.DepthFirstSearchNode` makes the following updates to
`BaseNode`'s public API.

### Public Attributes

##### branch_method
String for how nodes are branched. Set to "pseudo cost".

### Public Methods

##### branch
In the case of `PseudoCostBranchNode` objects, we want them to be branched on
the variable with the highest pseudo cost (i.e. average change in bound relative
to the depth of the cut created by branching) that also violates its integrality
constraint. We accomplish this by having `PseudoCostBranchNode.branch` do as follows:
* For each integer variable that has a fractional value in the optimal solution
  of the node's LP relaxation, multiply the depth of the cut created by branching
  up or down by the respective pseudo cost to determine the expected changes in
  bound by branching up and down, respectively.
* Branch (i.e. call `BaseNode._base_branch`) on the index with the largest
  minimum expected change.

For this function to work, however, we need to have a way of instantiating and
updating the pseudo costs. Since these costs are reevaluated after bounding
each node (i.e. solving its relaxation), we need to update `PseudoCostBranchNode.branch`
to allow for this.
  
##### bound
As far as bounding is concerned, we do not need to make any changes to how we
solve the node's LP relaxation or update its related attributes. Rather, we need
to ensure that after each solve we record information that helps us instantiate
and update psuedo costs. We can do so as follows:
* Bound the node in the usual fashion (i.e. solve its LP relaxation with 
  `BaseNode._base_bound`)
* **If** the LP relaxation is feasible
    * Strong branch on each integer index that is fractional and does not yet
      have pseudo costs. Instantiate the pseudo costs for this index as the 
      change in bound relative to the depth of cut that were both made by strong
      branching.
    * **If** strong branching had previously been run for the index that was
      branched on to create this node, update the pseudo cost for this index
      and branching direction as the average change in bound relative to depth
      of cut added over initial strong branching and all subsequent branching
      on this index and direction. 
      
For those who need a refresher, strong branching is where we branch on a given
index and record the changes in bound after completing a given number of simplex
iterations. As shown above, we can use this change in bound as an estimator
of how "effective" branching on this variable might be.
