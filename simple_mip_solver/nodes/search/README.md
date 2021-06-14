# search

As mentioned in the parent packages' READMEs, this subpackage is where we define
the different `BaseNode` subclasses to override the search method used by
the priority node queue in Branch and Bound. It is intended that each module
corresponds to one subclass where the public API is only overwritten for search
(e.g. `__lt__` and `__eq__`) methods, unless otherwise stated.

## depth_first.DepthFirstSearchNode
`BranchAndBound` instantiated with a node subclassed from
`depth_first.DepthFirstSearchNode` will use Depth First as its priority for
exploring nodes in the branch and bound tree. `depth_first.DepthFirstSearchNode`
makes the following updates to `BaseNode`'s public API.

### Public Attributes

##### search_method
String for how nodes are prioritized. Set to "Depth First".

### Public Methods

##### search
In the case of `DepthFirstSearchNode` objects, we want them to be prioritized in
our Branch and Bound node queue such that those at deeper depths in the search
tree are explored first (i.e. searched "depth first"). Since a priority queue pops
an object that is evaluated to be less than or equal to all other objects in the
queue, we implement "best first" by doing the following:
* `__lt__` returns `True` if the current object has a deeper depth in the branch
  and bound tree than the object it is being compared to, else `False`
* `__eq__` returns `True` if the current object has the same depth in the branch
  and bound tree than the object it is being compared to, else `False`
