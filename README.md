# simple_mip_solver

This package serves as a lightweight mixed integer programming solver as to
facilitate an easy testing ground for new branching and cutting schemes. You can
use the package as follows:
```python
from coinor.cuppy.milpInstance import MILPInstance
from simple_mip_solver import BranchAndBound

# define matrix A and vectors b, c, l, and u, which represent our model
model = MILPInstance(A=A, b=b, c=c, l=l, u=u)
bb = BranchAndBound(model)
bb.solve()
if bb.solution:
    print(f'solution: {bb.solution}')
    print(f'objective value: {bb.objective_value}')
```
Note, right now I do not currently have this on pypi, so clone the repo and
put the repo's path in a `.pth` file so the above runs.