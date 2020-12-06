numVars = 2
numCons = 5
A = [#[4, 1],
     [1, 4],
#     [1, -1],
     [-1, 0],
     [0, -1]] 

b = [#28,
     27,
#     1,
     0,
     0]

c = [2, 5]
obj_val = 2 
sense = ('Max', '<=')
integerIndices = [0, 1]

if __name__ == '__main__':

    try:
        from coinor.cuppy.cuttingPlanes import solve, gomoryCut
        from coinor.cuppy.milpInstance import MILPInstance
    except ImportError:
        from src.cuppy.cuttingPlanes import solve, gomoryCut
        from src.cuppy.milpInstance import MILPInstance

    m = MILPInstance(A = A, b = b, c = c, 
                     sense = sense, integerIndices = integerIndices,
                     numVars = numVars)
    
    solve(m, whichCuts = [(gomoryCut, {})], display = True, debug_print = True)
