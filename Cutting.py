import numpy as np


class GomoryMIR():
    def __init__(self):
        pass

    def cutting(self, model, node):
        cuts = []
        lp = node.cylp
        sol = lp.primalVariableSolution['x']
        rowInds = list(range(lp.nConstraints))
        flag_Integer = np.concatenate((model.int_index_array, np.array([False] * lp.nConstraints)))
        flag_notInteger = ~flag_Integer
        flag_notBasic = lp.varNotBasic
        flag_Basic = lp.varIsBasic
        for row in rowInds:
            basicVarInd = lp.basicVariables[row]
            if (basicVarInd in model.int_index) and (
                    np.abs(np.around(sol[basicVarInd]) - sol[basicVarInd]) > model.episilon):
                f0 = sol[basicVarInd] - np.floor(sol[basicVarInd])
                row_tableau = lp.tableau[row]
                f = row_tableau - np.floor(row_tableau)
                f[flag_Basic] = 0

                fj_le = f * ((f <= f0) & flag_Integer & flag_notBasic)
                fj_g = f0 / (1 - f0) * (1 - f) * ((f > f0) & flag_Integer & flag_notBasic)
                aj_g = row_tableau * ((row_tableau > 0) & flag_notInteger & flag_notBasic)
                aj_l = -f0 / (1 - f0) * row_tableau * ((row_tableau < 0) & flag_notInteger & flag_notBasic)
                coeff = fj_le + fj_g + aj_g + aj_l

                pi = coeff[:lp.nVariables] - coeff[lp.nVariables:] * lp.coefMatrix
                pi0 = f0 - np.dot(coeff[lp.nVariables:], lp.constraintsUpper)
                cuts.append((-pi, -pi0))
        return cuts, []


Cuttings = {'GomoryMIR': GomoryMIR}
