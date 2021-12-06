from coinor.cuppy.milpInstance import MILPInstance
import numpy as np
import os
import pandas as pd
from pathlib import Path
import time

from simple_mip_solver import BranchAndBound, PseudoCostBranchNode


def main(cut_offs, in_fldr, out_file='warm_start_comparison.csv'):
    assert ((np.array([0] + cut_offs)) < (np.array(cut_offs + [float('inf')]))).all(), \
        'please put cut off sizes in increasing order'
    Path(out_file).unlink(missing_ok=True)  # delete output file if it exists

    for i, file in enumerate(os.listdir(in_fldr)):
        print(f'running test {i + 1}')
        warm_bb = {}
        data = {}
        pth = os.path.join(in_fldr, file)
        model = MILPInstance(file_name=pth)
        # cold started branch and bound
        cold_bb = BranchAndBound(model, PseudoCostBranchNode, pseudo_costs={})

        for c in cut_offs:
            cold_bb.node_limit = c
            cold_bb.solve()
            start = time.process_time()
            pi, pi0 = cold_bb.find_strong_disjunctive_cut(0)
            cglp_time = time.process_time() - start

            # warm start branch and bound with disjunctive cut after <c> nodes
            A = np.append(cold_bb.root_node.lp.coefMatrix.toarray(), [pi], axis=0)
            b = np.append(cold_bb.root_node.lp.constraintsLower, pi0, axis=0)
            warm_model = MILPInstance(
                A=A, b=b, c=cold_bb.root_node.lp.objective,
                l=cold_bb.root_node.lp.variablesLower.copy(),
                u=cold_bb.root_node.lp.variablesUpper.copy(), sense=['Min', '>='],
                integerIndices=cold_bb.root_node._integer_indices,
                numVars=cold_bb.root_node.lp.nVariables
            )

            # get data to compare starts and progress after <c> node evaluations
            # for both warm and cold starts
            warm_bb[c] = BranchAndBound(warm_model, PseudoCostBranchNode, node_limit=c, pseudo_costs={})
            warm_bb[c].solve()
            data[c] = {
                'cold initial lower bound': cold_bb.root_node.objective_value,
                'warm initial lower bound': warm_bb[c].root_node.objective_value,
                'cold cut off lower bound': cold_bb.dual_bound,
                'warm cut off lower bound': warm_bb[c].dual_bound,
                'cut off time': cold_bb.solve_time,
                'cglp time': cglp_time
            }

            # get data on warm start termination
            warm_bb[c].node_limit = float('inf')
            warm_bb[c].solve()
            data[c]['warm evaluated nodes'] = warm_bb[c].evaluated_nodes
            data[c]['warm solve time'] = warm_bb[c].solve_time
            data[c]['total restart solve time'] = data[c]['cut off time'] + \
                data[c]['cglp time'] + warm_bb[c].solve_time
            data[c]['total restart evaluated nodes'] = cold_bb.evaluated_nodes + \
                warm_bb[c].evaluated_nodes
            # dual gap - update all these for that
            data[c]['warm initial gap'] = \
                abs(warm_bb[c].objective_value - data[c]['warm initial lower bound']) / \
                abs(warm_bb[c].objective_value)
            data[c]['warm cut off gap'] = \
                abs(warm_bb[c].objective_value - data[c]['warm cut off lower bound']) / \
                abs(warm_bb[c].objective_value)
            data[c]['warm objective value'] = warm_bb[c].objective_value

        # get data on cold start termination
        cold_bb.node_limit = float('inf')
        cold_bb.solve()
        for c in cut_offs:
            assert cold_bb.dual_bound <= warm_bb[c].primal_bound + .01 and \
                   cold_bb.primal_bound + .01 >= warm_bb[c].dual_bound, \
                   'gaps should overlap'
            data[c]['cold initial gap'] = \
                abs(cold_bb.objective_value - data[c]['cold initial lower bound']) / \
                abs(cold_bb.objective_value)
            data[c]['cold cut off gap'] = \
                abs(cold_bb.objective_value - data[c]['cold cut off lower bound']) / \
                abs(cold_bb.objective_value)
            data[c]['cold evaluated nodes'] = cold_bb.evaluated_nodes
            data[c]['cold solve time'] = cold_bb.solve_time
            data[c]['cold objective value'] = cold_bb.objective_value
            data[c]['initial gap improvement ratio'] = \
                (data[c]['cold initial gap'] - data[c]['warm initial gap']) / \
                data[c]['cold initial gap']
            data[c]['cut off gap improvement ratio'] = \
                (data[c]['cold cut off gap'] - data[c]['warm cut off gap']) / \
                data[c]['cold cut off gap']
            data[c]['warm evaluated nodes ratio'] = \
                (data[c]['cold evaluated nodes'] - data[c]['warm evaluated nodes']) / \
                data[c]['cold evaluated nodes']
            data[c]['warm solve time ratio'] = \
                (data[c]['cold solve time'] - data[c]['warm solve time']) / \
                data[c]['cold solve time']
            data[c]['total restart evaluated nodes ratio'] = \
                (data[c]['cold evaluated nodes'] - data[c]['total restart evaluated nodes']) / \
                data[c]['cold evaluated nodes']
            data[c]['total restart solve time ratio'] = \
                (data[c]['cold solve time'] - data[c]['total restart solve time']) / \
                data[c]['cold solve time']

        # append this test to our file
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.names = ['cut off']
        df.reset_index(inplace=True)
        df['test number'] = [i]*len(cut_offs)

        # rearrange columns
        cols = [
            'test number', 'cut off',

            'initial gap improvement ratio', 'cut off gap improvement ratio',
            'warm evaluated nodes ratio', 'total restart evaluated nodes ratio',
            'warm solve time ratio', 'total restart solve time ratio',

            'cold objective value', 'cold initial lower bound', 'cold initial gap',
            'cold cut off lower bound', 'cold cut off gap',
            'warm objective value', 'warm initial lower bound', 'warm initial gap',
            'warm cut off lower bound', 'warm cut off gap',

            'cold evaluated nodes', 'warm evaluated nodes', 'total restart evaluated nodes',
            
            'cold solve time', 'cut off time', 'cglp time', 'warm solve time',
            'total restart solve time'
        ]
        df = df[cols]
        with open(out_file, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False)


if __name__ == '__main__':
    in_fldr = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scale_8_models')
    main([4, 16], in_fldr)














