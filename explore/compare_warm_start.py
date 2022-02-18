import itertools

from coinor.cuppy.milpInstance import MILPInstance
import numpy as np
import os
import pandas as pd
from pathlib import Path
import time

from simple_mip_solver import BranchAndBound, PseudoCostBranchNode,\
    CuttingPlaneBoundPseudoCostBranchNode as CPBPCBNode
from simple_mip_solver.utils.cut_generating_lp import CutGeneratingLP


def main(cut_offs, in_fldr, out_file='warm_start_comparison.csv', mip_gap=.01):
    # check cut offs sorted in increasing order
    assert ((np.array([0] + cut_offs)) < (np.array(cut_offs + [float('inf')]))).all(), \
        'please put cut off sizes in increasing order'

    # delete output file if it exists
    Path(out_file).unlink(missing_ok=True)

    # I think I just need to create the different run options in the same test
    for i, file in enumerate(os.listdir(in_fldr)):
        print(f'\nrunning test {i + 1}')
        warm_bb = {}
        data = {}
        pth = os.path.join(in_fldr, file)
        model = MILPInstance(file_name=pth)
        # cold started branch and bound
        cold_bb = BranchAndBound(model, PseudoCostBranchNode, pseudo_costs={},
                                 mip_gap=mip_gap)

        if (i != 3):
            continue

        for cut_off in cut_offs:

            # solve cold start branch and bound to the current cut off
            cold_bb.node_limit = cut_off
            cold_bb.solve()

            # generate cglp for warm started instances
            start = time.process_time()
            cglp = CutGeneratingLP(bb=cold_bb, root_id=0)
            pi, pi0 = cglp.solve()
            cglp_init_time = time.process_time() - start

            for cglp_constraints, cglp_bounds in itertools.product(['cumulative', 'fixed'], ['cumulative', 'fixed']):
                # set the key (k) so as we add more pks the code stays readable
                k = (cut_off, cglp_constraints, cglp_bounds)
                print(k)
                if k != (16, 'cumulative', 'cumulative'):
                    continue

                # warm start branch and bound with disjunctive cut after <c> nodes
                # reinstantiate to avoid cuts sticking to underlying LP when recycling
                A = np.append(cold_bb.root_node.lp.coefMatrix.toarray(), [pi], axis=0)
                b = np.append(cold_bb.root_node.lp.constraintsLower, [pi0], axis=0)
                warm_model = MILPInstance(
                    A=A, b=b, c=cold_bb.root_node.lp.objective,
                    l=cold_bb.root_node.lp.variablesLower.copy(),
                    u=cold_bb.root_node.lp.variablesUpper.copy(), sense=['Min', '>='],
                    integerIndices=cold_bb.root_node._integer_indices,
                    numVars=cold_bb.root_node.lp.nVariables
                )

                # get data to compare starts and progress after <c> node evaluations
                # for both warm and cold starts
                warm_bb[k] = BranchAndBound(
                    warm_model, CPBPCBNode, node_limit=cut_off, pseudo_costs={},
                    mip_gap=mip_gap, cglp=cglp, cut_generating_lp=True,
                    cglp_cumulative_constraints=cglp_constraints == 'cumulative',
                    cglp_cumulative_bounds=cglp_bounds == 'cumulative'
                )
                warm_bb[k].solve()
                # get data on warm start up to cut off
                data[k] = {
                    'variables': cold_bb.root_node.lp.nVariables,
                    'constraints': cold_bb.root_node.lp.nConstraints,
                    'elements': np.sum(cold_bb.root_node.lp.coefMatrix.toarray() != 0),
                    'cold initial dual bound': cold_bb.root_node.objective_value,
                    'warm initial dual bound': warm_bb[k].root_node.objective_value,
                    'cold cut off dual bound': cold_bb.dual_bound,
                    'warm cut off dual bound': warm_bb[k].dual_bound,
                    'cut off time': cold_bb.solve_time,
                    'cglp init time': cglp_init_time
                }

                # get data on warm start termination
                warm_bb[k].node_limit = float('inf')
                warm_bb[k].solve()
                data[k]['warm evaluated nodes'] = warm_bb[k].evaluated_nodes
                data[k]['warm solve time'] = warm_bb[k].solve_time
                data[k]['total restart solve time'] = data[k]['cut off time'] + \
                    data[k]['cglp init time'] + warm_bb[k].solve_time
                data[k]['total restart evaluated nodes'] = cold_bb.evaluated_nodes + \
                    warm_bb[k].evaluated_nodes
                # dual gap - update all these for that
                data[k]['warm initial gap'] = \
                    abs(warm_bb[k].objective_value - data[k]['warm initial dual bound']) / \
                    abs(warm_bb[k].objective_value)
                data[k]['warm cut off gap'] = \
                    abs(warm_bb[k].objective_value - data[k]['warm cut off dual bound']) / \
                    abs(warm_bb[k].objective_value)
                data[k]['warm objective value'] = warm_bb[k].objective_value
                data[k]['failed cglps'] = len([
                    n for n in warm_bb[k].tree.get_node_instances(warm_bb[k].tree.nodes)
                    if n.lp_feasible and not n.mip_feasible and n.cglp.cylp_failure
                ])
                data[k]['null cglps'] = len([
                    n for n in warm_bb[k].tree.get_node_instances(warm_bb[k].tree.nodes)
                    if n.lp_feasible and not n.mip_feasible and not n.cglp_cut_added
                ])
                data[k]['run cglps'] = len([
                    n for n in warm_bb[k].tree.get_node_instances(warm_bb[k].tree.nodes)
                    if n.lp_feasible and not n.mip_feasible
                ])

        # get data on cold start termination
        cold_bb.node_limit = float('inf')
        cold_bb.solve()
        for k in data:
            assert cold_bb.dual_bound <= warm_bb[k].primal_bound + .01 and \
                   cold_bb.primal_bound + .01 >= warm_bb[k].dual_bound, \
                   'gaps should overlap'
            data[k]['cold initial gap'] = \
                abs(cold_bb.objective_value - data[k]['cold initial dual bound']) / \
                abs(cold_bb.objective_value)
            data[k]['cold cut off gap'] = \
                abs(cold_bb.objective_value - data[k]['cold cut off dual bound']) / \
                abs(cold_bb.objective_value)
            data[k]['cold evaluated nodes'] = cold_bb.evaluated_nodes
            data[k]['cold solve time'] = cold_bb.solve_time
            data[k]['cold objective value'] = cold_bb.objective_value
            data[k]['initial gap improvement ratio'] = \
                (data[k]['cold initial gap'] - data[k]['warm initial gap']) / \
                data[k]['cold initial gap']
            data[k]['cut off gap improvement ratio'] = \
                (data[k]['cold cut off gap'] - data[k]['warm cut off gap']) / \
                data[k]['cold cut off gap']
            data[k]['warm evaluated nodes ratio'] = \
                (data[k]['cold evaluated nodes'] - data[k]['warm evaluated nodes']) / \
                data[k]['cold evaluated nodes']
            data[k]['warm solve time ratio'] = \
                (data[k]['cold solve time'] - data[k]['warm solve time']) / \
                data[k]['cold solve time']
            data[k]['total restart evaluated nodes ratio'] = \
                (data[k]['cold evaluated nodes'] - data[k]['total restart evaluated nodes']) / \
                data[k]['cold evaluated nodes']
            data[k]['total restart solve time ratio'] = \
                (data[k]['cold solve time'] - data[k]['total restart solve time']) / \
                data[k]['cold solve time']

        # append this test to our file
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.names = ['cut off', 'cglp_constraints', 'cglp_bounds']
        df.reset_index(inplace=True)
        df['test number'] = [i]*len(data)

        # rearrange columns
        cols = [
            # test differentiators (cuts/bounds are cumulative or fixed)
            'test number', 'cut off', 'cglp_constraints', 'cglp_bounds',

            # dual gap comparison after initial (root) node solved
            'initial gap improvement ratio', 'cut off gap improvement ratio',
            'warm evaluated nodes ratio', 'total restart evaluated nodes ratio',
            'warm solve time ratio', 'total restart solve time ratio',

            # dual gap comparison after <cut off> nodes solved
            'cold objective value', 'cold initial dual bound', 'cold initial gap',
            'cold cut off dual bound', 'cold cut off gap',
            'warm objective value', 'warm initial dual bound', 'warm initial gap',
            'warm cut off dual bound', 'warm cut off gap',

            # node evaluation comparison at <mip_gap> mip_gap
            'cold evaluated nodes', 'warm evaluated nodes', 'total restart evaluated nodes',

            # run time comparison at <mip_gap> mip_gap
            'cold solve time', 'cut off time', 'cglp init time', 'warm solve time',
            'total restart solve time',

            # checks on reliability and application of cglps
            'failed cglps', 'null cglps', 'run cglps'
        ]
        df = df[cols]
        with open(out_file, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False)


if __name__ == '__main__':
    in_fldr = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scale_8_models')
    main([4, 16], in_fldr)
