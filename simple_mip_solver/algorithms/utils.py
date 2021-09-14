from coinor.cuppy.milpInstance import MILPInstance
from cylp.py.utils.sparseUtil import csc_matrixPlus
import inspect
from typing import Any, List, TypeVar, Dict, Tuple

U = TypeVar('U', bound='Utils')


class Utils:
    """Parent class to those used to solve Mixed Integer Linear Programs. Contains
    utility functions useful across multiple classes of algorithms"""
    def __init__(self: U, model: MILPInstance, Node: Any, node_attributes: List[str],
                 node_funcs: List[str], **kwargs: Any):
        # model asserts
        assert isinstance(model, MILPInstance), 'model must be cuppy MILPInstance'
        self.model = self._convert_constraints_to_greq(model)
        self._swapped_constraint_direction = model.sense != self.model.sense

        # Node asserts
        assert inspect.isclass(Node), 'Node must be a class'
        # ensures Node constructor has the args we need and no other required ones
        root_node = Node(lp=self.model.lp, integer_indices=self.model.integerIndices,
                         lower_bound=-float('inf'), idx=0)
        for attribute in node_attributes:
            assert hasattr(root_node, attribute), f'Node needs a {attribute} attribute'
        for func in node_funcs:
            c = getattr(root_node, func, None)
            assert callable(c), f'Node needs a {func} function'

        # kwargs asserts
        assert 'next_node_idx' not in kwargs, 'key next_node_idx is reserved for use by solver'

        # instantiate
        self._Node = Node
        self._root_node = root_node
        self._evaluated_nodes = 0
        kwargs['next_node_idx'] = 1  # put in kwargs so node methods can update this number
        self._kwargs = kwargs
        self._M = 999999999

    @staticmethod
    def _convert_constraints_to_greq(model: MILPInstance) -> MILPInstance:
        """ If constraints are of the form A <= b, convert them to A >= b

        :return: Updated MILPInstance with constraints turned around
        """
        if model.sense == '<=':
            # all problems converted to minimization via lp.objective in MILPInstance init
            if isinstance(model.A, csc_matrixPlus):
                model.A = model.A.toarray()
            return MILPInstance(A=-model.A, b=-model.b, c=model.lp.objective,
                                l=model.l, u=model.u, integerIndices=model.integerIndices,
                                sense=['Min', '>='], numVars=len(model.c))
        else:
            return model

    def _process_rtn(self: U, rtn: Dict[str, Any]):
        """ Assign the values of <rtn> to their keyed attributes

        :param rtn:
        :return:
        """
        assert isinstance(rtn, dict), 'rtn must be a dictionary'
        assert all(isinstance(k, str) for k in rtn), 'rtn keys must be strings'
        for k, v in rtn.items():
            self._kwargs[k] = v
