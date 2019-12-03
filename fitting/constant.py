# -*- coding: utf-8 -*-
"""
Constant.
"""

from fitting.func_class import FitFunc
import numpy as np
from numba import njit, jit


class Constant(FitFunc):
    """ Constant
    """

    param_defn = ["c"]
    parameter_unit = {"c": "Amplitude (a.u.)"}
    fn_type = "bground"

    # =================================

    @staticmethod
    @njit(fastmath=True)
    def base_fn(x, c):
        """ speed tested multiple methods, this was the fastest """
        return np.empty(np.shape(x)).fill(c)

    # =================================

    @staticmethod
    @njit
    def grad_fn(x, c):
        """ Compute the grad of the residue, excluding PL as a param
        {output shape: (len(x), 1)}
        """
        J = np.empty((x.shape[0], 1))
        J[:, 0] = 0
        return J
