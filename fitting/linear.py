# -*- coding: utf-8 -*-
"""
Linear.
"""

from fitting.func_class import FitFunc
import numpy as np
from numba import njit


class Linear(FitFunc):
    """ Constant
    """

    param_defn = ["c", "m"]
    parameter_unit = {"c": "Amplitude (a.u.)", "m": "Amplitude per Freq (a.u.)"}
    fn_type = "bground"

    #    def __init__(self, num_peaks):
    #        super().__init__(num_peaks)

    # =================================

    # speed tested, marginally faster with fastmath off (idk why)
    @staticmethod
    @njit(fastmath=False)
    def base_fn(x, c, m):
        return m * x + c

    # =================================

    @staticmethod
    @njit(fastmath=True)
    def grad_fn(x, c, m):
        """ Compute the grad of the residue, excluding PL as a param
        {output shape: (len(x), 1)}
        """
        J = np.empty((x.shape[0], 2))
        J[:, 0] = 1
        J[:, 1] = x
        return J


# -*- coding: utf-8 -*-
