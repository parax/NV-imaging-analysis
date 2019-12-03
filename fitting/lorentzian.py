# -*- coding: utf-8 -*-
"""
Sum of abitary number of Lorentzian functions.
"""

from fitting.func_class import FitFunc
import numpy as np
from numba import njit


class Lorentzian(FitFunc):
    """ Sum of abitary number of lorentzian Class.
    Define number of lorentzians on instansiation.
    Call instance as instance(x, inital_paramater_list) where
    inital_parameter_list is a list of inital guesses for each lorenzian's
    fwhm, pos, amp in that order, i.e [fwhm, pos, amp, fwhm2, pos2, amp2...].
    """

    param_defn = ["fwhm", "pos", "amp"]
    parameter_unit = {"fwhm": "Freq (MHz)", "pos": "Freq (MHz)", "amp": "Amp (a.u.)"}
    fn_type = "peak"

    #    def __init__(self, num_peaks):
    #        super().__init__(num_peaks)

    # =================================

    # fastmath gives a ~10% speed up on my testing
    @staticmethod
    @njit(fastmath=True)
    def base_fn(x, fwhm, pos, amp):
        hwhmsqr = (fwhm ** 2) / 4
        return amp * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)

    # =================================

    @staticmethod
    @njit
    def grad_fn(x, fwhm, pos, amp):
        """ Compute the grad of the residue, excluding PL as a param
        {output shape: (len(x), 3)}
        """
        # Lorentzian: a*g^2/ ((x-c)^2 + g^2)
        J = np.empty((x.shape[0], 3), dtype=np.float32)
        g = fwhm / 2
        c = pos
        a = amp
        J[:, 0] = ((2 * a * g) / (g ** 2 + (x - c) ** 2)) - (
            (2 * a * g ** 3) / (g ** 2 + (x - c) ** 2) ** 2
        )
        J[:, 1] = (2 * a * g ** 2 * (x - c)) / (g ** 2 + (x - c) ** 2) ** 2
        J[:, 2] = g ** 2 / ((x - c) ** 2 + g ** 2)
        return J
