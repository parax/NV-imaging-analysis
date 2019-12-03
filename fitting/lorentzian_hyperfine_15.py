# -*- coding: utf-8 -*-
"""
Sum of abitary number of Lorentzian functions.
"""

from fitting.func_class import FitFunc
from numba import njit


class Lorentzian_hyperfine_15(FitFunc):
    """ Sum of abitary number of lorentzian Class.
    Define number of lorentzians on instansiation.
    Call instance as instance(x, inital_paramater_list) where
    inital_parameter_list is a list of inital guesses for each lorenzian's
    fwhm, pos, amp in that order, i.e [fwhm, pos, amp, fwhm2, pos2, amp2...].
    """

    param_defn = ["fwhm", "pos", "amp_1_hyp", "amp_2_hyp"]
    parameter_unit = {
        "fwhm": "Frequency (MHz)",
        "pos": "Frequency (MHz)",
        "amp_1_hyp": "Amplitude (a.u.)",
        "amp_2_hyp": "Amplitude (a.u.)",
    }
    fn_type = "peak"

    def __init__(self, num_peaks):
        super().__init__(num_peaks)

    # =================================

    # A15 para = 3.03 MHz
    @staticmethod
    @njit(fastmath=True)
    def base_function(x, fwhm, pos, amp_1_hyp, amp_2_hyp):
        hwhmsqr = fwhm ** 2 / 4
        return amp_1_hyp * hwhmsqr / (
            (x - pos - (3.03 / 2)) ** 2 + hwhmsqr
        ) + amp_2_hyp * hwhmsqr / ((x - pos + (3.03 / 2)) ** 2 + hwhmsqr)
