# -*- coding: utf-8 -*-
"""
Sum of abitary number of Lorentzian functions.

-- Renamed as FitModel (as Generators are a specific thing in python)
"""

from fitting.lorentzian import Lorentzian
from fitting.lorentzian_hyperfine_14 import Lorentzian_hyperfine_14
from fitting.lorentzian_hyperfine_15 import Lorentzian_hyperfine_15
from fitting.constant import Constant
from fitting.linear import Linear
import matplotlib.pyplot as plt
import numpy as np

available_peak_types = {
    "lorentzian": Lorentzian,
    "lorentzian_hyperfine_14": Lorentzian_hyperfine_14,
    "lorentzian_hyperfine_15": Lorentzian_hyperfine_15,
    "constant": Constant,
    "linear": Linear,
}

# ============================================================================


class FitModel:
    """ Sum of abitary number of lorentzian Class.
    Define number of lorentzians on instansiation.
    Call instance as instance(x, inital_paramater_list) where
    inital_parameter_list is a list of inital guesses for each lorenzian's
    fwhm, pos, amp in that order, i.e [fwhm, pos, amp, fwhm2, pos2, amp2...].
    """

    def __init__(self, fn, num_fns, guess_dict, bound_dict, fit_param_ar, fit_param_bound_ar):
        """ num_fns is the number of functions in each lineshape/individual fn we pass """

        # Deal with the user passing single fns not as single lists
        try:
            self.peaks = available_peak_types[fn](num_fns)
            self.fn = [fn]
            self.num_fns = [num_fns]
        except TypeError:
            peak_chain = None
            self.param_keys = []
            # Loading these in reverse fixed some issue but I don't remember what.
            # It shouldn't matter *shurg*
            for single_fn, single_num_fns in zip(fn[::-1], num_fns[::-1]):
                peak_chain = available_peak_types[single_fn](single_num_fns, peak_chain)
            self.peaks = peak_chain
            # we reverse these for simplicity for other users, as the chain is reversed.
            self.fn = fn[::-1]
            self.num_fns = num_fns[::-1]

        self.init_guesses = guess_dict
        self.init_bounds = bound_dict
        self.fit_param_ar = fit_param_ar
        self.fit_param_bound_ar = fit_param_bound_ar

    # =================================

    def residuals_scipy(self, fit_param_ar, sweep_val, pl_val):
        return self.peaks(sweep_val, fit_param_ar) - pl_val

    # =================================

    def jacobian_scipy(self, fit_param_ar, sweep_val, pl_val):
        return self.peaks.jacobian(sweep_val, fit_param_ar)


# ============================================================================
# Data extraction functions

# TODO: Should these be static members or just stay in the madual like this.... not clear to me


def get_pixel_fitting_results(fit_model, fit_results, roi_shape):
    """Take the fit result data and back it down into a dictonary of arrays
    with keys representing the peak parameters (i.e. fwhm, pos, amp).  Each
    array is 3D [z,x,y] with the z dimentions representing the peak number and
    x and y the lateral map positions.  Any added functions (background ect are
    handeled by this in the same way.  I.e. with a linear background there will
    be 'm' and 'c' keys with a shape (1,x,y)."""
    fit_image_results = {}
    # Populate the dictonary with the correct size empty arrays using np.zeros.
    for num_fns, single_fn in zip(fit_model.num_fns, fit_model.fn):
        fn_params = available_peak_types[single_fn].param_defn
        for parameter_key in fn_params:
            fit_image_results[parameter_key] = (
                np.zeros((num_fns, roi_shape[0], roi_shape[1])) * np.nan
            )
    # Fill the arrays element-wise from the results function, which returns a
    # 1D array of flatterned best-fit parameters.
    for (x, y), result in fit_results:  # zip(fit_results[1::2], fit_results[::2]
        # num tracks position in the 1D array, as we loop over different fns
        # and parameters.
        num = 0
        for num_fns, single_fn in zip(fit_model.num_fns, fit_model.fn):
            fn_params = available_peak_types[single_fn].param_defn
            for idx in range(num_fns):
                for parameter_key in fn_params:
                    fit_image_results[parameter_key][idx, x, y] = result.x[
                        num
                    ]  # removed result()  might need to add it back when going multiproccesing
                    num += 1
    return fit_image_results


# =================================


def gen_init_guesses(options):
    guess_dict = {}
    bound_dict = {}
    peaks = available_peak_types[options["lineshape"]](options["num_peaks"])
    for param_key in peaks.param_defn:
        # < TODO > add a auto guess for the peak positions
        guess = options[param_key + "_guess"]
        val = guess
        if param_key + "_range" in options:
            if len(guess) > 1:
                val_b = [
                    [x - options[param_key + "_range"], x + options[param_key + "_range"]]
                    for x in guess
                ]
            else:
                val_b = [
                    guess - options[param_key + "_range"],
                    guess + options[param_key + "_range"],
                ]
        elif param_key + "_bounds" in options:
            val_b = options[param_key + "_bounds"]
        else:
            val_b = [[0, 0]]
        if val is not None:
            guess_dict[param_key] = val
            bound_dict[param_key] = np.array(val_b)
        else:
            raise RuntimeError(
                "I'm not sure what this means... I know "
                + "it's bad though... Don't put 'None' as "
                + "a param guess."
            )
    return guess_dict, bound_dict


# =================================


def gen_fit_params(options, guess_dict, bound_dict):

    num_fns = options["num_peaks"]
    fn = options["lineshape"]
    peaks = available_peak_types[fn](num_fns)

    fit_param_ar = np.array([])
    fit_param_bound_ar = np.array([])

    bg_function_name = options["bg_function"]
    bg_params = options["bg_parameter_guess"]
    bg_bounds = options["bg_parameter_bounds"]

    # TODO generalise for abitary number of BG functions
    for n in range(len(available_peak_types[bg_function_name].param_defn)):
        fit_param_ar = np.append(fit_param_ar, bg_params[n])
        fit_param_bound_ar = np.append(fit_param_bound_ar, bg_bounds[n])

    for n in range(num_fns):
        to_append = np.zeros(len(peaks.param_defn))
        to_append_bounds = np.zeros((len(peaks.param_defn), 2))
        for position, key in enumerate(peaks.param_defn):
            try:
                to_append[position] = guess_dict[key][n]
            except (TypeError, KeyError):
                to_append[position] = guess_dict[key]
            if len(bound_dict[key].shape) == 2:
                to_append_bounds[position] = bound_dict[key][n]
            else:
                to_append_bounds[position] = bound_dict[key]

        fit_param_ar = np.append(fit_param_ar, to_append)
        fit_param_bound_ar = np.append(fit_param_bound_ar, to_append_bounds)
    # This is a messy way to deal with this, why not keep its shape as you append.
    fit_param_bound_ar = fit_param_bound_ar.reshape(
        len(bg_params) + options["num_peaks"] * len(peaks.param_defn), 2
    )

    return fit_param_ar, fit_param_bound_ar


# =================================


def fit_model_test_plot(fit_model, x):
    # self._generate_parameters_list()
    plt.plot(fit_model.peaks(x, fit_model.fit_param_ar))
