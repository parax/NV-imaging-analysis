# -*- coding: utf-8 -*-

"""

This worker handles the basic fitting methods for WidefieldProcessor.
- Insert better docstring here to explain:
-- the parallel functions
-- the serialisation requirement, how we've optimised the code
-- least_squares and how it fits, what a residual is (basic explanation)
-- the fitmodel etc.
-- jacobian methods and why you might use it
-- other fit options

"""

# ============================================================================

import fitting.fit_model as FM

# ============================================================================

import numpy as np
from scipy.optimize import least_squares

import concurrent.futures

from tqdm import tqdm
from itertools import repeat
import warnings

# ============================================================================


class WorkerFitting(object):
    """ docstring
    """

    # TODO change input here

    def __init__(self, options):
        self.options = options

        guess_dict, bound_dict = FM.gen_init_guesses(options)
        fit_param_ar, fit_param_bound_ar = FM.gen_fit_params(options, guess_dict, bound_dict)
        self.fit_model = FM.FitModel(
            [options["lineshape"], options["bg_function"]],
            [options["num_peaks"], 1],
            guess_dict,
            bound_dict,
            fit_param_ar,
            fit_param_bound_ar,
        )
        self.options["fit_param_defn"] = self.fit_model.peaks.param_defn
        self.options["fit_parameter_unit"] = self.fit_model.peaks.parameter_unit

    # =================================

    def fit_roi(self, sig_norm, sweep_list):
        """ docstring """

        # fit *all* pl data (i.e. summing over FOV)
        # collapse to just pl_ar (as function of sweep, 1D)
        pl_roi = np.nansum(np.nansum(sig_norm, 2), 1)
        pl_roi = pl_roi / np.max(pl_roi)
        # affine parameter sweep axis vector for plotting fit
        sweep_vector = np.linspace(min(sweep_list), max(sweep_list))

        init_guess = self.fit_model.fit_param_ar
        fit_bounds = (
            self.fit_model.fit_param_bound_ar[:, 0],
            self.fit_model.fit_param_bound_ar[:, 1],
        )

        self.fit_options = {
            "method": self.options["fit_method"],
            "verbose": self.options["verbose_fitting"],
            "gtol": self.options["fit_gtol"],
            "xtol": self.options["fit_xtol"],
            "ftol": self.options["fit_ftol"],
            "loss": self.options["loss_fn"],
        }
        if self.options["fit_method"] != "lm":
            self.fit_options["bounds"] = fit_bounds
            self.fit_options["verbose"] = self.options["verbose_fitting"]

        if self.options["scale_x"]:
            self.fit_options["x_scale"] = "jac"
        else:
            self.options["scale_x"] = False

        # define jacobian option for least_squares fitting
        if self.fit_model.jacobian_scipy is None or not self.options["use_analytic_jac"]:
            self.fit_options["jac"] = self.options["fit_jac_acc"]
        else:
            self.fit_options["jac"] = self.fit_model.jacobian_scipy

        fitting_results = least_squares(
            self.fit_model.residuals_scipy,
            init_guess,
            args=(sweep_list, pl_roi),
            **self.fit_options
        )

        best_fit_result = fitting_results.x
        fit_sweep_vector = np.linspace(np.min(sweep_vector), np.max(sweep_vector), 1000)
        scipy_best_fit = self.fit_model.peaks(fit_sweep_vector, best_fit_result)
        init_fit = self.fit_model.peaks(fit_sweep_vector, init_guess)

        return pl_roi, sweep_vector, best_fit_result, scipy_best_fit, init_fit, fit_sweep_vector

    # =================================

    def fit_pixels(self, sig_norm, best_fit_result, sweep_list, ROI_shape):
        """ docstring"""
        fit_opt = self.fit_options
        threads = self.options["threads"]
        sweep_ar = np.array(sweep_list)

        data_length = np.shape(sig_norm)[1] * np.shape(sig_norm)[2]

        # this makes low binning work (idk why), else do chunksize = 1
        chunksize = int(data_length / (threads * 100))

        if not chunksize:
            warnings.warn("chunksize was 0, setting to 1")
            chunksize = 1

        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            fit_results = list(
                tqdm(
                    executor.map(
                        to_squares_wrapper,
                        repeat(self.fit_model.residuals_scipy),
                        repeat(best_fit_result),
                        repeat(sweep_ar),
                        my_gen(sig_norm),
                        repeat(fit_opt),
                        chunksize=chunksize,
                    ),
                    ascii=True,
                    mininterval=1,
                    total=data_length,
                    unit=" PX",
                    disable=(not self.options["show_progressbar"]),
                )
            )

        # alternate method:
        # import multiprocessing as mp
        # with mp.Pool(threads) as pool:
        #     fit_results = list(
        #         tqdm(
        #             pool.imap(
        #                 to_squares_wrapper_mp,
        #                 zip(
        #                     repeat(self.fit_model.residuals_scipy),
        #                     repeat(best_fit_result),
        #                     repeat(sweep_ar),
        #                     my_gen(sig_norm),
        #                     repeat(fit_opt),
        #                 ),
        #                 chunksize=chunksize,
        #             ),
        #             total=data_length,
        #             ascii=True,
        #             unit=" PX",
        #             disable=(not self.options["show_progressbar"]),
        #         )
        #     )

        self.fit_image_results = FM.get_pixel_fitting_results(
            self.fit_model, fit_results, ROI_shape
        )

    # =================================

    def save_fitted_data(self):
        """ Save fit param arrays as txt files
        """
        for parameter_key in self.fit_model.peaks.param_defn:
            for idx in range(self.options["num_peaks"]):
                np.savetxt(
                    self.options["output_dir"]
                    + "/peak_"
                    + str(parameter_key)
                    + "_"
                    + str(idx + 1)
                    + ".txt",
                    self.fit_image_results[parameter_key][idx, :, :],
                )


def to_squares_wrapper(fun, p0, sweep_val, pl_val, kwargs={}):
    return ((pl_val[0], pl_val[1]), least_squares(fun, p0, args=(sweep_val, pl_val[2]), **kwargs))


def to_squares_wrapper_mp(args):
    fun, p0, sweep_val, pl_val, kwargs = args
    return ((pl_val[0], pl_val[1]), least_squares(fun, p0, args=(sweep_val, pl_val[2]), **kwargs))


def my_gen(our_array):
    len_z, len_x, len_y = np.shape(our_array)
    for x in range(len_x):
        for y in range(len_y):
            yield [x, y, our_array[:, x, y]]


# =================================
# for reference:

# def fit_pixels_old(self, sig_norm, best_fit_result, sweep_list, ROI_shape):
#     """ docstring """

#     sweep_ar = np.array(sweep_list)
#     sig_norm_cpy = sig_norm
#     best_fit_result_cpy = best_fit_result
#     fit_opt = self.fit_options
#     threads = self.options["threads"]

#     pl_ar_map_temp = {
#         str(a) + "," + str(b): [(a, b), sig_norm_cpy[:, a, b]]
#         for (a, b) in np.ndindex((sig_norm_cpy[0].shape))
#     }
#     pl_ar_map = {k: v for k, v in pl_ar_map_temp.items() if ~np.isnan(v[1]).any()}

#     fit_results = []

#     with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
#         for idx, pl_data in pl_ar_map.values():
#             fit_opt["args"] = (sweep_ar, pl_data)
#             fit_results += [
#                 executor.submit(
#                     least_squares,
#                     *[self.fit_model.residuals_scipy, best_fit_result_cpy],
#                     **fit_opt
#                 )
#             ]
#             fit_results += [idx]
#         # progress bar that works in parallel
#         for a in tqdm(
#             concurrent.futures.as_completed(fit_results[::2]),
#             total=len(fit_results[::2]),
#             ascii=True,
#             unit=" PX",
#             disable=(not self.options["show_progressbar"]),
#         ):
#             pass

#     self.fit_image_results = FM.get_pixel_fitting_results(
#         self.fit_model, fit_results, ROI_shape
#     )
