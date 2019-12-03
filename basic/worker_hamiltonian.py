# -*- coding: utf-8 -*-

"""
Module docstring here

"""
__author__ = "David Broadway"
# ============================================================================

from ham_fitting.ham_model_generator import HamModelGenerator as HMG

# ============================================================================

import numpy as np
from scipy.optimize import least_squares
import concurrent.futures
from tqdm import tqdm
import skimage as sciimg

# ============================================================================


class WorkerHamiltonian(object):
    """ docstring
    """

    def __init__(self, options):
        self.options = options
        self.ham_model = HMG(self.options)

    # =================================

    def setup_fit(self):
        self.init_guess = self.ham_model.fit_param_list
        self.fit_bounds = (
            self.ham_model.fit_param_bound_list[:, 0],
            self.ham_model.fit_param_bound_list[:, 1],
        )

        # fit_options as fed to least_squares
        # self.fit_options = {"method": self.options["fit_method"]}
        self.fit_options = {
            "method": self.options["fit_method"],
            "verbose": self.options["verbose_fitting"],
            "gtol": self.options["fit_gtol"],
            "xtol": self.options["fit_xtol"],
            "ftol": self.options["fit_ftol"],
            "loss": self.options["loss_fn"],
        }

    # =================================

    def recon_linecut(self, line_cut_data):
        """ docstring """
        if np.isnan(line_cut_data).any():
            raise ValueError(
                "There appears to be NaNs in your line cut. Check that the line cut position and "
                "width are contained within the image size."
            )
        self.setup_fit()
        if "approx_bxyz" in self.options["recon_method"]:
            return None
        self.line_cut_fit_results = np.zeros(
            (len(self.ham_model.model.param_defn), len(line_cut_data[0]))
        )
        for pixel in range(len(line_cut_data[0])):
            self.fit_options["args"] = [
                [line_cut_data[idx][pixel] for idx in range(len(line_cut_data))]
            ]
            fitting_results = least_squares(
                self.ham_model.residuals_scipy, self.init_guess, **self.fit_options
            )
            self.line_cut_fit_results[:, pixel] = fitting_results.x
        return self.line_cut_fit_results

    # =================================

    def recon_pixels(self, data, options):
        # making copies of everything to minimise pickling other classes in
        # ProcessPoolExecutor (i.e. WP,DW,FW)
        import copy

        shape = data[0, ::].shape
        data_cpy = data.copy()
        init_guess = self.init_guess.copy()
        fit_opt = self.fit_options.copy()
        threads = copy.copy(options["threads"])
        data_map_temp = {
            str(a) + "," + str(b): [(a, b), data_cpy[:, a, b]]
            for (a, b) in np.ndindex((data_cpy[1].shape))
        }
        data_map = {k: v for k, v in data_map_temp.items() if ~np.isnan(v[1]).any()}
        fit_results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            for idx, data in data_map.values():
                fit_opt["args"] = [data.tolist()]
                fit_results += [
                    executor.submit(
                        least_squares, *[self.ham_model.residuals_scipy, init_guess], **fit_opt
                    )
                ]
                fit_results += [idx]
            # progress bar that works in parallel
            for a in tqdm(
                concurrent.futures.as_completed(fit_results[::2]),
                total=len(fit_results[::2]),
                ascii=True,
                unit=" PX",
                disable=(not self.options["show_progressbar"]),
            ):
                pass
        self.fit_image_results = self.get_pixel_fitting_results(fit_results, shape)

        return self.fit_image_results

    # =================================

    def get_pixel_fitting_results(self, fit_results, shape):
        fit_image_results = {}
        for parameter_key in self.ham_model.model.param_defn:
            fit_image_results[parameter_key] = np.zeros(shape) * np.nan
        for (x, y), result in zip(fit_results[1::2], fit_results[::2]):
            idx = 0
            for parameter_key in self.ham_model.model.param_defn:
                fit_image_results[parameter_key][x, y] = result.result().x[idx]
                idx += 1
        return fit_image_results

    # =================================

    def save_fitted_data(self):
        """ Save fit param arrays as txt files
        """
        for parameter_key in self.fit_model.peaks.parameter_definition:
            for idx in range(self.options["num_peaks"]):
                np.savetxt(
                    self.output_dir + "/peak_" + str(parameter_key) + "_" + str(idx + 1) + ".txt",
                    self.fit_image_results[parameter_key][idx, :, :],
                )

    # =================================

    def edge_detection(self, image):
        edges = sciimg.filters.sobel(image)
        return edges
