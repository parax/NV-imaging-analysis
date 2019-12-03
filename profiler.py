# -*- coding: utf-8 -*-

"""
Frame work for testing and validation of code.

NOTE turn off figure plotting if you want accurate timings (the time the plot is up before
you close it counts as time spent in script)
"""


__author__ = "Nikolai Dontschuk"

import basic.processor as WP
import basic.worker_plotting as WrkPlt
import basic.misc as misc

from fitting.fit_model import gen_init_guesses, gen_fit_params, FitModel

from scipy.optimize import least_squares

import numpy as np
import sys
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================================


def main(__spec__=None):

    # get command line argument (which case to profile)
    if len(sys.argv) > 1:
        case = int(sys.argv[1])
    else:
        # if case is None (no arguments supplied at command line) then it runs the first one
        case = 1

    if case < 10:
        profile_widefield(case)
    elif case >= 10 and case < 20:
        profile_single_sweep(case)
    else:
        # to be added for profiling reconstruction etc. Reserve 20-29 for reconstruction...
        raise NotImplementedError


# ============================================================================


def profile_widefield(case):
    # TODO changed / to \\? replace with PurePath
    if case is None or case == 1:
        test_options = misc.json_to_dict("tests/test_low_field_options.json")

    elif case == 2:
        test_options = misc.json_to_dict("tests/test_single_peak_options.json")
    elif case == 3:
        test_options = misc.json_to_dict("tests/test_8peak_options.json")
    else:
        raise NotImplementedError("that's not a case we've implemented yet...")

    if not Path.is_file(Path(test_options["filepath"])) or not Path.is_file(
        Path(test_options["filepath"] + "_metaSpool.txt")
    ):
        raise FileNotFoundError(
            "expected test files to be in the project root's parent folder, under a folder called "
            + "'test_data'. The files expected, etc. are noted in the \\docs folder under "
            + "'test_suite."
        )

    print(
        f"running case {case} (profile_widefield) with {test_options['num_bins']} binning, "
        + f"fit_pixels: {test_options['fit_pixels']}, analytic jacobian: "
        + f"{test_options['use_analytic_jac']}\n"
    )

    wp = WP.Processor(test_options)

    wp.process_file()

    wp.fit_data()

    if test_options["make_plots"]:

        plt_image = WrkPlt.Image(test_options)
        plt_image(filename="PL_previous_binning", pl_image=True)
        plt_image.single_image(wp.dw.image, "PL_image", cbar_title="PL (counts)")

    if test_options["show_plots"]:
        plt.show()
    misc.dict_to_json(wp.options, "saved_options.json", wp.options["output_dir"])


# ============================================================================


def profile_single_sweep(case):

    if case == 10:
        test_options = misc.json_to_dict("tests\\test_single_sweep_options.json")
        test_data = np.loadtxt("..\\test_data\\single_sweep.txt").T

    print(
        f"running case {case} (profile_single_sweep), "
        + f"analytic jacobian: {test_options['use_analytic_jac']}"
    )

    # ====================================================================

    test_guess_dict, test_bound_dict = gen_init_guesses(test_options)

    test_fit_param_ar, test_fit_param_bound_ar = gen_fit_params(
        test_options, test_guess_dict, test_bound_dict
    )

    test_FM = FitModel(
        ["lorentzian", "linear"],
        [8, 1],
        test_guess_dict,
        test_bound_dict,
        test_fit_param_ar,
        test_fit_param_bound_ar,
    )

    if test_options["make_plots"]:
        fig = plt.figure("Single Sweep Fit and Residual")
        frame1 = fig.add_axes((0.1, 0.3, 0.8, 0.6))
        frame1.set_xticklabels([])
        plt.plot(test_data[0], test_data[1], ls="", marker="+", ms=2, label="Test Spectra")

        x = np.linspace(min(test_data[0]), max(test_data[0]), 1000)
        plt.plot(
            x, test_FM.peaks(x, test_FM.fit_param_ar), ls="--", ms=2, c="k", label="Initial Guess"
        )

    if test_options["use_analytic_jac"]:
        jac = test_FM.jacobian_scipy
    else:
        jac = "2-point"

    for i in range(500):
        fitting_results = least_squares(
            test_FM.residuals_scipy, test_fit_param_ar, args=(test_data[0], test_data[1]), jac=jac
        )

    if test_options["make_plots"]:
        plt.plot(x, test_FM.peaks(x, fitting_results.x), ls="-", c="r", label="Scipy Best Fit")
        plt.legend(frameon=False)
        frame2 = fig.add_axes((0.1, 0.1, 0.8, 0.2))
        plt.scatter(test_data[0], fitting_results.fun, c="#06470c", s=1, label="Residual")
        plt.legend(frameon=False, loc="upper left")

    if test_options["make_plots"] and test_options["show_plots"]:
        plt.show()


# ============================================================================

if __name__ == "__main__":
    main()
