# -*- coding: utf-8 -*-

"""
Module docstring here

"""

__author__ = "David Broadway"

# ============================================================================

import basic.reconstructor as HR
import basic.misc as misc

import matplotlib.pyplot as plt
import os


# ============================================================================


def main(__spec__=None):
    plt.close("all")
    # do this for processing code, different for BNV plotting
    # could eg do both in one run (with different params and WP's)
    options = misc.json_to_dict("options/reconstruction.json")

    hr = HR.Reconstructor(options)

    hr.load_fitted_data()

    hr.bnv_extraction()

    hr.get_all_line_cuts()

    if options["recon_method"] != "BNV":
        hr.fit_linecut(direction="hor")
        hr.fit_linecut(direction="vert")

        hr.fit_pixels()
        hr.save_fitted_data()
        hr.subtract_ref_data()
        hr.plot_results()

    if options["bxyz_propagation"]:
        hr.bxyz_from_bnv(hr.bnv, nv_axis=options["bxyz_prop_bnv_axis"])
        b_prop = {
            "bx": hr.b_xyz_prop[0, ::],
            "by": hr.b_xyz_prop[1, ::],
            "bz": hr.b_xyz_prop[2, ::],
        }

    if options["magnetisation_propagation"]:

        hr.magnetisation_backwards_propagation(
            hr.bnv,
            b_type=options["mag_b_type"],
            bnv_axis=options["mag_bnv_axis"],
            mag_axis=options["mag_axis"],
        )
        # hr.magnetisation_backwards_propagation(hr.b_xyz_prop[2, ::], b_type="bz", mag_axis="m_z")

    if options["current_propagation"]:
        if "bnv" in options["curr_b_type"]:
            hr.current_backwards_propagation(hr.bnv, b_type="bnv")
        else:
            if options["curr_use_vector"]:
                if options["curr_use_bxyz_prop"]:
                    vector_b = b_prop
                else:
                    vector_b = {
                        "bx": hr.fit_image_results["bx"],
                        "by": hr.fit_image_results["by"],
                        "bz": hr.fit_image_results["bz"],
                    }

                hr.current_backwards_propagation(
                    hr.fit_image_results[options["curr_b_type"]],
                    b_type=options["curr_b_type"],
                    vector_b=vector_b,
                )

            else:
                hr.current_backwards_propagation(
                    hr.fit_image_results[options["curr_b_type"]], b_type=options["curr_b_type"]
                )

    misc.dict_to_json(hr.options, "saved_options.json", path_to_dir=hr.options["output_dir"])

    if options["show_plots"]:
        plt.show()


if __name__ == "__main__":
    main()
