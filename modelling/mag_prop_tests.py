# -*- coding: utf-8 -*-

"""
"""

__author__ = "David Broadway"

# ============================================================================

import basic.worker_propagator as WrkProp
import basic.worker_plotting as WrkPlt
import basic.misc as misc
from modelling import sim_b_from_magnetisation as sim_b_from_mag

# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import os

# ============================================================================
matplotlib.rcParams.update({"font.size": 8})
matplotlib.use("Qt5Agg")


# =================================


def main(__spec__=None):
    sim_type = "mag"
    # Get options
    options, plt_opts_gen, plt_opts_mag, plt_opts_b, plt_opts_curr = define_options(
        sim_type=sim_type
    )
    # get the magnetic field
    if sim_type is "mag":
        bxyz, mag = sim_b_from_mag.flake(
            options,
            plt_opts_b=plt_opts_b,
            plt_opts_mag=plt_opts_mag,
            b_limit=None,
            filter=options["filter"],
            sigma=options["sigma"],
            rebin=options["rebin"],
            num_bins=options["num_bins"],
            phi_error=0,
        )

        # define unv and get bnv
        unv, bnv = calc_bnv(0, 0, 54.7, 0, bxyz)
        # Plot the bnv map
        plt_b = WrkPlt.Image(options=options, title="bnv", **plt_opts_b)
        plt_b.single_image(bnv, **plt_opts_b)

        prop_bnv = WrkProp.WorkerPropagator(options, unv)
        bxyz_prop = prop_bnv.b_xyz(bnv, nv_axis=0)

        # ==== Propagate the bnv map into bxyz ==== #
        if options["use_prop_bxyz"]:
            # bxyz = {"bx": bxyz_prop[0], "by": bxyz_prop[1], "bz": bxyz_prop[2]}
            bxyz["bz"] = bxyz_prop[2]
            plt_bx_prop = WrkPlt.Image(options=options, title="prop Bx", **plt_opts_b)
            plt_bx_prop.single_image(bxyz_prop[0], **plt_opts_b)

            plt_by_prop = WrkPlt.Image(options=options, title="prop By", **plt_opts_b)
            plt_by_prop.single_image(bxyz_prop[1], **plt_opts_b)

            plt_bz_prop = WrkPlt.Image(options=options, title="prop Bz", **plt_opts_b)
            plt_bz_prop.single_image(bxyz_prop[2], **plt_opts_b)

        # ==== Propagate the b maps into magnetisation ==== #
        compare_different_propagation_types(options, mag, bnv, unv, bxyz, bxyz_prop, plt_opts_mag)

        # check_unv_theta_error(options, plt_opts_mag, bxyz, theta_range=20, loop_size=5)

        # check_unv_phi_error(options, plt_opts_mag, bxyz, phi_range=50, loop_size=5)

        # compare_in_plane_theta_guesses(options, plt_opts_mag, bxyz, theta_range=4, loop_size=21)
    else:
        bx, by, bz, jx, jy, jnorm = sim_b_from_mag.current(
            options,
            angle_1=10,
            angle_2=60,
            filter=options["filter"],
            sigma=options["sigma"],
            rebin=options["rebin"],
            num_bins=options["num_bins"],
            plt_opts_b=plt_opts_b,
            plt_opts_curr=plt_opts_curr,
            plot_linecuts=True,
            plot_images=True,
        )
        # define unv and get bnv
        bxyz = {"bx": bx, "by": -by, "bz": bz}
        j_sim = {"jx": jx, "jy": jy, "jnorm": jnorm}
        unv, bnv = calc_bnv(0, 0, 54.7, 0, bxyz)

        compare_different_curr_propagation_types(options, j_sim, bnv, unv, bxyz, plt_opts_curr)

    misc.dict_to_json(options, "saved_options.json", path_to_dir=options["output_dir"])

    plt.show()


# =================================


def compare_different_curr_propagation_types(options, j_sim, bnv, unv, bxyz, plt_opts_curr):
    # ==== Propagate the b maps into magnetisation ==== #
    # mag from bnv
    jx_bnv, jy_bnv, j_norm_bnv = reconstruct_curr(
        bnv, options, u_proj=unv, title="bnv", plt_opts=plt_opts_curr
    )

    # mag from bz
    jx_bz, jy_bz, j_norm_bz = reconstruct_curr(
        bxyz["bz"], options, u_proj=[0, 0, 1], title="bz", plt_opts=plt_opts_curr
    )

    # mag from by
    jx_bx, jy_bx, j_norm_bx = reconstruct_curr(
        bxyz["bx"], options, u_proj=[1, 0, 0], vector_b=bxyz, title="bxy", plt_opts=plt_opts_curr
    )
    # take line cuts
    maps = [j_sim["jnorm"], j_norm_bnv, j_norm_bz, j_norm_bx]
    labels = ["exact", "bnv", "bz", "bx"]
    title = "jnorm line cuts"
    plot_multiple_curr_line_cuts(maps, labels, title, options)

    maps = [j_sim["jy"], jy_bnv, jy_bz, jy_bx]
    labels = ["exact", "bnv", "bz", "bxy"]
    title = "jy curr line cuts"
    plot_multiple_curr_line_cuts(maps, labels, title, options)

    maps = [j_sim["jx"], jx_bnv, jx_bz, jx_bx]
    labels = ["exact", "bnv", "bz", "bxy"]
    title = "jx curr line cuts"
    plot_multiple_curr_line_cuts(maps, labels, title, options)


# =================================


def compare_different_propagation_types(options, mag, bnv, unv, bxyz, bxyz_prop, plt_opts_mag):
    # ==== Propagate the b maps into magnetisation ==== #
    # mag from bnv
    mag_nv = reconstruct_mz(
        bnv, options, u_proj=unv, title="(bnv)", mag_axis="m_z", plt_opts=plt_opts_mag
    )

    # mag from bnv
    mag_nv_prop = reconstruct_mz(
        bxyz_prop[2],
        options,
        u_proj=[0, 0, 1],
        title="(bnv to bz)",
        mag_axis="m_z",
        plt_opts=plt_opts_mag,
    )
    # mag_nv = mag_nv - np.min(mag_nv)
    # mag from bz
    mag_z = reconstruct_mz(
        bxyz["bz"], options, u_proj=[0, 0, 1], title="(bz)", mag_axis="m_z", plt_opts=plt_opts_mag
    )
    # mag_z = mag_z - np.nanmin(mag_z)
    # bx and by
    mag_planar = reconstruct_mz(
        bxyz["bz"],
        options,
        vector_b=bxyz,
        u_proj=[0, 0, 1],
        title="(b planar)",
        mag_axis="m_z",
        plt_opts=plt_opts_mag,
    )
    # mag_planar = mag_planar - np.nanmin(mag_planar)

    # take line cuts
    maps = [mag, mag_nv, mag_z, mag_planar, mag_nv_prop]
    labels = ["exact", "bnv", "bz", "b planar", "bn to bz"]
    title = "mag line cuts"
    plot_multiple_line_cuts(maps, labels, title, options)

    maps = [mag - mag_nv, mag - mag_z, mag - mag_planar, mag_nv_prop - mag_planar]
    labels = ["bnv", "bz", "b planar", "bn to bz"]
    title = "mag diff line cuts"
    plot_multiple_line_cuts(maps, labels, title, options)


# =================================


def plot_multiple_line_cuts(maps, labels, title, options):
    """ takes a list of maps and plots a horizontal line cut through the central pixel"""

    image_dimension = 1e6 * options["flake_width_x"] * options["sim_size_factor"]

    num_maps = len(maps)

    plt_spec = WrkPlt.Spectra(options)
    plt_spec("mag propagation line cut")

    for idx in range(num_maps):
        centre_pixel = int(len(maps[idx]) / 2)
        line_cut = WrkPlt.get_line_cuts(maps[idx], centre_pixel, centre_pixel, 5)

        x_linecut_values = np.linspace(-image_dimension, image_dimension, len(line_cut["hor"]))
        plt_spec.add_to_plot(
            line_cut["hor"], x_array=x_linecut_values, linestyle="-", label=labels[idx]
        )

    plt_spec.style_spectra_ax(title, "position (um)", r"Mag $\left(\mu_B nm^{-2}\right)$")


# =================================


def plot_multiple_curr_line_cuts(maps, labels, title, options):
    """ takes a list of maps and plots a horizontal line cut through the central pixel"""

    image_dimension = 1e6 * options["curr_width"] * options["sim_size_factor"] / 2

    num_maps = len(maps)

    plt_spec = WrkPlt.Spectra(options)
    plt_spec("mag propagation line cut")

    for idx in range(num_maps):
        centre_pixel = int(len(maps[idx]) / 2)
        line_cut = WrkPlt.get_line_cuts(maps[idx], centre_pixel, centre_pixel, 5)

        x_linecut_values = np.linspace(-image_dimension, image_dimension, len(line_cut["hor"]))
        plt_spec.add_to_plot(
            line_cut["hor"], x_array=x_linecut_values, linestyle="-", label=labels[idx]
        )

    plt_spec.style_spectra_ax(title, "position (um)", "current (A/m)")


# =================================


def compare_in_plane_theta_guesses(options, plt_opts, bxyz, theta_range=0, loop_size=0):
    map_size = len(bxyz["bz"])

    theta = np.linspace(-(theta_range / 2), theta_range / 2, loop_size) + options["in_plane_angle"]

    magnetisation = np.zeros((loop_size, map_size, map_size))
    magnetisation_bck = np.zeros((loop_size, map_size, map_size))
    mag_std_flake = np.zeros(loop_size)
    mag_std_bck = np.zeros(loop_size)

    flake_idxs = [
        int((map_size - options["sim_number_points"]) / 2),
        int((map_size + options["sim_number_points"]) / 2),
    ]
    flake_remove = flake_idxs
    flake_remove[0] = flake_idxs[0] - 5
    flake_remove[1] = flake_idxs[1] + 5
    for idx in range(loop_size):
        magnetisation[idx, ::] = reconstruct_mz(
            bxyz["bz"],
            {**options, **{"in_plane_angle": theta[idx]}},
            u_proj=[0, 0, 1],
            title="(bz)",
            mag_axis="m_z",
            plt_opts=plt_opts,
        )
        magnetisation[idx, ::] = magnetisation[idx, ::] - np.nanmin(magnetisation[idx, ::])
        magnetisation_bck[idx, ::] = magnetisation[idx, ::]
        magnetisation_bck[
            idx, flake_remove[0] : flake_remove[1], flake_remove[0] : flake_remove[1]
        ] = 0
        mag_std_flake[idx] = np.nanstd(
            magnetisation[idx, flake_idxs[0] : flake_idxs[1], flake_idxs[0] : flake_idxs[1]]
        )
        mag_std_bck[idx] = np.nanstd(magnetisation_bck[idx, ::])
    centre_pixel = int(map_size / 2)

    plt_mag = WrkPlt.Image(options=options, title="Mag ", **{**plt_opts, **{"paper_figure": False}})
    plt_mag.multiple_images(magnetisation, title="mag_v_theta", **plt_opts)

    plt_mag_bck = WrkPlt.Image(
        options=options, title="Mag bck", **{**plt_opts, **{"paper_figure": False}}
    )
    plt_mag_bck.multiple_images(magnetisation_bck, title="mag_back", **plt_opts)

    plt_spec = WrkPlt.Spectra(options)
    plt_spec("mag propagation line cut")
    for idx in range(loop_size):
        mag_linecuts = WrkPlt.get_line_cuts(magnetisation[idx, ::], centre_pixel, centre_pixel, 5)
        mag_linecuts["hor"] = mag_linecuts["hor"] - np.min(mag_linecuts["hor"])
        plt_spec.add_to_plot(
            mag_linecuts["hor"], linestyle="-", label=r"$\Delta\theta =$ " + str(theta[idx])
        )

    plt_spec.style_spectra_ax("mag line cuts", "pixels", "magnetisation (units)")

    plt_spec = WrkPlt.Spectra(options, paper_figure=True)
    plt_spec("mag std")
    plt_spec.add_to_plot(mag_std_flake, x_array=theta, linestyle="-", label="flake")
    plt_spec.add_to_plot(mag_std_bck, x_array=theta, linestyle="-", label="background")
    plt_spec.style_spectra_ax(
        "Standard deviation and theta guess", r"$\theta$", "Standard deviation"
    )


# =================================


def check_unv_theta_error(options, plt_opts, bxyz, theta_range=0, loop_size=0):
    map_size = len(bxyz["bx"])
    theta = np.linspace(-theta_range / 2, theta_range / 2, loop_size)
    magnetisation = np.zeros((loop_size, map_size, map_size))
    unv_base, bnv = calc_bnv(0, 0, 54.7, 0, bxyz)
    for idx in range(loop_size):
        unv, bnv = calc_bnv(theta[idx], 0, 54.7, 0, bxyz)
        magnetisation[idx, ::] = reconstruct_mz(
            bnv,
            options,
            u_proj=unv_base,
            title="M_z (bnv theta = " + str(theta[idx]) + ")",
            mag_axis="m_z",
            plt_opts=plt_opts,
        )
    centre_pixel = int(map_size / 2)

    plt_spec = WrkPlt.Spectra(options)
    plt_spec("mag propagation line cut")
    for idx in range(loop_size):
        mag_linecuts = WrkPlt.get_line_cuts(magnetisation[idx, ::], centre_pixel, centre_pixel, 5)
        mag_linecuts["hor"] = mag_linecuts["hor"] - np.min(mag_linecuts["hor"])
        plt_spec.add_to_plot(
            mag_linecuts["hor"], linestyle="-", label=r"$\Delta\theta =$ " + str(theta[idx])
        )

    plt_spec.style_spectra_ax("mag line cuts", "pixels", "magnetisation (units)")


# =================================


def check_unv_phi_error(options, plt_opts, bxyz, phi_range=0, loop_size=0):
    map_size = len(bxyz["bx"])
    phi = np.linspace(-phi_range / 2, phi_range / 2, loop_size)
    magnetisation = np.zeros((loop_size, map_size, map_size))
    unv_base, bnv = calc_bnv(0, 0, 54.7, 0, bxyz)
    for idx in range(loop_size):
        unv, bnv = calc_bnv(0, 0, 54.7, phi[idx], bxyz)
        magnetisation[idx, ::] = reconstruct_mz(
            bnv,
            options,
            u_proj=unv_base,
            title="M_z (bnv phi = " + str(phi[idx]) + ")",
            mag_axis="m_z",
            plt_opts=plt_opts,
        )
    centre_pixel = int(map_size / 2)

    plt_spec = WrkPlt.Spectra(options)
    plt_spec("mag propagation line cut")
    for idx in range(loop_size):
        mag_linecuts = WrkPlt.get_line_cuts(magnetisation[idx, ::], centre_pixel, centre_pixel, 5)
        mag_linecuts["hor"] = mag_linecuts["hor"] - np.min(mag_linecuts["hor"])
        plt_spec.add_to_plot(mag_linecuts["hor"], linestyle="-", label=r"$\phi =$ " + str(phi[idx]))

    plt_spec.style_spectra_ax("mag line cuts", "pixels", "magnetisation (units)")


# =================================


def reconstruct_mz(
    bmap, options, u_proj=None, vector_b=None, title="axis", mag_axis=None, plt_opts=None
):
    if u_proj is None:
        u_proj = [0, 0, 0]
    if mag_axis is None:
        mag_axis = "m_z"
    prop = WrkProp.WorkerPropagator(options, u_proj)
    mag = prop.magnetisation(bmap, vector_b=vector_b, u_proj=u_proj, magnetisation_axis=mag_axis)

    plt_mag = WrkPlt.Image(options=options, title="Mag " + title, **plt_opts)
    plt_opts = {**{"cbar_title": r"Mag $\left(\mu_B nm^{-2}\right)$"}, **plt_opts}
    plt_mag.single_image(mag, **plt_opts)
    return mag


# =================================


def reconstruct_curr(bmap, options, u_proj=None, vector_b=None, title="axis", plt_opts=None):
    prop = WrkProp.WorkerPropagator(options, u_proj)
    jx_bz, jy_bz, j_norm_bz = prop.current(bmap, u_proj=u_proj, vector_b=vector_b)
    titles = ["jx", "jy", "j_norm"]
    plt_mag = WrkPlt.Image(options=options, title="curr " + title, **plt_opts)
    plt_opts = {**{"cbar_title": r"Currrent (A/m)"}, **plt_opts}
    plt_mag.multiple_images([jx_bz, jy_bz, j_norm_bz], title=titles, **plt_opts)
    return jx_bz, jy_bz, j_norm_bz


# =================================


def calc_bnv(theta, phi, base_theta, base_phi, bxyz):
    """ defines the unv vector and applies this to obtain bnv"""
    unv = [
        np.sin(np.deg2rad(base_theta + theta)) * np.cos(np.deg2rad(base_phi + phi)),
        np.sin(np.deg2rad(base_theta + theta)) * np.sin(np.deg2rad(base_phi + phi)),
        np.cos(np.deg2rad(base_theta + theta)),
    ]
    bnv = unv[0] * bxyz["bx"] + unv[1] * bxyz["by"] + unv[2] * bxyz["bz"]
    return unv, bnv


# =================================


def define_options(sim_type="mag"):
    """ Reads in all of the different option json files and makes dictionaries. """
    if sim_type is "mag":
        options = misc.json_to_dict("opts_mag_testing.json")
        # the times 2 in the pixel size is required to match the magetisation simulation.
        options["raw_pixel_size"] = (
            2 * 1e6 * options["flake_width_x"] / options["sim_number_points"]
        )
    else:
        options = misc.json_to_dict("opts_curr_testing.json")
        options["raw_pixel_size"] = 1e6 * options["curr_width"] / options["sim_number_points"]

    current_dir = os.getcwd()
    home_dir = str(Path(current_dir).parent)

    # plotting options
    plt_def_opts = misc.json_to_dict(home_dir + "/options/plt_default.json")
    plt_opts_gen = {**plt_def_opts, **misc.json_to_dict(home_dir + "/options/plt_general.json")}
    plt_opts_mag = {**plt_def_opts, **misc.json_to_dict(home_dir + "/options/plt_mag.json")}
    plt_opts_curr = {**plt_def_opts, **misc.json_to_dict(home_dir + "/options/plt_curr.json")}
    plt_opts_b = {**plt_def_opts, **misc.json_to_dict(home_dir + "/options/plt_b.json")}

    # For the simulation we need to define the pixel sizes
    options["total_bin"] = 1

    if options["rebin"]:
        options["raw_pixel_size"] = options["raw_pixel_size"] * options["num_bins"]

    if not os.path.exists(options["data_dir"]):
        os.mkdir(options["data_dir"])
    return options, plt_opts_gen, plt_opts_mag, plt_opts_b, plt_opts_curr


if __name__ == "__main__":
    main()
