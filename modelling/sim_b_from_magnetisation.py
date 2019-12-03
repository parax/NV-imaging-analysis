# -*- coding: utf-8 -*-

"""
This collection of functions is used to calculate the magnetic field maps from various different
sources. 

< TODO >
Add more complex geometries to this other than a square flake
"""

__author__ = "David Broadway"

# ============================================================================

import basic.worker_plotting as WrkPlt
import basic.background_removal as bkc_rem

# ============================================================================

import numpy as np
from PIL import Image

# ============================================================================


def flake(
    options,
    plt_opts_b=None,
    plt_opts_mag=None,
    b_limit=None,
    filter=None,
    sigma=0,
    rebin=False,
    num_bins=0,
    phi_error=0,
):
    """ Function that takes a magnetic flake (square) and determines the magnetic field that
    should be emitted as a given standoff.

    Parameter:

    plt_opts_b (dictionary): Options for plotting magnetic field images used in plot worker

    plt_opts_mag (dictionary): Options for plotting magnetisation images used in plot worker

    b_limit (float): Value of which magnetic fields above this value are set to equal this value.
        Acting to simulate missing some magnetic field due to gradients.

    filter (boolean): Apply an Gaussian image filter to the magnetic field maps convolving the
        image with a Gaussian with a width given by sigma.

    sigma (float): width of the Gaussian used in the filter.

    rebin (boolean): Rebin the magnetic fields using the num_bins before performing the
    magnetisation propagation.

    num_bins (int modulo 2): number of bins used to rebin the magnetic field.

    phi_error (float): angle between the magnetic material and the surface of the diamond.
        Assuming this angle is only along the x-direction.
        0 ignores this parameter.

    Returns

    bxyz (dictionary of 2D arrays, float): {"bx": bx, "by": by, "bz": bz} contains the magnetic
    field
        maps of the different directions

    mag_array (2D array, float): contains the image of the simulated magnetisation.

    """
    # Define the magnetisation vector depending on the type of magnetic flake.
    # i.e. either out of plane (mag_z) or in plane (planar mag)
    if options["mag_z"]:
        mag = [0, 0, options["magnetisation_mag"]]
    else:
        angle = np.deg2rad(options["magnetisation_angle"])
        mag = [
            options["magnetisation_mag"] * np.cos(angle),
            options["magnetisation_mag"] * np.sin(angle),
            0,
        ]
    # define the x and y vectors
    xy_dim = options["flake_width_x"] * options["sim_size_factor"]
    size = options["sim_number_points"] * options["sim_size_factor"]
    x_vec = np.linspace(-xy_dim, xy_dim, num=size)
    y_vec = np.linspace(-xy_dim, xy_dim, num=size)

    # calculate magnetic field
    xv, yv = np.meshgrid(x_vec, y_vec)
    if phi_error is not 0:
        z_vec = x_vec * np.tan(np.deg2rad(phi_error))
        zv, _ = np.meshgrid(z_vec, y_vec)
        zv = zv + np.min(zv)
        zv = options["flake_stand_off"] * np.ones((size, size)) + zv
    else:
        zv = options["flake_stand_off"] * np.ones((size, size))
    pos = [xv, -yv, zv]
    flake_dim = [options["flake_width_x"], options["flake_width_y"], options["flake_thickness"] / 2]
    bx, by, bz = ComputeHfromM(mag, pos, flake_dim)

    # Calculate the magnetisation array
    start_flake = int((len(bx) - options["sim_number_points"]) / 2)
    end_flake = int((len(bx) + options["sim_number_points"]) / 2)
    mag_array = np.zeros((size, size))
    mag_array[start_flake:end_flake, start_flake:end_flake] = options["magnetisation_mag"]
    # conversion of units
    unit_conversion = 1e-18 / 9.27e-24  # m^2 -> nm^2 = 1e-18 and A -> uB/m^2 = 9.27e-24
    mag_array = mag_array * unit_conversion * options["flake_thickness"]

    # Convert to gauss
    bx = bx * 1e4
    by = by * 1e4
    bz = bz * 1e4

    if filter:
        bx = bkc_rem.image_filtering(bx, sigma=sigma, options=options, return_filter=True)
        by = bkc_rem.image_filtering(by, sigma=sigma, options=options, return_filter=True)
        bz = bkc_rem.image_filtering(bz, sigma=sigma, options=options, return_filter=True)
        mag_array = bkc_rem.image_filtering(
            mag_array, sigma=sigma, options=options, return_filter=True
        )

    if rebin:
        bx = rebin_image(bx, num_bins)
        by = rebin_image(by, num_bins)
        bz = rebin_image(bz, num_bins)
        mag_array = rebin_image(mag_array, num_bins)

    if options["add_noise"]:
        b_mag = np.max(np.sqrt(bx ** 2 + by ** 2 + bz ** 2))
        noise_sigma = b_mag * options["noise_percentage"]
        bx = add_noise_to_image(bx, options["noise_percentage"], sigma=noise_sigma)
        by = add_noise_to_image(by, options["noise_percentage"], sigma=noise_sigma)
        bz = add_noise_to_image(bz, options["noise_percentage"], sigma=noise_sigma)

    if b_limit:
        bx[bx > b_limit] = b_limit
        bx[bx < b_limit] = -b_limit
        by[by > b_limit] = b_limit
        by[by < b_limit] = -b_limit
        bz[bz > b_limit] = b_limit
        bz[bz < b_limit] = -b_limit

    plt_bx = WrkPlt.Image(options=options, title="Bx", **plt_opts_b)
    plt_bx.single_image(bx, **plt_opts_b)

    plt_by = WrkPlt.Image(options=options, title="By", **plt_opts_b)
    plt_by.single_image(by, **plt_opts_b)

    plt_bz = WrkPlt.Image(options=options, title="Bz", **plt_opts_b)
    plt_bz.single_image(bz, **plt_opts_b)

    plt_mag = WrkPlt.Image(options=options, title="sim_mag", **plt_opts_mag)
    plt_mag.single_image(mag_array, **plt_opts_mag)
    return {"bx": bx, "by": by, "bz": bz}, mag_array


# =================================


def current(
    options,
    angle_1=0,
    angle_2=0,
    b_limit=None,
    filter=None,
    sigma=0,
    rebin=False,
    num_bins=1,
    plt_opts_b=None,
    plt_opts_curr=None,
    plot_linecuts=False,
    plot_images=False,
):
    # define the x and y vectors
    xy_dim = options["curr_width"] * options["sim_size_factor"]
    size = options["sim_number_points"] * options["sim_size_factor"]
    x_vec = np.linspace(-xy_dim, xy_dim, num=size)
    bx, bz, jy = compute_b_from_current(options, x_vec)

    if plot_linecuts:
        plt_spec = WrkPlt.Spectra(options)
        plt_spec("B from curr sim line cut")

        plt_spec.add_to_plot(bx, x_array=x_vec * 1e6, linestyle="-", label="analytic bx")
        plt_spec.add_to_plot(bz, x_array=x_vec * 1e6, linestyle="-", label="analytic bz")
        plt_spec.style_spectra_ax("b curr linecut", "position (um)", "magnetic field (G)")

        plt_curr = WrkPlt.Spectra(options)
        plt_curr("curr sim line cut")
        plt_curr.add_to_plot(jy, x_array=x_vec * 1e6, linestyle="-", label="analytic jy")
        plt_curr.style_spectra_ax("curr linecut", "position (um)", "current density (A/m)")

    # make 2D map
    map_size = (
        options["sim_size_factor"] * options["sim_number_points"],
        options["sim_size_factor"] * options["sim_number_points"],
    )

    bx = bx * np.ones(map_size)
    bz = bz * np.ones(map_size)
    jy = jy * np.ones(map_size)

    angle_1 = angle_1 - 90
    angle_2 = angle_2 - 90

    bz_im = Image.fromarray(bz)
    bz = np.asarray(bz_im.rotate(angle_1, resample=Image.BICUBIC, expand=False)) + np.asarray(
        bz_im.rotate(angle_2, resample=Image.BICUBIC, expand=False)
    )

    bx_im = Image.fromarray(bx)

    bx = np.asarray(bx_im.rotate(angle_1, resample=Image.BICUBIC, expand=False)) * np.sin(
        np.deg2rad(angle_1 + 90)
    ) + np.asarray(bx_im.rotate(angle_2, resample=Image.BICUBIC, expand=False)) * np.sin(
        np.deg2rad(angle_2 + 90)
    )

    by = -np.asarray(bx_im.rotate(angle_1, resample=Image.BICUBIC, expand=False)) * np.cos(
        np.deg2rad(angle_1 + 90)
    ) - np.asarray(bx_im.rotate(angle_2, resample=Image.BICUBIC, expand=False)) * np.cos(
        np.deg2rad(angle_2 + 90)
    )

    jy_im = Image.fromarray(jy)
    jx = np.asarray(jy_im.rotate(angle_1, resample=Image.BICUBIC, expand=False)) * np.cos(
        np.deg2rad(angle_1 + 90)
    ) + np.asarray(jy_im.rotate(angle_2, resample=Image.BICUBIC, expand=False)) * np.cos(
        np.deg2rad(angle_2 + 90)
    )
    jy = np.asarray(jy_im.rotate(angle_1, resample=Image.BICUBIC, expand=False)) * np.sin(
        np.deg2rad(angle_1 + 90)
    ) + np.asarray(jy_im.rotate(angle_2, resample=Image.BICUBIC, expand=False)) * np.sin(
        np.deg2rad(angle_2 + 90)
    )

    s = len(jx)
    adj = int(round((s - s * np.cos(np.deg2rad(45))) / 2))

    bx = bx[adj : s - adj, adj : s - adj]
    by = by[adj : s - adj, adj : s - adj]
    bz = bz[adj : s - adj, adj : s - adj]

    jx = jx[adj : s - adj, adj : s - adj]
    jy = jy[adj : s - adj, adj : s - adj]
    jnorm = np.sqrt(jy ** 2 + jx ** 2)

    if filter:
        bx = bkc_rem.image_filtering(bx, sigma=sigma, options=options, return_filter=True)
        by = bkc_rem.image_filtering(by, sigma=sigma, options=options, return_filter=True)
        bz = bkc_rem.image_filtering(bz, sigma=sigma, options=options, return_filter=True)

    if rebin:
        bx = rebin_image(bx, num_bins)
        by = rebin_image(by, num_bins)
        bz = rebin_image(bz, num_bins)

    if options["add_noise"]:
        b_mag = np.max(np.sqrt(bx ** 2 + by ** 2 + bz ** 2))
        noise_sigma = b_mag * options["noise_percentage"]
        bx = add_noise_to_image(bx, options["noise_percentage"], sigma=noise_sigma)
        by = add_noise_to_image(by, options["noise_percentage"], sigma=noise_sigma)
        bz = add_noise_to_image(bz, options["noise_percentage"], sigma=noise_sigma)

    if b_limit:
        bx[bx > b_limit] = b_limit
        bx[bx < b_limit] = -b_limit
        by[by > b_limit] = b_limit
        by[by < b_limit] = -b_limit
        bz[bz > b_limit] = b_limit
        bz[bz < b_limit] = -b_limit

    if plot_images:

        plt_b_fields = WrkPlt.Image(options=options, title="Magnetic fields", **plt_opts_b)
        titles = ["Bx", "By", "Bz"]
        plt_b_fields.multiple_images([bx, by, bz], title=titles, **plt_opts_b)

        plt_j = WrkPlt.Image(options=options, title="current sim", **plt_opts_curr)
        titles = ["Jx", "Jy", "Jnorm"]
        plt_j.multiple_images([jx, jy, jnorm], title=titles, **plt_opts_curr)

    return bx, by, bz, jx, jy, jnorm


# =================================


def compute_b_from_current(options, x_vec):
    # Analytic field
    zp = options["curr_stand_off"]
    width = options["curr_width"]
    curr = options["curr"]
    mu0 = np.pi * 4e-7
    bx_analytic = (
        -mu0
        * curr
        / (2 * np.pi * width)
        * (np.arctan((width - 2 * x_vec) / (2 * zp)) + np.arctan((width + 2 * x_vec) / (2 * zp)))
    )
    bz_analytic = (
        mu0
        * curr
        / (4 * np.pi * width)
        * np.log(
            ((width - 2 * x_vec) ** 2 + (2 * zp) ** 2) / ((width + 2 * x_vec) ** 2 + (2 * zp) ** 2)
        )
    )
    jy_analytic = (
        curr
        / (np.pi * width)
        * (
            np.arctan((width - 2 * x_vec) / (2 * 1e-9))
            + np.arctan((width + 2 * x_vec) / (2 * 1e-9))
        )
    )
    return bx_analytic * 1e4, bz_analytic * 1e4, jy_analytic


# =================================


def rebin_image(image, num_bins):
    """ Rebins the image

    Parameters:
        image (2D array, float): image to be rebinned

        num_bins (int modulo 2): number of bins to average over and rebin.

    Returns:
        image_bin  (2D array, float): rebinned image
    """
    height, width = image.shape
    image_bin = np.nansum(
        np.nansum(
            np.reshape(image, [int(height / num_bins), num_bins, int(width / num_bins), num_bins]),
            1,
        ),
        2,
    ) / (num_bins ** 2)
    return image_bin


# ==================================================================
# ========== Functions for calculating magnetic fields =============
# ==================================================================


def ComputeHfromM(mag, pos, flake_dim):
    """ computes the magnetic field from a magnetised material

    Parameters:
        mag (list, float): The magnitude of the magnetisation of the material in each cartesian
        coordinate [x, y, z]

        pos (list of 2D arrays, float): Dimensions of the image to compute the magnetic field
        over [x, y, z]

        flake_dim (list of 2D arrays, float): Dimensions of the magnetic material. Assumes that the
        z-dimension is half of the actual width of the material.

    Returns:
        hx, hy, hz  (2D arrays, float): values of the magnetic field generated by the magnetic
        material
    """
    hx = 0
    hy = 0
    hz = 0

    if mag[0] is not 0:
        hx = hx + mag[0] * computehx(
            pos[0], pos[1], pos[2], flake_dim[0], flake_dim[1], flake_dim[2]
        )
        hy = hy + mag[0] * computehy(
            pos[0], pos[1], pos[2], flake_dim[0], flake_dim[1], flake_dim[2]
        )
        hz = hz + mag[0] * computehz(
            pos[0], pos[1], pos[2], flake_dim[0], flake_dim[1], flake_dim[2]
        )

    if mag[1] is not 0:
        hx = hx - mag[1] * computehy(
            pos[1], -pos[0], pos[2], flake_dim[1], flake_dim[0], flake_dim[2]
        )
        hy = hy + mag[1] * computehx(
            pos[1], -pos[0], pos[2], flake_dim[1], flake_dim[0], flake_dim[2]
        )
        hz = hz + mag[1] * computehz(
            pos[1], -pos[0], pos[2], flake_dim[1], flake_dim[0], flake_dim[2]
        )

    if mag[2] is not 0:
        hx = hx - mag[2] * computehz(
            pos[2], pos[1], -pos[0], flake_dim[2], flake_dim[1], flake_dim[0]
        )
        hy = hy + mag[2] * computehy(
            pos[2], pos[1], -pos[0], flake_dim[2], flake_dim[1], flake_dim[0]
        )
        hz = hz + mag[2] * computehx(
            pos[2], pos[1], -pos[0], flake_dim[2], flake_dim[1], flake_dim[0]
        )
    return hx, hy, hz


# =================================


def computehx(x, y, z, a, b, c):
    """ computes the x component of the magnetic field

    Parameters:
        x, y, z (2D array, float): Dimensions of the image to compute the magnetic field over

        a, b, c (2D array, float): Dimensions of the magnetic material. Assumes that the
        z-dimension (c) is half of the actual width of the material.

    Returns:
        hx (2D array, float): values of the magnetic field generated by the magnetic material
    """
    hx_plus = (
        np.arctan(
            (y + b) * (z + c) / ((x + a) * np.sqrt(((x + a) ** 2 + (y + b) ** 2 + (z + c) ** 2)))
        )
        + np.arctan(
            (y - b) * (z + c) / ((x - a) * np.sqrt(((x - a) ** 2 + (y - b) ** 2 + (z + c) ** 2)))
        )
        + np.arctan(
            (y + b) * (z - c) / ((x - a) * np.sqrt(((x - a) ** 2 + (y + b) ** 2 + (z - c) ** 2)))
        )
        + np.arctan(
            (y - b) * (z - c) / ((x + a) * np.sqrt(((x + a) ** 2 + (y - b) ** 2 + (z - c) ** 2)))
        )
    )

    hx_minus = (
        np.arctan(
            (y - b) * (z - c) / ((x - a) * np.sqrt(((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2)))
        )
        + np.arctan(
            (y + b) * (z + c) / ((x - a) * np.sqrt(((x - a) ** 2 + (y + b) ** 2 + (z + c) ** 2)))
        )
        + np.arctan(
            (y - b) * (z + c) / ((x + a) * np.sqrt(((x + a) ** 2 + (y - b) ** 2 + (z + c) ** 2)))
        )
        + np.arctan(
            (y + b) * (z - c) / ((x + a) * np.sqrt(((x + a) ** 2 + (y + b) ** 2 + (z - c) ** 2)))
        )
    )

    return -(hx_plus - hx_minus) * 1e-7


# =================================


def computehy(x, y, z, a, b, c):
    """ computes the y component of the magnetic field

    Parameters:
        x, y, z (2D array, float): Dimensions of the image to compute the magnetic field over

        a, b, c (2D array, float): Dimensions of the magnetic material. Assumes that the
        z-dimension (c) is half of the actual width of the material.

    Returns:
        hy (2D array, float): values of the magnetic field generated by the magnetic material
    """
    Hy_up = (
        ((z + c) - np.sqrt((x + a) ** 2 + (y + b) ** 2 + (z + c) ** 2))
        / ((z + c) + np.sqrt((x + a) ** 2 + (y + b) ** 2 + (z + c) ** 2))
        * ((z + c) - np.sqrt((x - a) ** 2 + (y - b) ** 2 + (z + c) ** 2))
        / ((z + c) + np.sqrt((x - a) ** 2 + (y - b) ** 2 + (z + c) ** 2))
        * ((z - c) - np.sqrt((x + a) ** 2 + (y - b) ** 2 + (z - c) ** 2))
        / ((z - c) + np.sqrt((x + a) ** 2 + (y - b) ** 2 + (z - c) ** 2))
        * ((z - c) - np.sqrt((x - a) ** 2 + (y + b) ** 2 + (z - c) ** 2))
        / ((z - c) + np.sqrt((x - a) ** 2 + (y + b) ** 2 + (z - c) ** 2))
    )

    Hy_down = (
        ((z - c) - np.sqrt((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2))
        / ((z - c) + np.sqrt((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2))
        * ((z + c) - np.sqrt((x - a) ** 2 + (y + b) ** 2 + (z + c) ** 2))
        / ((z + c) + np.sqrt((x - a) ** 2 + (y + b) ** 2 + (z + c) ** 2))
        * ((z + c) - np.sqrt((x + a) ** 2 + (y - b) ** 2 + (z + c) ** 2))
        / ((z + c) + np.sqrt((x + a) ** 2 + (y - b) ** 2 + (z + c) ** 2))
        * ((z - c) - np.sqrt((x + a) ** 2 + (y + b) ** 2 + (z - c) ** 2))
        / ((z - c) + np.sqrt((x + a) ** 2 + (y + b) ** 2 + (z - c) ** 2))
    )

    return -0.5 * np.log(Hy_up / Hy_down) * 1e-7


# =================================


def computehz(x, y, z, a, b, c):
    """ computes the z component of the magnetic field which is just a transformation of the hy
    equations

    Parameters:
        x, y, z (2D array, float): Dimensions of the image to compute the magnetic field over

        a, b, c (2D array, float): Dimensions of the magnetic material. Assumes that the
        z-dimension (c) is half of the actual width of the material.

    Returns:
        hz (2D array, float): values of the magnetic field generated by the magnetic material
    """
    return computehy(x, z, y, a, c, b)


# =================================


def add_noise_to_image(image, sigma_percentage, sigma=None):
    if sigma is None:
        sigma = np.max(np.abs(image)) * sigma_percentage / 100
    else:
        sigma = sigma / 100
    noise = np.random.normal(0, sigma, image.shape)
    noisey_image = image + noise
    return noisey_image
