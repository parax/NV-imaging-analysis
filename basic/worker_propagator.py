# -*- coding: utf-8 -*-

"""
Module docstring here

"""

__author__ = "David Broadway"

# ============================================================================

import basic.worker_plotting as WrkPlt
import basic.background_removal as BkgRem

# ============================================================================

import matplotlib.pyplot as plt
import matplotlib

import numpy as np


# ============================================================================


class WorkerPropagator(object):
    """ Class for dealing with all different types of propagation b xyz, magnetisation, and current
    mapping
    """

    def __init__(self, options, unv):
        self.options = options
        self.pixel_size_for_fft = self.options["raw_pixel_size"] * self.options["total_bin"] * 1e-6
        self.unv = unv
        self.min_k = self.pixel_size_for_fft / 2 * np.pi
        self.magnetisation_axis_dict = {"m_x": 0, "m_y": 1, "m_z": 2}
        # Initialise data types
        self.kx = np.array([])
        self.ky = np.array([])
        self.k = np.array([])

    # =================================
    # == magnetic field propagation  ==
    # =================================

    def b_xyz(self, bnv, nv_axis=0):
        """ Propagates the bnv data to bxyz.
        Parameters:
            bnv (2D array, float): image/map of the Bnv data.

            nv_axis (list, float): the unv orientation e.g. [0.57, 0, 0.87]

        Returns:
            bx_region, by_region, bz_region (2D arrays, float): images/maps of the bxyz magnetic
            fields after propagation.
        """
        # pad the image for better FFT
        padded_bnv, padding_roi = self.pad_image(bnv)
        # Perform the FFT
        fft_bnv = np.fft.fftshift(np.fft.fft2(padded_bnv))
        # define the k vectors
        self.kx, self.ky, self.k = self.define_k_vectors(fft_bnv.shape)
        # get the transformations
        bnv2bx, bnv2by, bnv2bz = self.define_b_xyz_transformation(nv_axis)
        # ==== Define filter ==== #
        img_filter = self.get_image_filter()
        # Define all of the inverse transformation matrices
        bnv2bx = self.remove_invalid_elements(img_filter * bnv2bx)
        bnv2by = self.remove_invalid_elements(img_filter * bnv2by)
        bnv2bz = self.remove_invalid_elements(img_filter * bnv2bz)
        # transform into xyz
        fft_bx = fft_bnv * bnv2bx
        fft_by = fft_bnv * bnv2by
        fft_bz = fft_bnv * bnv2bz
        # remove the DC component as it is lost in the FFT anyway
        # fft_bx[self.k < self.min_k] = 0
        # fft_by[self.k < self.min_k] = 0
        # fft_bz[self.k < self.min_k] = 0
        # fourier transform back into real space
        bx = np.fft.ifft2(np.fft.ifftshift(fft_bx)).real
        by = np.fft.ifft2(np.fft.ifftshift(fft_by)).real
        bz = np.fft.ifft2(np.fft.ifftshift(fft_bz)).real
        # Readout the non padded region
        if self.options["fft_padding"]:
            bx_region = bx[padding_roi[0], padding_roi[1]]
            by_region = by[padding_roi[0], padding_roi[1]]
            bz_region = bz[padding_roi[0], padding_roi[1]]
        else:
            bx_region = bx
            by_region = by
            bz_region = bz
        return bx_region, by_region, bz_region

    # =================================

    def define_b_xyz_transformation(self, nv_axis):
        """
        Defines the transformation that takes bnv to b_xyz
        """
        try:
            unv = self.unv[nv_axis, ::]
        except TypeError:
            unv = self.unv

        bnv2bx = 1 / (unv[0] + unv[1] * self.ky / self.kx + 1j * unv[2] * self.k / self.kx)
        bnv2by = 1 / (unv[0] * self.kx / self.ky + unv[1] + 1j * unv[2] * self.k / self.ky)
        bnv2bz = 1 / (-1j * unv[0] * self.kx / self.k - 1j * unv[1] * self.ky / self.k + unv[2])
        return bnv2bx, bnv2by, bnv2bz

    # =================================
    # === Magnetisation propagation ===
    # =================================

    def magnetisation(self, b_image, vector_b=None, u_proj=None, magnetisation_axis=None):
        """ takes a magnetic image and returns the magnetisation that produced it
        """
        if u_proj is None:
            # If no projection axis is given assume we are using bnv from the NV with the largest
            # splitting
            u_proj = self.unv[0]
        # Define the direction of the magnetisation to be propagated to
        if magnetisation_axis is None:
            # if no desired magnetisation is given assume magnetised in the z-axis
            magnetisation_axis = "m_z"
        # Define the mag axis to pass to the transformation
        mag_axis = self.magnetisation_axis_dict[magnetisation_axis]

        if vector_b:
            fft_m, padding_roi = self.mag_multi_map_transformation(vector_b, mag_axis)
        else:
            fft_m, padding_roi = self.mag_single_map_transformation(b_image, u_proj, mag_axis)

        # ==== inverse FFT ==== #
        mz = np.fft.ifft2(np.fft.ifftshift(fft_m)).real

        # Readout the non padded region
        if self.options["fft_padding"]:
            mz = mz[padding_roi[0], padding_roi[1]]

        # conversion into more useful units
        # m^2 -> nm^2 = 1e-18
        # A -> uB/m^2 = 9.27e-24
        unit_conversion = 1e-18 / 9.27e-24
        mz = mz * unit_conversion
        if self.options["in_plane_propagation"]:
            theta = np.deg2rad(self.options["in_plane_angle"])
            mz = self.in_plane_background_normalisation(mz, theta, 20)
        return mz

    # =================================

    def mag_single_map_transformation(self, b_image, u_proj, mag_axis):
        """
        Deals with the transformation of a single b map into both Jx and Jy maps. Takes the fourier
        transform of the b map (fft_b_image) and the direction of the b map (u_proj).
        """
        # === pad the image for better FFT === #
        padded_b_image, padding_roi = self.pad_image(b_image * 1e-4)

        # === Perform the FFT === #
        fft_b_image = np.fft.fftshift(np.fft.fft2(padded_b_image))

        # ==== define k vectors ==== #
        self.kx, self.ky, self.k = self.define_k_vectors(fft_b_image.shape)
        # ==== define transformation matrix ==== #
        d_matrix = self.define_magnetisation_transformation()
        # Transformation for unv or single axis
        m_to_b = (
            u_proj[0] * d_matrix[mag_axis, 0, ::]
            + u_proj[1] * d_matrix[mag_axis, 1, ::]
            + u_proj[2] * d_matrix[mag_axis, 2, ::]
        )

        if self.options["in_plane_propagation"]:
            # if the flake is magnetised in plane than use this transformation instead
            b_axis = np.nonzero(u_proj)[0]
            theta = np.deg2rad(self.options["in_plane_angle"])
            m_to_b = np.zeros(d_matrix[0, 0, ::].shape, dtype="complex128")
            for idx in b_axis:
                m_to_b += (
                    np.cos(theta) * d_matrix[0, int(idx), ::]
                    + np.sin(theta) * d_matrix[1, int(idx), ::]
                )

        # ==== Define filter ==== #
        img_filter = self.get_image_filter()

        # Apply filter and Replace all nans and infs with zero
        b_to_m = self.remove_invalid_elements(img_filter / m_to_b)

        # remove the DC component as it is lost in the FFT anyway
        fft_b_image[np.abs(self.k) < self.options["k_vector_epsilon"]] = 0

        # Transform the bnv into current density
        fft_m = b_to_m * fft_b_image

        # === plot the stages of the fourier transformation process === #
        self.plot_fft_stages(b_to_m, fft_b_image, fft_m, plt_stages=self.options["plot_fft_stages"])
        return fft_m, padding_roi

    # =================================

    def mag_multi_map_transformation(self, vector_b, mag_axis):

        # === pad the image for better FFT === #
        padded_bx_image, padding_roi = self.pad_image(vector_b["bx"] * 1e-4)
        padded_by_image, padding_roi = self.pad_image(vector_b["by"] * 1e-4)

        # === Perform the FFT === #
        fft_bx_image = np.fft.fftshift(np.fft.fft2(padded_bx_image))
        fft_by_image = np.fft.fftshift(np.fft.fft2(padded_by_image))

        # ==== define k vectors ==== #
        self.kx, self.ky, self.k = self.define_k_vectors(fft_bx_image.shape)

        # ==== define transformation matrix ==== #
        d_matrix = self.define_magnetisation_transformation()

        # ==== Define filter ==== #
        img_filter = self.get_image_filter()

        # Define the dtype
        m_to_bx = np.zeros(d_matrix[0, 0, ::].shape, dtype="complex128")
        m_to_by = np.zeros(d_matrix[0, 0, ::].shape, dtype="complex128")

        # Define all of the inverse transformation matrices
        if self.options["in_plane_propagation"]:
            # if the flake is magnetised in plane than use this transformation instead
            theta = np.deg2rad(self.options["in_plane_angle"])
            m_to_bx += np.cos(theta) * d_matrix[0, 0, ::] + np.sin(theta) * d_matrix[1, 0, ::]
            m_to_by += np.cos(theta) * d_matrix[0, 1, ::] + np.sin(theta) * d_matrix[1, 1, ::]
            # m_to_bx += np.cos(theta) * d_matrix[0, 0, ::]
            # m_to_by += np.sin(theta) * d_matrix[1, 1, ::]
        else:
            m_to_bx += d_matrix[mag_axis, 0, ::]
            m_to_by += d_matrix[mag_axis, 1, ::]

        # Get m_z from b_xyz
        fft_m_bx = fft_bx_image * img_filter / m_to_bx
        fft_m_by = fft_by_image * img_filter / m_to_by
        fft_m = (fft_m_bx + fft_m_by) / 2

        # remove the DC component as it is lost in the FFT anyway
        fft_m[np.abs(self.k) < self.options["k_vector_epsilon"]] = 0

        # Replace troublesome pixels in fourier space
        x_idxs = np.argwhere(np.abs(self.kx) < self.options["k_vector_epsilon"])
        y_idxs = np.argwhere(np.abs(self.ky) < self.options["k_vector_epsilon"])
        for idx in x_idxs:
            fft_m[idx[0], idx[1]] = fft_m_by[idx[0], idx[1]]
        for idx in y_idxs:
            fft_m[idx[0], idx[1]] = fft_m_bx[idx[0], idx[1]]
        fft_m = self.remove_invalid_elements(fft_m)
        return fft_m, padding_roi

    # =================================

    def define_magnetisation_transformation(self):
        """
        Defines the transformation matrix that takes magnetisation to b such that
        b_xyz = d_matrix . m_xyz
        """
        mu0 = 4 * np.pi * 1e-7
        # Selector for if the exponential factor is included.
        if self.options["use_stand_off"]:
            exp_factor = np.exp(-1 * self.k * self.options["stand_off"])
        else:
            exp_factor = 1
        # Definition of the transformation matrix
        d_matrix = (
            (mu0 / 2)
            * exp_factor
            * np.array(
                [
                    [-(self.kx ** 2) / self.k, -(self.kx * self.ky) / self.k, -1j * self.kx],
                    [-self.kx * self.ky / self.k, -(self.ky ** 2 / self.k), -1j * self.ky],
                    [-1j * self.kx, -1j * self.ky, self.k],
                ]
            )
        )
        return d_matrix

    # =================================

    def in_plane_background_normalisation(self, image, theta, edge_pixels_used):
        """
        function to subtract the mean value of the background line by line defined by the average
        over the number of pixels at the edge of the image

        parameters:

        image (2D array, float): image to be normalised.

        theta (float, radians): Angle of the magnetisation used for the propagation

        edge_pixels_used (int): Number of pixels at the edge of the image to use for the
        normalisation

        return:

        image (2D array, float): Image after the line by line subtraction
        """
        size = image.shape
        norm_line = size[0] - 1 - np.tan(theta) * range(size[0])
        norm_idxs = [round(x) for x in norm_line.tolist()]
        ii = 0
        for idx in range(round((size[0]) - 1)):
            norm_idxs_new = []
            for x in norm_idxs:
                if x - ii in range(size[0] - 1):
                    norm_idxs_new.append(x - ii)

            x_idxs = [round(x) for x in range(len(norm_idxs_new))]
            image_cut = image[(norm_idxs_new, x_idxs)]
            image[(norm_idxs_new, x_idxs)] = (
                image_cut
                - (
                    np.mean(image_cut[0:edge_pixels_used])
                    + np.mean(image_cut[-edge_pixels_used:-1])
                )
                / 2
            )
            ii += 1
        ii = 0
        for idx in range(round((size[0]) - 1)):
            ii += 1
            norm_idxs_new = []
            for x in norm_idxs:
                if x + ii in range(size[0] - 1):
                    norm_idxs_new.append(x + ii)

            x_idxs = [size[0] - 1 - round(x) for x in range(len(norm_idxs_new))]

            image_cut = image[(norm_idxs_new, x_idxs[::-1])]
            image[(norm_idxs_new, x_idxs[::-1])] = (
                image_cut
                - (
                    np.mean(image_cut[0:edge_pixels_used])
                    + np.mean(image_cut[-edge_pixels_used:-1])
                )
                / 2
            )
        return image

    # =================================
    # ====== Current propagation ======
    # =================================

    def current(self, b_image, u_proj=None, vector_b=None):
        """ Takes a b_map or the vector b maps (bx, by, bz) and returns the current density
        jx, jy, jnorm) responsible for producing the magnetic field.

        u_proj defines the axis the b map is measured about. This could be a nv axis or standard
        cartesian coords. If no axis is given the assumption is made that you are using a b map from
        an NV axis that corresponds to the furthest split NV calculated from the magnetic field
        parameters.
        """
        b_image = b_image * 1e-4  # converts gauss into tesla
        if u_proj is None:
            # If no projection axis is given assume we are using bnv from the NV with the largest
            # splitting
            u_proj = self.unv[0]

        # === pad the image for better FFT === #
        padded_b_image, padding_roi = self.pad_image(b_image)
        # === Perform the FFT === #
        fft_b_image = np.fft.fftshift(np.fft.fft2(padded_b_image))
        fft_b_image = self.remove_invalid_elements(fft_b_image)
        # ==== define k vectors ==== #
        self.kx, self.ky, self.k = self.define_k_vectors(fft_b_image.shape)

        if vector_b:
            fft_jx, fft_jy = self.current_multi_map_transformation(vector_b)
        else:
            fft_jx, fft_jy = self.current_single_map_transformation(fft_b_image, u_proj)

        # ==== inverse FFT ==== #
        jx = np.fft.ifft2(np.fft.ifftshift(fft_jx)).real
        jy = np.fft.ifft2(np.fft.ifftshift(fft_jy)).real
        # Readout the non padded region
        if self.options["fft_padding"]:
            jx = jx[padding_roi[0], padding_roi[1]].T
            jy = jy[padding_roi[0], padding_roi[1]].T

        # normal from corner
        jx = jx - np.mean(jx[0:50, 0:50])
        jy = jy - np.mean(jy[0:50, 0:50])
        # jy = jy - np.mean(jy[0:20, 0:20])
        # convert to Amp/ micron
        j_norm = np.sqrt(jx ** 2 + jy ** 2)
        return jx, jy, j_norm

    # =================================

    def current_single_map_transformation(self, fft_b_image, u_proj):
        """
        Deals with the transformation of a single b map into both Jx and Jy maps. Takes the fourier
        transform of the b map (fft_b_image) and the direction of the b map (u_proj).
        """
        # ==== define transformation matrix ==== #
        b_to_jx, b_to_jy = self.define_current_transformation(u_proj)

        # ==== Define filter ==== #
        img_filter = self.get_image_filter()
        # Apply filter and
        # Replace all nans and infs with zero
        b_to_jx = self.remove_invalid_elements(img_filter * b_to_jx)
        b_to_jy = self.remove_invalid_elements(img_filter * b_to_jy)

        # remove the DC component as it is lost in the FFT anyway
        # fft_b_image[np.abs(self.k) < 2 * self.min_k] = 0

        # Transform the bnv into current density
        fft_jx = b_to_jx * fft_b_image
        fft_jy = b_to_jy * fft_b_image

        # === plot the stages of the fourier transformation process === #
        self.plot_fft_stages(
            b_to_jx, fft_b_image, fft_jx, plt_stages=self.options["plot_fft_stages"]
        )

        return fft_jx, fft_jy

    # =================================

    def current_multi_map_transformation(self, fft_b_images):
        """
        Deals with the transformation of a vector b map into both Jx and Jy maps.
        """
        # === pad the image for better FFT === #
        # Also convert gauss into tesla
        padded_bx_image, padding_roi = self.pad_image(fft_b_images["bx"] * 1e-4)
        padded_by_image, padding_roi = self.pad_image(fft_b_images["by"] * 1e-4)
        padded_bz_image, padding_roi = self.pad_image(fft_b_images["bz"] * 1e-4)

        # === Perform the FFT === #
        fft_bx_image = np.fft.fftshift(np.fft.fft2(padded_bx_image))
        fft_bx_image = self.remove_invalid_elements(fft_bx_image)
        fft_by_image = np.fft.fftshift(np.fft.fft2(padded_by_image))
        fft_by_image = self.remove_invalid_elements(fft_by_image)
        fft_bz_image = np.fft.fftshift(np.fft.fft2(padded_bz_image))
        fft_bz_image = self.remove_invalid_elements(fft_bz_image)

        # ==== define transformation matrix ==== #
        bx_to_jx, bx_to_jy = self.define_current_transformation([1, 0, 0], bx_only=True)
        by_to_jx, by_to_jy = self.define_current_transformation([0, 1, 0], by_only=True)
        bz_to_jx, bz_to_jy = self.define_current_transformation([0, 0, 1])

        # ==== Define filter ==== #
        img_filter = self.get_image_filter()
        # Apply filter and
        # Replace all nans and infs with zero
        bx_to_jy = self.remove_invalid_elements(img_filter * bx_to_jy)
        by_to_jx = self.remove_invalid_elements(img_filter * by_to_jx)
        bz_to_jx = self.remove_invalid_elements(img_filter * bz_to_jx)
        bz_to_jy = self.remove_invalid_elements(img_filter * bz_to_jy)

        if "bz" in self.options["curr_ignore_axis"]:
            # Transform the bnv into current density
            fft_jx = by_to_jx * fft_by_image
            fft_jy = bx_to_jy * fft_bx_image
        else:
            # Transform the bnv into current density
            fft_jx = by_to_jx * fft_by_image + bz_to_jx * fft_bz_image
            fft_jy = bx_to_jy * fft_bx_image + bz_to_jy * fft_bz_image

        # # Replace troublesome pixels in fourier space
        # x_idxs = np.argwhere(np.abs(self.kx) < self.options["k_vector_epsilon"])
        # y_idxs = np.argwhere(np.abs(self.ky) < self.options["k_vector_epsilon"])
        # for idx in x_idxs:
        #     fft_jx[idx[0], idx[1]] = bz_to_jx[idx[0], idx[1]] * fft_bz_image[idx[0], idx[1]]
        # for idx in y_idxs:
        #     fft_jy[idx[0], idx[1]] = bz_to_jy[idx[0], idx[1]] * fft_bz_image[idx[0], idx[1]]

        return fft_jx, fft_jy

    # =================================

    def define_current_transformation(self, u_proj, bx_only=False, by_only=False):
        """
        Defines the transformation matrix that takes b to J
        """
        mu0 = np.pi * 4e-7
        if self.options["use_stand_off"]:
            exp_factor = np.exp(-1 * self.k * self.options["stand_off"])
        else:
            exp_factor = 1
        g = -mu0 / 2 * exp_factor

        b_to_jx = self.ky / (
            g * (u_proj[1] * self.ky - u_proj[0] * self.kx + 1j * u_proj[2] * self.k)
        )
        b_to_jy = self.kx / (
            g * (u_proj[0] * self.kx - u_proj[1] * self.ky - 1j * u_proj[2] * self.k)
        )
        if by_only:
            b_to_jx = (0 * self.ky + 1) / g
            b_to_jy = 0
        if bx_only:
            b_to_jx = 0
            b_to_jy = (0 * self.kx + 1) / g
        return b_to_jx, b_to_jy

    # =================================
    # ======= General FFT funcs =======
    # =================================

    def pad_image(self, image):
        """
        Apply a padding to the image to prepare for the FFT

        Parameters:
            image (2D array): image to have padding applied too

        Returns:
            padded_image (2D array): image with addition padding if request, otherwise returns
            the original image.

            padding_roi (2D array): is a meshgrid of indices that contain the all of the non
            padded elements. This is used for plotting in different sections of the code.

        """
        # get the shape of the image
        image_size = image.shape
        if self.options["fft_padding"]:
            # define the padding size
            y_pad = self.options["padding_factor"] * image_size[1]
            x_pad = self.options["padding_factor"] * image_size[0]

            # < TODO > check why the padding sometimes causes a transpose in the data (maybe a
            #  dict vs list issue).
            # performing the padding
            padded_image = np.pad(
                image,
                mode=self.options["padding_mode"],
                pad_width=((y_pad // 2, y_pad // 2), (x_pad // 2, x_pad // 2)),
            )
        else:
            padded_image = image
        # Define the region of interest for plotting
        padded_shape = padded_image.shape
        centre = [padded_shape[0] // 2, padded_shape[1] // 2]
        x = [
            np.linspace(
                centre[0] - image_size[0] / 2,
                centre[0] + image_size[0] / 2,
                image_size[0],
                dtype=int,
            )
        ]
        y = [
            np.linspace(
                centre[1] - image_size[1] / 2,
                centre[1] + image_size[1] / 2,
                image_size[1],
                dtype=int,
            )
        ]
        padding_roi = np.meshgrid(x, y)
        # plot the padded image if requested.
        if self.options["plot_padded_image"]:
            plt_trans = WrkPlt.Image(options=self.options, title="padded image")
            plt_trans.single_image(padded_image, sub_back=False, colour_map="viridis")
        return padded_image, padding_roi

    # =================================

    def define_k_vectors(self, fft_image_shape):
        """
        Function for defining the correct fourier space k vectors with the correct pixel sizes.
        Note: pixel size is taken from the options_reconstructions json file.

        parameters:

            fft_image_shape (tuple): size of the image to have the fourier transform applied to it.

        Returns:

            kx, ky, k (2D array, float64): fourier space vectors for calculations. The negative
            ky value is returned to maintain the correction image rotation.
        """

        # scaling for the k vectors so they are in the right units
        scaling = np.float64(2 * np.pi / self.pixel_size_for_fft)
        # get the fft frequencies and shift the ordering and forces type to be float64
        kx_vec = scaling * np.fft.fftshift(np.fft.fftfreq(fft_image_shape[0]))
        ky_vec = scaling * np.fft.fftshift(np.fft.fftfreq(fft_image_shape[1]))
        # Include a small factor in the k vectors to remove division by zero issues (min_k)
        # Make a meshgrid to pass back
        if self.options["use_k_vector_epsilon"]:
            kx, ky = np.meshgrid(
                kx_vec + self.options["k_vector_epsilon"], ky_vec + self.options["k_vector_epsilon"]
            )
        else:
            kx, ky = np.meshgrid(kx_vec, ky_vec)
        # Define the k mag vector
        k = np.sqrt(kx ** 2 + ky ** 2)
        # Take the negative of ky to maintain the correct image rotation
        return kx, -ky, k

    # =================================

    def get_image_filter(self):
        """ Computes a hanning image filter with both low and high pass filters.

        Returns:
            img_filter (2d array, float): bandpass filter to remove artifacts in the FFT process.
        """
        # Define Hanning filter to prevent noise amplification at frequencies higher than the
        # spatial resolution
        if self.options["hanning_filter"]:
            # New hanning filter method
            hx = np.hanning(self.k.shape[0])
            hy = np.hanning(self.k.shape[0])
            img_filter = np.sqrt(np.outer(hx, hy))
            # apply frequency cutoffs
            if self.options["use_hanning_high_cutoff"]:
                k_cut_high = 2 * np.pi / self.options["lambda_high_cutoff"]
                img_filter[(self.k > k_cut_high)] = 0
            if self.options["use_hanning_low_cutoff"]:
                k_cut_low = 2 * np.pi / self.options["lambda_low_cutoff"]
                img_filter[(self.k < k_cut_low)] = 0
        else:
            img_filter = 1
        return img_filter

    # =================================

    def plot_fft_stages(self, transformation, fft_image, fft_image_transformed, plt_stages=False):
        """ plots the various stages in the transformation process in fourier space """
        if plt_stages:
            title = "FFT trans"
            plots = [
                transformation.real,
                transformation.imag,
                fft_image.imag ** 2 + fft_image.real ** 2,
                fft_image_transformed.imag ** 2 + fft_image_transformed.real ** 2,
            ]
            titles = ["transformation real", "transformation imag", "FFT", "transformed"]
            plt_trans = WrkPlt.Image(options=self.options, title=title)
            plt_trans.multiple_images(plots, title=titles, sub_back=False, colour_map="viridis")

    # =================================
    # ===      Static methods       ===
    # =================================

    @staticmethod
    def remove_invalid_elements(array):
        """ replaces NaNs and infs with zero"""
        idxs = np.logical_or(np.isnan(array), np.isinf(array))
        array[idxs] = 0
        return array
