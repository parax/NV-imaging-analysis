# -*- coding: utf-8 -*-

"""
Hamiltonian reconstructor
---

A work in progress.

Docstrings should be written when this is complete...

"""
__author__ = "David Broadway"

# ============================================================================

import numpy as np

# ============================================================================

import basic.worker_data as WrkData
import basic.worker_plotting as WrkPlt
import basic.worker_hamiltonian as WrkHam
import basic.background_removal as sub_bg
import basic.worker_propagator as WrkProp
import basic.misc as misc
import os


# ============================================================================


class Reconstructor(object):
    """ Class for dealing with all reconstuction of the NV Hamiltonian from frequency fits of ODMR
    """

    def __init__(self, options):
        # Define all of the options dictionaries
        self.options = options
        self.prev_options = {}
        self.plt_def_opts = misc.json_to_dict("options/plt_default.json")
        self.plt_opts_gen = {**self.plt_def_opts, **misc.json_to_dict("options/plt_general.json")}
        self.plt_opts_mag = {**self.plt_def_opts, **misc.json_to_dict("options/plt_mag.json")}
        self.plt_opts_b = {**self.plt_def_opts, **misc.json_to_dict("options/plt_b.json")}
        self.plt_opts_e = {**self.plt_def_opts, **misc.json_to_dict("options/plt_e.json")}
        self.plt_opts_curr = {**self.plt_def_opts, **misc.json_to_dict("options/plt_curr.json")}
        # define consistent parameters
        self.output_dir = None
        self.is_processed = True

        # === LOAD DATA === #
        # create a data worker
        self.wd = WrkData.WorkerData(self.options, is_processed=self.is_processed)
        # Load data sets
        # < TODO > fix why loading the data here doesn't work
        self.load_fitted_data()

        # create a hamiltonian worker
        self.wh = WrkHam.WorkerHamiltonian(self.options)
        self.ham = self.wh.ham_model
        self.unv = self.ham.model.nv_signed_ori

        # Initialise data types
        self.peak_fit_roi = np.array([])
        self.hor_line_cuts = {}
        self.vert_line_cuts = {}
        self.hor_pixels = np.array([])
        self.vert_pixels = np.array([])
        self.bnv = np.array([])
        self.nv_d_shift = np.array([])
        self.b_xyz_prop = np.array([])
        self.fit_image_results = np.array([])
        self.ROI = np.array([])
        self.line_cut_fit_results = np.array([])

    # =================================

    def load_fitted_data(self):
        """ Function to load the data from the previous fitting """
        # Load the dataset
        self.wd.load_dataset()
        self.peak_fit_roi, self.ROI = self.wd.transform_dataset()
        # update the options to include the values that are obtained in the data workers
        self.options = self.wd.get_options()
        self.prev_options = self.wd.get_previous_options()

        if self.options["auto_read_b"]:
            end_string = self.options["filepath"].find("_processed")
            previous_dir = self.options["filepath"][0:end_string]
            self.wd.read_meta_data(filepath=previous_dir)
            self.options = {
                **self.options,
                **{
                    "b_mag": self.wd.metadata["Field Strength (G)"],
                    "b_theta": self.wd.metadata["Theta (deg)"],
                    "b_phi": self.wd.metadata["Phi (def)"],
                },
            }
        # === plot the previous fitted data === #
        if self.options["plot_previous_data"] == 1:
            # Plot the PL image
            plt_image = WrkPlt.Image(
                options=self.options,
                previous_options=self.prev_options,
                title="PL_previous_binning",
                filename="PL_previous_binning",
                pl_image=True,
            )
            plt_image.single_image(self.wd.image, cbar_title="PL (counts)")

            # Plot all of the other fitted parameters
            for parameter_key in self.prev_options["fit_param_defn"]:
                plt_pd = WrkPlt.Image(
                    options=self.options,
                    previous_options=self.prev_options,
                    title="previous fit " + parameter_key,
                )
                colourbar_label = self.prev_options["fit_parameter_unit"][parameter_key]
                plt_pd.multiple_images(
                    self.peak_fit_roi[parameter_key],
                    title="peak " + str(parameter_key),
                    cbar_title=colourbar_label,
                    colour_map="viridis",
                    ROI=self.ROI,
                )

    # =================================

    def bnv_extraction(self):
        """ Function for extracting out bnv and d from the base frequencies """
        num_peaks = len(self.peak_fit_roi["pos"])  # define the number of peaks in the data
        # Define plot worker for bnv
        plt_bnv = WrkPlt.Image(
            options=self.options, previous_options=self.prev_options, title="BNV", **self.plt_opts_b
        )
        if num_peaks == 1:
            # Define a temp variable
            temp_peak_pos = self.peak_fit_roi["pos"]
            # get the BNV value from the frequency vector
            self.bnv, self.nv_d_shift = self.get_bnv_and_d_shift(temp_peak_pos)
            self.subtract_ref_bnv_data()
            # plot bnv results
            plt_bnv.single_image(self.bnv, ROI=self.ROI, **self.plt_opts_b)
        elif num_peaks == 2:  # if there is only one NV axis measured
            # Define a temp variable
            temp_peak_pos = np.array([self.peak_fit_roi["pos"][0], self.peak_fit_roi["pos"][1]])
            # get the BNV value from the frequency vector
            self.bnv, self.nv_d_shift = self.get_bnv_and_d_shift(temp_peak_pos)
            self.subtract_ref_bnv_data()
            # plot bnv results
            plt_bnv.single_image(self.bnv, ROI=self.ROI, **self.plt_opts_b)
        elif num_peaks > 2:
            # get the BNV value from the frequency vectors
            self.bnv, self.nv_d_shift = self.multi_get_bnv_and_d_shift(self.peak_fit_roi)
            self.subtract_ref_bnv_data()
            # plot bnv results
            plt_bnv.multiple_images(
                self.bnv, ROI=self.ROI, **{**self.plt_opts_b, **{"sub_back": False}}
            )
        if self.nv_d_shift is not None:  # if d shift exists then plot it
            plt_d = WrkPlt.Image(
                options=self.options,
                previous_options=self.prev_options,
                title="Zero field splitting",
                **self.plt_opts_gen
            )
            plt_opts = {
                **{"cbar_title": "Zero field, D (MHz)", "ROI": self.ROI},
                **self.plt_opts_gen,
            }
            if num_peaks > 2:
                plt_d.multiple_images(self.nv_d_shift, **{**plt_opts, **{"sub_back": False}})
            else:
                plt_d.single_image(self.nv_d_shift, **plt_opts)

    # =================================

    def subtract_ref_bnv_data(self):
        """ subtracts the separate reference measurement from the fitted values """
        if self.options["subtract_ref"]:
            for idx in range(4):
                path = str(self.options["filepath_ref"]) + "/BNV " + str(idx) + ".txt"
                self.bnv[idx, ::] = self.bnv[idx, ::] - np.loadtxt(path)
                np.savetxt(
                    os.path.join(self.options["data_dir"], "BNV " + str(idx) + " sub ref.txt"),
                    self.bnv[idx, ::],
                )

    # =================================

    def get_all_line_cuts(self):
        """ Takes all of the fit parameters and obtains a line cut of the data in both directions
        """
        # Define the number of pixels that is used for plotting
        # < TODO > check if this is even required. Surely plotting doesn't need this
        hor_size = np.size(self.peak_fit_roi["pos"], 1)
        vert_size = np.size(self.peak_fit_roi["pos"], 2)
        self.hor_pixels = np.linspace(1, hor_size, hor_size)
        self.vert_pixels = np.linspace(1, vert_size, vert_size)

        # Get linecuts of all the previous fitted data
        for key in self.prev_options["fit_param_defn"]:
            peak_pos_line_cuts = WrkPlt.get_mulitple_line_cuts(self.peak_fit_roi[key], self.options)
            self.hor_line_cuts[key] = [peak_pos_line_cuts[key]["hor"] for key in peak_pos_line_cuts]
            self.vert_line_cuts[key] = [
                peak_pos_line_cuts[key]["vert"] for key in peak_pos_line_cuts
            ]
        # Get the line cuts of the bnv data
        bnv_line_cuts = WrkPlt.get_mulitple_line_cuts(self.bnv, self.options)
        self.hor_line_cuts["BNV"] = [bnv_line_cuts[key]["hor"] for key in bnv_line_cuts]
        self.vert_line_cuts["BNV"] = [bnv_line_cuts[key]["vert"] for key in bnv_line_cuts]

    # =================================

    def fit_linecut(self, direction="hor"):
        """ Function to take line cuts of the relevant data and then fits said data """
        # picks the correct key for the reconstuction method
        if self.options["recon_method"] in ["BNV", "approx_bxyz"]:
            key = "BNV"
        else:
            key = "pos"
        # Gets the line_cut_data and the pixels
        if direction == "hor":
            # Define the line cut data set
            line_cut_data = [peak for peak in self.hor_line_cuts[key]]
            pixels = self.hor_pixels
        else:
            # Define the line cut data set
            line_cut_data = [peak for peak in self.vert_line_cuts[key]]
            pixels = self.vert_pixels

        # === perform the line cut fit === #
        self.line_cut_fit_results = self.wh.recon_linecut(line_cut_data)

        # === Plot the results === #
        if self.options["plot_line_cuts"]:
            # < TODO > update plot worker for spectra to include the make_plots option
            # Plot the BNV line cuts
            plt_spec = WrkPlt.Spectra(self.options, previous_options=self.prev_options)
            plt_spec("BNV linecuts sub mean")
            plt_spec.multiple_data(line_cut_data, pixels, direction, sub_mean=True, linestyle=".-")
            plt_spec.style_spectra_ax(direction + " BNV line cuts", "pixels", "frequency (MHz)")

            # Plot the Bxyz line cuts
            plt_bxyz_line = WrkPlt.Spectra(self.options, previous_options=self.prev_options)
            plt_bxyz_line("hor Bxyz sub mean")
            b_idxs = [
                self.ham.model.param_defn.index("bx"),
                self.ham.model.param_defn.index("by"),
                self.ham.model.param_defn.index("bz"),
            ]
            data_to_plot = np.array(
                [
                    self.line_cut_fit_results[b_idxs[0]],
                    self.line_cut_fit_results[b_idxs[1]],
                    self.line_cut_fit_results[self.ham.model.param_defn.index("bz")],
                ]
            )
            labels = ["bx", "by", "bz"]
            plt_bxyz_line.multiple_data(data_to_plot, pixels, labels, sub_mean=True, linestyle=".-")
            plt_bxyz_line.style_spectra_ax(
                direction + " B fit line cuts", "pixels", "Magnetic field (G)"
            )

            # === plot D field === #
            if "d" in self.ham.model.param_defn:
                d_idx = self.ham.model.param_defn.index("d")
                plt_bxyz_line = WrkPlt.Spectra(self.options, previous_options=self.prev_options)
                plt_bxyz_line("hor D sub mean")
                label = "d"
                plt_bxyz_line.add_to_plot(
                    self.line_cut_fit_results[d_idx], linestyle=".-", label=label
                )
                plt_bxyz_line.style_spectra_ax(
                    direction + " D fit line cuts", "pixels", "Zero field splitting (MHz)"
                )

            # === plot E field === #
            if "ez" in self.ham.model.param_defn:
                e_idx = self.ham.model.param_defn.index("ez")
                plt_bxyz_line = WrkPlt.Spectra(self.options, previous_options=self.prev_options)
                plt_bxyz_line("hor E sub mean")
                label = "ez"
                plt_bxyz_line.add_to_plot(
                    self.line_cut_fit_results[e_idx], linestyle=".-", label=label
                )
                plt_bxyz_line.style_spectra_ax(
                    direction + " E fit line cuts", "pixels", "Electric field (kV/cm)"
                )

    # =================================

    def fit_pixels(self):
        """ Fit all of the pixels and then plot the results """
        if self.options["recon_reload"]:
            self.fit_image_results = {}
            for fit_param in self.ham.model.param_defn:
                path = str(self.options["output_dir"]) + "/data/raw fitted " + fit_param + ".txt"
                self.fit_image_results[fit_param] = np.loadtxt(path)
            return
        if self.options["fit_pixels"]:
            # perform the pixel fitting
            if self.options["recon_method"] in ["BNV", "approx_bxyz"]:
                # if recon method is BNV or approx_bxyz use just the BNV value
                self.fit_image_results = self.wh.recon_pixels(self.bnv, self.options)
            else:
                # else we need to use the frequencies as the hamiltonian is used.
                self.fit_image_results = self.wh.recon_pixels(
                    self.peak_fit_roi["pos"], self.options
                )

    # =================================

    def save_fitted_data(self):
        """ subtracts the separate reference measurement from the fitted values """
        if self.options["fit_pixels"]:
            for fit_param in self.ham.model.param_defn:
                np.savetxt(
                    os.path.join(self.options["data_dir"], "raw fitted " + fit_param + ".txt"),
                    self.fit_image_results[fit_param],
                )

    # =================================

    def subtract_ref_data(self):
        """ subtracts the separate reference measurement from the fitted values """
        if self.options["subtract_ref"]:
            for fit_param in self.ham.model.param_defn:
                path = str(self.options["filepath_ref"]) + "/raw fitted " + fit_param + ".txt"
                self.fit_image_results[fit_param] = self.fit_image_results[fit_param] - np.loadtxt(
                    path
                )
                np.savetxt(
                    os.path.join(self.options["data_dir"], "fitted " + fit_param + " sub ref.txt"),
                    self.fit_image_results[fit_param],
                )

    # =================================

    def magnetisation_backwards_propagation(self, b_map, b_type="bnv", mag_axis=None, bnv_axis=0):
        """ takes a magnetic field map and propagate it to obtain the magnetisation """
        if b_map.ndim == 3:
            # checks if more than one bnv map has been passed and picks the correct axis
            b_map = b_map[bnv_axis]
        # Define the propagator class
        prop = WrkProp.WorkerPropagator(self.options, self.unv)
        # Define the projection
        b_type_proj = {"bnv": self.unv[bnv_axis], "bx": [1, 0, 0], "by": [0, 1, 0], "bz": [0, 0, 1]}
        u_proj = b_type_proj[b_type]
        # fit background Bmap
        if self.options["subtract_b_background"]:
            b_map = sub_bg.remove_background(
                b_map,
                model=self.options["subtract_b_background_model"],
                results="basic",
                title="B_map (" + b_type + ") for m prop",
                plot_fit=True,
                options=self.options,
                **{**self.plt_opts_b, **{"paper_figure": False}}
            )
        # get mz from the b map
        mz = prop.magnetisation(b_map, u_proj=u_proj, magnetisation_axis=mag_axis)
        # Apply a simple guassian filtering
        # mz_filtered = sub_bg.image_filtering(mz, sigma=0.1)
        # Plot the results
        plt_pd = WrkPlt.Image(options=self.options, title="Mag from " + b_type, **self.plt_opts_mag)
        colourbar_label = r"Magnetisation, $M_z$ $\left(\mu_B nm^{-2}\right)$"
        plt_pd.single_image(mz, cbar_title=colourbar_label, ROI=self.ROI, **self.plt_opts_mag)
        return mz

    # =================================

    def current_backwards_propagation(self, b_map, b_type="bnv", bnv_axis=0, vector_b=None):
        """ takes a magnetic field map and propagate it to obtain the magnetisation """
        if b_map.ndim == 3:
            # checks if more than one bnv map has been passed and picks the correct axis
            b_map = b_map[bnv_axis]
        # Define the propagator class
        prop = WrkProp.WorkerPropagator(self.options, self.unv)
        # Define the projection
        b_type_proj = {"bnv": self.unv[bnv_axis], "bx": [1, 0, 0], "by": [0, 1, 0], "bz": [0, 0, 1]}
        u_proj = b_type_proj[b_type]
        if vector_b:
            b_type = "total vector"
        # fit background Bmap
        if self.options["subtract_b_background"]:
            b_map = sub_bg.remove_background(
                b_map,
                model=self.options["subtract_b_background_model"],
                results="basic",
                title="B_map (" + b_type + ") for m prop",
                plot_fit=True,
                options=self.options,
                **{**self.plt_opts_b, **{"paper_figure": False}}
            )
        # get mz from the b map
        jx, jy, j_norm = prop.current(b_map, u_proj=u_proj, vector_b=vector_b)
        # Apply a simple guassian filtering
        # mz_filtered = sub_bg.image_filtering(mz, sigma=0.1)
        # Plot the results
        plt_pd = WrkPlt.Image(options=self.options, title="jx from " + b_type)
        colourbar_label = r"Current density, $j_x$ $\left(A/m\right)$"
        plt_pd.single_image(jx, cbar_title=colourbar_label, ROI=self.ROI, **self.plt_opts_curr)

        plt_pd = WrkPlt.Image(options=self.options, title="jy from " + b_type)
        colourbar_label = r"Current density, $j_y$ $\left(A/m\right)$"
        plt_pd.single_image(jy, cbar_title=colourbar_label, ROI=self.ROI, **self.plt_opts_curr)

        plt_pd = WrkPlt.Image(options=self.options, title="j_norm from " + b_type)
        colourbar_label = r"Current density, $j_{\rm norm}$ $\left(A/m\right)$"
        plt_pd.single_image(j_norm, cbar_title=colourbar_label, ROI=self.ROI, **self.plt_opts_curr)

        self.plot_multiple_curr_line_cuts(jx, jy, j_norm)
        return jx, jy

    def plot_multiple_curr_line_cuts(self, jx, jy, j_norm):
        """ takes a list of maps and plots a horizontal line cut through the central pixel"""

        image_dimension = (
            len(jx) * self.options["raw_pixel_size"] * self.options["total_bin"] * 1e-6
        )
        x_linecut_values = np.linspace(-image_dimension, image_dimension, len(jx))

        plt_spec = WrkPlt.Spectra(self.options)
        plt_spec("Curr propagation line cuts")

        # Jx
        line_cut = WrkPlt.get_line_cuts(
            jx,
            self.options["linecut_hor_pos"],
            self.options["linecut_ver_pos"],
            self.options["linecut_thickness"],
        )
        plt_spec.add_to_plot(line_cut["hor"], x_array=x_linecut_values, linestyle="-", label="jx")
        # Jy
        line_cut = WrkPlt.get_line_cuts(
            jy,
            self.options["linecut_hor_pos"],
            self.options["linecut_ver_pos"],
            self.options["linecut_thickness"],
        )
        plt_spec.add_to_plot(line_cut["hor"], x_array=x_linecut_values, linestyle="-", label="jy")

        line_cut = WrkPlt.get_line_cuts(
            j_norm,
            self.options["linecut_hor_pos"],
            self.options["linecut_ver_pos"],
            self.options["linecut_thickness"],
        )
        plt_spec.add_to_plot(
            line_cut["hor"], x_array=x_linecut_values, linestyle="-", label="j_norm"
        )
        plt_spec.style_spectra_ax("j prop linecuts", "position (um)", "current (A/m)")

    # =================================

    def detect_edges(self, image, threshold=1):
        """ Function to try and detect edges of objects using the scikit package"""
        edges = self.wh.edge_detection(image)
        thres_min_idx = edges < threshold
        edges[thres_min_idx] = np.nan

        plt_edge = WrkPlt.Image(self.options, title="Edge detection")
        plt_opts = {
            **self.plt_opts_gen,
            **{"colourbar_label": "Edges height (a.u.)", "colour_map": "Greys", "ROI": self.ROI},
        }
        plt_edge.single_image(edges, **plt_opts)

    # =================================

    def bxyz_from_bnv(self, bnv, nv_axis=0):
        """
        Function that takes the bnv matrix and propagates this in fourier space to get the bxyz
        magnetic fields.
        """
        if bnv.ndim == 3:
            # checks if more than one bnv map has been passed and picks the correct axis
            bnv = bnv[nv_axis]
        # create instance of the propagator
        prop = WrkProp.WorkerPropagator(self.options, self.unv)
        # get the propagated bxyz
        bx, by, bz = prop.b_xyz(bnv, nv_axis=nv_axis)
        # Combine results for use elsewhere
        self.b_xyz_prop = np.array([bx, by, bz])
        # make array for plotting that contained bnv as well
        b_plotting = np.array([bnv, bx, by, bz])
        # create instance of plt worker
        plt_b = WrkPlt.Image(
            options=self.options, previous_options=self.prev_options, title="bnv prop to xyz"
        )
        # Define the various titles
        title = ["BNV propagation", "Propagated Bx", "Propagated By", "Propagated Bz"]
        # perform the plotting based off the plot options for magnetic fields

        plt_b.multiple_images(
            b_plotting, title=title, roi=self.ROI, **{**self.plt_opts_b, **{"sub_back": False}}
        )

    # =================================

    def multi_get_bnv_and_d_shift(self, peak_fits):
        num_peaks = len(peak_fits["pos"])  # define the number of peaks in the data
        # of the results
        # there is more than one b_nv in the data and as such we need to preallocate the size
        shape_x = np.size(peak_fits["pos"], 1)
        shape_y = np.size(peak_fits["pos"], 2)
        # preallocate for index referencing
        bnv = np.zeros((int(num_peaks / 2), shape_x, shape_y))
        nv_d_shift = np.zeros((int(num_peaks / 2), shape_x, shape_y))

        for idx in range(int(num_peaks / 2)):
            temp_peak_pos = np.array([peak_fits["pos"][idx], peak_fits["pos"][-idx - 1]])
            # get the BNV value from the frequency vector
            bnv[idx, ::], nv_d_shift[idx, ::] = self.get_bnv_and_d_shift(temp_peak_pos)
        return bnv, nv_d_shift

    # =================================

    def plot_results(self):
        if self.options["make_plots"]:
            for param_key in self.ham.model.param_defn:
                colourbar_label = self.ham.model.param_unit[param_key]
                title = self.ham.model.param_title[param_key]
                plt_pd = WrkPlt.Image(options=self.options, title=title)
                opts = self.plt_opts_gen
                if "b" in param_key:
                    opts = self.plt_opts_b
                if "e" in param_key:
                    opts = self.plt_opts_e
                plt_opts = {
                    **opts,
                    **{"sub_title": title, "cbar_title": colourbar_label, "ROI": self.ROI},
                }
                plt_pd.single_image(self.fit_image_results[param_key], **plt_opts)

    # =================================
    # ===     Static functions      ===
    # =================================

    @staticmethod
    def get_bnv_and_d_shift(peak_fits):
        """ gets the bnv and d shift of the data without any fitting"""
        gamma = 2.8  # MHz/G
        try:
            # If there is two peaks then get bnv as the difference and the d shift as the average
            bnv = np.abs(peak_fits[0, :, :] - peak_fits[1, :, :]) / (2 * gamma)
            d_shift = (peak_fits[0, :, :] + peak_fits[1, :, :]) / 2
        except IndexError:
            # if there is only one peak convert to magnetic field and ignore d
            bnv = np.abs(peak_fits[0, :, :]) / (2 * gamma)
            d_shift = None
        return bnv, d_shift
