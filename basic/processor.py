# -*- coding: utf-8 -*-

"""
processor
---

This module is designed as a clean object-oriented interface for processing the
data from the widefield NV centre setups. The hierarchy used is a hub-and-spokes
style where WidefieldProcessor (WP) delegates tasks out to separate Workers.
Most data should be sent through the options dictionary or possibly as an
array. Care should be taken to ensure data is accessed from child workers via
methods and not directly through __getattr__. This is to ensure the object
structure is not cyclic, in particular due to the pickling/serialising
requirements of the parallelising tools. If the objects reference cyclically
then the whole process will slow down by a lot.

TODO
    - remove all of the .fw. and .dw. calls here

"""
__author__ = "Sam Scholten"

# ============================================================================

import basic.worker_data as WD
import basic.worker_fitting as WF
import basic.worker_plotting as WP
import basic.misc as misc

# ============================================================================


class Processor(object):
    """ General object that handles the processing of raw widefield data,
    or given the is_processed flag, can read in already processed data.
    """

    def __init__(self, options, is_processed=False):
        self.options = options
        self.output_dir = None
        self.is_processed = is_processed
        self.plt_def_opts = misc.json_to_dict("options/plt_default.json")
        self.plt_opts_gen = {**self.plt_def_opts, **misc.json_to_dict("options/plt_general.json")}
        self.plt_opts_nv_pl = {**self.plt_def_opts, **misc.json_to_dict("options/plt_nv_pl.json")}

    # =================================

    def process_file(self):
        self.dw = WD.WorkerData(self.options)
        self.dw.load_dataset()
        self.dw.transform_dataset()
        self.options = self.dw.get_options()

    # =================================

    def plot_area_spectra(self):
        area_plots = WP.SpectraComparison(self.options)
        area_plots(data_worker=self.dw, filename="Area spectra comparision")
        area_plots.compare_regions()

        pixel_whole_plots = WP.SpectraComparison(self.options)
        pixel_whole_plots(
            data_worker=self.dw, filename="Single pix v whole spectra comparision.png"
        )
        pixel_whole_plots.compare_regions(area_plots=False)

    # =================================

    def fit_data(self):
        # Fit the ROI and make the plot
        self.fw = WF.WorkerFitting(self.options)

        self.pl_roi, self.sweep_vector, self.best_fit_result, self.scipy_best_fit, self.init_fit, self.fit_sweep_vector = self.fw.fit_roi(
            self.dw.sig_norm, self.dw.sweep_list
        )

        if self.options["make_plots"]:
            plotting_roi = WP.Spectra(self.options)

            plotting_roi("Full ROI fitting")

            plotting_roi.add_to_plot(self.pl_roi, x_array=self.dw.sweep_list, label="Data")
            plotting_roi.add_to_plot(
                self.init_fit, x_array=self.fit_sweep_vector, linestyle="k--", label="Initial fit"
            )
            plotting_roi.add_to_plot(
                self.scipy_best_fit, x_array=self.fit_sweep_vector, linestyle="r-", label="Best fit"
            )
            plotting_roi.style_spectra_ax("Full ROI fitting", "Frequency (MHz)", "PL (norm.)")

        # fit all of the pixels
        # -- ask user if they want to proceed here
        # -- (and/or add override in params)
        if self.options["fit_pixels"]:

            self.fw.fit_pixels(
                self.dw.sig_norm, self.best_fit_result, self.dw.sweep_list, self.dw.image_ROI.shape
            )

            # < TODO > fix this so it also plots the background
            for parameter_key in self.fw.fit_model.peaks.param_defn:
                colourbar_label = self.fw.fit_model.peaks.parameter_unit[parameter_key]
                plt_pd = WP.Image(options=self.options, title=parameter_key, filename=parameter_key)
                plt_pd.multiple_images(
                    self.fw.fit_image_results[parameter_key],
                    subtitle="peak " + str(parameter_key),
                    cbar_title=colourbar_label,
                    **self.plt_opts_gen
                )

    # =================================

    def plot_images(self):
        """ plot the full ROI and rebinned images """
        if self.options["make_plots"]:
            full_pl = WP.Image(options=self.options, title="PL", filename="PL_image", pl_image=True)
            full_pl.single_image(
                self.dw.image,
                sub_title="PL_image",
                cbar_title="PL (counts)",
                full_image=True,
                **self.plt_opts_nv_pl
            )
            roi_pl = WP.Image(
                options=self.options,
                title="PL (roi)",
                roi=True,
                filename="PL_bin_image",
                pl_image=True,
            )
            roi_pl.single_image(
                self.dw.image_ROI,
                sub_title="PL_bin_image",
                cbar_title="PL (counts)",
                **self.plt_opts_nv_pl
            )
