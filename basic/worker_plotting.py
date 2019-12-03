# -*- coding: utf-8 -*-

"""
plot_worker
----

A work in progress, designed to be a helper class to widefield processor, but
also any future workers.

# ======
# Don't plt.show() until the end of main
# One plotworker per plot!
# ======

"""
__author__ = "David Broadway"

# ============================================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import SI_LENGTH
import os
import simplejson

from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

# ============================================================================

import basic.background_removal as sub_bck

# ============================================================================
# <TODO> make matplotlib style files
# < TODO > fix plot worker to take just plotting options

matplotlib.use("Qt5Agg")


class PlotWorker(object):
    """ Docstring
    """

    def __init__(
        self,
        options,
        previous_options=None,
        title=None,
        roi=False,
        filename=None,
        paper_figure=False,
        **kwargs,
    ):
        self.options = options
        self.previous_options = previous_options
        self.roi = roi
        self.title = title
        self.filename = filename
        if paper_figure:
            plt.style.use("paper_image_figure")
        else:
            plt.style.use("classic")
            matplotlib.rcParams.update({"font.size": 8})

        if self.title:
            self.fig = plt.figure(self.title)
            self.ax = self.fig.gca()
        if self.options["output_dir"] is None:
            raise RuntimeError("No folder found - import data first")

    # =================================

    def style_axis_title(self, ax=None, spectra_plot=False):
        """ removes scientific notation from the colourbar axis and places it
        in the unit label.
        Note that it does this by finding where the ( ) are so if the colour
        title is changed to no parentheses this will break
        """

        # <TODO> add an error checker for cbar existence
        # tight layout needed to draw offset text grabbed below, but that
        # pushes the cbar label off to the right edge, so we bring it in a bit
        # this is a *slow* method, but fast enough (& there's no other way...)
        # < NOTE > to be seen whether this effects other plots
        if ax is None:
            ax = self.ax
        plt.subplots_adjust(right=0.85)
        old_title = ax.get_ylabel()
        # fix label offset
        scale = ax.yaxis.get_offset_text().get_text()
        insert_idx = old_title.find("(") + 1
        new_title = old_title[:insert_idx] + str(scale) + " " + old_title[insert_idx:]
        if scale:
            if spectra_plot is False:
                ax.set_ylabel(new_title, rotation=270)
            else:
                ax.set_ylabel(new_title)

    # =================================

    def style_image_ax(
        self, im, fig=None, ax=None, ax_title="", cbar_title="", pixel_size=1, full_image=False
    ):
        """ function that enforces the style of image figures
        """
        # set the title for the ax
        ax.set_title(ax_title)
        # removes the ticks from both axes
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # === add a colour bar === #
        cbar = self.add_colorbar(im, fig, ax)
        cbar.ax.set_ylabel(cbar_title, rotation=270)

        cbar.outline.set_linewidth(0.5)
        # === add a scale bar === #
        # import matplotlib.font_manager as fm
        # fontprops = fm.FontProperties(size=6)
        self.scalebar = ScaleBar(pixel_size, "um", SI_LENGTH, font_properties={"size": 6})
        self.scalebar.location = "lower left"
        self.scalebar.box_alpha = 0.5
        ax.add_artist(self.scalebar)

        # add a circular boundary if required
        if self.options["ROI"] == "Circle" and not full_image:
            circ = Circle(
                (self.options["ROI_radius"], self.options["ROI_radius"]),
                self.options["ROI_radius"],
                color="k",
                fill=False,
                linewidth=1,
            )
            ax.add_patch(circ)
            ax.axis("off")

        elif self.previous_options is not None:
            if self.previous_options["ROI"] == "Circle":
                if self.options["num_bins"]:
                    radius = self.previous_options["ROI_radius"] / self.options["num_bins"]
                else:
                    radius = self.previous_options["ROI_radius"]
                circ = Circle((radius, radius), radius, edgecolor="k", fill=False, linewidth=2)
                ax.add_patch(circ)
                ax.axis("off")

        # Styles the colour bar axis title to remove auto SI notation and places the SI notations
        # into the units section of the title i.e. (SI unit)
        self.style_axis_title(ax=cbar.ax)

    # =================================

    def style_spectra_ax(self, title, xlabel, ylabel, ax=None):
        """ Defines the style of spectrum plots axes
        """
        if ax is None:
            ax = self.ax
        pad_y = 0.05
        ybins = 4
        xbins = 6
        self.title = title
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # Gets the limits of the plot
        line = ax.lines
        x_min = np.nanmin([np.nanmin(line[idx].get_xdata()) for idx in range(len(line))])
        x_max = np.nanmax([np.nanmax(line[idx].get_xdata()) for idx in range(len(line))])
        y_min = np.nanmin([np.nanmin(line[idx].get_ydata()) for idx in range(len(line))])
        y_max = np.nanmax([np.nanmax(line[idx].get_ydata()) for idx in range(len(line))])
        # pads the y axis a little
        pad_y = pad_y * (y_max - y_min)
        # sets the axis limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min - pad_y, y_max + pad_y])
        # sets the ticks on insdie of axes
        ax.tick_params(direction="in", bottom=True, top=True, left=True, right=True, labelsize=6)
        # Defines the number of ticks
        ax.locator_params(axis="y", nbins=ybins)
        ax.locator_params(axis="x", nbins=xbins)
        ax.tick_params(width=1)
        ax.legend(prop={"size": 6})
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(3.1, 3)
        if self.options["save_plots"]:
            self._save_figure_and_data(title, ax=ax)

    # =================================

    @staticmethod
    def add_colorbar(im, fig, ax, aspect=20, pad_fraction=1, **kwargs):
        """Add a vertical color bar to an image plot."""
        divider = make_axes_locatable(ax)
        width = axes_size.AxesY(ax, aspect=1.0 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        cbar = fig.colorbar(im, cax=cax, **kwargs)
        tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.linewidth = 0.5
        cbar.ax.tick_params(direction="in", labelsize=6, size=2)

        return cbar

    # =================================

    @staticmethod
    def _add_patch_square_centre(ax, area_c, area_size, label=None, edgecolor="b"):
        """  add the ROI rectangles"""
        rect_corner = [int(area_c[0] - area_size / 2), int(area_c[1] - area_size / 2)]
        rect = patches.Rectangle(
            (rect_corner[0], rect_corner[1]),
            int(area_size),
            int(area_size),
            linewidth=1,
            edgecolor=edgecolor,
            facecolor="none",
        )
        ax.add_patch(rect)
        if label:
            # Add label for the square
            plt.text(
                area_c[0],
                area_c[1] - area_size / 2 - 2,
                label,
                {"color": edgecolor, "fontsize": 10, "ha": "center", "va": "center"},
            )

    # ===============================

    @staticmethod
    def _add_patch_rect(
        ax, rect_corner_x, rect_corner_y, size_x, size_y, label=None, edgecolor="b"
    ):
        """  add the ROI rectangle to a ax and labels the rectangle if a label has been given """
        rect = patches.Rectangle(
            (rect_corner_x, rect_corner_y),
            int(size_x),
            int(size_y),
            linewidth=1,
            edgecolor=edgecolor,
            facecolor="none",
        )
        ax.add_patch(rect)
        if label:
            plt.text(
                rect_corner_x + size_x / 2,
                rect_corner_y - 2,
                label,
                {"color": edgecolor, "fontsize": 10, "ha": "center", "va": "bottom"},
            )

    # =================================

    def save_as_txt(self, image, filename=None):
        """ save the image array as a txt file (filename inc. pattern?)"""
        if filename is None:
            filename = self.filename
        if self.options["save_plots"]:
            np.savetxt(os.path.join(self.options["data_dir"], filename + ".txt"), image)

    # =================================

    def save_figure(self, filename=None):
        """ save figure (filename inc. pattern?)"""
        # < TODO > needs to error checkers in here for existence
        if filename is None:
            filename = self.filename
        if filename is not None and self.options["save_plots"]:
            self.fig.savefig(
                os.path.join(
                    self.options["output_dir"], filename + "." + self.options["save_fig_type"]
                ),
                bbox_inches="tight",
                dpi=self.options["dpi"],
                transparent=True,
            )
            self.fig.savefig(
                os.path.join(self.options["data_dir"], filename + ".pdf"),
                dpi=self.options["dpi"],
                bbox_inches="tight",
                transparent=True,
            )

    # =================================

    def _save_figure_and_data(self, fig_name, ax=None):
        if ax is None:
            ax = self.ax
        all_plotted_data = ax.lines
        data_labels = ax.get_legend_handles_labels()[1]
        # get data as dict
        data_to_save_dict = {}
        idx = 0
        for key in data_labels:
            data_to_save_dict[key + "_sweep_data"] = all_plotted_data[idx].get_xdata().tolist()
            data_to_save_dict[key + "_PL_data"] = all_plotted_data[idx].get_ydata().tolist()
            idx += 1
        f = open(str(self.options["data_dir"]) + str("/" + fig_name + ".txt"), "w")
        f.write(simplejson.dumps(data_to_save_dict, indent=4, sort_keys=True))
        f.close()
        self.fig.savefig(
            os.path.join(
                self.options["output_dir"], fig_name + "." + self.options["save_fig_type"]
            ),
            bbox_inches="tight",
            dpi=self.options["dpi"],
            transparent=True,
        )
        self.fig.savefig(
            os.path.join(self.options["output_dir"], fig_name + ".pdf"),
            bbox_inches="tight",
            dpi=self.options["dpi"],
            transparent=True,
        )


# ============================================================================
# ======================== Image plotting class ==============================
# ============================================================================


class Image(PlotWorker):
    """  Subclass of PlotWorker that deals with the plotting of images.
    """

    # =================================

    def __init__(self, pl_image=False, **kwargs):
        """  """
        self.pl_image = pl_image
        super().__init__(**kwargs)

    # =================================

    def single_image(
        self,
        image,
        sub_title=None,
        colour_map="Greys_r",
        cbar_title="",
        figax=None,
        ROI=None,
        symmetric_axis=False,
        c_range=None,
        auto_range=True,
        sub_back=False,
        sub_back_type="mean",
        sub_back_multiple_times=1,
        filter_image=False,
        filter_type="gaussian",
        filter_sigma=0,
        full_image=False,
        save_figure=False,
        **kwargs,
    ):
        """ function for plotting a single image plot
        """
        # if make_plots is false do nothing
        if self.options["make_plots"] is False:
            return
        # if the fig and ax are passed then override the values made in the parent __init__
        if figax is not None:
            self.fig, self.ax = figax
        # if no sub title is given then use the value made in the parent __init__
        if sub_title is None:
            sub_title = self.title

        # Remove the background of the image if requested
        if sub_back:
            upper_thres = kwargs["upper_threshold"]
            lower_thres = kwargs["lower_threshold"]

            for ii in range(sub_back_multiple_times):
                opts = {
                    **kwargs,
                    **{"upper_threshold": upper_thres[ii], "lower_threshold": lower_thres[ii]},
                }
                image = sub_bck.remove_background(
                    image,
                    model=sub_back_type,
                    options=self.options,
                    results="basic",
                    plot_fit=sub_back,
                    cbar_title=cbar_title,
                    title=sub_title + " (rb)" + str(ii),
                    colour_map=colour_map,
                    **opts,
                )

        # If a region of interest is passed to the functions and it hasn't already been already
        # applied to the image then use the ROI
        try:
            self.options["use_ROI_for_fit"]
            if self.options["use_ROI_for_fit"] is False and ROI:
                image = image[ROI[0], ROI[1]]
        except:
            image = image

        if filter_image:
            upper_thres = kwargs["upper_threshold"]
            lower_thres = kwargs["lower_threshold"]
            opts = {
                **kwargs,
                **{"upper_threshold": upper_thres[-1], "lower_threshold": lower_thres[-1]},
            }
            image = sub_bck.image_filtering(
                image,
                flt_type=filter_type,
                sigma=filter_sigma,
                options=self.options,
                plot_fit=filter_image,
                cbar_title=cbar_title,
                title=sub_title + " (filter)",
                colour_map=colour_map,
                **opts,
            )

        # Defines the range for the colour bar
        if symmetric_axis:
            c_range = [-np.nanmax(np.abs(image)), np.nanmax(np.abs(image))]
        elif auto_range:
            c_range = [np.nanmin(image), np.nanmax(image)]

        im = self.ax.imshow(image, cmap=colour_map, vmin=c_range[0], vmax=c_range[1])

        pixel_size = self.options["raw_pixel_size"] * self.options["total_bin"]

        if self.pl_image:
            if self.options["annotate_plots"]:
                self.annotate_image()
            if not self.roi:
                pixel_size = self.options["raw_pixel_size"] * self.options["original_bin"]

        self.style_image_ax(
            im,
            fig=self.fig,
            ax=self.ax,
            ax_title=sub_title,
            cbar_title=cbar_title,
            pixel_size=pixel_size,
            full_image=full_image,
        )
        self.save_as_txt(image, filename=sub_title)
        if save_figure:
            self.save_figure(filename=sub_title)
        return

    # =================================

    def multiple_images(
        self, data, title=None, cbar_title=None, ROI=None, colour_map="Greys_r", **kwargs
    ):
        """ docstring
        """
        kwargs["save_figure"] = False
        if self.options["make_plots"] is False:
            return
        if type(title) is list:
            sub_title = title[0]
        else:
            sub_title = title

        size = len(data)
        if size == 1:
            self.single_image(
                data[0, :, :],
                sub_title=self.title + " 0",
                colour_map=colour_map,
                cbar_title=cbar_title,
                ROI=ROI,
                **kwargs,
            )
        elif size == 2:
            for idx in range(size):
                ax = plt.subplot(1, 2, idx + 1)
                # print(idx+1)
                if isinstance(title, list):
                    sub_title = title[idx]
                else:
                    sub_title = self.title + " " + str(idx)
                if isinstance(cbar_title, list):
                    cbar_subtitle = cbar_title[idx]
                else:
                    cbar_subtitle = cbar_title
                self.single_image(
                    data[idx, :, :],
                    sub_title=sub_title,
                    cbar_title=cbar_subtitle,
                    colour_map=colour_map,
                    figax=(self.fig, ax),
                    ROI=ROI,
                    **kwargs,
                )
        else:
            if int(np.sqrt(size) + 0.5) ** 2 == size:
                s_num = np.sqrt(size)
            else:
                s_num = np.sqrt(size) + 1
            for idx in range(size):
                ax = plt.subplot(s_num, s_num, idx + 1)
                # print(idx+1)
                if isinstance(title, list):
                    sub_title = title[idx]
                else:
                    sub_title = self.title + " " + str(idx)
                if isinstance(cbar_title, list):
                    cbar_subtitle = cbar_title[idx]
                else:
                    cbar_subtitle = cbar_title
                try:
                    plt_data = data[idx, ::]
                except:
                    plt_data = data[idx]
                self.single_image(
                    plt_data,
                    sub_title=sub_title,
                    colour_map=colour_map,
                    cbar_title=cbar_subtitle,
                    figax=(self.fig, ax),
                    ROI=ROI,
                    **kwargs,
                )
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        plt.tight_layout()
        # === Save the figure ===#
        self.save_figure(filename=self.title)

    def annotate_image(self):
        """ Function for adding coloured rectangles and labels to image plots for when either a
        smaller region of interest is used for other plots or we are looking at a comparision
        between different regions
        """
        if self.roi:
            # Add square for first region
            self._add_patch_square_centre(
                self.ax,
                self.options["area_1_centre"],
                self.options["area_1_size"],
                label="Area 1",
                edgecolor="b",
            )
            # Add square for second region
            self._add_patch_square_centre(
                self.ax,
                self.options["area_2_centre"],
                self.options["area_2_size"],
                label="Area 2",
                edgecolor="tab:orange",
            )

        else:
            binning = self.options["num_bins"]
            if binning == 0:
                binning = 1
            if self.options["ROI"] == "Square":
                size = self.options["ROI_radius"] * binning * 2
                corner = [
                    self.options["ROI_centre"][0] * binning - size / 2,
                    self.options["ROI_centre"][1] * binning - size / 2,
                ]

                self._add_patch_rect(
                    self.ax, corner[0], corner[1], size, size, label="ROI", edgecolor="r"
                )

            elif self.options["ROI"] == "Circle":
                radius = self.options["ROI_radius"] * binning
                centre = self.options["ROI_centre"]
                circ = Circle(
                    (centre[0], centre[1]), radius, edgecolor="r", fill=False, linewidth=2
                )
                plt.text(
                    centre[0] - radius,
                    centre[1] - radius,
                    r"ROI",
                    {"color": "red", "fontsize": 10, "ha": "left", "va": "top"},
                )
                self.ax.add_patch(circ)

            elif self.options["ROI"] == "Rectangle":
                start_x = binning * (
                    self.options["ROI_centre"][0] - self.options["ROI_rect_size"][0]
                )
                start_y = binning * (
                    self.options["ROI_centre"][1] - self.options["ROI_rect_size"][1]
                )
                size_x = 2 * binning * self.options["ROI_rect_size"][0]
                size_y = 2 * binning * self.options["ROI_rect_size"][1]

                self._add_patch_rect(
                    self.ax, start_x, start_y, size_x, size_y, label="ROI", edgecolor="r"
                )


# ============================================================================
# ========================  Spectra plotting classes =========================
# ============================================================================


class Spectra(PlotWorker):
    """ Docstring
    """

    def __call__(self, filename=None, figax=None):
        """ , filename including pattern (i.e. PL.png) """

        if self.options["output_dir"] is None:
            raise RuntimeError("No folder found - import data first")

        self.filename = filename
        if figax is not None:
            self.fig, self.ax = figax
        else:
            self.fig, self.ax = plt.subplots()
        return self.fig

    # =================================

    def add_to_plot(self, new_pl_array, x_array=None, linestyle=".", label="", ax=None):
        """ Add additional data to a spectra plot
        """
        if ax is None:
            ax = self.ax
        if x_array is None:
            ax.plot(new_pl_array, linestyle, label=label)
        else:
            ax.plot(x_array, new_pl_array, linestyle, label=label)

    # =================================

    def multiple_data(self, data, base_title=None, sub_mean=False, linestyle="."):
        for idx in range(len(data)):
            if isinstance(base_title, list):
                label = base_title[idx]
            else:
                label = base_title + " " + str(idx)
            if sub_mean:
                self.add_to_plot(data[idx] - np.mean(data[idx]), label=label, linestyle=linestyle)
            else:
                self.add_to_plot(data[idx], label=label, linestyle=linestyle)


class SpectraComparison(PlotWorker):
    """ Class is used to plot spectra from different areas of the image
    """

    # =================================

    def __call__(self, data_worker=None, filename=None):
        """ , filename including pattern (i.e. PL.png) """
        self.dw = data_worker
        self.sweep_list = self.dw.sweep_list
        self.sig = self.dw.sig
        self.ref = self.dw.ref
        if self.options["output_dir"] is None:
            raise RuntimeError("No folder found - import data first")
        if filename is None:
            raise RuntimeError("fix me")
        self.filename = filename
        if self.options["used_ref"] is True:
            self.fig, self.ax = plt.subplots(
                2, 2, figsize=(12, 6), dpi=100, facecolor="w", edgecolor="k"
            )
        else:
            self.fig, self.ax = plt.subplots()

    def _get_data_to_plot(self, area_plots=True):
        if area_plots is False:
            self.area_1_sig = self.sig[
                :, self.options["single_pixel_check"][0], self.options["single_pixel_check"][1]
            ]
            self.area_1_ref = self.ref[
                :, self.options["single_pixel_check"][0], self.options["single_pixel_check"][1]
            ]

            self.area_2_sig = np.nansum(np.nansum(self.sig, 2), 1)
            self.area_2_ref = np.nansum(np.nansum(self.ref, 2), 1)

            self.data_titles = ["Pixel", "Whole"]
        else:
            area_roi_1 = self.dw.define_area_roi_centre(
                self.options["area_1_centre"], self.options["area_1_size"]
            )
            area_roi_2 = self.dw.define_area_roi_centre(
                self.options["area_2_centre"], self.options["area_2_size"]
            )

            self.area_1_sig = np.nansum(np.nansum(self.sig[:, area_roi_1[0], area_roi_1[1]], 2), 1)
            self.area_1_ref = np.nansum(np.nansum(self.ref[:, area_roi_1[0], area_roi_1[1]], 2), 1)

            self.area_2_sig = np.nansum(np.nansum(self.sig[:, area_roi_2[0], area_roi_2[1]], 2), 1)
            self.area_2_ref = np.nansum(np.nansum(self.ref[:, area_roi_2[0], area_roi_2[1]], 2), 1)
            self.data_titles = ["Area 1", "Area 2"]

        area_1_sub = self.area_1_sig - self.area_1_ref
        area_1_sub = area_1_sub - min(area_1_sub)
        self.area_1_sub = area_1_sub / max(area_1_sub)

        area_2_sub = self.area_2_sig - self.area_2_ref
        area_2_sub = area_2_sub - min(area_2_sub)
        self.area_2_sub = area_2_sub / max(area_2_sub)

        self.div_area_1 = self.area_1_sig / self.area_1_ref
        self.div_area_2 = self.area_2_sig / self.area_2_ref

    # =================================

    def _get_data_to_plot_no_ref(self, area_plots=True):
        if area_plots is False:
            self.area_1_sig = self.sig[
                :, self.options["single_pixel_check"][0], self.options["single_pixel_check"][1]
            ]
            self.area_1_sig = self.area_1_sig / max(self.area_1_sig)

            self.area_2_sig = np.nansum(np.nansum(self.sig, 2), 1)
            self.area_2_sig = self.area_2_sig / max(self.area_2_sig)
            self.data_titles = ["Pixel", "Whole"]
        else:
            area_ROI_1 = self.dw.define_area_roi_centre(
                self.options["area_1_centre"], self.options["area_1_size"]
            )
            self.area_1_sig = np.nansum(np.nansum(self.sig[:, area_ROI_1[0], area_ROI_1[1]], 2), 1)
            self.area_1_sig = self.area_1_sig / max(self.area_1_sig)

            area_ROI_2 = self.dw.define_area_roi_centre(
                self.options["area_2_centre"], self.options["area_2_size"]
            )
            self.area_2_sig = np.nansum(np.nansum(self.sig[:, area_ROI_2[0], area_ROI_2[1]], 2), 1)

            self.area_2_sig = self.area_2_sig / max(self.area_2_sig)
            self.data_titles = ["Area 1", "Area 2"]

    # =================================

    def compare_regions(self, area_plots=True):
        if self.options["used_ref"] is True:
            self._get_data_to_plot(area_plots=area_plots)
        else:
            self._get_data_to_plot_no_ref(area_plots=area_plots)

        # define the plotting figure
        plotting_area = Spectra(self.options)
        plotting_area("Area_plots.png", figax=(self.fig, self.ax))
        if self.options["used_ref"] is True:
            # Plot the first region
            plotting_area.add_to_plot(
                self.area_1_sig,
                x_array=self.sweep_list,
                label="sig",
                ax=self.ax[0, 0],
                linestyle=".-",
            )
            plotting_area.add_to_plot(
                self.area_1_ref,
                x_array=self.sweep_list,
                label="ref",
                ax=self.ax[0, 0],
                linestyle=".-",
            )

            plotting_area.style_spectra_ax(
                "Sig & ref from " + self.data_titles[0],
                "Frequency (MHz)",
                "PL (counts.)",
                ax=self.ax[0, 0],
            )
            # Plot the second region
            plotting_area.add_to_plot(
                self.area_2_sig,
                x_array=self.sweep_list,
                label="sig",
                ax=self.ax[0, 1],
                linestyle=".-",
            )
            plotting_area.add_to_plot(
                self.area_2_ref,
                x_array=self.sweep_list,
                label="ref",
                ax=self.ax[0, 1],
                linestyle=".-",
            )
            plotting_area.style_spectra_ax(
                "Sig & ref from " + self.data_titles[1],
                "Frequency (MHz)",
                "PL (counts.)",
                ax=self.ax[0, 1],
            )

            # plot the subtraction normalisation
            plotting_area.add_to_plot(
                self.area_1_sub,
                x_array=self.sweep_list,
                label=self.data_titles[0],
                ax=self.ax[1, 0],
                linestyle=".-",
            )
            plotting_area.add_to_plot(
                self.area_2_sub,
                x_array=self.sweep_list,
                label=self.data_titles[1],
                ax=self.ax[1, 0],
                linestyle=".-",
            )
            plotting_area.style_spectra_ax(
                "Subtraction normalisation", "Frequency (MHz)", "PL (counts.)", ax=self.ax[1, 0]
            )

            # plot the division normalisation
            plotting_area.add_to_plot(
                self.div_area_1,
                x_array=self.sweep_list,
                label=self.data_titles[0],
                ax=self.ax[1, 1],
                linestyle=".-",
            )
            plotting_area.add_to_plot(
                self.div_area_2,
                x_array=self.sweep_list,
                label=self.data_titles[1],
                ax=self.ax[1, 1],
                linestyle=".-",
            )
            plotting_area.style_spectra_ax(
                "Division normalisation", "Frequency (MHz)", "PL (counts.)", ax=self.ax[1, 1]
            )

            # Make better looking plots
            plt.subplots_adjust(
                top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35
            )
            plt.tight_layout()
            self.style_axis_title(ax=self.ax[0, 0], spectra_plot=True)
            self.style_axis_title(ax=self.ax[0, 1], spectra_plot=True)
        else:
            plotting_area.add_to_plot(
                self.area_1_sig, x_array=self.sweep_list, label=self.data_titles[0], linestyle=".-"
            )
            plotting_area.add_to_plot(
                self.area_2_sig, x_array=self.sweep_list, label=self.data_titles[1], linestyle=".-"
            )
            plotting_area.style_spectra_ax(self.filename[:-4], "Frequency (MHz)", "PL (counts.)")
            plt.tight_layout()

        if self.options["save_plots"]:
            self.fig.savefig(os.path.join(self.options["output_dir"], self.filename))


# ============================================================================


def get_mulitple_line_cuts(images, opts):
    """ Takes multiple images and returns the horizontal and vertical line cuts for each image """
    hor_pos = opts["linecut_hor_pos"]
    vert_pos = opts["linecut_ver_pos"]
    thickness = opts["linecut_thickness"]
    line_cuts = {}
    for idx in range(len(images)):
        try:
            line_cuts[str(idx)] = get_line_cuts(images[idx, ::], hor_pos, vert_pos, thickness)
        except:
            line_cuts[str(idx)] = get_line_cuts(images, hor_pos, vert_pos, thickness)
    return line_cuts


def get_line_cuts(image, hor_pos, vert_pos, thickness):
    """ Takes an image and returns a dictionary of the line cuts"""

    return {
        "hor": np.nanmean(image[hor_pos - thickness : hor_pos + thickness, :], axis=0),
        "vert": np.nanmean(image[:, vert_pos - thickness : vert_pos + thickness], axis=1),
    }
