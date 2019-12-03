# -*- coding: utf-8 -*-

"""
DataWorker
---

Designed to do all of the io work for the WidefieldProcessor. Specifically
reading the raw data and then trasforming it. All args should be stored in
the options dictionary passed on init, or bool types. Similarly to pass any
objects back, methods should be written for DataWorker for simplicity of
reading and rebugging.

"""
__author__ = "Sam Scholten"

# ============================================================================

import numpy as np
import os
import re
import warnings
import json
import pathlib

# ============================================================================

import basic.misc as misc

# ============================================================================


class WorkerData(object):
    """worker_data.py
    Mostly what is called here is just load_dataset and transform_dataset,
    the rest are helpers.
    """

    def __init__(self, options, is_processed=None):
        self.options = options
        self.is_processed = is_processed

    # =================================

    def get_raw_fit_options(self):
        """ Get the dictionaries that were used for the inital ODMR fitting
        """
        filepath = os.path.normpath(self.options["filepath"] + "/saved_options.json")
        f = open(filepath)
        json_str = f.read()
        self.prev_options = json.loads(json_str)

    # =================================

    # < TODO > make a new subclass for the this worker for reading in data
    # < TODO > make a new subclass for data manipulation

    def reload_dataset(self):
        # < TODO > finish this function for both processed and raw datasets
        self.options["filepath_data"] = self.options["filepath"] + "\\data"
        # === Read in the previous options === #
        self.get_raw_fit_options()

        # === get the binning for reference
        self.options["original_bin"] = self.prev_options["total_bin"]
        if int(self.options["num_bins"]) == 0:
            self.options["total_bin"] = self.options["original_bin"]
        else:
            self.options["total_bin"] = self.options["original_bin"] * int(self.options["num_bins"])

        # === Define the new file path
        output_dir = pathlib.PurePosixPath(
            self.options["filepath"]
            + "/"
            + self.options["recon_method"]
            + "_bin_"
            + str(self.options["total_bin"])
        )
        if not os.path.exists(output_dir):
            raise ("You are trying to reload a dataset that does not exist")
        self.options["output_dir"] = output_dir

    # =================================

    def load_dataset(self):
        """ load raw data, metadata and makes new processed (output) folder
        """
        if self.is_processed:
            # TODO sam change to dict_to_json? -- how to specify filename
            self.options["filepath_data"] = self.options["filepath"] + "/data"
            # === Read in the previous options === #
            self.get_raw_fit_options()

            # === initialise the dictionary for the prev fitted data === #
            self.peak_fit = {}
            shape = self._read_processed_data(self.prev_options["fit_param_defn"][0] + " 0").shape
            for fit_key in self.prev_options["fit_param_defn"]:
                self.peak_fit[fit_key] = np.zeros(
                    (self.prev_options["num_peaks"], shape[0], shape[1])
                )

            # === Read in the previous fitted data === #
            for fit_key in self.prev_options["fit_param_defn"]:
                for idx in range(self.prev_options["num_peaks"]):
                    self.peak_fit[fit_key][idx, :, :] = self._read_processed_data(
                        fit_key + " " + str(idx)
                    )
            # === load previous image data
            self.image = np.loadtxt(self.options["filepath_data"] + "/PL_bin_image.txt")

            # === get the new binning for reference
            self.options["original_bin"] = self.prev_options["total_bin"]
            if int(self.options["num_bins"]) == 0:
                self.options["total_bin"] = self.options["original_bin"]
            else:
                self.options["total_bin"] = self.options["original_bin"] * int(
                    self.options["num_bins"]
                )

            # === Define the new file path
            output_dir = pathlib.PurePosixPath(
                self.options["filepath"]
                + "/"
                + self.options["recon_method"]
                + "_bin_"
                + str(self.options["total_bin"])
            )
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            self.options["output_dir"] = output_dir
        # === load image data
        else:
            with open(os.path.normpath(self.options["filepath"]), "r") as fid:
                self.raw_data = np.fromfile(fid, dtype=np.float32())[2:]

            self.read_meta_data()

            # === make output_dir for processed data
            bin_conversion = [1, 2, 4, 8, 16, 32]
            self.options["original_bin"] = bin_conversion[int(self.metadata["Binning"])]
            self.options["total_bin"] = self.options["original_bin"] * int(self.options["num_bins"])

            output_dir = pathlib.PurePosixPath(
                self.options["filepath"] + "_processed" + "_bin_" + str(self.options["total_bin"])
            )
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            self.options["output_dir"] = output_dir

        self.options["data_dir"] = output_dir / "data"
        if not os.path.exists(self.options["data_dir"]):
            os.mkdir(self.options["data_dir"])

    def read_meta_data(self, filepath=None):
        # === get metadata
        # first read off the sweep (i.e. freq/tau times) string, then read
        # out each pair of key:value items in the metadata file. If the
        # value is numeric, store it as a float, else keep the string.
        # Store sweep in a list, metadata in a dict
        # ===
        if filepath is None:
            filepath = self.options["filepath"]
        with open(os.path.normpath(filepath + "_metaSpool.txt"), "r") as fid:
            sweep_str = fid.readline().rstrip().split("\t")
            self.sweep_list = [float(i) for i in sweep_str]

            rest_str = fid.read()
            matches = re.findall(
                r"^([a-zA-Z0-9_ _/+()#-]+):([a-zA-Z0-9_ _/+()#-]+)", rest_str, re.MULTILINE
            )
            self.metadata = {a: misc.failfloat(b) for (a, b) in matches}

    # =================================

    def transform_dataset(self):
        """ docstring
        """
        # reshapes raw data from 1D to 3D, then rebins if requested
        # just does the rebin step if its already been processed
        if self.is_processed:
            # just rebin, already shaped - do we need to change the lines
            # below this if statement? Or does the ROI still apply?
            self.peak_fit_bin = {}
            shape = self.peak_fit["pos"].shape
            for fit_key in self.prev_options["fit_param_defn"]:
                bins = self.options["num_bins"]
                if bins == 0:
                    bins = 1
                self.peak_fit_bin[fit_key] = np.zeros(
                    (shape[0], int(shape[1] / bins), int(shape[2] / bins))
                )

            for fit_key in self.prev_options["fit_param_defn"]:
                for idx in range(self.prev_options["num_peaks"]):
                    self.peak_fit_bin[fit_key][idx] = self._rebin_image(
                        is_processed=True, image=self.peak_fit[fit_key][idx, ::]
                    )
        else:
            self._reshape_raw()
            self._rebin_image(False)

        self._define_roi()
        mask = create_circular_mask(self.options)
        self.peak_fit_roi = {}

        if self.is_processed:
            if self.options["use_ROI_for_fit"]:
                for fit_key in self.prev_options["fit_param_defn"]:
                    self.peak_fit_roi[fit_key] = self.peak_fit_bin[fit_key][
                        :, self.ROI[0], self.ROI[1]
                    ]
                    if self.options["ROI"] == "Circle":
                        for idx in range(self.prev_options["num_peaks"]):
                            self.peak_fit_roi[fit_key][idx, ~mask] = np.nan
                self.image_ROI = self.image
                return self.peak_fit_roi, self.ROI
            else:
                self.image_ROI = self.image
                return self.peak_fit_bin, self.ROI
        else:
            self._remove_unwanted_sweeps()

        if self.options["ROI"] == "Circle":
            # apply mask
            self.sig[:, ~mask] = np.nan
            self.ref[:, ~mask] = np.nan
            self.sig_norm[:, ~mask] = np.nan
            self.image_ROI[:, ~mask] = np.nan

        # After this transform method, these arrays won't have sweep data.
        # (i.e. outside this module)
        # < NOTE > is image different for processed data? - if so, explain here
        self.image_ROI = self.image_ROI.sum(0)
        self.image = self.image.sum(0)

    # =================================

    def _reshape_raw(self):
        """ reshape the 1D data into 3D array, idx: [f, y, x] """
        # flag for later
        self.options["used_ref"] = False

        if self.options["ignore_ref"]:
            # use every second element
            data_points = len(2 * self.sweep_list)

        try:
            data_points = len(self.sweep_list)
            self.image = np.reshape(
                self.raw_data,
                [data_points, int(self.metadata["AOIHeight"]), int(self.metadata["AOIWidth"])],
            )
        except ValueError:
            # if the ref is used then there's 2* the number of sweeps
            data_points = 2 * len(self.sweep_list)
            if self.options["ignore_ref"]:
                # use every second element
                self.image = np.reshape(
                    self.raw_data[::2],
                    [data_points, int(self.metadata["AOIHeight"]), int(self.metadata["AOIWidth"])],
                )
            else:
                self.image = np.reshape(
                    self.raw_data,
                    [data_points, int(self.metadata["AOIHeight"]), int(self.metadata["AOIWidth"])],
                )
                warnings.warn(
                    "Detected that dataset has reference. "
                    + "Continuing processing using the reference."
                )
                self.options["used_ref"] = True
        # Transpose the dataset to get the correct x and y orientations
        for idx in range(len(self.image)):
            self.image[idx, ::] = self.image[idx, ::].transpose()

    # =================================

    def _rebin_image(self, is_processed=False, image=None, num_bins=None):
        """ Using reshaping and summation to apply additional binning to the
        image
        At this point image and image_bin still have sweep data (summed over
        later)
        """
        if not num_bins:
            num_bins = self.options["num_bins"]
        if image is None:
            image = self.image
        if not num_bins:
            image_bin = image
            self.image_bin = image_bin
        else:
            if num_bins % 2:
                raise misc.ParamError("The binning parameter needs to be a multiple of 2.")
            if is_processed:
                height, width = image.shape
                if height % 2:
                    image = image[1:, 1:]
                    height -= 1
                    width -= 1
                    warnings.warn(
                        "Processed data had odd size. Removed "
                        + "first element in both dimensions to fix."
                    )
                if num_bins != 0:
                    image_bin = np.nansum(
                        np.nansum(
                            np.reshape(
                                image,
                                [int(height / num_bins), num_bins, int(width / num_bins), num_bins],
                            ),
                            1,
                        ),
                        2,
                    ) / (num_bins ** 2)
                    image_bin[image_bin == 0] = np.nan
                self.image_bin = image_bin
                return image_bin
            else:
                data_points = image.shape[0]
                height = image.shape[1]
                width = image.shape[2]
                # height, width = self.image.shape[1:2]
                image_bin = (
                    np.reshape(
                        image,
                        [
                            data_points,
                            int(height / num_bins),
                            num_bins,
                            int(width / num_bins),
                            num_bins,
                        ],
                    )
                    .sum(2)
                    .sum(3)
                )
                self.image_bin = image_bin
        # define sig and ref differently if we're using a ref
        if not is_processed:
            if self.options["used_ref"]:
                self.sig = image_bin[::2, :, :]
                self.ref = image_bin[1::2, :, :]
                if self.options["normalisation"] == "sub":
                    self.sig_norm = self.sig - self.ref
                elif self.options["normalisation"] == "div":
                    self.sig_norm = self.sig / self.ref
                else:
                    raise KeyError("bad normalisation option")
            else:

                self.sig = self.ref = image_bin
                self.sig_norm = self.sig / np.max(self.sig, 0)
        return self.image_bin

    # =================================

    def _define_roi(self):
        """ holds the if statements for the different ROI shape options
        - see _define_area_roi below
        """
        try:
            size_h, size_w = self.image_bin.shape[1:]
        except:
            size_h, size_w = self.image_bin.shape
        if self.options["ROI"] == "Full":
            self.ROI = self.define_area_roi(0, 0, size_w - 1, size_h - 1)
        elif self.options["ROI"] == "Square":
            self.ROI = self.define_area_roi_centre(
                self.options["ROI_centre"], 2 * self.options["ROI_radius"]
            )
        elif self.options["ROI"] == "Circle":
            self.ROI = self.define_area_roi_centre(
                self.options["ROI_centre"], 2 * self.options["ROI_radius"]
            )
        elif self.options["ROI"] == "Rectangle":
            start_x = self.options["ROI_centre"][0] - self.options["ROI_rect_size"][0]
            start_y = self.options["ROI_centre"][1] - self.options["ROI_rect_size"][1]
            end_x = self.options["ROI_centre"][0] + self.options["ROI_rect_size"][0]
            end_y = self.options["ROI_centre"][1] + self.options["ROI_rect_size"][1]
            self.ROI = self.define_area_roi(start_x, start_y, end_x, end_y)

    # =================================

    def _remove_unwanted_sweeps(self):
        rem_start = self.options["remove_start_sweep"]
        rem_end = self.options["remove_end_sweep"]
        self.image_ROI = self.image_bin[:, self.ROI[0], self.ROI[1]]
        self.sig = self.sig[rem_start : -1 - rem_end, self.ROI[0], self.ROI[1]]
        self.ref = self.ref[rem_start : -1 - rem_end, self.ROI[0], self.ROI[1]]
        self.sig_norm = self.sig_norm[rem_start : -1 - rem_end, self.ROI[0], self.ROI[1]]
        self.sweep_list = np.asarray(self.sweep_list[rem_start : -1 - rem_end])

    # =================================

    def _read_processed_data(self, fitted_param=None):
        """ helper function """
        return np.loadtxt(self.options["filepath_data"] + "/" + fitted_param + ".txt")

    # =================================

    def get_options(self):
        return self.options

    # =================================

    def get_previous_options(self):
        return self.prev_options

    # =================================

    @staticmethod
    def define_area_roi(start_x, start_y, end_x, end_y):
        """ Makes a list with a mesh that defines the an ROI
        This ROI can be simply applied to the 2D image through direct
        indexing, e.g new_image = image(:,ROI[0],ROI[1]) with shink the
        ROI of the image.
        """
        x = [np.linspace(start_x, end_x, end_x - start_x + 1, dtype=int)]
        y = [np.linspace(start_y, end_y, end_y - start_y + 1, dtype=int)]
        xv, yv = np.meshgrid(x, y)
        return [yv, xv]

    # =================================

    @staticmethod
    def define_area_roi_centre(centre, size):
        x = [np.linspace(centre[0] - size / 2, centre[0] + size / 2, size + 1, dtype=int)]
        y = [np.linspace(centre[1] - size / 2, centre[1] + size / 2, size + 1, dtype=int)]
        xv, yv = np.meshgrid(x, y)
        return [yv, xv]


# ============================================================================
# Helper Functions that belong in this module


def create_circular_mask(options):
    """ This function defines a circular mask that can be used to remove
    unwanted edge data.
    Example:
        image[:,~mask] = np.nan
    Here the image now = NaN where ever the mask gives a False value.
    """
    h = w = int(1 + 2 * options["ROI_radius"])
    radius = int(options["ROI_radius"])
    center = options["ROI_radius"]
    if options["ROI"] == "Circle":
        Y, X = np.ogrid[:h, :w]
        distr_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
        mask = distr_from_center <= radius
    else:
        mask = np.array([[True] * w for x in range(h)])
    return mask


# =================================


class DataShapeError(Exception):
    pass
