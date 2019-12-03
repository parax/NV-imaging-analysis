# -*- coding: utf-8 -*-

"""
DataLoader
---

"""
__author__ = "David Broadway"

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


class DataLoader(object):
    """load_data.py
    Called to load_datasets
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
        self.options["filepath_data"] = self.options["filepath"] + "/data"
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
            shape = self._read_processed_data(self.prev_options["fit_param_defn"][0] + " 1").shape
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

    def read_meta_data(self):
        # === get metadata
        # first read off the sweep (i.e. freq/tau times) string, then read
        # out each pair of key:value items in the metadata file. If the
        # value is numeric, store it as a float, else keep the string.
        # Store sweep in a list, metadata in a dict
        # ===
        with open(os.path.normpath(self.options["filepath"] + "_metaSpool.txt"), "r") as fid:
            sweep_str = fid.readline().rstrip().split("\t")
            self.sweep_list = [float(i) for i in sweep_str]

            rest_str = fid.read()
            matches = re.findall(
                r"^([a-zA-Z0-9_ _/+()#-]+):([a-zA-Z0-9_ _/+()#-]+)", rest_str, re.MULTILINE
            )
            self.metadata = {a: misc.failfloat(b) for (a, b) in matches}

    # =================================

    def _read_processed_data(self, fitted_param=None):
        """ helper function """
        return np.loadtxt(self.options["filepath_data"] + "\\" + "peak " + fitted_param + ".txt")

    # =================================

    def get_options(self):
        return self.options

    # =================================

    def get_previous_options(self):
        return self.prev_options
