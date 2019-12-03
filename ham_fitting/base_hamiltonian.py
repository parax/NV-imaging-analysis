import numpy as np
import math

# from collections import OrderedDict
"""

NB: In the scipy least_squares documentation, x is a vector specifying the
independent parameters (i.e. fit_parameters_list), whereas here x is the sweep
variable (freq, tau time etc.) - updated to call 'x' 'sweep_val' to
clear this up

"""
__author__ = "David Broadway"


class BaseHamiltonian(object):
    """ Parent Class for fit arbitary peak fit functions
    """

    parameter_definition = []
    parameter_unit = {}
    gamma = 2.8
    nv_axes_CVD = [
        {"nv_number": 1, "ori": [np.sqrt(2 / 3), 0, np.sqrt(1 / 3)]},
        {"nv_number": 2, "ori": [-np.sqrt(2 / 3), 0, np.sqrt(1 / 3)]},
        {"nv_number": 3, "ori": [0, np.sqrt(2 / 3), -np.sqrt(1 / 3)]},
        {"nv_number": 4, "ori": [0, -np.sqrt(2 / 3), -np.sqrt(1 / 3)]},
    ]

    nv_axes_HPHT = [
        {"nv_number": 1, "ori": [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]},
        {"nv_number": 2, "ori": [-np.sqrt(1 / 3), -np.sqrt(1 / 3), np.sqrt(1 / 3)]},
        {"nv_number": 3, "ori": [np.sqrt(1 / 3), -np.sqrt(1 / 3), -np.sqrt(1 / 3)]},
        {"nv_number": 4, "ori": [-np.sqrt(1 / 3), np.sqrt(1 / 3), -np.sqrt(1 / 3)]},
    ]

    def __init__(self, params):
        # applied magnetic field parameters
        self.options = params
        self.diamond_type = params["diamond_type"]
        self.theta = params["b_theta"]
        self.phi = params["b_phi"]
        self.mag = params["b_mag"]
        self.theta_rad = math.radians(self.theta)
        self.phi_rad = math.radians(self.phi)
        # magnetic field initial guesses
        self.b_guess = 0
        # NV orientation arrays
        self.nv_ori = np.zeros((4, 3))
        self.nv_signs = np.zeros(4)
        self.nv_signed_ori = np.zeros((4, 3))
        self.calc_nv_ori()

    # =================================

    def get_parameter_definition(self):
        return self.parameter_definition

    def get_parameter_unit(self):
        return self.parameter_unit

    def calc_nv_ori(self):
        # get the cartesian magnetic fields
        bx = self.mag * np.sin(self.theta_rad) * np.cos(self.phi_rad)
        by = self.mag * np.sin(self.theta_rad) * np.sin(self.phi_rad)
        bz = self.mag * np.cos(self.theta_rad)
        # uses these values for the initial guesses
        self.b_guess = {}
        self.b_guess["bx"] = bx
        self.b_guess["by"] = by
        self.b_guess["bz"] = bz
        # Get the NV orientations B magnitude and sign
        if self.diamond_type == "HPHT":
            self.nv_axes = self.nv_axes_HPHT
        else:
            self.nv_axes = self.nv_axes_CVD
        for key in range(len(self.nv_axes)):
            projection = np.dot(self.nv_axes[key]["ori"], [bx, by, bz])
            self.nv_axes[key]["mag"] = np.abs(projection)
            self.nv_axes[key]["sign"] = np.sign(projection)
        # Sort the dictionary in the correct order
        sorted_dict = sorted(self.nv_axes, key=lambda x: x["mag"], reverse=True)
        # define the nv orientation list for the fit
        for idx in range(len(sorted_dict)):
            self.nv_ori[idx, :] = sorted_dict[idx]["ori"]
            self.nv_signs[idx] = sorted_dict[idx]["sign"]
            self.nv_signed_ori[idx, :] = (
                np.array(sorted_dict[idx]["ori"]) * sorted_dict[idx]["sign"]
            )
        if self.options["use_unv"]:
            self.nv_signed_ori = np.array(self.options["unv"])
        # Calculate the inverse of the nv orientation matrix
        self.nv_signed_ori_inv = self.nv_signed_ori.copy()
        self.nv_signed_ori_inv[self.nv_signed_ori_inv == 0] = np.inf
        self.nv_signed_ori_inv = 1 / self.nv_signed_ori_inv

    def get_spherical_b(self, xyz):
        """ takes b from the cartesian coord and return b in spherical
        coords
        """
        spherical = np.zeros(xyz.shape)
        xy = xyz[0] ** 2 + xyz[1] ** 2
        spherical[0] = np.sqrt(xy + xyz[2] ** 2)
        # for elevation angle defined from Z-axis down
        spherical[1] = math.degrees(np.arctan2(np.sqrt(xy), xyz[2]))
        spherical[2] = math.degrees(np.arctan2(xyz[1], xyz[0]))
        return spherical

    # =================================

    @staticmethod
    def hamiltonian(peak_fits, *fit_params):
        raise NotImplementedError(
            "You shouldn't be here, go away. You MUST override " + "base_fn, check your spelling."
        )
