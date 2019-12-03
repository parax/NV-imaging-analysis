# -*- coding: utf-8 -*-

__author__ = "David Broadway"

# =================================

from ham_fitting.hamiltonians import BNV
from ham_fitting.hamiltonians import ApproxBxyz
from ham_fitting.hamiltonians import Bxyz
from ham_fitting.hamiltonians import BxyzStress
from ham_fitting.hamiltonians import BxyzEz

# =================================

import numpy as np

# =================================


class HamModelGenerator(object):
    """ Takes the reconstruction method (recon_method) that is contained in the options dictionary
    and returns the correct hamiltonian with correctly set initial guesses for the scipy fitting
    package.
    """

    available_ham_types = {
        "BNV": BNV,
        "approx_bxyz": ApproxBxyz,
        "bxyz": Bxyz,
        "bxyz_ez": BxyzEz,
        "bxyz_stress": BxyzStress,
    }

    # =================================

    def __init__(self, options):

        self.options = options
        if self.options["recon_method"] not in self.available_ham_types:
            key_str = "\n"
            for key in self.available_ham_types:
                key_str = key_str + key + "\n"
            raise ValueError(
                "Your reconstruction method does not exist! " + "Choose from one of:" + key_str
            )

        self.model = self.available_ham_types[self.options["recon_method"]](self.options)

        self.initial_guess_dict = {}
        self.bound_dict = {}
        self.fit_param_list = np.array([])
        self.fit_param_bound_list = np.array([])
        if self.options["recon_method"] != "BNV":
            self.gen_init_guesses()
            self.gen_fit_param_list()

    # =================================

    def gen_init_guesses(self):
        for parameter_key in self.model.param_defn:
            if parameter_key in parameter_key:
                if self.options["auto_b_guess"] and "b" in parameter_key:
                    guess = self.model.b_guess[parameter_key]
                else:
                    guess = self.options[parameter_key + "_guess"]
                val = guess
                val_b = [
                    [guess - self.options[parameter_key + "_range"]],
                    [guess + self.options[parameter_key + "_range"]],
                ]
            else:
                val = 0
                val_b = [[0], [0]]
            if val is not None:
                self.initial_guess_dict[parameter_key] = val
                self.bound_dict[parameter_key] = np.array(val_b)
            else:
                raise RuntimeError(
                    "I'm not sure what this means... I know "
                    + "it's bad though... Don't put 'None' as "
                    + "a param guess."
                )

    def gen_fit_param_list(self):
        """ Need to fix the bounds """

        to_append = np.zeros(len(self.model.param_defn))
        to_append_bounds = np.zeros((len(self.model.param_defn), 2))
        for position, key in enumerate(self.model.param_defn):
            to_append[position] = self.initial_guess_dict[key]
            to_append_bounds[position] = [self.bound_dict[key][0], self.bound_dict[key][1]]
        self.fit_param_list = np.append(self.fit_param_list, to_append)
        self.fit_param_bound_list = np.append(self.fit_param_bound_list, to_append_bounds)
        self.fit_param_bound_list = self.fit_param_bound_list.reshape(len(self.model.param_defn), 2)

    # =================================

    def residuals_scipy(self, fit_params_list, pos_fits):
        return self.model.hamiltonian(fit_params_list) - pos_fits
