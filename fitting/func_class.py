# -*- coding: utf-8 -*-
"""

NB: In the scipy least_squares documentation, x is a vector specifying the
independent parameters (i.e. fit_parameters_list), whereas here x is the sweep
variable (freq, tau time etc.) - updated to call 'x' 'sweep_val' to
clear this up

"""
from numba import jit
import numpy as np

# ============================================================================


class FitFunc:
    """ Parent class for fit arbitary peak fit functions
    num_fns is the number of functins in this FitFunc - not including backgrounds
    """

    param_defn = []

    def __init__(self, num_fns, chain_fitfunc=None):
        self.num_fns = num_fns
        if chain_fitfunc is None:
            self.chain_param_len = 0
            self.chain_fitfunc = ChainTerminator()
        else:
            self.chain_fitfunc = chain_fitfunc
            self.chain_param_len = len(chain_fitfunc.get_param_defn())

    # =================================

    def __call__(self, sweep_vec, fit_params):
        """ Returns the value of the fit function at sweep_val (i.e. freq, tau)
        for given fit_options.
        """
        chain_params, these_params = np.split(fit_params, [self.chain_param_len])
        newoptions = these_params.reshape(self.num_fns, len(self.param_defn))

        outx = np.zeros(np.shape(sweep_vec))
        for f_params in newoptions:
            outx += self.base_fn(sweep_vec, *f_params)
        #        try:
        return outx + self.chain_fitfunc(sweep_vec, chain_params)

    # =================================

    def jacobian(self, sweep_vec, fit_params):
        """ Returns the value of the fit functions jacobian at sweep_vals for
        given fit_params.
        shape: (len(sweep_val), num_fns*len(param_defn))
        """

        chain_params, params = np.split(fit_params, [self.chain_param_len])
        new_params = params.reshape(self.num_fns, len(self.param_defn))

        try:
            ftype = self.fn_type
        except AttributeError:
            raise AttributeError("You need to define the type of your function - peak or bground")

        # if ftype == "terminator":
        #     return self.grad_fn(sweep_vec)

        for i, f_params in enumerate(new_params):
            # hmm this didn't do anything ?
            # if self.num_fns == 1 and ftype == "peak":
            #     return self.grad_fn(sweep_vec, *f_params[0])
            # elif self.num_fns == 1 and ftype == "bground":
            #     output = self.grad_fn(sweep_vec, *f_params)
            if not i:
                output = self.grad_fn(sweep_vec, *f_params)
            else:
                # stack on next peak's 'grad'/jacobian
                output = np.hstack((output, self.grad_fn(sweep_vec, *f_params)))

        if self.chain_fitfunc.fn_type == "terminator":
            # not adding pl term to jacobian...
            return output

        # stack on the next fit functions jacobian (recursively)
        # NOTE the chain fitfuncs have to be added at the start as that's how its defined
        # in fit_model (gen_fit_params), UNTESTED for more than one bground fn
        return np.hstack((self.chain_fitfunc.jacobian(sweep_vec, chain_params), output))

    # =================================

    def get_param_defn(self):
        """ Returns the chained parameter defintions.  Not sure if used and
        should be considered for removal or renaming as it is confusinigly similar
        to the static member variable param.defn which does not include chained
        functions."""
        try:
            return self.param_defn + self.chain_fitfunc.get_param_defn()
        except (AttributeError):
            return self.param_defn

    # =================================

    @staticmethod
    def base_fn(sweep_vec, *fit_params):
        raise NotImplementedError(
            "You shouldn't be here, go away. You MUST override " + "base_fn, check your spelling."
        )

    # =================================

    @staticmethod
    def grad_fn(sweep_vec, *fit_params):
        """ if you want to use a grad_fn override this in the subclass """
        return None


# ============================================================================


class ChainTerminator(FitFunc):
    """
    Ends the chain of arbitrary fit functions. This needs to be here as we don't want
    circular dependencies.
    """

    param_defn = []
    parameter_unit = {}
    fn_type = "terminator"
    chain_fitfunc = None

    # override the init for FitFunc
    def __init__(self):
        self.chain_param_len = 0
        self.num_fns = 0

    def __call__(self, *anything):
        """ contributes nothing to the residual """
        return 0

    @staticmethod
    def base_fn(*anything):
        raise NotImplementedError("you shouldn't be here")

    @staticmethod
    def grad_fn(sweep_vec, *anything):
        """ hstack the PL term onto the jacobian """
        return -np.ones(sweep_vec.shape[0], dtype=np.float32).reshape(-1, 1)
