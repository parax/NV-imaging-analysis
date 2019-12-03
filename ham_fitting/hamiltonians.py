# -*- coding: utf-8 -*-
"""
"""
__author__ = "David Broadway"

# =================================

from ham_fitting.base_hamiltonian import BaseHamiltonian

# =================================

import numpy as np

# =================================
# ============== Bnv ==============
# =================================


class BNV(BaseHamiltonian):
    """ Class for dealing with just BNV data. As no fitting is done on the bnv data there is no
    additional functions.
    """

    param_defn = ["BNV", "D"]
    param_title = {"BNV": "BNV", "D": "Zero field splitting"}
    param_unit = {"BNV": "Magnetic field, BNV (G)", "D": "Frequency (MHz)"}

    def __init__(self, params):
        super().__init__(params)


# =================================
# ====== Aproximate  Bxyz =========
# =================================


class ApproxBxyz(BaseHamiltonian):
    """ Class that takes an approximation to calculate the magnetic field. In this case only
    magnetic fields that are aligned with the NV are considered and thus a simple dot product can be
    used.
    """

    param_defn = ["bx", "by", "bz"]
    param_title = {"bx": "fitted Bx", "by": "fitted By", "bz": "fitted Bz"}
    param_unit = {
        "bx": "Magnetic field, Bx (G)",
        "by": "Magnetic field, By (G)",
        "bz": "Magnetic field, Bz (G)",
    }

    def __init__(self, params):
        super().__init__(params)

    # =================================

    def hamiltonian(self, bxyz):
        return np.dot([bxyz[0], bxyz[1], bxyz[2]], self.nv_signed_ori.T)


# =================================
# ============= Bxyz ==============
# =================================


class Bxyz(BaseHamiltonian):
    """ Hamiltonian for bxyz and D fitting.
    """

    param_defn = ["d", "bx", "by", "bz"]
    param_title = {"d": "fitted D", "bx": "fitted bx", "by": "fitted by", "bz": "fitted bz"}
    param_unit = {
        "d": "Zero field splitting (MHz)",
        "bx": "Magnetic field, Bx (G)",
        "by": "Magnetic field, by (G)",
        "bz": "Magnetic field, Bz (G)",
    }

    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    sy = np.array([[0, -1j, 0], [1j, 0, 1j], [0, 1j, 0]]) / np.sqrt(2)
    sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    sx_array = np.array([sx, sx, sx, sx]).T
    sy_array = np.array([sy, sy, sy, sy]).T
    sz_array = np.array([sz, sz, sz, sz]).T

    def __init__(self, params):
        super().__init__(params)
        self.rotations = np.zeros((4, 3, 3))
        self.nv_frequencies = np.zeros(8)
        for ii in range(4):
            zrot = self.nv_signed_ori[ii].T
            yrot = np.cross(zrot, self.nv_signed_ori[-ii - 1].T)
            yrot = yrot / np.linalg.norm(yrot)
            xrot = np.cross(yrot, zrot)
            self.rotations[ii, ::] = [xrot, yrot, zrot]

    # =================================

    def hamiltonian(self, fit_params):
        """ Hamiltonain of the NV spin using only the zero field splitting D and the magnetic field
        bxyz. Takes the fit_params in the order [D, bx, by, bz] and returns the nv frequencies.
        """
        Hzero = fit_params[0] * (self.sz * self.sz)
        for ii in range(4):
            bx = np.dot(fit_params[1:4], self.rotations[ii, 0, ::])
            by = np.dot(fit_params[1:4], self.rotations[ii, 1, ::])
            bz = np.dot(fit_params[1:4], self.rotations[ii, 2, ::])

            HB = self.gamma * (bx * self.sx + by * self.sy + bz * self.sz)
            freq, length = np.linalg.eig(Hzero + HB)
            freq = np.sort(np.real(freq))
            self.nv_frequencies[ii] = np.real(freq[1] - freq[0])
            self.nv_frequencies[7 - ii] = np.real(freq[2] - freq[0])
        return self.nv_frequencies


# =================================
# =========== Bxyz Exyz ===========
# =================================


class BxyzEz(BaseHamiltonian):
    """ Hamiltonian for bxyz, Exyz and D fitting.
    """

    # < TODO > test the E field hamiltonain fitting

    param_defn = ["d", "bx", "by", "bz", "ez"]
    param_title = {
        "d": "fitted D",
        "bx": "fitted bx",
        "by": "fitted by",
        "bz": "fitted bz",
        "ez": "fitted ez",
    }
    param_unit = {
        "d": "Zero field splitting (MHz)",
        "bx": "Magnetic field, Bx (G)",
        "by": "Magnetic field, by (G)",
        "bz": "Magnetic field, Bz (G)",
        "ez": "Electric field, Ez (V/cm)",
    }

    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    sy = np.array([[0, -1j, 0], [1j, 0, 1j], [0, 1j, 0]]) / np.sqrt(2)
    sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    sx_array = np.array([sx, sx, sx, sx]).T
    sy_array = np.array([sy, sy, sy, sy]).T
    sz_array = np.array([sz, sz, sz, sz]).T

    def __init__(self, params):
        super().__init__(params)
        self.dgs_para = 0.35  # Hz cm V ^ -1
        self.dgs_perp = 17  # Hz cm V ^ -1
        # convert to MHz
        self.dgs_para = self.dgs_para * 1e-6
        self.dgs_perp = self.dgs_perp * 1e-6

        self.rotations = np.zeros((4, 3, 3))
        self.nv_frequencies = np.zeros(8)
        for ii in range(4):
            zrot = self.nv_signed_ori[ii].T
            yrot = np.cross(zrot, self.nv_signed_ori[-ii - 1].T)
            yrot = yrot / np.linalg.norm(yrot)
            xrot = np.cross(yrot, zrot)
            self.rotations[ii, ::] = [xrot, yrot, zrot]

    # =================================

    def hamiltonian(self, fit_params):
        """ Hamiltonain of the NV spin using only the zero field splitting D, the magnetic field
        bxyz and electric field exyz. Takes the fit_params in the order [D, bx, by, bz, ex, ey, ez]
        and returns the nv frequencies.
        """
        if self.options["constant_d"]:
            fit_params[0] = self.options["d_guess"]
        Hzero = fit_params[0] * (self.sz * self.sz)
        e_to_rot = [0, 0, fit_params[4]]
        for ii in range(4):
            bx = np.dot(fit_params[1:4], self.rotations[ii, 0, ::])
            by = np.dot(fit_params[1:4], self.rotations[ii, 1, ::])
            bz = np.dot(fit_params[1:4], self.rotations[ii, 2, ::])

            ex = np.dot(e_to_rot, self.rotations[ii, 0, ::])
            ey = np.dot(e_to_rot, self.rotations[ii, 1, ::])
            ez = np.dot(e_to_rot, self.rotations[ii, 2, ::])

            HB = self.gamma * (bx * self.sx + by * self.sy + bz * self.sz)
            HE = (
                self.dgs_para * ez * (self.sz * self.sz)
                + self.dgs_perp * (ey * (self.sx * self.sy + self.sy * self.sx))
                - self.dgs_perp * ex * (self.sx ** 2 - self.sy ** 2)
            )
            freq, length = np.linalg.eig(Hzero + HB + HE)
            freq = np.sort(np.real(freq))
            self.nv_frequencies[ii] = np.real(freq[1] - freq[0])
            self.nv_frequencies[-ii - 1] = np.real(freq[2] - freq[0])
        return self.nv_frequencies


# =================================
# ========== Bxyz stress ==========
# =================================


class BxyzStress(BaseHamiltonian):
    """ Hamiltonian for bxyz, stress and D fitting.
    """

    # < TODO > test the stress hamiltonain fitting

    param_defn = ["d", "bx", "by", "bz", "sigma_axial", "sigma_xy", "sigma_xz", "sigma_yz"]
    param_title = {
        "d": "fitted D",
        "bx": "fitted bx",
        "by": "fitted by",
        "bz": "fitted bz",
        "sigma_axial": "fitted axial stress",
        "sigma_xy": "fitted sigma_xy",
        "sigma_xz": "fitted sigma_xz",
        "sigma_yz": "fitted sigma_yz",
    }
    param_unit = {
        "d": "Zero field splitting (MHz)",
        "bx": "Magnetic field, Bx (G)",
        "by": "Magnetic field, by (G)",
        "bz": "Magnetic field, Bz (G)",
        "sigma_axial": "axial stress (MPa)",
        "sigma_xy": "shear stress (MPa)",
        "sigma_xz": "shear stress (MPa)",
        "sigma_yz": "shear stress (MPa)",
    }

    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    sy = np.array([[0, -1j, 0], [1j, 0, 1j], [0, 1j, 0]]) / np.sqrt(2)
    sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    sx_array = np.array([sx, sx, sx, sx]).T
    sy_array = np.array([sy, sy, sy, sy]).T
    sz_array = np.array([sz, sz, sz, sz]).T

    def __init__(self, params):
        super().__init__(params)
        self.a1 = 4.86 * 1e-3  # MHz / MPa
        self.a2 = -3.7 * 1e-3  # MHz / MPa
        self.b = -(2.3 / 2) * 1e-3  # MHz / MPa
        self.c = (3.5 / 2) * 1e-3  # MHz / MPa

        self.rotations = np.zeros((4, 3, 3))
        self.nv_frequencies = np.zeros(8)
        for ii in range(4):
            zrot = self.nv_signed_ori[ii].T
            yrot = np.cross(zrot, self.nv_signed_ori[-ii - 1].T)
            yrot = yrot / np.linalg.norm(yrot)
            xrot = np.cross(yrot, zrot)
            self.rotations[ii, ::] = [xrot, yrot, zrot]

    # =================================

    def calculate_effective_electric_field(self, sigma):

        # < TODO > rewrite this to calculate from the actual rotations.
        # separate out stress terms
        sigma_axial = sigma[0]
        sigma_xy = sigma[1]
        sigma_xz = sigma[2]
        sigma_yz = sigma[3]

        # Define uni axial stress contributions. Using hydrostatic stress assumption
        # (sigma_xx = sigma_yy = sigma_zz) means Funix = Funiy = 0
        # Funix = self.b * (-sigma_xx - sigma_yy + 2 * sigma_zz)
        # Funiy = self.b * (sigma_xx - sigma_yy);
        Funix = 0
        Funiy = 0
        Funiz = self.a1 * sigma_axial

        # Define stress contribution for
        # === NV axis 1 === #
        Feff1x = self.c * (2 * sigma_xy - sigma_xz - sigma_yz) + Funix
        Feff1y = np.sqrt(3) * (self.c * (-sigma_xz + sigma_yz)) + Funiy
        Feff1z = 2 * self.a2 * (sigma_xy + sigma_xz + sigma_yz) + Funiz
        Feff1 = [Feff1x, Feff1y, Feff1z]

        # === AXIS 2 === #
        Feff2x = self.c * (-2 * sigma_xy + sigma_xz - sigma_yz) + Funix
        Feff2y = np.sqrt(3) * (self.c * (sigma_xz + sigma_yz)) + Funiy
        Feff2z = 2 * self.a2 * (-sigma_xy - sigma_xz + sigma_yz) + Funiz
        Feff2 = [Feff2x, Feff2y, Feff2z]

        # === AXIS 3 === #
        Feff3x = self.c * (-2 * sigma_xy - sigma_xz + sigma_yz) + Funix
        Feff3y = np.sqrt(3) * (self.c * (-sigma_xz - sigma_yz)) + Funiy
        Feff3z = 2 * self.a2 * (-sigma_xy + sigma_xz - sigma_yz) + Funiz
        Feff3 = [Feff3x, Feff3y, Feff3z]

        # === AXIS 4 === #
        Feff4x = self.c * (2 * sigma_xy + sigma_xz + sigma_yz) + Funix
        Feff4y = np.sqrt(3) * (self.c * (sigma_xz - sigma_yz)) + Funiy
        Feff4z = 2 * self.a2 * (sigma_xy - sigma_xz - sigma_yz) + Funiz
        Feff4 = [Feff4x, Feff4y, Feff4z]

        F_eff = [Feff1, Feff2, Feff3, Feff4]
        # this loop just orders the effective electric field correctly
        for ii in range(4):
            if self.nv_signed_ori[ii, 0] == 0 and self.nv_signed_ori[ii, 1] > 0:
                F_eff[ii] = Feff2
            elif self.nv_signed_ori[ii, 0] == 0 and self.nv_signed_ori[ii, 1] < 0:
                F_eff[ii] = Feff3
            elif self.nv_signed_ori[ii, 1] == 0 and self.nv_signed_ori[ii, 0] > 0:
                F_eff[ii] = Feff4
            else:
                F_eff[ii] = Feff1

        return F_eff

    def hamiltonian(self, fit_params):
        """ Hamiltonain of the NV spin using only the zero field splitting D, the magnetic field
        bxyz and electric field exyz. Takes the fit_params in the order [D, bx, by, bz, ex, ey, ez]
        and returns the nv frequencies.
        """
        if self.options["constant_d"]:
            fit_params[0] = self.options["d_guess"]
        Hzero = fit_params[0] * (self.sz * self.sz)
        F_eff = self.calculate_effective_electric_field(fit_params[4:8])
        for ii in range(4):
            bx = np.dot(fit_params[1:4], self.rotations[ii, 0, ::])
            by = np.dot(fit_params[1:4], self.rotations[ii, 1, ::])
            bz = np.dot(fit_params[1:4], self.rotations[ii, 2, ::])

            FXeff = F_eff[ii][0]
            FYeff = F_eff[ii][1]
            FZeff = F_eff[ii][2]

            HB = self.gamma * (bx * self.sx + by * self.sy + bz * self.sz)
            H_sigma = (
                FZeff * (self.sz * self.sz)
                + FXeff * (self.sx * self.sx - self.sy * self.sy)
                + FYeff * (self.sx * self.sy + self.sy * self.sx)
            )
            freq, length = np.linalg.eig(Hzero + HB + H_sigma)
            freq = np.sort(np.real(freq))
            self.nv_frequencies[ii] = np.real(freq[1] - freq[0])
            self.nv_frequencies[-ii - 1] = np.real(freq[2] - freq[0])
        return self.nv_frequencies
