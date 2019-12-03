import numpy as np
import json
from ham_fitting.ham_model_generator import HamModelGenerator as HMG


def read_parameters(filename):
    path = "C:\Widefield_processing_code\Python\Process_widefield"
    f = open(path + filename)
    json_str = f.read()
    param_dict = json.loads(json_str)
    return param_dict


params = read_parameters("/options/reconstruction.json")

ham = HMG(params)

print(ham.initial_guess_dict)
params = np.array(
    [
        ham.initial_guess_dict["d"],
        ham.initial_guess_dict["bx"],
        ham.initial_guess_dict["by"],
        ham.initial_guess_dict["bz"],
        ham.initial_guess_dict["sigma_axial"],
        ham.initial_guess_dict["sigma_xy"],
        ham.initial_guess_dict["sigma_xz"],
        ham.initial_guess_dict["sigma_yz"],
    ]
)
print(ham.model.hamiltonian(params))
print(ham.model.nv_ori)
