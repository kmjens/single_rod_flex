import itertools
import signac
import numpy as np

def grid(gridspec):
    '''
    Input: dictionary whose keys correspond to parameter names.
    Keys are associated with iterable values that each parameter could be.
    Outputs: sequence of dictionaries where each has unique combo of parameter values.
    '''
    for values in itertools.product(*gridspec.values()):
        yield dict(zip(gridspec.keys(), values))

gridspec = {
    "N_mesh":   [1e3],
    "R":        [10], # Radius of spherical mesh
    "kT":       [0.2],
    "gamma":    [50],
    "k_bend":   [1000],
    "k_area":   [10000],
    "dt":       [0.0005],
    "N_active": [1,100],
    "sigma":    [1], # active rod const sphere diam (will become coupled to len_ration for N_active = 1. 
    "ratio_len":[3], # [2, 3, 4], # ratio of mesh diam to rod length (decides rod const. particle)
    "v0":       [40],
    "runtime":  [2e5], #[5e6],
    
    # To flatten (approx cylinders)
    "num_filler":  [0], #[0,10], #number of filler particles to flatten spherocylinder
    "filler_diam_ratio": [0.25], # ratio of filler particle diam to A diam 

    # To add gravity
    "gravity_strength": [0], #abs value gravity strength
    
    # To make chiral:
    "rand_orient":  ['False'],
    "active_angle": [0], #angle in degrees
    "torque_mag":   [0],
}

if __name__ == "__main__":
    project = signac.init_project()
    for i, statepoint in enumerate(grid(gridspec)):
        statepoint["seed"] = i
        print("Opening job", statepoint)
        job = project.open_job(statepoint).init()
