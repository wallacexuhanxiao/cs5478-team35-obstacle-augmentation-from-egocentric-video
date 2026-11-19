"""Named-DOF ordering for numpy or torch arrays.

retarget_to_sim_indices[j] is the SIMULATION index of retarget joint j.
Therefore reading uses this gather, while writing uses its inverse gather.
"""


def to_sim_order(retarget_values, retarget_to_sim_indices):
    return retarget_values[..., retarget_to_sim_indices.argsort()]


def to_retarget_order(sim_values, retarget_to_sim_indices):
    return sim_values[..., retarget_to_sim_indices]
