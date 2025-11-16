"""Script used to generate the `data/helmholtz_sw_comsol/p_b.npz` and `data/helmholtz_sw_comsol/p_s.npz` files using COMSOL."""

import mph
from comsol.helmholtz_sw_scattering import (
    create_new_model,
    ModelParameters,
    get_results,
)
import numpy as np


if __name__ == "__main__":

    # Create a COMSOL client
    client = mph.Client(port=2036)

    # Define model parameters
    parameters = ModelParameters(
        R_PML="1.8 [m]",
        freq_start="1000 [Hz]",
        freq_step="5 [Hz]",
        freq_stop="3000 [Hz]",
    )

    # Create a new COMSOL model and solve it
    model = create_new_model(client, parameters)
    model.solve()

    # Evaluate frequency, extinction, and scattering cross-sections
    z, freq, p_s, p_b = get_results(model)

    # z has a (N_freq, N_z) shape. We assert that for different frequencies, z is the same.
    assert np.allclose(z[0], z)

    # Save the results to .npz files
    mask = (
        z[0] > 0.02
    )  # to remove points too close to the scatterer, where the mesh is extremelly dense
    np.savez("p_b.npz", p=np.conj(p_b.T[mask]), freq=freq, z=z[0][mask])
    np.savez("p_s.npz", p=np.conj(p_s.T[mask]), freq=freq, z=z[0][mask])
