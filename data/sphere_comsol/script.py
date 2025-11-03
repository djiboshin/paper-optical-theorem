"""Script used to generate the `data/sphere_comsol/sigma.csv` file using COMSOL."""

import mph
from comsol.sphere_scattering import create_new_model, ModelParameters
import pandas as pd


if __name__ == "__main__":

    # Create a COMSOL client
    client = mph.Client(port=2036)

    # Define model parameters
    parameters = ModelParameters(
        freq_start="0.01[Hz]",
        freq_step="0.002[Hz]",
        freq_stop="0.5[Hz]",
        mesh_h="0.1[m]",
        R_PML="15[m]",
        L_PML="5[m]",
        R_int="8[m]",
    )

    # Create a new COMSOL model and solve it
    model = create_new_model(client, parameters)
    model.solve()

    # Evaluate frequency, extinction, and scattering cross-sections
    freq = model.evaluate("freq", "Hz")
    sigma_ext = model.evaluate("W_ext / I0", "m^2")
    sigma_sc = model.evaluate("W_sc / I0", "m^2")

    # Assemble results into a DataFrame
    df = pd.DataFrame(
        {
            "freq": freq,
            "sigma_ext": sigma_ext,
            "sigma_sc": sigma_sc,
        }
    )

    # Save the DataFrame to CSV
    df.to_csv(
        "sigma.csv",
        index=False,
        float_format="%.17g",
    )
