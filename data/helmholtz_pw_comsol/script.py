"""Script used to generate the `data/helmholtz_pw_comsol/sigma.csv` file using COMSOL."""

import mph
from comsol.helmholtz_pw_scattering import create_new_model, ModelParameters
import pandas as pd
import tqdm


if __name__ == "__main__":

    # Create a COMSOL client
    client = mph.Client(port=2036)

    # Define model parameters
    parameters = ModelParameters(
        freq_start="1600 [Hz]",
        freq_step="5 [Hz]",
        freq_stop="2400 [Hz]",
    )

    # Create a new COMSOL model and solve it
    model = create_new_model(client, parameters)

    d_wall_p_list = [
        "1.8[mm]",
        "1.9[mm]",
        "2.0[mm]",
        "2.1[mm]",
        "2.2[mm]",
        "2.3[mm]",
        "2.4[mm]",
    ]
    dfs = []

    for d_wall_p in tqdm.tqdm(d_wall_p_list):

        model.parameter("d_wall_p", d_wall_p)
        model.solve()

        # Evaluate frequency, extinction, and scattering cross-sections
        freq = model.evaluate("freq", "Hz")
        sigma_ext = model.evaluate("W_ext / I0", "m^2")
        sigma_sc = model.evaluate("W_sc / I0", "m^2")
        d_wall_p = model.evaluate("d_wall_p", "mm")

        # Assemble results into a DataFrame
        dfs.append(
            pd.DataFrame(
                {
                    "freq": freq,
                    "sigma_ext": sigma_ext,
                    "sigma_sc": sigma_sc,
                    "d_wall_p": d_wall_p,
                }
            )
        )

    df = pd.concat(dfs)
    # Save the DataFrame to CSV
    df.to_csv(
        "sigma.csv",
        index=False,
        float_format="%.17g",
    )
