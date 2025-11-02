import torch


class ModelPB(torch.nn.Module):
    def __init__(
        self,
        p0_prime_abs: torch.Tensor,
        p0_prime_phase: torch.Tensor,
        R_prime: torch.Tensor,
        R_prime_max: torch.Tensor | float,
        c: torch.Tensor,
        L: torch.Tensor,
        L_max: torch.Tensor | float,
        r_hat_abs: torch.Tensor,
        r_hat_phase: torch.Tensor,
    ):
        """Model for background pressure field.

        Args:
            p0_prime_abs (torch.Tensor): $|p_0'|$, tensor of shape `(N_f,)`.
            p0_prime_phase (torch.Tensor): $\\text{arg}(p_0')$, tensor of shape `(N_f,)`.
            R_prime (torch.Tensor): $R'$, tensor of shape `(N_f,)`.
            R_prime_max (torch.Tensor | float): Maximum allowed $R'$, tensor of shape `(N_f,)` or scalar.
            c (torch.Tensor): Speed of sound in the host medium (scalar).
            L (torch.Tensor): Distance to the chamber wall (scalar).
            L_max (torch.Tensor | float): Maximum distance to the chamber wall (scalar).
            r_hat_abs (torch.Tensor): $|\\hat{r}|$, tensor of shape `(N_f,)`.
            r_hat_phase (torch.Tensor): $\\text{arg}(\\hat{r})$, tensor of shape `(N_f,)`.

        Note:
            All parameters with `_abs` are converted to positive values via `softplus`.
            Parameters with a `_max` suffix are constrained by a sigmoid to stay within [0, max].
        """
        super().__init__()
        self._p0_prime_abs = p0_prime_abs
        self._p0_prime_phase = p0_prime_phase
        self._R_prime = R_prime
        self._R_prime_max = R_prime_max
        self._c = c
        self._L = L
        self._L_max = L_max
        self._r_hat_abs = r_hat_abs
        self._r_hat_phase = r_hat_phase

    @property
    def p0_prime(self) -> torch.Tensor:
        amp = torch.nn.functional.softplus(self._p0_prime_abs)
        return amp * torch.exp(1j * self._p0_prime_phase)

    @property
    def R_prime(self) -> torch.Tensor:
        return self._R_prime_max * torch.sigmoid(self._R_prime)  # bounds parameter

    @property
    def L(self) -> torch.Tensor:
        return self._L_max * torch.sigmoid(self._L)  # bounds parameter

    @property
    def r_hat(self) -> torch.Tensor:
        amp = torch.nn.functional.softplus(self._r_hat_abs)
        return amp * torch.exp(1j * self._r_hat_phase)

    @property
    def c(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._c)

    def forward(self, freq: torch.Tensor, z_prime: torch.Tensor) -> torch.Tensor:
        """Predict the field.

        Args:
            freq (torch.Tensor): Frequencies, tensor of shape `(N_f,)`.
            z_prime (torch.Tensor): z-coordinates, tensor of shape `(N_z,)`.

        Returns:
            torch.Tensor: Calculated pressure field, tensor of shape `(N_z, N_f)`.
        """

        k = 2 * torch.pi * freq / self.c

        p0_prime = self.p0_prime[None, ...]
        R_prime = self.R_prime[None, ...]
        r_hat = self.r_hat[None, ...]

        z_prime = z_prime[..., None]

        res = p0_prime * (
            R_prime * torch.exp(1j * k * z_prime) / (z_prime + R_prime)
            + r_hat * torch.exp(1j * k * (self.L - z_prime))
        )
        return res

    def __repr__(self) -> str:
        def get_stat(value: torch.Tensor) -> str:
            return f"mean={value.mean().item():.3f}, min={value.min().item():.3f}, max={value.max().item():.3f}"

        s = "Parameters:\n"
        s += f"  |p0_prime|: {get_stat(self.p0_prime.abs())}\n"
        s += f"     R_prime: {get_stat(self.R_prime)}\n"
        s += f"           c: {self.c.item():.3f}\n"
        s += f"           L: {self.L.item():.3f}\n"
        s += f"     |r_hat|: {get_stat(self.r_hat.abs())}"
        return super().__repr__() + "\n" + s


class ModelY(torch.nn.Module):
    def __init__(
        self,
        R_prime: torch.Tensor,
        c: torch.Tensor,
        Delta: torch.Tensor,
        Delta_max: torch.Tensor | float,
        sigma: torch.Tensor,
        A_im: torch.Tensor,
        C: torch.Tensor,
    ) -> None:
        """Model for the $y$ quantity.

        Args:
            R_prime (torch.Tensor): $R'$, tensor of shape `(N_f,)`.
            c (torch.Tensor): Speed of sound in the host medium (scalar).
            Delta (torch.Tensor): $\\Delta$ (scalar).
            Delta_max (torch.Tensor | float): Maximum allowed value for $\\Delta$ (scalar).
            sigma (torch.Tensor): Extinction cross-section, tensor of shape `(N_f,)`.
            A_im (torch.Tensor): $\\text{Im}A$, tensor of shape `(N_f,)`.
            C (torch.Tensor): $\\bar{C}$, tensor of shape `(N_f,)`.
        """
        super().__init__()
        self.R_prime = R_prime
        self.c = c
        self._Delta = Delta
        self._Delta_max = Delta_max
        self._sigma = sigma
        self.A_im = A_im
        self.C = C

    @property
    def sigma(self):
        return torch.nn.functional.softplus(self._sigma)  # bounds parameter

    @property
    def Delta(self):
        return self._Delta_max * torch.sigmoid(self._Delta)  # bounds parameter

    def forward(self, freq: torch.Tensor, z_prime: torch.Tensor) -> torch.Tensor:
        """Predict the $y$ quantity.

        Args:
            freq (torch.Tensor): Frequencies, tensor of shape `(N_f,)`.
            z_prime (torch.Tensor): z-coordinates, tensor of shape `(N_z,)`.

        Returns:
            torch.Tensor: Calculated $y$, tensor of shape `(N_z, N_f)`.
        """
        k = 2 * torch.pi * freq / self.c

        R_prime = self.R_prime[None, ...]
        A_im = self.A_im[None, ...]

        d = z_prime[..., None] + self.Delta

        return (
            R_prime
            / (R_prime - self.Delta)
            * (
                k**2 / (4 * torch.pi) * self.sigma / (k * d)
                + A_im / (k * d) ** 2
                + self.C
            )
        )

    def __repr__(self) -> str:
        def get_stat(value: torch.Tensor) -> str:
            return f"mean={value.mean().item():.3f}, min={value.min().item():.3f}, max={value.max().item():.3f}"

        s = "Parameters:\n"
        s += f"  Delta: {self.Delta.item():.3f}\n"
        s += f"  sigma: {get_stat(self.sigma)}\n"
        s += f"   A_im: {get_stat(self.A_im)}\n"
        s += f"      C: {get_stat(self.C)}"
        return super().__repr__() + "\n" + s
