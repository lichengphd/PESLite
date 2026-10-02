"""Voltage-adaptive SRF-PLL add-on with frequency tracking."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Optional

from . import ANGLE, FREQUENCY, V_AB, ConfigError, Loop, register_loop_type

__all__ = ["VoltageAdaptivePLL"]


@register_loop_type
class VoltageAdaptivePLL(Loop):
    """PLL with filtered measured-voltage magnitude and grid-frequency tracking.

    With ``alpha = 2*pi*bandwidth``, the equations in pu are::

        eps = imag(v * exp(-j*theta)) / u_g
        omega = omega_g + 2*alpha*eps
        d(theta)/dt = omega
        d(omega_g)/dt = alpha**2*eps
        d(u_g)/dt = 2*alpha*(real(v * exp(-j*theta)) - u_g)

    This differs from PESLite's default rated-voltage SRF-PLL because the measured voltage updates
    the normalising magnitude.  ``theta`` remains unwrapped to match PESLite's state convention.
    """

    @dataclass(frozen=True, kw_only=True)
    class Params:
        bandwidth: float
        period: Optional[float] = None
        type: str = "voltage_adaptive_pll"

        def __post_init__(self):
            if not math.isfinite(self.bandwidth) or self.bandwidth <= 0.0:
                raise ConfigError(f"bandwidth must be finite and positive, got {self.bandwidth}")

    type = "voltage_adaptive_pll"
    role = "pll"
    outputs_from_state = True
    inputs = {"v": V_AB}
    outputs = {"theta": ANGLE, "frame": ANGLE, "omega": FREQUENCY}
    continuous_path_inputs = ("v",)
    state_names = {"theta": "theta", "omega_g": "omega_g", "u_g_pu": "u_g"}

    def __init__(self, cfg, unit, startup):
        super().__init__(cfg, unit, startup)
        alpha = 2.0 * math.pi * cfg.bandwidth
        self.kp, self.ki = 2.0 * alpha, alpha * alpha
        self.w0, self.T = unit.base.w0, cfg.period
        self.theta, self.omega_g, self.u_g = 0.0, self.w0, 1.0
        self.omega = self.w0

    def initial_outputs(self):
        return {"theta": self.theta, "frame": self.theta, "omega": self.omega}

    def _evaluate(self, voltage):
        measured = voltage * cmath.exp(-1j * self.theta)
        error = measured.imag / self.u_g if self.u_g > 0.0 else 0.0
        self.omega = self.omega_g + self.kp * error
        return measured, error

    def update(self, _t, inputs):
        frame = self.theta
        measured, error = self._evaluate(inputs["v"])
        self.theta += self.T * self.omega
        self.omega_g += self.T * self.ki * error
        self.u_g += self.T * self.kp * (measured.real - self.u_g)
        return {"theta": self.theta, "frame": frame, "omega": self.omega}

    def continuous(self, _t, inputs):
        outputs, derivatives = {}, {}
        self.continuous_into(_t, inputs, outputs, derivatives)
        return outputs, derivatives

    def continuous_into(self, _t, inputs, outputs, derivatives):
        result = self.continuous_path(inputs["v"])
        outputs.update(theta=result[0], frame=result[1], omega=result[2])
        derivatives.update(theta=result[3], omega_g=result[4], u_g_pu=result[5])

    def continuous_path(self, voltage):
        """Allocation-free positional form used by the compiled continuous controller graph."""
        measured, error = self._evaluate(voltage)
        return (self.theta, self.theta, self.omega, self.omega,
                self.ki * error, self.kp * (measured.real - self.u_g))

    def reset_integrator(self):
        self.omega_g = self.w0
