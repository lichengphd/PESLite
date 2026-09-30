"""The output stage of the controller: from the dq voltage command to the duty ratios of the three phases.

The command is rotated into alpha-beta, turned into modulating signals by the PWM method (``spwm``,
``svpwm`` or a custom one), limited, and fed back to the current loop when the limit is reached
(anti-windup).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from .blocks import abc2complex, complex2abc, peak_abs, phases

__all__ = ["PWMMethod", "SignalLimiter", "PWM_METHODS", "spwm", "svpwm", "ModulationLimiter", "OutputStage"]


class PWMMethod(Protocol):
    """Map an alpha-beta voltage command (V) and a positive dc voltage (V) to modulating signals ``m_abc``.

    Must be deterministic and return three finite real values, shape ``(3,)``, possibly outside [-1, 1].
    """

    def __call__(self, u_ab: complex, u_dc: float) -> NDArray[np.float64]: ...


class SignalLimiter(Protocol):
    """Map three modulating signals to three finite limited signals, shape ``(3,)``.

    Must be deterministic; it may be called on tentative commands and keeps no saturation state.
    """

    def __call__(self, m_abc: NDArray[np.float64]) -> NDArray[np.float64]: ...


def spwm(u_ab: complex, u_dc: float) -> NDArray[np.float64]:
    """Sinusoidal PWM: return ``2 * u_abc / u_dc`` (no zero sequence)."""
    a, b, c = phases(u_ab)
    return np.array([2.0 * a / u_dc, 2.0 * b / u_dc, 2.0 * c / u_dc])


def svpwm(u_ab: complex, u_dc: float) -> NDArray[np.float64]:
    """Continuous space-vector PWM: SPWM signals with min-max zero-sequence injection."""
    m = spwm(u_ab, u_dc)
    return m - 0.5 * (float(m.max()) + float(m.min()))


PWM_METHODS = {"spwm": spwm, "svpwm": svpwm}


class ModulationLimiter:
    """Return a copy of ``m_abc`` scaled uniformly so that ``max(abs(m_abc)) <= limit``."""

    def __init__(self, limit: float) -> None:
        self.limit = limit

    def __call__(self, m_abc: NDArray[np.float64]) -> NDArray[np.float64]:
        m = np.array(m_abc, dtype=float, copy=True)
        max_abs = peak_abs(m)
        if max_abs > self.limit:
            m *= self.limit / max_abs
        return m


class _ConfiguredLimiter:
    """Distinguish an omitted limiter (use the configuration) from explicit None."""


CONFIGURED = _ConfiguredLimiter()


def _signals(value: Any, component: str) -> np.ndarray:
    """``value`` as three finite real modulating signals, or a ValueError naming ``component``."""
    message = f"{component} must return three finite real modulating signals with shape (3,)"
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError
        m = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if m.shape != (3,) or not all(map(math.isfinite, m.tolist())):
        raise ValueError(message)
    return m


class OutputStage:
    """Duty ratios from the dq voltage command of a unit's controller.

    pwm_method: ``None`` uses ``cfg.pwm.method``.
    limiter: omitted uses ``ModulationLimiter(cfg.pwm.modulation_limit)``; ``None`` disables the limit
    and its anti-windup. ``u_init_ab``: the alpha-beta voltage (V) of the start-up modulation.
    """

    def __init__(self, cfg: Any, u_init_ab: complex, pwm_method: Optional[PWMMethod] = None, *,
                 limiter: SignalLimiter | None | _ConfiguredLimiter = CONFIGURED) -> None:
        self.v_base, self.v_dc_base = cfg.base.v_phase_peak, cfg.dc_base.v
        self.pwm_method = PWM_METHODS[cfg.pwm.method] if pwm_method is None else pwm_method
        self.limiter = ModulationLimiter(cfg.pwm.modulation_limit) if limiter is CONFIGURED else limiter
        if not callable(self.pwm_method):
            raise TypeError("pwm_method must be callable")
        if self.limiter is not None and not callable(self.limiter):
            raise TypeError("limiter must be callable or None")
        self.n_updates = 0
        self.n_saturated = 0
        self.first_saturation_t: float | None = None
        self.saturated = False
        self._memo = None  # cached evaluation for the current control instant
        # start-up modulation from u_init_ab and the rated dc voltage
        self.align_startup(u_init_ab, cfg.dclink.vdc_ref)

    def initial_duty(self) -> np.ndarray:
        """The duty ratios of the start-up modulation."""
        return 0.5 * (1.0 + self.m_init)

    def align_startup(self, u_ab: complex, u_dc: float) -> None:
        """Set the start-up modulation that reproduces ``u_ab`` (V, alpha-beta) from the dc voltage ``u_dc`` (V)."""
        self.modulate(0.0, u_ab / self.v_base, 0.0, u_dc / self.v_dc_base, count=False)
        self.m_init = self.m_abc.copy()

    def new_instant(self) -> None:
        """Start a new control instant (clear the cached evaluation)."""
        self._memo = None

    def _evaluate(self, command: complex, rot: complex, u_dc: float) -> tuple[np.ndarray, bool]:
        """Modulating signals and saturation for a dq command (pu), cached for this control instant."""
        memo = self._memo
        if memo is not None and memo[0] == command and memo[1] == (rot, u_dc):
            return memo[2], memo[3]
        u_ab = command * rot * self.v_base
        if u_dc > 0.0:
            m = _signals(self.pwm_method(u_ab, u_dc), "pwm_method")
        else:
            u = complex2abc(u_ab)
            peak = float(np.max(np.abs(u)))
            m = u / peak if peak > 0.0 else np.zeros(3)
        saturated = False
        if self.limiter is not None:
            limited = _signals(self.limiter(m.copy()), "limiter")
            saturated = limited.tolist() != m.tolist() or (u_dc <= 0.0 and u_ab != 0j)  # both finite, shape (3,)
            m = limited
        self._memo = (command, (rot, u_dc), m, saturated)
        return m, saturated

    def modulate(self, t: float, u_cmd_dq: complex, theta: float, u_dc: float, cc: Any = None,
                 extra_dq: Optional[complex] = None, *, count: bool = True) -> np.ndarray:
        """Convert a dq voltage command to duty ratios, with the current loop's anti-windup.

        u_cmd_dq, extra_dq: voltage command (pu, AC base); theta: frame angle (rad); u_dc: dc voltage (pu, DC base).
        cc: current loop for anti-windup (``"conditional"``: roll back its integration, or ``"backcalc"``).
        count: ``False`` leaves the publication and saturation counters unchanged.
        """
        rot = complex(math.cos(theta), math.sin(theta))
        u_dc = u_dc * self.v_dc_base  # PWM receives DC volts; loop feedback stays pu
        u_dq = u_cmd_dq if extra_dq is None else u_cmd_dq + extra_dq
        m_abc, saturated = self._evaluate(u_dq, rot, u_dc)
        if cc is not None and cc.antiwindup == "conditional" and saturated:
            u_dq = cc.rollback() if extra_dq is None else cc.rollback() + extra_dq
            m_abc, saturated = self._evaluate(u_dq, rot, u_dc)
        if cc is not None and cc.antiwindup == "backcalc" and saturated:
            u_lim_dq = abc2complex(m_abc) * u_dc / (2.0 * self.v_base) * complex(math.cos(theta), -math.sin(theta))
            cc.backcalculate(u_dq, u_lim_dq)
        self.m_abc, self.saturated = m_abc, saturated
        if count:
            self.n_updates += 1
            if saturated:
                self.n_saturated += 1
                if self.first_saturation_t is None:
                    self.first_saturation_t = t
        return 0.5 * (1.0 + self.m_abc)
