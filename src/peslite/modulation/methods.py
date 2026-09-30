"""Compatibility access to controller output-stage PWM methods."""

from ..control.modulation import PWM_METHODS, spwm, svpwm

__all__ = ["PWM_METHODS", "make_pwm_method", "spwm", "svpwm"]


def make_pwm_method(name: str):
    """Return a built-in voltage-to-modulating-signal method by name."""
    try:
        return PWM_METHODS[name]
    except KeyError:
        raise KeyError(
            f"unknown pwm.method {name!r}; known: {sorted(PWM_METHODS)}"
        ) from None
