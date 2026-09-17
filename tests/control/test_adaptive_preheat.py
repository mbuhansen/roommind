"""Tests for adaptive pre-heat.

The optimizer only looks a few blocks ahead per decision, so a scheduled
comfort step further out stays invisible until it is too late to reach it.
Adaptive pre-heat asks the thermal model how long full-power heating needs and
starts that far in advance, with a floor derived from the heating system
profile the optimizer already uses.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from custom_components.roommind.const import TargetTemps
from custom_components.roommind.control.mpc_controller import (
    MODE_HEATING,
    MODE_IDLE,
    PREHEAT_MAX_MINUTES,
    MPCController,
)
from custom_components.roommind.control.mpc_optimizer import MPCPlan
from custom_components.roommind.control.thermal_model import RCModel

from .conftest import build_hass, make_room

# ~3 °C/h at 20 °C with outdoor 5 °C, so 2 °C takes roughly 45 min.
FAST_MODEL = RCModel(C=1.0, U=0.2, Q_heat=6.0, Q_cool=4.0)
# Equilibrium barely above the eco target — comfort is out of reach.
WEAK_MODEL = RCModel(C=1.0, U=0.2, Q_heat=3.1, Q_cool=4.0)

ECO = TargetTemps(heat=20.5, cool=27.0)
COMFORT = TargetTemps(heat=22.5, cool=24.0)


def _step_resolver(before: TargetTemps, after: TargetTemps, step_in_blocks: int):
    """Resolver switching from *before* to *after* *step_in_blocks* blocks ahead."""
    # Switch a little before the block boundary so test runtime cannot shift it.
    switch_ts = time.time() + step_in_blocks * 300 - 50
    return lambda ts: after if ts >= switch_ts else before


def _controller(
    monkeypatch,
    *,
    resolver,
    model: RCModel = FAST_MODEL,
    heating_system_type: str = "radiator",
    climate_mode: str = "auto",
    plan_action: str = MODE_IDLE,
) -> MPCController:
    mgr = MagicMock()
    mgr.get_model = MagicMock(return_value=model)
    ctrl = MPCController(
        build_hass(),
        make_room(climate_mode=climate_mode),
        model_manager=mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
        target_resolver=resolver,
        heating_system_type=heating_system_type,
    )
    power = 0.0 if plan_action == MODE_IDLE else 1.0
    fake_plan = MPCPlan(
        actions=[plan_action] * 24,
        temperatures=[20.5] * 25,
        power_fractions=[power] * 24,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    return ctrl


def test_preheat_waits_while_there_is_still_time(monkeypatch):
    """Comfort two hours out: the model has time to spare, so stay idle."""
    ctrl = _controller(monkeypatch, resolver=_step_resolver(ECO, COMFORT, 24))

    mode, pf = ctrl._evaluate_mpc(20.5, ECO)

    assert mode == MODE_IDLE
    assert pf == 0.0


def test_preheat_starts_when_the_estimate_no_longer_fits(monkeypatch):
    """Same room, same step, 40 min out: start now or arrive late."""
    ctrl = _controller(monkeypatch, resolver=_step_resolver(ECO, COMFORT, 8))

    mode, pf = ctrl._evaluate_mpc(20.5, ECO)

    assert mode == MODE_HEATING
    assert pf == 1.0
    # Direct devices must aim for the comfort target, not eco.
    assert ctrl.direct_setpoint_target(MODE_HEATING, 20.5) == 22.5


def test_preheat_start_scales_with_the_gap(monkeypatch):
    """A small gap starts later than a large one, for the same schedule step."""
    small_gap = 22.3  # 0.2 °C below comfort

    ctrl_small = _controller(monkeypatch, resolver=_step_resolver(ECO, COMFORT, 10))
    mode_small, _ = ctrl_small._evaluate_mpc(small_gap, ECO)

    ctrl_large = _controller(monkeypatch, resolver=_step_resolver(ECO, COMFORT, 10))
    mode_large, _ = ctrl_large._evaluate_mpc(19.0, ECO)

    assert mode_small == MODE_IDLE  # minutes away, 50 min of slack
    assert mode_large == MODE_HEATING  # needs longer than the 50 min left


def test_underfloor_starts_on_the_profile_floor(monkeypatch):
    """Underfloor charges its slab first, so it starts far earlier than the air model says."""
    # 80 min out: the air model alone would wait, the profile floor (30 + 60) does not.
    ctrl = _controller(
        monkeypatch,
        resolver=_step_resolver(ECO, COMFORT, 16),
        heating_system_type="underfloor",
    )

    mode, _ = ctrl._evaluate_mpc(20.5, ECO)

    assert mode == MODE_HEATING

    # A radiator in the same situation keeps waiting.
    radiator = _controller(monkeypatch, resolver=_step_resolver(ECO, COMFORT, 16))
    assert radiator._evaluate_mpc(20.5, ECO)[0] == MODE_IDLE


def test_unreachable_target_starts_at_the_cap(monkeypatch):
    """When the model never reaches comfort, heat as early as the cap allows."""
    just_inside = int(PREHEAT_MAX_MINUTES / 5) - 1  # 175 min ahead
    ctrl = _controller(
        monkeypatch,
        resolver=_step_resolver(ECO, COMFORT, just_inside),
        model=WEAK_MODEL,
    )

    assert ctrl._evaluate_mpc(20.5, ECO)[0] == MODE_HEATING

    # Beyond the cap nothing is pulled forward.
    far = _controller(
        monkeypatch,
        resolver=_step_resolver(ECO, COMFORT, int(PREHEAT_MAX_MINUTES / 5) + 6),
        model=WEAK_MODEL,
    )
    assert far._evaluate_mpc(20.5, ECO)[0] == MODE_IDLE


def test_no_preheat_when_already_warm_enough(monkeypatch):
    """Room already at the upcoming comfort target — nothing to pre-heat for."""
    ctrl = _controller(monkeypatch, resolver=_step_resolver(ECO, COMFORT, 8))

    assert ctrl._evaluate_mpc(22.6, ECO)[0] == MODE_IDLE


def test_no_preheat_without_a_step_up(monkeypatch):
    """A flat schedule (and a step down) never triggers a pre-heat."""
    flat = _controller(monkeypatch, resolver=lambda ts: ECO)
    assert flat._evaluate_mpc(20.5, ECO)[0] == MODE_IDLE

    step_down = _controller(monkeypatch, resolver=_step_resolver(COMFORT, ECO, 8))
    assert step_down._evaluate_mpc(22.5, COMFORT)[0] == MODE_IDLE


def test_off_blocks_never_pull_heating_forward(monkeypatch):
    """Blocks resolving to "off" (None) are not a step up."""
    off = TargetTemps(heat=None, cool=None)
    ctrl = _controller(monkeypatch, resolver=_step_resolver(ECO, off, 8))

    assert ctrl._evaluate_mpc(20.5, ECO)[0] == MODE_IDLE


def test_no_preheat_when_heating_is_not_allowed(monkeypatch):
    """A cool-only room must not be pulled into heating."""
    ctrl = _controller(
        monkeypatch,
        resolver=_step_resolver(ECO, COMFORT, 8),
        climate_mode="cool_only",
    )

    assert ctrl._evaluate_mpc(20.5, ECO)[0] == MODE_IDLE


def test_preheat_only_sets_the_target_when_already_heating(monkeypatch):
    """With the optimizer already heating, pre-heat just raises the direct target."""
    ctrl = _controller(
        monkeypatch,
        resolver=_step_resolver(ECO, COMFORT, 8),
        plan_action=MODE_HEATING,
    )

    mode, pf = ctrl._evaluate_mpc(20.0, ECO)

    assert mode == MODE_HEATING
    assert pf == 1.0
    assert ctrl.direct_setpoint_target(MODE_HEATING, 20.5) == 22.5
