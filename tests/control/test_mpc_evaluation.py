"""Tests for MPC evaluation: confidence transition, outdoor series, evaluate_mpc safety guards, target resolver, solar series."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from custom_components.roommind.const import TargetTemps
from custom_components.roommind.control.mpc_controller import (
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
    PREHEAT_MAX_MINUTES,
    MPCController,
)
from custom_components.roommind.control.mpc_optimizer import MPCPlan
from custom_components.roommind.control.thermal_model import RCModel, RoomModelManager

from .conftest import build_hass, make_room


@pytest.mark.asyncio
async def test_confidence_transition_threshold():
    """pred_std >= 0.5 -> bang-bang, pred_std < 0.5 -> MPC."""
    hass = build_hass()
    room = make_room()
    model_mgr = RoomModelManager()
    model_mgr.update("living_room", 18.5, 5.0, "heating", 5.0)
    model_mgr.update("living_room", 19.0, 5.0, "heating", 5.0)

    # Enough training data for MPC
    model_mgr.get_mode_counts = MagicMock(return_value=(100, 30, 0))
    # Mock a realistic trained model (2 EKF updates give alpha=_ALPHA_MIN which is
    # too low for the optimizer to distinguish heating from idle via T_eq clamping)
    model_mgr.get_model = MagicMock(return_value=RCModel(C=1.0, U=0.15, Q_heat=3.0, Q_cool=4.0))

    # Just above threshold — bang-bang
    model_mgr.get_prediction_std = MagicMock(return_value=0.5)
    ctrl = MPCController(
        hass,
        room,
        model_manager=model_mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    mode, pf = await ctrl.async_evaluate(current_temp=17.0, target_temp=21.0)
    assert mode == "heating"  # bang-bang also heats when cold
    assert pf == 1.0  # bang-bang: full power

    # Just below threshold — MPC path
    model_mgr.get_prediction_std = MagicMock(return_value=0.49)
    ctrl2 = MPCController(
        hass,
        room,
        model_manager=model_mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    mode2, pf2 = await ctrl2.async_evaluate(current_temp=17.0, target_temp=21.0)
    assert mode2 == "heating"  # MPC also heats when cold
    assert 0.0 < pf2 <= 1.0


# ---------------------------------------------------------------------------
# T4: _build_outdoor_series unit tests
# ---------------------------------------------------------------------------


class TestBuildOutdoorSeries:
    """Unit tests for MPCController._build_outdoor_series."""

    def test_constant_outdoor_no_forecast(self):
        """Without forecast, returns constant outdoor temp repeated n_blocks times."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        ctrl = MPCController(
            hass,
            room,
            model_manager=model_mgr,
            outdoor_temp=8.0,
            settings={},
            has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(10)
        assert series == [8.0] * 10

    def test_fallback_when_outdoor_temp_none_no_forecast(self):
        """Without forecast and outdoor_temp=None, uses DEFAULT_OUTDOOR_TEMP_FALLBACK."""
        from custom_components.roommind.control.mpc_controller import DEFAULT_OUTDOOR_TEMP_FALLBACK

        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        ctrl = MPCController(
            hass,
            room,
            model_manager=model_mgr,
            outdoor_temp=None,
            settings={},
            has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(5)
        assert series == [DEFAULT_OUTDOOR_TEMP_FALLBACK] * 5

    def test_forecast_used_when_available(self):
        """Block 0 = sensor, remaining blocks expand forecast[0] hourly entry."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [
            {"temperature": 5.0},
            {"temperature": 6.0},
            {"temperature": 7.0},
        ]
        ctrl = MPCController(
            hass,
            room,
            model_manager=model_mgr,
            outdoor_temp=8.0,
            outdoor_forecast=forecast,
            settings={},
            has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(3)
        # Block 0 = sensor; blocks 1-2 still within forecast[0]'s hour
        assert series == [8.0, 5.0, 5.0]

    def test_forecast_padded_when_shorter_than_n_blocks(self):
        """Short forecast: last hourly entry is repeated for the remaining blocks."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [
            {"temperature": 5.0},
            {"temperature": 6.0},
        ]
        ctrl = MPCController(
            hass,
            room,
            model_manager=model_mgr,
            outdoor_temp=8.0,
            outdoor_forecast=forecast,
            settings={},
            has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(5)
        # n_blocks=5 stays within forecast[0]'s hour after block 0 (sensor)
        assert series == [8.0, 5.0, 5.0, 5.0, 5.0]

    def test_forecast_truncated_when_longer_than_n_blocks(self):
        """n_blocks smaller than one hourly slot stays within forecast[0]."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [
            {"temperature": 5.0},
            {"temperature": 6.0},
            {"temperature": 7.0},
            {"temperature": 8.0},
            {"temperature": 9.0},
        ]
        ctrl = MPCController(
            hass,
            room,
            model_manager=model_mgr,
            outdoor_temp=10.0,
            outdoor_forecast=forecast,
            settings={},
            has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(3)
        # n_blocks=3: block 0 sensor + 2 of forecast[0]; forecast[1..4] not reached
        assert series == [10.0, 5.0, 5.0]

    def test_forecast_missing_temperature_key_uses_outdoor_temp(self):
        """Forecast entry without 'temperature' falls back to outdoor_temp."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [
            {"temperature": 5.0},
            {"condition": "cloudy"},  # no temperature key
            {"temperature": 7.0},
        ]
        ctrl = MPCController(
            hass,
            room,
            model_manager=model_mgr,
            outdoor_temp=8.0,
            outdoor_forecast=forecast,
            settings={},
            has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(3)
        # n_blocks=3 stays within forecast[0]'s hour; block 0 = sensor
        assert series == [8.0, 5.0, 5.0]

    def test_forecast_missing_temp_key_and_outdoor_none_uses_fallback(self):
        """Missing temperature + outdoor_temp=None falls back to DEFAULT for all blocks."""
        from custom_components.roommind.control.mpc_controller import DEFAULT_OUTDOOR_TEMP_FALLBACK

        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [
            {"condition": "cloudy"},  # no temperature key
        ]
        ctrl = MPCController(
            hass,
            room,
            model_manager=model_mgr,
            outdoor_temp=None,
            outdoor_forecast=forecast,
            settings={},
            has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(3)
        # outdoor_temp None → block 0 is not overridden; whole hour uses fallback
        assert all(v == DEFAULT_OUTDOOR_TEMP_FALLBACK for v in series)

    def test_block_zero_is_sensor_reading(self):
        """Block 0 always uses the real-time sensor, not forecast[0]."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [{"temperature": 15.6}, {"temperature": 14.0}]
        ctrl = MPCController(
            hass,
            room,
            model_manager=model_mgr,
            outdoor_temp=20.67,
            outdoor_forecast=forecast,
            settings={},
            has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(6)
        assert series == [20.67, 15.6, 15.6, 15.6, 15.6, 15.6]

    def test_hourly_forecast_expanded_to_blocks(self):
        """Each hourly forecast entry covers 60 // PLAN_DT_MINUTES blocks."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [{"temperature": 15.6}, {"temperature": 14.0}]
        ctrl = MPCController(
            hass,
            room,
            model_manager=model_mgr,
            outdoor_temp=20.67,
            outdoor_forecast=forecast,
            settings={},
            has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(24)
        # First hour (blocks 0-11): sensor + 11 of forecast[0]
        # Second hour (blocks 12-23): 12 of forecast[1]
        assert series == [20.67] + [15.6] * 11 + [14.0] * 12

    def test_outdoor_temp_none_uses_forecast_for_block_zero(self):
        """When sensor reading is missing, block 0 falls back to forecast[0]."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [{"temperature": 12.3}]
        ctrl = MPCController(
            hass,
            room,
            model_manager=model_mgr,
            outdoor_temp=None,
            outdoor_forecast=forecast,
            settings={},
            has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(3)
        assert series == [12.3, 12.3, 12.3]

    def test_pad_with_last_value_when_forecast_exhausted(self):
        """When n_blocks exceeds the forecast horizon, last value pads the tail."""
        hass = build_hass()
        room = make_room()
        model_mgr = RoomModelManager()
        forecast = [{"temperature": 5.0}, {"temperature": 6.0}]
        ctrl = MPCController(
            hass,
            room,
            model_manager=model_mgr,
            outdoor_temp=8.0,
            outdoor_forecast=forecast,
            settings={},
            has_external_sensor=True,
        )
        series = ctrl._build_outdoor_series(30)
        # 1 sensor + 11 of forecast[0] + 12 of forecast[1] + 6 padding (= forecast[1])
        assert series == [8.0] + [5.0] * 11 + [6.0] * 12 + [6.0] * 6


# ---------------------------------------------------------------------------
# _evaluate_mpc edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_mpc_none_inputs():
    """None current_temp or target_temp returns idle."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    mode, pf = ctrl._evaluate_mpc(None, TargetTemps(heat=21.0, cool=24.0))
    assert mode == MODE_IDLE
    assert pf == 0.0

    mode, pf = ctrl._evaluate_mpc(20.0, TargetTemps(heat=None, cool=None))
    assert mode == MODE_IDLE
    assert pf == 0.0


# ---------------------------------------------------------------------------
# _build_outdoor_series (additional)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _evaluate_mpc safety guard
# ---------------------------------------------------------------------------


def test_evaluate_mpc_safety_guard_heating_above_target(monkeypatch):
    """Safety guard overrides heating to idle when temp >= max(near_targets)."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=19.0,
        settings={},
        has_external_sensor=True,
    )
    # Mock optimizer to return HEATING so the safety guard is actually tested
    fake_plan = MPCPlan(
        actions=[MODE_HEATING] * 6,
        temperatures=[22.0] * 7,
        power_fractions=[0.8] * 6,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    # current_temp=22 >= target=21 → safety guard should override to idle
    mode, pf = ctrl._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=24.0))
    assert mode == MODE_IDLE
    assert pf == 0.0


def test_evaluate_mpc_safety_guard_cooling_below_target(monkeypatch):
    """Safety guard overrides cooling to idle when temp <= min(near_targets)."""
    hass = build_hass()
    room = make_room(acs=["climate.ac1"], thermostats=[], climate_mode="cool_only")
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=30.0,
        settings={},
        has_external_sensor=True,
    )
    # Mock optimizer to return COOLING so the safety guard is tested
    fake_plan = MPCPlan(
        actions=[MODE_COOLING] * 6,
        temperatures=[22.0] * 7,
        power_fractions=[0.8] * 6,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    # current_temp=22 <= target=23 → safety guard should override to idle
    mode, pf = ctrl._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=23.0))
    assert mode == MODE_IDLE
    assert pf == 0.0


def test_evaluate_mpc_safety_guard_respects_min_run_heating(monkeypatch):
    """Safety guard must NOT override heating when within minimum run window."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=21.0,  # near room temp so predicted idle temp won't drop below target-margin
        settings={},
        has_external_sensor=True,
        previous_mode=MODE_HEATING,
        heating_system_type="underfloor",
        mode_on_since=time.time() - 60,  # started 1 min ago (within 30-min window)
    )
    fake_plan = MPCPlan(
        actions=[MODE_HEATING] * 6,
        temperatures=[22.0] * 7,
        power_fractions=[0.8] * 6,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    # current_temp=22 >= target=21 but we're in min-run window → must keep heating
    mode, pf = ctrl._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=24.0))
    assert mode == MODE_HEATING


def test_evaluate_mpc_safety_guard_fires_after_min_run_heating(monkeypatch):
    """Safety guard must override heating when minimum run window has elapsed."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=19.0,
        settings={},
        has_external_sensor=True,
        previous_mode=MODE_HEATING,
        heating_system_type="underfloor",
        mode_on_since=time.time() - 2000,  # started 33 min ago (past 30-min window)
    )
    fake_plan = MPCPlan(
        actions=[MODE_HEATING] * 6,
        temperatures=[22.0] * 7,
        power_fractions=[0.8] * 6,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    mode, pf = ctrl._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=24.0))
    assert mode == MODE_IDLE
    assert pf == 0.0


def test_evaluate_mpc_safety_guard_fires_after_default_min_run_heating(monkeypatch):
    """Safety guard must override heating when default minimum run window has elapsed."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=19.0,
        settings={},
        has_external_sensor=True,
        previous_mode=MODE_HEATING,
        heating_system_type="",
        mode_on_since=time.time() - 660,  # started 11 min ago (past 10-min default window)
    )
    fake_plan = MPCPlan(
        actions=[MODE_HEATING] * 6,
        temperatures=[22.0] * 7,
        power_fractions=[0.8] * 6,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    mode, pf = ctrl._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=24.0))
    assert mode == MODE_IDLE
    assert pf == 0.0


# ---------------------------------------------------------------------------
# _predict_idle_drift
# ---------------------------------------------------------------------------


def test_predict_idle_drift_returns_model_prediction():
    """_predict_idle_drift predicts temperature with Q_active=0."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
        q_solar=0.3,
        q_residual=0.1,
        shading_factor=0.8,
        q_occupancy=0.5,
    )

    model = mgr.get_model("living_room")
    predicted = ctrl._predict_idle_drift(21.0, 30.0)
    expected = model.predict(
        21.0,
        5.0,
        Q_active=0.0,
        dt_minutes=30.0,
        q_solar=0.3 * 0.8,
        q_residual=0.1,
        q_occupancy=0.5,
    )
    assert predicted == pytest.approx(expected, abs=0.01)


def test_predict_idle_drift_uses_fallback_outdoor_temp():
    """When outdoor_temp is None, uses DEFAULT_OUTDOOR_TEMP_FALLBACK."""
    from custom_components.roommind.control.mpc_controller import (
        DEFAULT_OUTDOOR_TEMP_FALLBACK,
    )

    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=None,
        settings={},
        has_external_sensor=True,
    )

    model = mgr.get_model("living_room")
    predicted = ctrl._predict_idle_drift(21.0, 30.0)
    expected = model.predict(21.0, DEFAULT_OUTDOOR_TEMP_FALLBACK, Q_active=0.0, dt_minutes=30.0)
    assert predicted == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Prediction-aware safety guard
# ---------------------------------------------------------------------------


def test_safety_guard_allows_heating_when_prediction_dips(monkeypatch):
    """Safety guard allows HEATING when idle-drift predicts temp below target."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )

    fake_plan = MPCPlan(
        actions=[MODE_HEATING] * 24,
        temperatures=[21.0] * 25,
        power_fractions=[0.8] * 24,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )

    # current_temp=21.0 >= target=21.0 → guard entry triggers
    # With outdoor=5, 30-min idle prediction from 21°C dips well below 21 - 0.2
    # → guard should allow heating
    mode, pf = ctrl._evaluate_mpc(21.0, TargetTemps(heat=21.0, cool=25.0))
    assert mode == MODE_HEATING
    assert pf > 0.0


def test_safety_guard_suppresses_when_prediction_stays_warm(monkeypatch):
    """Safety guard suppresses HEATING when prediction stays above target."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=20.0,
        settings={},
        has_external_sensor=True,
    )

    fake_plan = MPCPlan(
        actions=[MODE_HEATING] * 24,
        temperatures=[22.0] * 25,
        power_fractions=[0.8] * 24,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )

    # current_temp=22 >= target=21, outdoor=20 → prediction stays warm → suppress
    mode, pf = ctrl._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=25.0))
    assert mode == MODE_IDLE
    assert pf == 0.0


def test_safety_guard_adaptive_horizon_underfloor(monkeypatch):
    """Underfloor heating uses guard horizon derived from min_run_blocks.

    On upstream (min_run_minutes=30): max(6, 6) = 6 blocks = 30 min.
    On dev (min_run_minutes=60): max(6, 12) = 12 blocks = 60 min.
    Either way, the prediction-aware guard still allows heating when
    the model predicts a dip.
    """
    hass = build_hass()
    room = make_room(heating_system_type="underfloor")
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
        heating_system_type="underfloor",
    )

    fake_plan = MPCPlan(
        actions=[MODE_HEATING] * 24,
        temperatures=[21.0] * 25,
        power_fractions=[0.8] * 24,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )

    # At outdoor=5, idle drift from 21°C drops below 21 - 0.2 = 20.8
    # → guard allows heating regardless of horizon length
    mode, pf = ctrl._evaluate_mpc(21.0, TargetTemps(heat=21.0, cool=25.0))
    assert mode == MODE_HEATING


def test_safety_guard_allows_cooling_when_prediction_rises(monkeypatch):
    """Safety guard allows COOLING when idle-drift predicts temp above target."""
    hass = build_hass()
    room = make_room(acs=["climate.ac1"], thermostats=[], climate_mode="cool_only")
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=35.0,
        settings={},
        has_external_sensor=True,
        q_solar=0.5,
    )

    fake_plan = MPCPlan(
        actions=[MODE_COOLING] * 24,
        temperatures=[24.0] * 25,
        power_fractions=[0.8] * 24,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )

    # current_temp=24 <= cool_target=24 → guard entry triggers
    # With outdoor=35 and solar=0.5, prediction rises above 24 + 0.2 → allow cooling
    mode, pf = ctrl._evaluate_mpc(24.0, TargetTemps(heat=21.0, cool=24.0))
    assert mode == MODE_COOLING
    assert pf > 0.0


def test_safety_guard_suppresses_cooling_when_prediction_stays_cool(monkeypatch):
    """Safety guard suppresses COOLING when prediction stays below target."""
    hass = build_hass()
    room = make_room(acs=["climate.ac1"], thermostats=[], climate_mode="cool_only")
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=23.0,
        settings={},
        has_external_sensor=True,
    )

    fake_plan = MPCPlan(
        actions=[MODE_COOLING] * 24,
        temperatures=[23.0] * 25,
        power_fractions=[0.8] * 24,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )

    # current_temp=23 <= cool_target=23, outdoor=23 → prediction stays stable → suppress
    mode, pf = ctrl._evaluate_mpc(23.0, TargetTemps(heat=21.0, cool=23.0))
    assert mode == MODE_IDLE
    assert pf == 0.0


# ---------------------------------------------------------------------------
# Hard overshoot ceiling (issue #152)
# ---------------------------------------------------------------------------


def test_hard_ceiling_overrides_heating_past_overshoot(monkeypatch):
    """Hard ceiling forces idle when temp > max(target) + 1°C, even during min-run."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=5.0,  # cold outdoor — model would predict dip and allow heating
        settings={},
        has_external_sensor=True,
        previous_mode=MODE_HEATING,
        heating_system_type="underfloor",
        mode_on_since=time.time() - 60,  # within 30-min min-run window
    )
    fake_plan = MPCPlan(
        actions=[MODE_HEATING] * 24,
        temperatures=[23.0] * 25,
        power_fractions=[0.8] * 24,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    # 23°C > 21°C + 1.0 → hard ceiling must fire, overriding min-run
    mode, pf = ctrl._evaluate_mpc(23.0, TargetTemps(heat=21.0, cool=25.0))
    assert mode == MODE_IDLE
    assert pf == 0.0


def test_hard_ceiling_does_not_fire_below_threshold(monkeypatch):
    """Hard ceiling does NOT fire when overshoot is below the ceiling threshold."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
    )
    fake_plan = MPCPlan(
        actions=[MODE_HEATING] * 24,
        temperatures=[21.5] * 25,
        power_fractions=[0.8] * 24,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    # 21.5°C >= 21°C → model-based guard entry, outdoor=5 → prediction dips → allow
    mode, pf = ctrl._evaluate_mpc(21.5, TargetTemps(heat=21.0, cool=25.0))
    assert mode == MODE_HEATING
    assert pf > 0.0


def test_hard_ceiling_overrides_cooling_past_overshoot(monkeypatch):
    """Hard ceiling forces idle when temp < min(cool_target) - 1°C."""
    hass = build_hass()
    room = make_room(acs=["climate.ac1"], thermostats=[], climate_mode="cool_only")
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=30.0,
        settings={},
        has_external_sensor=True,
    )
    fake_plan = MPCPlan(
        actions=[MODE_COOLING] * 24,
        temperatures=[21.0] * 25,
        power_fractions=[0.8] * 24,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    # 21°C < 23°C - 1.0 = 22°C → hard ceiling must fire
    mode, pf = ctrl._evaluate_mpc(21.0, TargetTemps(heat=20.0, cool=23.0))
    assert mode == MODE_IDLE
    assert pf == 0.0


def test_hard_ceiling_exact_boundary_does_not_fire(monkeypatch):
    """At exactly target + 1.0°C, hard ceiling does NOT fire (strict >)."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=20.0,  # warm outdoor → prediction stays warm → model-based guard suppresses
        settings={},
        has_external_sensor=True,
    )
    fake_plan = MPCPlan(
        actions=[MODE_HEATING] * 24,
        temperatures=[22.0] * 25,
        power_fractions=[0.8] * 24,
    )
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    # 22°C is exactly 21+1.0 → strict > means hard ceiling does NOT fire
    # Falls through to model-based guard which suppresses (outdoor=20 → warm prediction)
    mode, pf = ctrl._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=25.0))
    assert mode == MODE_IDLE
    assert pf == 0.0


# ---------------------------------------------------------------------------
# _evaluate_mpc without target_resolver
# ---------------------------------------------------------------------------


def test_evaluate_mpc_no_target_resolver():
    """Without target_resolver, uses flat target_series."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
        target_resolver=None,
    )
    # Call _evaluate_mpc — room is cold (17 vs target 21) → should heat
    mode, pf = ctrl._evaluate_mpc(17.0, TargetTemps(heat=21.0, cool=24.0))
    assert mode == MODE_HEATING
    # Should also store a last_plan
    assert ctrl.last_plan is not None


def test_evaluate_mpc_with_target_resolver():
    """With target_resolver, builds target_series from resolver calls."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
        target_resolver=lambda ts: TargetTemps(heat=21.0, cool=24.0),  # constant resolver
    )
    # Room is cold (17 vs target 21) → should heat
    mode, pf = ctrl._evaluate_mpc(17.0, TargetTemps(heat=21.0, cool=24.0))
    assert mode == MODE_HEATING
    assert ctrl.last_plan is not None


# ---------------------------------------------------------------------------
# direct_setpoint_target (pre-heat / pre-cool target for direct devices)
# ---------------------------------------------------------------------------


def _step_resolver(before: TargetTemps, after: TargetTemps, step_in_blocks: int):
    """Resolver that switches from *before* to *after* at block *step_in_blocks*."""
    # Switch slightly ahead of the block boundary so a few seconds of test runtime
    # between here and the resolver calls cannot shift the step by one block.
    switch_ts = time.time() + step_in_blocks * 300 - 50
    return lambda ts: after if ts >= switch_ts else before


def _preheat_ctrl(monkeypatch, resolver, action: str) -> MPCController:
    ctrl = MPCController(
        build_hass(),
        make_room(),
        model_manager=RoomModelManager(),
        outdoor_temp=10.0,
        settings={},
        has_external_sensor=True,
        target_resolver=resolver,
    )
    fake_plan = MPCPlan(actions=[action] * 24, temperatures=[21.0] * 25, power_fractions=[1.0] * 24)
    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    return ctrl


def test_direct_setpoint_target_preheat_uses_upcoming_comfort(monkeypatch):
    """Pre-heating for comfort 25 min ahead: direct devices aim for comfort, not eco."""
    eco = TargetTemps(heat=20.5, cool=27.0)
    comfort = TargetTemps(heat=22.5, cool=24.0)
    ctrl = _preheat_ctrl(monkeypatch, _step_resolver(eco, comfort, 5), MODE_HEATING)

    mode, _ = ctrl._evaluate_mpc(22.0, eco)

    assert mode == MODE_HEATING
    assert ctrl.direct_setpoint_target(MODE_HEATING, 20.5) == 22.5
    # Only the commanded mode is affected
    assert ctrl.direct_setpoint_target(MODE_IDLE, 20.5) == 20.5


def test_direct_setpoint_target_ignores_comfort_beyond_preheat_window(monkeypatch):
    """A target change further out than adaptive pre-heat looks does not raise the setpoint."""
    eco = TargetTemps(heat=18.0, cool=27.0)
    comfort = TargetTemps(heat=22.5, cool=24.0)
    beyond_cap = int(PREHEAT_MAX_MINUTES / 5) + 6
    ctrl = _preheat_ctrl(monkeypatch, _step_resolver(eco, comfort, beyond_cap), MODE_HEATING)

    mode, _ = ctrl._evaluate_mpc(17.0, eco)

    assert mode == MODE_HEATING
    assert ctrl.direct_setpoint_target(MODE_HEATING, 18.0) == 18.0


def test_direct_setpoint_target_not_raised_when_not_heating(monkeypatch):
    """No pre-heat target when the optimizer stays idle and no step-up is coming."""
    eco = TargetTemps(heat=20.5, cool=27.0)
    ctrl = _preheat_ctrl(monkeypatch, lambda ts: eco, MODE_IDLE)

    ctrl._evaluate_mpc(22.0, eco)

    assert ctrl.direct_setpoint_target(MODE_HEATING, 20.5) == 20.5


def test_direct_setpoint_target_never_lowers_heat_target(monkeypatch):
    """Comfort now, eco ahead: the current (higher) target is kept."""
    comfort = TargetTemps(heat=22.5, cool=24.0)
    eco = TargetTemps(heat=20.5, cool=27.0)
    ctrl = _preheat_ctrl(monkeypatch, _step_resolver(comfort, eco, 3), MODE_HEATING)

    mode, _ = ctrl._evaluate_mpc(21.0, comfort)

    assert mode == MODE_HEATING
    assert ctrl.direct_setpoint_target(MODE_HEATING, 22.5) == 22.5


def test_direct_setpoint_target_precool_uses_upcoming_cool_target(monkeypatch):
    """Pre-cooling mirrors pre-heating: direct ACs aim for the lower upcoming target."""
    eco = TargetTemps(heat=18.0, cool=27.0)
    comfort = TargetTemps(heat=21.0, cool=24.0)
    ctrl = _preheat_ctrl(monkeypatch, _step_resolver(eco, comfort, 4), MODE_COOLING)
    ctrl._get_can_heat_cool = lambda: (True, True)

    mode, _ = ctrl._evaluate_mpc(25.0, eco)

    assert mode == MODE_COOLING
    assert ctrl.direct_setpoint_target(MODE_COOLING, 27.0) == 24.0


def test_direct_setpoint_target_off_blocks_are_ignored(monkeypatch):
    """Blocks resolving to "off" (None) never become a pre-heat target."""
    now_target = TargetTemps(heat=20.0, cool=26.0)
    ctrl = _preheat_ctrl(
        monkeypatch,
        _step_resolver(now_target, TargetTemps(heat=None, cool=None), 2),
        MODE_HEATING,
    )

    mode, _ = ctrl._evaluate_mpc(19.5, now_target)

    assert mode == MODE_HEATING
    assert ctrl.direct_setpoint_target(MODE_HEATING, 20.0) == 20.0


# ---------------------------------------------------------------------------
# _build_solar_series with cloud_series
# ---------------------------------------------------------------------------


def test_build_solar_series_with_cloud():
    """Cloud series is expanded to 5-min blocks and passed to build_solar_series."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
        cloud_series=[50.0, 80.0],
        latitude=48.0,
        longitude=11.0,
    )
    series = ctrl._build_solar_series(30)
    assert len(series) == 30
    # All values should be non-negative floats
    assert all(isinstance(v, (int, float)) and v >= 0 for v in series)


def test_build_solar_series_short_cloud_extended():
    """Short cloud series is extended to fill n_blocks."""
    hass = build_hass()
    room = make_room()
    mgr = RoomModelManager()
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=5.0,
        settings={},
        has_external_sensor=True,
        cloud_series=[50.0],  # only 1 hour = 12 blocks
        latitude=48.0,
        longitude=11.0,
    )
    series = ctrl._build_solar_series(30)
    assert len(series) == 30


# ---------------------------------------------------------------------------
# Issue #131: UFH proactive pre-heating — end-to-end via _evaluate_mpc
# ---------------------------------------------------------------------------


def test_evaluate_mpc_ufh_preheats_before_setpoint_drop():
    """UFH room at target with cold outdoor: _evaluate_mpc picks HEATING proactively."""
    hass = build_hass()
    room = make_room(heating_system_type="underfloor")
    mgr = RoomModelManager()
    # Stub a well-trained model and enough data for MPC to activate.
    mgr.get_mode_counts = MagicMock(return_value=(100, 50, 0))
    mgr.get_prediction_std = MagicMock(return_value=0.3)
    mgr.get_model = MagicMock(return_value=RCModel(C=1.0, U=0.15, Q_heat=3.0, Q_cool=4.0))
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=5.0,
        settings={"comfort_weight": 70, "control_mode": "mpc"},
        has_external_sensor=True,
        heating_system_type="underfloor",
    )
    mode, pf = ctrl._evaluate_mpc(20.0, TargetTemps(heat=20.0, cool=20.0))
    assert mode == MODE_HEATING, f"Expected HEATING, got {mode}"
    assert pf > 0.0


def test_mpc_guard_horizon_extended_for_ufh(monkeypatch):
    """UFH guard uses the optimizer's extended lookahead for idle-drift prediction.

    At outdoor=19°C, room=21°C, target=21°C the 30-min drift stays above the
    20.8°C suppression threshold (would trigger old-style suppression), but the
    120-min drift dips below it (the fix allows heating).
    """
    hass = build_hass()
    room = make_room(heating_system_type="underfloor")
    mgr = RoomModelManager()
    mgr.get_mode_counts = MagicMock(return_value=(100, 50, 0))
    mgr.get_prediction_std = MagicMock(return_value=0.3)
    mgr.get_model = MagicMock(return_value=RCModel(C=1.0, U=0.15, Q_heat=3.0, Q_cool=4.0))
    ctrl = MPCController(
        hass,
        room,
        model_manager=mgr,
        outdoor_temp=19.0,
        settings={"comfort_weight": 70, "control_mode": "mpc"},
        has_external_sensor=True,
        heating_system_type="underfloor",
    )

    fake_plan = MPCPlan(
        actions=[MODE_HEATING] * 24,
        temperatures=[21.0] * 25,
        power_fractions=[0.8] * 24,
        lookahead_blocks=24,  # UFH lookahead as per fix
    )

    monkeypatch.setattr(
        "custom_components.roommind.control.mpc_controller.MPCOptimizer.optimize",
        lambda *a, **kw: fake_plan,
    )
    mode, _ = ctrl._evaluate_mpc(21.0, TargetTemps(heat=21.0, cool=25.0))
    assert mode == MODE_HEATING, "Extended guard horizon for UFH should allow heating at mild outdoor=19°C"


# ---------------------------------------------------------------------------
# Occupancy is kept out of the shading peak (#418)
# ---------------------------------------------------------------------------


def _occupancy_controller(q_occupancy: float) -> MPCController:
    """Controller whose model has a strong occupancy gain and a slow room."""
    mgr = RoomModelManager()
    mgr.get_model = MagicMock(return_value=RCModel(C=1.0, U=0.25, Q_heat=1.0, Q_cool=1.0, Q_occupancy=1.46))
    return MPCController(
        build_hass(),
        make_room(),
        model_manager=mgr,
        outdoor_temp=24.0,
        settings={},
        has_external_sensor=True,
        q_occupancy=q_occupancy,
    )


def test_simulate_plan_reproduces_the_plan_it_replays():
    """Replaying a plan with its own inputs must return the plan's trajectory."""
    from custom_components.roommind.control.mpc_optimizer import MPCOptimizer

    model = RCModel(C=1.0, U=0.25, Q_heat=1.0, Q_cool=1.0, Q_occupancy=1.46)
    optimizer = MPCOptimizer(model=model, can_heat=True, can_cool=True)
    outdoor = [24.0] * 24
    solar = [0.2] * 24
    residual = [0.3] * 24
    occupancy = [1.0] * 24
    plan = optimizer.optimize(
        T_room=22.0,
        T_outdoor_series=outdoor,
        heat_target_series=[21.0] * 24,
        cool_target_series=[24.0] * 24,
        dt_minutes=5,
        solar_series=solar,
        residual_series=residual,
        occupancy_series=occupancy,
    )

    replayed = optimizer.simulate_plan(
        plan,
        22.0,
        outdoor,
        5,
        solar_series=solar,
        residual_series=residual,
        occupancy_series=occupancy,
    )
    assert replayed == plan.temperatures


def test_predicted_peak_excludes_occupancy_heat():
    """The shading peak must not carry body heat, the control plan still does."""
    ctrl = _occupancy_controller(1.0)
    ctrl._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=24.0))

    assert ctrl.last_plan is not None
    plan_peak = max(ctrl.last_plan.temperatures[1:])
    assert ctrl.predicted_peak_temp is not None
    assert ctrl.predicted_peak_temp < plan_peak - 1.0


def test_control_plan_still_sees_occupancy():
    """Removing occupancy from the peak must not change the control decision."""
    occupied = _occupancy_controller(1.0)
    empty = _occupancy_controller(0.0)
    occupied._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=24.0))
    empty._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=24.0))

    assert occupied.last_plan is not None and empty.last_plan is not None
    # The optimizer's own trajectory diverges because it still gets the gain.
    assert max(occupied.last_plan.temperatures[1:]) > max(empty.last_plan.temperatures[1:])


def test_predicted_peak_matches_empty_room_peak():
    """An occupied and an empty room must offer the cover logic the same peak."""
    occupied = _occupancy_controller(1.0)
    empty = _occupancy_controller(0.0)
    occupied._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=24.0))
    empty._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=24.0))

    assert occupied.predicted_peak_temp == pytest.approx(empty.predicted_peak_temp, abs=0.05)


def test_unoccupied_room_uses_the_plan_trajectory_directly():
    """Without occupancy there is nothing to subtract, so no replay happens."""
    ctrl = _occupancy_controller(0.0)
    ctrl._evaluate_mpc(22.0, TargetTemps(heat=21.0, cool=24.0))

    assert ctrl._peak_temps_unoccupied is None
    assert ctrl.last_plan is not None
    assert ctrl.predicted_peak_temp == max(ctrl.last_plan.temperatures[1:])


def test_simulate_plan_replays_directed_actions():
    """Heating and cooling blocks must replay with their power, not as idle."""
    from custom_components.roommind.control.mpc_optimizer import MPCOptimizer

    model = RCModel(C=1.0, U=0.25, Q_heat=1.0, Q_cool=2.0, Q_occupancy=1.0)
    optimizer = MPCOptimizer(model=model, can_heat=False, can_cool=True)
    plan = MPCPlan(
        actions=[MODE_COOLING] * 4,
        temperatures=[28.0] * 5,
        power_fractions=[1.0] * 4,
    )
    replayed = optimizer.simulate_plan(plan, 28.0, [30.0] * 4, 5)

    assert replayed[0] == 28.0
    assert replayed[-1] < 28.0  # cooling actually removed heat

    heat_plan = MPCPlan(
        actions=[MODE_HEATING] * 4,
        temperatures=[18.0] * 5,
        power_fractions=[1.0] * 4,
    )
    idle_plan = MPCPlan(
        actions=[MODE_IDLE] * 4,
        temperatures=[18.0] * 5,
        power_fractions=[0.0] * 4,
    )
    heated = optimizer.simulate_plan(heat_plan, 18.0, [10.0] * 4, 5)
    idled = optimizer.simulate_plan(idle_plan, 18.0, [10.0] * 4, 5)
    assert heated[-1] > idled[-1]
