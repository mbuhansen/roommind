"""Tests for EKF training with TRV-reported valve position.

A self-regulating TRV (e.g. setpoint_mode="direct") often opens its valve only
partially while RoomMind commands heating.  Training with the commanded power
fraction then teaches the model a heating rate far below the real one; the
reported valve opening is used instead when it is conclusive.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.roommind.const import MODE_HEATING, MODE_IDLE

from .conftest import (
    SAMPLE_ROOM,
    _create_coordinator,
    _make_store_mock,
    make_mock_states_get,
)


def _trv(entity_id: str, valve_entity: str | None) -> dict:
    dev = {
        "entity_id": entity_id,
        "type": "trv",
        "role": "auto",
        "heating_system_type": "radiator",
        "idle_action": "off",
        "idle_fan_mode": "low",
        "setpoint_mode": "direct",
    }
    if valve_entity is not None:
        dev["valve_position_entity"] = valve_entity
    return dev


def _room(devices: list[dict]) -> dict:
    return {
        **SAMPLE_ROOM,
        "temperature_sensor": "sensor.living_room_temp",
        "devices": devices,
        "heating_system_type": "radiator",
        "thermostats": [d["entity_id"] for d in devices if d["type"] == "trv"],
        "acs": [d["entity_id"] for d in devices if d["type"] == "ac"],
        # valve protection exclusion must not affect valve-position training
        "valve_protection_exclude": [d["entity_id"] for d in devices],
        "comfort_temp": 21.0,
        "eco_temp": 17.0,
        "comfort_heat": 21.0,
        "eco_heat": 17.0,
        "comfort_cool": 24.0,
        "eco_cool": 27.0,
    }


def _heating_trv_state(hvac_action: str = "heating") -> tuple[str, dict]:
    return ("heat", {"hvac_action": hvac_action, "current_temperature": 19.5, "temperature": 21.0})


async def _train_kwargs(hass, mock_config_entry, room: dict, extra: dict, settings: dict | None = None):
    """Run one update (room cold → heating) and return the EKF training kwargs."""
    states_get = make_mock_states_get(
        temp="19.5",
        humidity="55.0",
        schedule_state="on",
        outdoor_temp="5.0",
        extra=extra,
    )
    store = _make_store_mock({room["area_id"]: room})
    if settings:
        store.get_settings.return_value = settings
    hass.data = {"roommind": {"store": store}}
    hass.states.get = MagicMock(side_effect=states_get)
    hass.services.async_call = AsyncMock()

    coordinator = _create_coordinator(hass, mock_config_entry)
    training_mock = MagicMock()
    coordinator._ekf_training.process = training_mock

    await coordinator._async_update_data()

    assert training_mock.called
    return training_mock.call_args.kwargs


class TestEkfValveFraction:
    @pytest.mark.asyncio
    async def test_mean_valve_opening_replaces_commanded_power(self, hass, mock_config_entry):
        room = _room([_trv("climate.trv_a", "sensor.valve_a"), _trv("climate.trv_b", "sensor.valve_b")])
        extra = {
            "climate.trv_a": _heating_trv_state(),
            "climate.trv_b": _heating_trv_state(),
            "sensor.valve_a": ("20", {}),
            "sensor.valve_b": ("40", {}),
        }
        kwargs = await _train_kwargs(hass, mock_config_entry, room, extra)

        assert kwargs["ekf_mode"] == MODE_HEATING
        assert kwargs["ekf_pf"] == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_trv_without_valve_entity_keeps_commanded_power(self, hass, mock_config_entry):
        room = _room([_trv("climate.trv_a", "sensor.valve_a"), _trv("climate.trv_b", None)])
        extra = {
            "climate.trv_a": _heating_trv_state(),
            "climate.trv_b": _heating_trv_state(),
            "sensor.valve_a": ("20", {}),
        }
        kwargs = await _train_kwargs(hass, mock_config_entry, room, extra)

        assert kwargs["ekf_mode"] == MODE_HEATING
        assert kwargs["ekf_pf"] == 1.0

    @pytest.mark.parametrize("valve_state", ["unavailable", "unknown", "open"])
    @pytest.mark.asyncio
    async def test_non_numeric_valve_keeps_commanded_power(self, hass, mock_config_entry, valve_state):
        room = _room([_trv("climate.trv_a", "sensor.valve_a")])
        extra = {"climate.trv_a": _heating_trv_state(), "sensor.valve_a": (valve_state, {})}
        kwargs = await _train_kwargs(hass, mock_config_entry, room, extra)

        assert kwargs["ekf_mode"] == MODE_HEATING
        assert kwargs["ekf_pf"] == 1.0

    @pytest.mark.asyncio
    async def test_missing_valve_entity_state_keeps_commanded_power(self, hass, mock_config_entry):
        room = _room([_trv("climate.trv_a", "sensor.valve_gone")])
        extra = {"climate.trv_a": _heating_trv_state()}
        kwargs = await _train_kwargs(hass, mock_config_entry, room, extra)

        assert kwargs["ekf_pf"] == 1.0

    @pytest.mark.asyncio
    async def test_room_with_ac_is_not_overridden(self, hass, mock_config_entry):
        ac = {
            "entity_id": "climate.ac",
            "type": "ac",
            "role": "auto",
            "heating_system_type": "",
            "idle_action": "off",
            "idle_fan_mode": "low",
            "setpoint_mode": "direct",
        }
        room = _room([_trv("climate.trv_a", "sensor.valve_a"), ac])
        room["climate_mode"] = "heat_only"
        extra = {
            "climate.trv_a": _heating_trv_state(),
            "climate.ac": ("off", {"hvac_modes": ["off", "cool"]}),
            "sensor.valve_a": ("20", {}),
        }
        kwargs = await _train_kwargs(hass, mock_config_entry, room, extra)

        assert kwargs["ekf_pf"] != pytest.approx(0.2)

    @pytest.mark.parametrize("valve_pct", ["0", "1.5"])
    @pytest.mark.asyncio
    async def test_closed_valve_trains_as_idle(self, hass, mock_config_entry, valve_pct):
        room = _room([_trv("climate.trv_a", "sensor.valve_a")])
        extra = {"climate.trv_a": _heating_trv_state(), "sensor.valve_a": (valve_pct, {})}
        kwargs = await _train_kwargs(hass, mock_config_entry, room, extra)

        assert kwargs["ekf_mode"] == MODE_IDLE
        assert kwargs["q_residual"] == 0.0

    @pytest.mark.asyncio
    async def test_valve_ignored_when_not_heating(self, hass, mock_config_entry):
        room = _room([_trv("climate.trv_a", "sensor.valve_a")])
        extra = {"climate.trv_a": _heating_trv_state("idle"), "sensor.valve_a": ("60", {})}
        states_get = make_mock_states_get(
            temp="23.0",  # above comfort → idle
            humidity="55.0",
            schedule_state="on",
            outdoor_temp="5.0",
            extra=extra,
        )
        store = _make_store_mock({room["area_id"]: room})
        hass.data = {"roommind": {"store": store}}
        hass.states.get = MagicMock(side_effect=states_get)
        hass.services.async_call = AsyncMock()
        coordinator = _create_coordinator(hass, mock_config_entry)
        training_mock = MagicMock()
        coordinator._ekf_training.process = training_mock

        await coordinator._async_update_data()

        kwargs = training_mock.call_args.kwargs
        assert kwargs["ekf_mode"] == MODE_IDLE
        assert kwargs["ekf_pf"] == 0.0

    @pytest.mark.asyncio
    async def test_learn_only_mode_uses_valve_opening(self, hass, mock_config_entry):
        room = _room([_trv("climate.trv_a", "sensor.valve_a")])
        extra = {"climate.trv_a": _heating_trv_state(), "sensor.valve_a": ("55", {})}
        kwargs = await _train_kwargs(
            hass,
            mock_config_entry,
            room,
            extra,
            settings={"climate_control_active": False, "outdoor_temp_sensor": "sensor.outdoor_temp"},
        )

        assert kwargs["ekf_mode"] == MODE_HEATING
        assert kwargs["ekf_pf"] == pytest.approx(0.55)

    @pytest.mark.asyncio
    async def test_valve_value_clamped_to_full_open(self, hass, mock_config_entry):
        room = _room([_trv("climate.trv_a", "sensor.valve_a")])
        extra = {"climate.trv_a": _heating_trv_state(), "sensor.valve_a": ("120", {})}
        kwargs = await _train_kwargs(hass, mock_config_entry, room, extra)

        assert kwargs["ekf_pf"] == 1.0
