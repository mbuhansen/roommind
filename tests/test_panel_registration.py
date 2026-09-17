"""Tests for sidebar panel registration and its cache-busting URL."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.roommind import _async_register_panel, _panel_cache_key
from custom_components.roommind.const import DOMAIN, VERSION


def _prepare(hass) -> None:
    hass.data[DOMAIN] = {}
    hass.http = MagicMock()
    hass.http.async_register_static_paths = AsyncMock()


def _registered_config(register_mock) -> dict:
    return register_mock.call_args.kwargs["config"]["_panel_custom"]


class TestPanelCacheKey:
    def test_digest_changes_with_file_contents(self, tmp_path):
        bundle = tmp_path / "roommind-panel.js"
        bundle.write_bytes(b"console.log('v1');")
        first = _panel_cache_key(bundle)

        bundle.write_bytes(b"console.log('v2');")
        second = _panel_cache_key(bundle)

        assert first != second
        assert len(first) == 8

    def test_same_contents_give_a_stable_digest(self, tmp_path):
        bundle = tmp_path / "roommind-panel.js"
        bundle.write_bytes(b"console.log('v1');")

        assert _panel_cache_key(bundle) == _panel_cache_key(bundle)

    def test_unreadable_file_falls_back_to_version(self, tmp_path):
        assert _panel_cache_key(tmp_path / "missing.js") == VERSION


class TestRegisterPanel:
    @pytest.mark.asyncio
    async def test_js_url_carries_the_bundle_digest(self, hass):
        _prepare(hass)
        with (
            patch("custom_components.roommind.async_register_built_in_panel") as register,
            patch("custom_components.roommind._panel_cache_key", return_value="abc12345"),
            patch.object(Path, "exists", return_value=True),
        ):
            await _async_register_panel(hass)

        assert _registered_config(register)["js_url"] == "/roommind/roommind-panel.js?v=abc12345"
        assert hass.data[DOMAIN]["panel_registered"] is True

    @pytest.mark.asyncio
    async def test_missing_bundle_skips_registration(self, hass):
        _prepare(hass)
        with (
            patch("custom_components.roommind.async_register_built_in_panel") as register,
            patch.object(Path, "exists", return_value=False),
        ):
            await _async_register_panel(hass)

        register.assert_not_called()
        assert DOMAIN in hass.data and not hass.data[DOMAIN].get("panel_registered")

    @pytest.mark.asyncio
    async def test_already_registered_is_a_no_op(self, hass):
        _prepare(hass)
        hass.data[DOMAIN]["panel_registered"] = True
        with patch("custom_components.roommind.async_register_built_in_panel") as register:
            await _async_register_panel(hass)

        register.assert_not_called()

    @pytest.mark.asyncio
    async def test_static_path_stays_unversioned(self, hass):
        """The query is only on the panel URL — the served path must not change."""
        _prepare(hass)
        with (
            patch("custom_components.roommind.async_register_built_in_panel"),
            patch("custom_components.roommind._panel_cache_key", return_value="abc12345"),
            patch.object(Path, "exists", return_value=True),
        ):
            await _async_register_panel(hass)

        static_config = hass.http.async_register_static_paths.call_args[0][0][0]
        assert static_config.url_path == "/roommind/roommind-panel.js"
