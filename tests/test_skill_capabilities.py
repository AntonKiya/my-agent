from pathlib import Path
from typing import cast

import pytest

import agent_service.skills.loader as skills_loader
from agent_service.skills import load_builtin_skill_capabilities


def test_builtin_skills_include_vkusvill_shopping_as_deferred_capability() -> None:
    capabilities = load_builtin_skill_capabilities()

    capability_by_id = {capability.id: capability for capability in capabilities}
    assert set(capability_by_id) == {
        "reminders",
        "tutu-travel",
        "vkusvill-shopping",
        "weather-forecast",
    }
    capability = capability_by_id["vkusvill-shopping"]
    assert capability.id == "vkusvill-shopping"
    assert capability.defer_loading is True
    description = cast(str, capability.get_description())
    assert "TRIGGER when: the user wants to buy food" in description
    instructions = cast(list[str], capability.get_instructions())
    assert len(instructions) == 1
    assert instructions[0].startswith("# VkusVill Shopping Skill")
    assert "---" not in instructions[0]
    assert "`mcp_vkusvill_vkusvill_products_search`" in instructions[0]
    assert "`mcp_vkusvill_vkusvill_cart_link_create`" in instructions[0]


def test_builtin_skills_include_weather_forecast_as_deferred_capability() -> None:
    capabilities = load_builtin_skill_capabilities()

    capability_by_id = {capability.id: capability for capability in capabilities}
    capability = capability_by_id["weather-forecast"]
    assert capability.defer_loading is True
    description = cast(str, capability.get_description())
    assert "TRIGGER when: the user asks about current weather" in description
    assert "a weather follow-up asking for details by hour/time of day" in description
    instructions = cast(list[str], capability.get_instructions())
    assert len(instructions) == 1
    assert instructions[0].startswith("# Weather Forecast Skill")
    assert "`get_weather_forecast(location, period, location_language)`" in instructions[0]
    assert "hourly data, a time of day, or a time window" in instructions[0]
    assert "по часам" in instructions[0]
    assert "вечером" in instructions[0]
    assert "Do not use `web_research` as the first source" in instructions[0]
    assert (
        "If the user asks for a forecast and the period is not clear, use `week`."
        in (instructions[0])
    )


def test_builtin_skills_include_tutu_travel_as_deferred_capability() -> None:
    capabilities = load_builtin_skill_capabilities()

    capability_by_id = {capability.id: capability for capability in capabilities}
    capability = capability_by_id["tutu-travel"]
    assert capability.defer_loading is True
    description = cast(str, capability.get_description())
    assert "TRIGGER when: the user asks for Tutu tickets" in description
    assert "checkout links" in description
    assert "details of a previously shown Tutu option" in description
    assert "Use Tutu tools instead of web tools" in description
    instructions = cast(list[str], capability.get_instructions())
    assert len(instructions) == 1
    assert instructions[0].startswith("# Tutu Travel Skill")
    assert "---" not in instructions[0]
    assert "## Trigger Policy" not in instructions[0]
    assert "`mcp_tutu_search_hotels`" in instructions[0]
    assert "`mcp_tutu_get_offer_details`" in instructions[0]
    assert "`mcp_tutu_create_checkout_link`" in instructions[0]


def test_builtin_skills_include_reminders_as_deferred_capability() -> None:
    capabilities = load_builtin_skill_capabilities()

    capability_by_id = {capability.id: capability for capability in capabilities}
    capability = capability_by_id["reminders"]
    assert capability.defer_loading is True
    description = cast(str, capability.get_description())
    assert "TRIGGER when: the user asks to be reminded" in description
    instructions = cast(list[str], capability.get_instructions())
    assert len(instructions) == 1
    assert instructions[0].startswith("Use the reminder tools")
    assert "`create_once_reminder`" in instructions[0]
    assert "`create_weekly_reminder`" in instructions[0]
    assert "`create_interval_window_reminder`" in instructions[0]
    assert "`get_current_time`" in instructions[0]


def test_builtin_skills_can_be_filtered_by_enabled_skill_ids() -> None:
    assert load_builtin_skill_capabilities(enabled_skill_ids=set()) == ()

    capabilities = load_builtin_skill_capabilities(enabled_skill_ids={"weather-forecast"})

    assert len(capabilities) == 1
    assert capabilities[0].id == "weather-forecast"


def test_skill_loader_rejects_paths_outside_skills_directory() -> None:
    with pytest.raises(ValueError, match="must stay within"):
        skills_loader._resolve_skill_path(Path("../config.py"))


def test_skill_loader_rejects_absolute_paths() -> None:
    with pytest.raises(ValueError, match="must be relative"):
        skills_loader._resolve_skill_path(Path("/tmp/SKILL.md"))
