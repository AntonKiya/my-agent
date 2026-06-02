from pathlib import Path

import pytest

import agent_service.skills.loader as skills_loader
from agent_service.skills import load_builtin_skill_capabilities


def test_builtin_skills_include_vkusvill_shopping_as_deferred_capability() -> None:
    capabilities = load_builtin_skill_capabilities()

    assert len(capabilities) == 1
    capability = capabilities[0]
    assert capability.id == "vkusvill-shopping"
    assert capability.defer_loading is True
    assert "TRIGGER when: the user wants to buy food" in capability.get_description()
    instructions = capability.get_instructions()
    assert len(instructions) == 1
    assert instructions[0].startswith("# VkusVill Shopping Skill")
    assert "---" not in instructions[0]
    assert "`mcp_vkusvill_vkusvill_products_search`" in instructions[0]
    assert "`mcp_vkusvill_vkusvill_cart_link_create`" in instructions[0]


def test_builtin_skills_can_be_filtered_by_enabled_skill_ids() -> None:
    assert load_builtin_skill_capabilities(enabled_skill_ids=set()) == ()

    capabilities = load_builtin_skill_capabilities(enabled_skill_ids={"vkusvill-shopping"})

    assert len(capabilities) == 1
    assert capabilities[0].id == "vkusvill-shopping"


def test_skill_loader_rejects_paths_outside_skills_directory() -> None:
    with pytest.raises(ValueError, match="must stay within"):
        skills_loader._resolve_skill_path(Path("../config.py"))


def test_skill_loader_rejects_absolute_paths() -> None:
    with pytest.raises(ValueError, match="must be relative"):
        skills_loader._resolve_skill_path(Path("/tmp/SKILL.md"))
