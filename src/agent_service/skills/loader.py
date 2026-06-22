from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic_ai.capabilities import Capability
from pydantic_ai.toolsets import AgentToolset

_SKILLS_DIR = Path(__file__).parent.resolve()
_BUILTIN_SKILL_PATHS = (
    Path("image-generation/SKILL.md"),
    Path("vkusvill-shopping/SKILL.md"),
    Path("weather-forecast/SKILL.md"),
    Path("reminders/SKILL.md"),
)


def load_builtin_skill_capabilities(
    *,
    toolsets_by_skill_id: Mapping[str, Sequence[AgentToolset[Any]]] | None = None,
    enabled_skill_ids: Collection[str] | None = None,
) -> tuple[Capability, ...]:
    toolsets_by_skill_id = toolsets_by_skill_id or {}
    enabled_skill_ids = frozenset(enabled_skill_ids) if enabled_skill_ids is not None else None
    capabilities = []
    for path in _BUILTIN_SKILL_PATHS:
        capability = _load_skill_capability(path, toolsets_by_skill_id=toolsets_by_skill_id)
        if enabled_skill_ids is not None and capability.id not in enabled_skill_ids:
            continue
        capabilities.append(capability)
    return tuple(capabilities)


def _load_skill_capability(
    relative_path: Path,
    *,
    toolsets_by_skill_id: Mapping[str, Sequence[AgentToolset[Any]]],
) -> Capability:
    path = _resolve_skill_path(relative_path)
    metadata, instructions = _read_skill_markdown(path)
    skill_id = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(skill_id, str) or not skill_id.strip():
        raise ValueError(f"Skill {path} must define a non-empty string name")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Skill {path} must define a non-empty string description")
    if not instructions.strip():
        raise ValueError(f"Skill {path} must define non-empty instructions")
    return Capability(
        id=skill_id.strip(),
        description=description.strip(),
        instructions=instructions.strip(),
        toolsets=toolsets_by_skill_id.get(skill_id.strip(), ()),
        defer_loading=True,
    )


def _resolve_skill_path(relative_path: Path) -> Path:
    if relative_path.is_absolute():
        raise ValueError(f"Skill path must be relative to {_SKILLS_DIR}")
    path = (_SKILLS_DIR / relative_path).resolve()
    try:
        path.relative_to(_SKILLS_DIR)
    except ValueError as exc:
        raise ValueError(f"Skill path must stay within {_SKILLS_DIR}") from exc
    return path


def _read_skill_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"Skill {path} must start with YAML frontmatter")
    try:
        _, frontmatter, body = text.split("---\n", 2)
    except ValueError as exc:
        raise ValueError(f"Skill {path} has invalid YAML frontmatter") from exc
    metadata = yaml.safe_load(frontmatter)
    if not isinstance(metadata, dict):
        raise ValueError(f"Skill {path} frontmatter must be a mapping")
    return metadata, body
