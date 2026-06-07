from pathlib import Path

import pytest

from agent_service.instructions import (
    AgentInstructionLoadError,
    load_base_agent_instructions,
)


def write_instruction_files(base_dir: Path, contents: dict[str, str]) -> None:
    for filename, text in contents.items():
        (base_dir / filename).write_text(text, encoding="utf-8")


def test_load_base_agent_instructions_preserves_product_order(tmp_path: Path) -> None:
    write_instruction_files(
        tmp_path,
        {
            "identity.md": "identity",
            "output_style.md": "output style",
            "safety.md": "safety",
        },
    )

    instructions = load_base_agent_instructions(tmp_path)

    assert instructions == ["identity", "output style", "safety"]


def test_load_base_agent_instructions_rejects_missing_file(tmp_path: Path) -> None:
    write_instruction_files(
        tmp_path,
        {
            "identity.md": "identity",
            "output_style.md": "output style",
        },
    )

    with pytest.raises(AgentInstructionLoadError, match="safety.md"):
        load_base_agent_instructions(tmp_path)


def test_load_base_agent_instructions_rejects_empty_file(tmp_path: Path) -> None:
    write_instruction_files(
        tmp_path,
        {
            "identity.md": "identity",
            "output_style.md": "   \n",
            "safety.md": "safety",
        },
    )

    with pytest.raises(AgentInstructionLoadError, match="output_style.md"):
        load_base_agent_instructions(tmp_path)
