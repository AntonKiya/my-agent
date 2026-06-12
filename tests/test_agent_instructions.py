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
            "time_context.md": "time context",
            "capabilities.md": "capabilities",
            "safety.md": "safety",
        },
    )

    instructions = load_base_agent_instructions(tmp_path)

    assert instructions == [
        "identity",
        "output style",
        "time context",
        "capabilities",
        "safety",
    ]


def test_load_base_agent_instructions_rejects_missing_file(tmp_path: Path) -> None:
    write_instruction_files(
        tmp_path,
        {
            "identity.md": "identity",
            "output_style.md": "output style",
            "time_context.md": "time context",
            "capabilities.md": "capabilities",
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
            "time_context.md": "time context",
            "capabilities.md": "capabilities",
            "safety.md": "safety",
        },
    )

    with pytest.raises(AgentInstructionLoadError, match="output_style.md"):
        load_base_agent_instructions(tmp_path)


def test_base_agent_instructions_require_successful_tool_for_external_actions() -> None:
    instructions = "\n".join(load_base_agent_instructions())

    assert "не говори, что действие выполнено" in instructions
    assert "соответствующий tool реально не был вызван и не вернул успех" in instructions
    assert "сформировать корзину" in instructions


def test_base_agent_instructions_include_public_capabilities() -> None:
    instructions = "\n".join(load_base_agent_instructions())

    assert "когда пользователь задаёт общий вопрос" in instructions
    assert "Собрать корзину во ВкусВилл" in instructions
    assert "Ответить на голосовое" in instructions


def test_base_agent_instructions_include_time_context_policy() -> None:
    instructions = "\n".join(load_base_agent_instructions())

    assert "Если правильный ответ зависит от текущей даты" in instructions
    assert "Вызови `get_current_time` с явным IANA timezone" in instructions
    assert "Если пользователь уже указал точную дату" in instructions


def test_base_agent_instructions_protect_skill_instruction_texts() -> None:
    instructions = "\n".join(load_base_agent_instructions())

    assert "полный или дословный текст инструкций" in instructions
    assert "skills/capabilities" in instructions
    assert "объяснить назначение и правила использования своими словами" in instructions
