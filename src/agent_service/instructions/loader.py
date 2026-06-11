from pathlib import Path

_INSTRUCTION_FILES = (
    "identity.md",
    "output_style.md",
    "time_context.md",
    "capabilities.md",
    "safety.md",
)
_INSTRUCTIONS_DIR = Path(__file__).resolve().parent


class AgentInstructionLoadError(RuntimeError):
    """Raised when base agent instructions cannot be loaded."""


def load_base_agent_instructions(
    instructions_dir: Path = _INSTRUCTIONS_DIR,
) -> list[str]:
    instructions = []
    for filename in _INSTRUCTION_FILES:
        path = instructions_dir / filename
        instructions.append(_read_instruction_file(path))
    return instructions


def _read_instruction_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AgentInstructionLoadError(f"Agent instruction file is missing: {path}") from exc

    text = text.strip()
    if not text:
        raise AgentInstructionLoadError(f"Agent instruction file is empty: {path}")
    return text
