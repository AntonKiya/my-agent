import html
import re

TELEGRAM_HTML_PARSE_MODE = "HTML"

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def markdown_to_telegram_html(text: str) -> str:
    """Render a conservative Markdown subset as Telegram-safe HTML."""
    rendered_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        line_body = line[:-1] if line.endswith("\n") else line
        line_break = "\n" if line.endswith("\n") else ""
        heading_match = _MARKDOWN_HEADING_RE.match(line_body)
        if heading_match is not None:
            rendered_lines.append(
                f"<b>{_render_inline_markdown(heading_match.group(2))}</b>{line_break}"
            )
            continue
        rendered_lines.append(f"{_render_inline_markdown(line_body)}{line_break}")
    return "".join(rendered_lines)


def _render_inline_markdown(text: str) -> str:
    rendered: list[str] = []
    cursor = 0
    while cursor < len(text):
        if text.startswith("**", cursor):
            end = text.find("**", cursor + 2)
            if end != -1:
                rendered.append(f"<b>{html.escape(text[cursor + 2 : end], quote=False)}</b>")
                cursor = end + 2
                continue

        rendered.append(html.escape(text[cursor], quote=False))
        cursor += 1
    return "".join(rendered)
