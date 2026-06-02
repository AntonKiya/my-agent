import html
import re
from urllib.parse import urlsplit

TELEGRAM_HTML_PARSE_MODE = "HTML"

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_MARKDOWN_CODE_FENCE_RE = re.compile(r"^```\w*\s*$")
_SUPPORTED_LINK_SCHEMES = frozenset({"http", "https"})


def markdown_to_telegram_html(text: str) -> str:
    """Render a conservative Markdown subset as Telegram-safe HTML."""
    rendered_lines: list[str] = []
    lines = text.splitlines(keepends=True)
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        line_body = line[:-1] if line.endswith("\n") else line
        line_break = "\n" if line.endswith("\n") else ""

        if _MARKDOWN_CODE_FENCE_RE.match(line_body):
            cursor, code = _render_code_block(lines, cursor)
            rendered_lines.append(code)
            continue

        if line_body.startswith(">"):
            cursor, quote = _render_blockquote(lines, cursor)
            rendered_lines.append(quote)
            continue

        heading_match = _MARKDOWN_HEADING_RE.match(line_body)
        if heading_match is not None:
            rendered_lines.append(
                f"<b>{_render_inline_markdown(heading_match.group(2))}</b>{line_break}"
            )
            cursor += 1
            continue
        rendered_lines.append(f"{_render_inline_markdown(line_body)}{line_break}")
        cursor += 1
    return "".join(rendered_lines)


def _render_inline_markdown(text: str) -> str:
    rendered: list[str] = []
    cursor = 0
    while cursor < len(text):
        if text.startswith("**", cursor):
            end = text.find("**", cursor + 2)
            if end > cursor + 2:
                rendered.append(f"<b>{_render_inline_markdown(text[cursor + 2 : end])}</b>")
                cursor = end + 2
                continue

        if text.startswith("~~", cursor):
            end = text.find("~~", cursor + 2)
            if end > cursor + 2:
                rendered.append(f"<s>{_render_inline_markdown(text[cursor + 2 : end])}</s>")
                cursor = end + 2
                continue

        if text.startswith("||", cursor):
            end = text.find("||", cursor + 2)
            if end > cursor + 2:
                rendered.append(
                    f"<tg-spoiler>{_render_inline_markdown(text[cursor + 2 : end])}</tg-spoiler>"
                )
                cursor = end + 2
                continue

        if text[cursor] == "`":
            end = text.find("`", cursor + 1)
            if end > cursor + 1:
                rendered.append(f"<code>{html.escape(text[cursor + 1 : end], quote=False)}</code>")
                cursor = end + 1
                continue

        if text[cursor] == "[":
            link = _render_link(text, cursor)
            if link is not None:
                rendered_text, cursor = link
                rendered.append(rendered_text)
                continue

        if text[cursor] in {"*", "_"} and _is_emphasis_start(text, cursor):
            delimiter = text[cursor]
            end = _find_emphasis_end(text, delimiter, cursor + 1)
            if end > cursor + 1:
                rendered.append(f"<i>{_render_inline_markdown(text[cursor + 1 : end])}</i>")
                cursor = end + 1
                continue

        rendered.append(html.escape(text[cursor], quote=False))
        cursor += 1
    return "".join(rendered)


def _render_code_block(lines: list[str], cursor: int) -> tuple[int, str]:
    opening_line = lines[cursor]
    code_lines: list[str] = []
    cursor += 1
    while cursor < len(lines):
        line = lines[cursor]
        line_body = line[:-1] if line.endswith("\n") else line
        if _MARKDOWN_CODE_FENCE_RE.match(line_body):
            line_break = "\n" if line.endswith("\n") else ""
            code = "".join(code_lines)
            if not code:
                return cursor + 1, html.escape(f"{opening_line}{line}", quote=False)
            return (
                cursor + 1,
                f"<pre>{html.escape(code, quote=False)}</pre>{line_break}",
            )
        code_lines.append(line)
        cursor += 1
    if not code_lines:
        return cursor, html.escape(opening_line, quote=False)
    return cursor, f"<pre>{html.escape(''.join(code_lines), quote=False)}</pre>"


def _render_blockquote(lines: list[str], cursor: int) -> tuple[int, str]:
    quote_lines: list[str] = []
    original_lines: list[str] = []
    trailing_line_break = ""
    while cursor < len(lines):
        line = lines[cursor]
        line_body = line[:-1] if line.endswith("\n") else line
        line_break = "\n" if line.endswith("\n") else ""
        if not line_body.startswith(">"):
            break
        original_lines.append(line)
        quote_line = line_body[1:]
        if quote_line.startswith(" "):
            quote_line = quote_line[1:]
        quote_lines.append(f"{_render_inline_markdown(quote_line)}{line_break}")
        trailing_line_break = line_break
        cursor += 1
    rendered_quote = "".join(quote_lines).rstrip(chr(10))
    if not rendered_quote:
        return cursor, html.escape("".join(original_lines), quote=False)
    return (
        cursor,
        f"<blockquote>{rendered_quote}</blockquote>{trailing_line_break}",
    )


def _render_link(text: str, cursor: int) -> tuple[str, int] | None:
    label_end = text.find("](", cursor + 1)
    if label_end == -1:
        return None
    url_end = text.find(")", label_end + 2)
    if url_end == -1:
        return None

    label = text[cursor + 1 : label_end]
    url = text[label_end + 2 : url_end]
    if not label or not _is_safe_link_url(url):
        return None

    rendered = (
        f'<a href="{html.escape(url, quote=True)}">'
        f"{_render_inline_markdown(label)}"
        "</a>"
    )
    return rendered, url_end + 1


def _is_safe_link_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return parsed.scheme in _SUPPORTED_LINK_SCHEMES and bool(parsed.netloc)


def _is_emphasis_start(text: str, cursor: int) -> bool:
    if cursor + 1 >= len(text) or text[cursor + 1].isspace():
        return False
    if cursor > 0 and text[cursor - 1].isalnum():
        return False
    return True


def _find_emphasis_end(text: str, delimiter: str, cursor: int) -> int:
    while cursor < len(text):
        end = text.find(delimiter, cursor)
        if end == -1:
            return -1
        if end > 0 and not text[end - 1].isspace():
            return end
        cursor = end + 1
    return -1
