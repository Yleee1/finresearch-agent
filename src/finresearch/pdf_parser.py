from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .schemas import Document

_WHITESPACE = re.compile(r"[ \t\u3000]+")
_PAGE_NUMBER = re.compile(r"^(?:第\s*)?[—-]?\s*\d+\s*[—-]?(?:\s*页)?$")
_SENTENCE = re.compile(r"[^。！？；!?]+[。！？；!?]?")
_NUMBERISH = re.compile(r"^[\d,.$%+\-–—~～/：:（）()]+$")


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


def normalize_text(text: str) -> str:
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _looks_like_table(lines: list[str]) -> bool:
    if len(lines) < 6:
        return False
    short_ratio = sum(len(line) <= 12 for line in lines) / len(lines)
    number_ratio = sum(bool(_NUMBERISH.fullmatch(line)) for line in lines) / len(lines)
    return short_ratio >= 0.6 or number_ratio >= 0.2


def normalize_block_text(text: str) -> str:
    """Repair visual line wraps in prose while preserving table-like rows."""

    normalized = normalize_text(text)
    lines = normalized.splitlines()
    if len(lines) <= 1 or _looks_like_table(lines):
        return normalized

    output = lines[0]
    for line in lines[1:]:
        previous = output[-1:] if output else ""
        current = line[:1]
        separator = ""
        if previous and current and not (
            "\u3400" <= previous <= "\u9fff" and "\u3400" <= current <= "\u9fff"
        ):
            separator = " "
        output += separator + line
    return output


def extract_pdf_pages(path: str | Path) -> list[PageText]:
    """Extract text blocks in visual reading order while preserving page numbers."""

    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires PyMuPDF. Install with: pip install -e '.[ingest]'"
        ) from exc

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    pages: list[PageText] = []
    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text_blocks: list[str] = []
            for block in page.get_text("blocks", sort=True):
                block_type = block[6]
                if block_type != 0:
                    continue
                cleaned = normalize_block_text(block[4])
                if cleaned:
                    text_blocks.append(cleaned)
            pages.append(PageText(page_number=page_index + 1, text="\n".join(text_blocks)))
    return pages


def remove_repeated_marginal_lines(
    pages: list[PageText],
    *,
    margin_lines: int = 2,
    min_page_ratio: float = 0.15,
) -> list[PageText]:
    """Remove repeated headers/footers without deleting repeated body text."""

    if margin_lines <= 0:
        raise ValueError("margin_lines must be positive")
    if not 0 < min_page_ratio <= 1:
        raise ValueError("min_page_ratio must be between 0 and 1")

    candidates: Counter[str] = Counter()
    page_lines: list[list[str]] = []
    for page in pages:
        lines = [line for line in page.text.splitlines() if line.strip()]
        page_lines.append(lines)
        marginal = set(lines[:margin_lines] + lines[-margin_lines:])
        candidates.update(marginal)

    threshold = max(3, math.ceil(len(pages) * min_page_ratio))
    repeated = {
        line
        for line, frequency in candidates.items()
        if frequency >= threshold and len(line) <= 120
    }

    cleaned_pages: list[PageText] = []
    for page, lines in zip(pages, page_lines):
        last_start = max(0, len(lines) - margin_lines)
        kept: list[str] = []
        for index, line in enumerate(lines):
            is_margin = index < margin_lines or index >= last_start
            if is_margin and (line in repeated or _PAGE_NUMBER.fullmatch(line)):
                continue
            kept.append(line)
        cleaned_pages.append(PageText(page.page_number, "\n".join(kept)))
    return cleaned_pages


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap_chars
    return pieces


def _semantic_units(text: str, max_chars: int) -> list[str]:
    units: list[str] = []
    for paragraph in text.splitlines():
        for match in _SENTENCE.finditer(paragraph.strip()):
            sentence = match.group(0).strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                units.extend(_split_long_text(sentence, max_chars, 0))
    return units


def _split_semantic_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    units = _semantic_units(text, max_chars)
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for unit in units:
        separator_length = 1 if current else 0
        if current and current_length + separator_length + len(unit) > max_chars:
            chunks.append("\n".join(current))
            overlap: list[str] = []
            overlap_length = 0
            for previous in reversed(current):
                added = len(previous) + (1 if overlap else 0)
                if overlap_length + added > overlap_chars:
                    break
                overlap.insert(0, previous)
                overlap_length += added
            current = overlap
            current_length = overlap_length
            if current and current_length + 1 + len(unit) > max_chars:
                current = []
                current_length = 0
        if current:
            current_length += 1
        current.append(unit)
        current_length += len(unit)

    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_page_texts(
    pages: list[PageText],
    *,
    document_id: str,
    title: str,
    company: str,
    report_year: int,
    source: str,
    max_chars: int = 800,
    overlap_chars: int = 120,
    strategy: str = "fixed",
) -> list[Document]:
    """Build page-local chunks so every answer can cite an exact PDF page."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars - 1")
    if strategy not in {"fixed", "semantic"}:
        raise ValueError("strategy must be 'fixed' or 'semantic'")

    chunks: list[Document] = []
    for page in pages:
        if not page.text.strip():
            continue
        splitter = _split_semantic_text if strategy == "semantic" else _split_long_text
        for piece_index, piece in enumerate(splitter(page.text, max_chars, overlap_chars), start=1):
            chunk_id = f"{document_id}-p{page.page_number:04d}-c{piece_index:02d}"
            chunks.append(
                Document(
                    doc_id=chunk_id,
                    document_id=document_id,
                    title=title,
                    text=piece,
                    source=source,
                    company=company,
                    report_year=report_year,
                    page_start=page.page_number,
                    page_end=page.page_number,
                )
            )
    linked: list[Document] = []
    for index, chunk in enumerate(chunks):
        previous = chunks[index - 1] if index > 0 else None
        following = chunks[index + 1] if index + 1 < len(chunks) else None
        linked.append(
            Document(
                doc_id=chunk.doc_id,
                title=chunk.title,
                text=chunk.text,
                source=chunk.source,
                document_id=chunk.document_id,
                company=chunk.company,
                report_year=chunk.report_year,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                prev_chunk_id=(
                    previous.doc_id
                    if previous is not None and previous.document_id == chunk.document_id
                    else ""
                ),
                next_chunk_id=(
                    following.doc_id
                    if following is not None and following.document_id == chunk.document_id
                    else ""
                ),
            )
        )
    return linked


def parse_and_chunk_pdf(
    path: str | Path,
    *,
    clean_margins: bool = False,
    **chunk_options,
) -> list[Document]:
    pages = extract_pdf_pages(path)
    if clean_margins:
        pages = remove_repeated_marginal_lines(pages)
    return chunk_page_texts(pages, **chunk_options)
