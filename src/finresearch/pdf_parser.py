from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class PageTable:
    page_number: int
    table_index: int
    rows: tuple[tuple[str, ...], ...]
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    page_height: float = 0.0


@dataclass(frozen=True)
class StructuredPage:
    page_number: int
    text: str
    tables: tuple[PageTable, ...] = ()


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


def _normalize_cell(value: object) -> str:
    if value is None:
        return ""
    return normalize_text(str(value)).replace("\n", " ").strip()


def _normalize_table_rows(rows: list[list[object]]) -> tuple[tuple[str, ...], ...]:
    normalized = [[_normalize_cell(cell) for cell in row] for row in rows]
    normalized = [row for row in normalized if any(row)]
    if not normalized:
        return ()

    width = max(len(row) for row in normalized)
    rectangular = [row + [""] * (width - len(row)) for row in normalized]
    nonempty_columns = [
        index for index in range(width) if any(row[index] for row in rectangular)
    ]
    return tuple(
        tuple(row[index] for index in nonempty_columns) for row in rectangular
    )


def _is_usable_table(rows: tuple[tuple[str, ...], ...]) -> bool:
    """Reject layout false positives while keeping ordinary financial tables."""

    if len(rows) < 2 or len(rows[0]) < 2:
        return False
    nonempty_cells = [cell for row in rows for cell in row if cell]
    if len(nonempty_cells) < 4:
        return False
    # PyMuPDF may treat a large prose region as a one-cell table. Such regions
    # are safer in the prose chunker than in a row-oriented table chunk.
    return max(map(len, nonempty_cells), default=0) <= 600


def _block_belongs_to_table(
    block_bbox: tuple[float, float, float, float],
    table_bbox: tuple[float, float, float, float],
) -> bool:
    x0, y0, x1, y1 = block_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    return tx0 <= center_x <= tx1 and ty0 <= center_y <= ty1


def extract_structured_pdf_pages(path: str | Path) -> list[StructuredPage]:
    """Extract prose and usable tables separately without duplicating table text."""

    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires PyMuPDF. Install with: pip install -e '.[ingest]'"
        ) from exc

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    pages: list[StructuredPage] = []
    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            extracted_tables: list[PageTable] = []
            table_bboxes: list[tuple[float, float, float, float]] = []
            try:
                detected = page.find_tables().tables
            except (AttributeError, RuntimeError, ValueError):
                detected = []

            for table_index, table in enumerate(detected, start=1):
                rows = _normalize_table_rows(table.extract())
                if not _is_usable_table(rows):
                    continue
                bbox = tuple(float(value) for value in table.bbox)
                extracted_tables.append(
                    PageTable(
                        page_number,
                        table_index,
                        rows,
                        bbox,  # type: ignore[arg-type]
                        float(page.rect.height),
                    )
                )
                table_bboxes.append(bbox)  # type: ignore[arg-type]

            text_blocks: list[str] = []
            for block in page.get_text("blocks", sort=True):
                if block[6] != 0:
                    continue
                block_bbox = tuple(float(value) for value in block[:4])
                if any(
                    _block_belongs_to_table(block_bbox, bbox)  # type: ignore[arg-type]
                    for bbox in table_bboxes
                ):
                    continue
                cleaned = normalize_block_text(block[4])
                if cleaned:
                    text_blocks.append(cleaned)

            pages.append(
                StructuredPage(
                    page_number=page_number,
                    text="\n".join(text_blocks),
                    tables=tuple(extracted_tables),
                )
            )
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


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _render_table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    width = len(header)

    def render(row: tuple[str, ...]) -> str:
        padded = row + ("",) * (width - len(row))
        return "| " + " | ".join(_escape_markdown_cell(cell) for cell in padded[:width]) + " |"

    output = [render(header), "| " + " | ".join("---" for _ in header) + " |"]
    output.extend(render(row) for row in rows)
    return "\n".join(output)


def _pack_table_rows(
    header: tuple[str, ...],
    data_rows: tuple[tuple[str, ...], ...],
    max_chars: int,
) -> list[tuple[int, int, str]]:
    """Pack whole rows and repeat the header in every emitted table chunk."""

    if not data_rows:
        return [(0, 0, _render_table(header, []))]

    packed: list[tuple[int, int, str]] = []
    current: list[tuple[str, ...]] = []
    start_row = 1
    for row_number, row in enumerate(data_rows, start=1):
        candidate = current + [row]
        if current and len(_render_table(header, candidate)) > max_chars:
            packed.append((start_row, row_number - 1, _render_table(header, current)))
            current = [row]
            start_row = row_number
        else:
            current = candidate
    if current:
        packed.append((start_row, start_row + len(current) - 1, _render_table(header, current)))
    return packed


def _table_signature(header: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(re.sub(r"\s+", "", cell).lower() for cell in header)


def _link_table_chunks(chunks: list[Document]) -> list[Document]:
    grouped: dict[str, list[Document]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.table_id, []).append(chunk)

    linked: list[Document] = []
    for table_chunks in grouped.values():
        ordered = sorted(
            table_chunks,
            key=lambda item: (item.page_start or 0, item.row_start or 0, item.doc_id),
        )
        for index, chunk in enumerate(ordered):
            linked.append(
                replace(
                    chunk,
                    prev_chunk_id=ordered[index - 1].doc_id if index else "",
                    next_chunk_id=(
                        ordered[index + 1].doc_id if index + 1 < len(ordered) else ""
                    ),
                )
            )
    return linked


def chunk_structured_pages(
    pages: list[StructuredPage],
    *,
    document_id: str,
    title: str,
    company: str,
    report_year: int,
    source: str,
    max_chars: int = 800,
    overlap_chars: int = 120,
    strategy: str = "semantic",
) -> list[Document]:
    """Chunk prose semantically and tables by complete row groups."""

    prose_chunks = chunk_page_texts(
        [PageText(page.page_number, page.text) for page in pages],
        document_id=document_id,
        title=title,
        company=company,
        report_year=report_year,
        source=source,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        strategy=strategy,
    )

    table_chunks: list[Document] = []
    table_counter = 0
    recent_signatures: dict[tuple[str, ...], tuple[int, str]] = {}
    recent_layouts: list[
        tuple[
            int,
            str,
            tuple[str, ...],
            tuple[float, float, float, float],
            float,
        ]
    ] = []
    for page in pages:
        continued_table_ids: set[str] = set()
        for table in page.tables:
            first_row = table.rows[0]
            width = len(first_row)
            previous_layout = next(
                (
                    layout
                    for layout in reversed(recent_layouts)
                    if layout[0] == page.page_number - 1
                    and layout[1] not in continued_table_ids
                    and layout[4] > 0
                    and layout[3][3] >= layout[4] * 0.85
                    and abs(layout[3][0] - table.bbox[0]) <= 25
                    and abs(layout[3][2] - table.bbox[2]) <= 25
                ),
                None,
            )
            continues_from_previous_page = bool(
                previous_layout is not None
                and table.page_height > 0
                and table.bbox[1] <= table.page_height * 0.12
            )
            if continues_from_previous_page:
                assert previous_layout is not None
                table_id = previous_layout[1]
                continued_table_ids.add(table_id)
                header = previous_layout[2]
                compact_header = tuple(cell for cell in header if cell)
                if len(header) != width and len(compact_header) == width:
                    header = compact_header
                data_rows = table.rows[1:] if first_row == header else table.rows
            else:
                header = first_row
                data_rows = table.rows[1:]

            signature = _table_signature(header)
            if not continues_from_previous_page:
                previous = recent_signatures.get(signature)
                if (
                    any(signature)
                    and previous is not None
                    and previous[0] == page.page_number - 1
                    and previous[1] not in continued_table_ids
                ):
                    table_id = previous[1]
                else:
                    table_counter += 1
                    table_id = f"{document_id}-table-{table_counter:04d}"
            recent_signatures[signature] = (page.page_number, table_id)
            recent_layouts.append(
                (
                    page.page_number,
                    table_id,
                    header,
                    table.bbox,
                    table.page_height,
                )
            )

            for part_index, (row_start, row_end, text) in enumerate(
                _pack_table_rows(header, data_rows, max_chars),
                start=1,
            ):
                table_chunks.append(
                    Document(
                        doc_id=(
                            f"{table_id}-p{page.page_number:04d}-part{part_index:02d}"
                        ),
                        document_id=document_id,
                        title=title,
                        text=text,
                        source=source,
                        company=company,
                        report_year=report_year,
                        section_path="table",
                        page_start=page.page_number,
                        page_end=page.page_number,
                        chunk_type="table",
                        table_id=table_id,
                        table_header=header,
                        row_start=row_start,
                        row_end=row_end,
                    )
                )

    combined = prose_chunks + _link_table_chunks(table_chunks)
    return sorted(
        combined,
        key=lambda item: (
            item.page_start or 0,
            0 if item.chunk_type == "text" else 1,
            item.doc_id,
        ),
    )


def parse_and_chunk_pdf(
    path: str | Path,
    *,
    clean_margins: bool = False,
    structure_aware_tables: bool = False,
    **chunk_options,
) -> list[Document]:
    if structure_aware_tables:
        structured_pages = extract_structured_pdf_pages(path)
        if clean_margins:
            cleaned = remove_repeated_marginal_lines(
                [PageText(page.page_number, page.text) for page in structured_pages]
            )
            cleaned_by_page = {page.page_number: page.text for page in cleaned}
            structured_pages = [
                StructuredPage(
                    page_number=page.page_number,
                    text=cleaned_by_page[page.page_number],
                    tables=page.tables,
                )
                for page in structured_pages
            ]
        return chunk_structured_pages(structured_pages, **chunk_options)

    pages = extract_pdf_pages(path)
    if clean_margins:
        pages = remove_repeated_marginal_lines(pages)
    return chunk_page_texts(pages, **chunk_options)
