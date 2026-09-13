from __future__ import annotations

import argparse

from .io import write_documents
from .pdf_parser import parse_and_chunk_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse a PDF into citation-ready JSONL chunks")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--company", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap-chars", type=int, default=120)
    parser.add_argument("--strategy", choices=("fixed", "semantic"), default="fixed")
    parser.add_argument("--clean-margins", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    documents = parse_and_chunk_pdf(
        args.pdf,
        document_id=args.document_id,
        title=args.title,
        company=args.company,
        report_year=args.year,
        source=args.pdf,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        strategy=args.strategy,
        clean_margins=args.clean_margins,
    )
    write_documents(args.output, documents)
    print(f"wrote {len(documents)} chunks to {args.output}")


if __name__ == "__main__":
    main()
