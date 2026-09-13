import unittest

from finresearch.pdf_parser import (
    PageText,
    chunk_page_texts,
    normalize_block_text,
    normalize_text,
    remove_repeated_marginal_lines,
)


class TextNormalizationTests(unittest.TestCase):
    def test_normalizes_whitespace_and_empty_lines(self):
        self.assertEqual(normalize_text("研发   投入\n\n  542 亿元 "), "研发 投入\n542 亿元")

    def test_repairs_chinese_visual_line_wrap(self):
        self.assertEqual(
            normalize_block_text("凭\n借工程师的努力，公司持续创新。"),
            "凭借工程师的努力，公司持续创新。",
        )

    def test_preserves_table_like_lines(self):
        table = "项目\n产能\n在建产能\n产能利用率\n676\n219\n76.33%\n516"
        self.assertEqual(normalize_block_text(table), table)


class PageChunkingTests(unittest.TestCase):
    def test_preserves_page_metadata(self):
        chunks = chunk_page_texts(
            [PageText(8, "研发投入为542亿元")],
            document_id="byd-2024",
            title="比亚迪2024年报",
            company="比亚迪",
            report_year=2024,
            source="report.pdf",
            max_chars=100,
            overlap_chars=10,
        )
        self.assertEqual(chunks[0].doc_id, "byd-2024-p0008-c01")
        self.assertEqual(chunks[0].page_start, 8)
        self.assertEqual(chunks[0].company, "比亚迪")

    def test_long_page_uses_overlap(self):
        chunks = chunk_page_texts(
            [PageText(1, "abcdefghij")],
            document_id="doc",
            title="title",
            company="company",
            report_year=2024,
            source="report.pdf",
            max_chars=6,
            overlap_chars=2,
        )
        self.assertEqual([chunk.text for chunk in chunks], ["abcdef", "efghij"])

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_page_texts(
                [],
                document_id="doc",
                title="title",
                company="company",
                report_year=2024,
                source="report.pdf",
                max_chars=100,
                overlap_chars=100,
            )

    def test_semantic_strategy_keeps_short_sentences_whole(self):
        chunks = chunk_page_texts(
            [PageText(1, "第一句话完整。第二句话也完整。第三句话仍然完整。")],
            document_id="doc",
            title="title",
            company="company",
            report_year=2024,
            source="report.pdf",
            max_chars=14,
            overlap_chars=0,
            strategy="semantic",
        )
        self.assertEqual(chunks[0].text, "第一句话完整。")
        self.assertFalse(chunks[1].text.startswith("句话"))


class MarginCleaningTests(unittest.TestCase):
    def test_removes_repeated_header_and_page_numbers(self):
        pages = [
            PageText(index, f"公司年度报告\n第{index}页正文\n{index}")
            for index in range(1, 5)
        ]
        cleaned = remove_repeated_marginal_lines(pages, min_page_ratio=0.5)
        self.assertEqual(cleaned[0].text, "第1页正文")
        self.assertEqual(cleaned[3].text, "第4页正文")


if __name__ == "__main__":
    unittest.main()
