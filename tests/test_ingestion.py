import unittest

from finresearch.pdf_parser import (
    PageTable,
    PageText,
    StructuredPage,
    chunk_page_texts,
    chunk_structured_pages,
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


class StructuredTableChunkingTests(unittest.TestCase):
    def test_splits_tables_by_rows_and_repeats_header(self):
        table = PageTable(
            page_number=8,
            table_index=1,
            rows=(
                ("项目", "2024年", "2023年"),
                ("营业收入", "100", "80"),
                ("研发投入", "20", "15"),
                ("净利润", "10", "8"),
            ),
        )
        chunks = chunk_structured_pages(
            [StructuredPage(8, "正文说明。", (table,))],
            document_id="byd-2024",
            title="比亚迪2024年报",
            company="比亚迪",
            report_year=2024,
            source="report.pdf",
            max_chars=75,
            overlap_chars=10,
        )
        tables = [chunk for chunk in chunks if chunk.chunk_type == "table"]

        self.assertGreater(len(tables), 1)
        self.assertEqual({chunk.table_id for chunk in tables}, {"byd-2024-table-0001"})
        self.assertTrue(all("| 项目 | 2024年 | 2023年 |" in chunk.text for chunk in tables))
        self.assertEqual(tables[0].row_start, 1)
        self.assertEqual(tables[-1].row_end, 3)
        self.assertEqual(tables[0].next_chunk_id, tables[1].doc_id)
        self.assertEqual(tables[1].prev_chunk_id, tables[0].doc_id)

    def test_reuses_table_id_for_matching_header_on_next_page(self):
        pages = [
            StructuredPage(
                8,
                "",
                (PageTable(8, 1, (("项目", "金额"), ("营业收入", "100"))),),
            ),
            StructuredPage(
                9,
                "",
                (PageTable(9, 1, (("项目", "金额"), ("净利润", "10"))),),
            ),
        ]
        chunks = chunk_structured_pages(
            pages,
            document_id="doc",
            title="年报",
            company="公司",
            report_year=2024,
            source="report.pdf",
            max_chars=200,
            overlap_chars=10,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].table_id, chunks[1].table_id)
        self.assertEqual(chunks[0].next_chunk_id, chunks[1].doc_id)

    def test_continuation_page_inherits_previous_header(self):
        pages = [
            StructuredPage(
                8,
                "",
                (
                    PageTable(
                        8,
                        1,
                        (("项目", "金额"), ("营业收入", "100")),
                        (0, 700, 500, 820),
                        842,
                    ),
                ),
            ),
            StructuredPage(
                9,
                "",
                (
                    PageTable(
                        9,
                        1,
                        (("净利润", "10"), ("研发投入", "20")),
                        (0, 70, 500, 300),
                        842,
                    ),
                ),
            ),
        ]
        chunks = chunk_structured_pages(
            pages,
            document_id="doc",
            title="年报",
            company="公司",
            report_year=2024,
            source="report.pdf",
            max_chars=200,
            overlap_chars=10,
        )

        self.assertEqual({chunk.table_id for chunk in chunks}, {"doc-table-0001"})
        self.assertIn("| 项目 | 金额 |", chunks[1].text)
        self.assertIn("| 净利润 | 10 |", chunks[1].text)

    def test_only_one_table_can_continue_the_same_previous_table(self):
        pages = [
            StructuredPage(
                8,
                "",
                (
                    PageTable(
                        8,
                        1,
                        (("项目", "金额"), ("营业收入", "100")),
                        (0, 700, 500, 820),
                        842,
                    ),
                ),
            ),
            StructuredPage(
                9,
                "",
                (
                    PageTable(
                        9,
                        1,
                        (("净利润", "10"), ("研发投入", "20")),
                        (0, 40, 500, 160),
                        842,
                    ),
                    PageTable(
                        9,
                        2,
                        (("现金流", "30"), ("存货", "40")),
                        (0, 70, 500, 210),
                        842,
                    ),
                ),
            ),
        ]
        chunks = chunk_structured_pages(
            pages,
            document_id="doc",
            title="年报",
            company="公司",
            report_year=2024,
            source="report.pdf",
            max_chars=200,
            overlap_chars=10,
        )

        self.assertEqual(len({chunk.doc_id for chunk in chunks}), len(chunks))
        self.assertEqual(chunks[0].table_id, chunks[1].table_id)
        self.assertNotEqual(chunks[1].table_id, chunks[2].table_id)

    def test_matching_header_cannot_reuse_an_already_continued_table(self):
        full_header = ("", "项目", "", "2024年", "", "2023年")
        pages = [
            StructuredPage(
                8,
                "",
                (
                    PageTable(
                        8,
                        1,
                        (full_header, ("", "营业收入", "", "100", "", "80")),
                        (0, 700, 500, 820),
                        842,
                    ),
                ),
            ),
            StructuredPage(
                9,
                "",
                (
                    PageTable(
                        9,
                        1,
                        (("净利润", "10", "8"), ("研发投入", "20", "15")),
                        (0, 40, 500, 160),
                        842,
                    ),
                    PageTable(
                        9,
                        2,
                        (full_header, ("", "存货", "", "40", "", "30")),
                        (0, 200, 500, 400),
                        842,
                    ),
                ),
            ),
        ]
        chunks = chunk_structured_pages(
            pages,
            document_id="doc",
            title="年报",
            company="公司",
            report_year=2024,
            source="report.pdf",
            max_chars=300,
            overlap_chars=10,
        )

        self.assertEqual(len({chunk.doc_id for chunk in chunks}), len(chunks))
        self.assertEqual(chunks[0].table_id, chunks[1].table_id)
        self.assertNotEqual(chunks[1].table_id, chunks[2].table_id)


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
