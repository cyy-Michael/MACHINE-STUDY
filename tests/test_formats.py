# -*- coding: utf-8 -*-
"""格式读取层测试。

全部用临时文件构造，**不依赖仓库里的 `数据集/` 目录** —— 数据集本体不进
仓库，所以这些用例在任何一台机器上都应该能跑。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.data.formats import (
    DatasetFormatError,
    decode_bytes,
    read_arff,
    read_delimited,
    read_table,
)


class DelimitedTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, text: str, encoding: str = "utf-8") -> Path:
        path = self.dir / name
        path.write_bytes(text.encode(encoding))
        return path

    def test_headerless_file_keeps_every_row(self):
        """无表头文件绝不能吃掉首行。

        这里曾经真的丢过数据：缓存键把布尔 ``False`` 转成了字符串 `"False"`，
        而 ``bool("False")`` 是 ``True``，于是 iris/car/credit 各自少了一行。
        """
        path = self._write("a.data", "1,2,x\n3,4,y\n5,6,z\n")
        table = read_delimited(path, delimiter=",", has_header=False)
        self.assertEqual(len(table.rows), 3)
        self.assertEqual(table.rows[0], ["1", "2", "x"])
        self.assertEqual(table.header, ["column_0", "column_1", "column_2"])

    def test_headered_file_drops_only_the_header(self):
        path = self._write("a.csv", "a,b,c\n1,2,x\n3,4,y\n")
        table = read_delimited(path, delimiter=",", has_header=True)
        self.assertEqual(table.header, ["a", "b", "c"])
        self.assertEqual(len(table.rows), 2)

    def test_semicolon_delimiter(self):
        path = self._write("a.csv", "a;b\n1;2\n")
        table = read_delimited(path, delimiter=";", has_header=True)
        self.assertEqual(table.header, ["a", "b"])
        self.assertEqual(table.rows[0], ["1", "2"])

    def test_two_trailing_empty_columns_are_stripped(self):
        """AirQualityUCI 的表头以两个分号结尾，会多出两列恒空的特征。"""
        path = self._write("a.csv", "a;b;;\n1;2;;\n3;4;;\n")
        table = read_delimited(path, delimiter=";", has_header=True)
        self.assertEqual(table.header, ["a", "b"])
        self.assertTrue(all(len(row) == 2 for row in table.rows))

    def test_fully_blank_rows_are_dropped(self):
        """AirQualityUCI 尾部有 114 行只有分号的空行。"""
        path = self._write("a.csv", "a;b\n1;2\n;;\n;;\n3;4\n")
        table = read_delimited(path, delimiter=";", has_header=True)
        self.assertEqual(len(table.rows), 2)

    def test_extra_empty_field_is_removed_to_restore_width(self):
        """行宽错位且多余字段为空时，删掉它恢复列对齐。

        这是 chronic_kidney_disease.arff 的真实一行：25 列的位置出现了 26 个
        字段，多出的那个是空的。删掉之后所有值都落回合法取值域
        （dm=no, cad=no, appet=good, pe=no, class=notckd），所以能确定是多余的。
        """
        text = (
            "75,70,1.020,0,0,normal,normal,notpresent,notpresent,107,48,0.8,"
            "144,3.5,13.6,46,10300,4.8,no,,no,no,good,no,no,notckd\n"
        )
        path = self._write("ckd.data", text)
        table = read_delimited(
            path, delimiter=",", has_header=False, expected_columns=25
        )
        self.assertEqual(table.repaired, 1)
        self.assertEqual(len(table.rows[0]), 25)
        self.assertEqual(table.rows[0][-1], "notckd")
        self.assertEqual(table.rows[0][19], "no")

    def test_surplus_field_with_content_is_truncated_not_guessed(self):
        """多余字段有内容时不猜，直接截断并计数。"""
        path = self._write("a.data", "1,2,3\n4,5,6,EXTRA\n")
        table = read_delimited(
            path, delimiter=",", has_header=False, expected_columns=3
        )
        self.assertEqual(table.repaired, 1)
        self.assertEqual(table.rows[1], ["4", "5", "6"])

    def test_short_rows_are_padded(self):
        path = self._write("a.data", "1,2,3\n4,5\n")
        table = read_delimited(
            path, delimiter=",", has_header=False, expected_columns=3
        )
        self.assertEqual(table.repaired, 1)
        self.assertEqual(table.rows[1], ["4", "5", ""])

    def test_comment_prefix_is_skipped(self):
        path = self._write("a.csv", "%note\na,b\n1,2\n")
        table = read_delimited(path, delimiter=",", has_header=True, comment_prefix="%")
        self.assertEqual(table.header, ["a", "b"])

    def test_missing_file_raises(self):
        with self.assertRaises(DatasetFormatError):
            read_delimited(self.dir / "nope.csv")


class ArffTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, text: str) -> Path:
        path = self.dir / "a.arff"
        path.write_text(text, encoding="utf-8")
        return path

    def test_attribute_names_and_data(self):
        path = self._write(
            "@relation t\n"
            "@attribute 'age' numeric\n"
            "@attribute rbc {normal,abnormal}\n"
            "@attribute class {ckd,notckd}\n"
            "@data\n"
            "48,normal,ckd\n"
            "7,?,notckd\n"
        )
        table = read_arff(path)
        self.assertEqual(table.header, ["age", "rbc", "class"])
        self.assertEqual(len(table.rows), 2)
        self.assertEqual(table.rows[1], ["7", "?", "notckd"])

    def test_comments_and_blank_lines_are_ignored(self):
        path = self._write(
            "% a comment\n"
            "@relation t\n\n"
            "@attribute a numeric\n"
            "@attribute b numeric\n"
            "@data\n"
            "1,2\n"
            "\n"
            "3,4\n"
        )
        table = read_arff(path)
        self.assertEqual(len(table.rows), 2)

    def test_missing_attribute_declaration_raises(self):
        with self.assertRaises(DatasetFormatError):
            read_arff(self._write("@data\n1,2\n"))


class EncodingDetectionTest(unittest.TestCase):
    def test_gb18030_fallback(self):
        """数据集里混着 ANSI 编码的文件，utf-8 解不了时要能退回 gb18030。"""
        raw = "名称,数量\n苹果,3\n".encode("gb18030")
        text, encoding = decode_bytes(raw)
        self.assertEqual(encoding, "gb18030")
        self.assertIn("苹果", text)

    def test_utf8_bom_is_handled(self):
        raw = "a,b\n1,2\n".encode("utf-8-sig")
        text, encoding = decode_bytes(raw)
        self.assertEqual(encoding, "utf-8-sig")
        self.assertTrue(text.startswith("a,b"))


class ReadTableDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_unknown_format_raises(self):
        path = self.dir / "a.xyz"
        path.write_text("1,2\n", encoding="utf-8")
        with self.assertRaises(DatasetFormatError):
            read_table(path, format="xyz")

    def test_alias_formats(self):
        path = self.dir / "a.txt"
        path.write_text("1,2\n", encoding="utf-8")
        for alias in ("txt", "data", "delimited"):
            with self.subTest(alias=alias):
                table = read_table(path, format=alias)
                self.assertEqual(table.rows, [["1", "2"]])


if __name__ == "__main__":
    unittest.main()
