# -*- coding: utf-8 -*-
"""原始文件读取层：脏 CSV / ARFF / Excel。

本层只做一件事——把文件变成「表头 + 字符串矩阵」。类型推断、缺失值填充、
类别编码全部留给 encoding 层，这样每种格式的坑都集中在这一个文件里。

已知的真实数据坑（都不是假设，是本仓库数据集实测出来的）：

* ``AirQualityUCI.csv`` 末尾有两个恒为空的分号列，且文件尾部有 114 行
  只有分号的空行 —— 不处理会多出 2 个全缺失特征、多出 114 条空样本。
* ``Concrete_Data.xls`` 是旧版 BIFF 二进制 Excel，需要 ``xlrd``（openpyxl 读不了）。
* ``iris.data`` / ``car.data`` / ``crx.data`` 没有表头，首行是数据不是列名。
* ARFF 的类别声明是 ``{a,b,c}``，且缺失值统一写作 ``?``。
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import NamedTuple

# 数据集混杂了 ASCII / UTF-8 / ANSI 编码，按可靠性从高到低尝试。
# latin-1 永不解码失败，必须放在最后兜底。
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "latin-1")

SUPPORTED_FORMATS = ("csv", "arff", "excel")


class DatasetFormatError(RuntimeError):
    """文件缺失或格式无法解析。"""


class Table(NamedTuple):
    """统一中间表示：列名 + 字符串单元格。"""

    header: list[str]
    rows: list[list[str]]
    encoding: str
    format: str
    repaired: int = 0


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------

def decode_bytes(raw: bytes, encodings: tuple[str, ...] = ENCODINGS) -> tuple[str, str]:
    """按候选编码解码，返回 (文本, 实际使用的编码)。"""
    for enc in encodings:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise DatasetFormatError("无法解码文件：utf-8 / gb18030 / latin-1 全部失败")


def _drop_blank_rows(rows: list[list[str]]) -> list[list[str]]:
    """丢掉所有单元格都为空白的行（AirQualityUCI 尾部 114 行）。"""
    return [row for row in rows if any(cell.strip() for cell in row)]


def _strip_trailing_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    """裁掉恒为空白的尾列（AirQualityUCI 尾部两个分号列）。

    只裁「所有行都为空」的列，避免误伤某列恰好最后几行为空的情况。
    """
    if not rows:
        return rows
    width = max(len(row) for row in rows)
    while width > 1:
        index = width - 1
        if all((row[index].strip() if index < len(row) else "") == "" for row in rows):
            width -= 1
        else:
            break
    return [row[:width] for row in rows]


def _normalise(rows: list[list[str]]) -> list[list[str]]:
    return _strip_trailing_empty_columns(_drop_blank_rows(rows))


def _align_rows(
    rows: list[list[str]],
    width: int,
    *,
    skip_first: bool = False,
) -> tuple[list[list[str]], int]:
    """把每行对齐到 ``width`` 列，返回 ``(对齐后的行, 被修复的行数)``。

    真实数据里存在「多一个逗号」的行。``chronic_kidney_disease.arff`` 有 3 行
    宽度是 26 而不是 25：多出的那个字段是空的，删掉之后整行才成立
    （``htn=no, dm=no, cad=no, appet=good, pe=no, ane=no, class=notckd``）。
    所以多余的**空**单元格会被删掉，而不是把整行丢掉。

    这个修复有边界：只有当多余字段确实是空的时候才删。要是多余字段有内容，
    说明列错位的原因未知，此时截断并在 ``repaired`` 里计数，绝不静默猜。
    """
    head = rows[:1] if skip_first else []
    body = rows[1:] if skip_first else rows

    aligned: list[list[str]] = []
    repaired = 0
    for row in body:
        if len(row) == width:
            aligned.append(row)
            continue
        repaired += 1
        if len(row) > width:
            surplus = len(row) - width
            kept: list[str] = []
            removed = 0
            for cell in row:
                if removed < surplus and not cell.strip() and len(row) - removed > width:
                    removed += 1
                    continue
                kept.append(cell)
            aligned.append(kept[:width])
        else:
            aligned.append(row + [""] * (width - len(row)))

    return head + aligned, repaired


def _resolve_width(
    rows: list[list[str]],
    *,
    has_header: bool,
    expected_columns: int | None,
) -> int:
    """决定基准列数：元数据优先，其次表头长度，最后取多数行的宽度。

    无表头文件不能用 ``max()``——一行多一个逗号就会把基准拉高，导致
    所有正常行都被当成「少一列」而补空。
    """
    if expected_columns:
        return expected_columns
    if has_header and rows:
        return len(rows[0])
    counts: dict[int, int] = {}
    for row in rows:
        counts[len(row)] = counts.get(len(row), 0) + 1
    return max(counts, key=lambda width: (counts[width], -width))


# --------------------------------------------------------------------------
# 分隔符文本（CSV / TXT / DATA）
# --------------------------------------------------------------------------

def read_delimited(
    path: str | Path,
    *,
    delimiter: str = ",",
    has_header: bool = False,
    comment_prefix: str | None = None,
    expected_columns: int | None = None,
) -> Table:
    path = Path(path)
    if not path.exists():
        raise DatasetFormatError(f"数据集文件不存在：{path}")
    text, encoding = decode_bytes(path.read_bytes())

    if comment_prefix:
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith(comment_prefix)
        )

    rows = [row for row in csv.reader(io.StringIO(text), delimiter=delimiter)]
    rows = _normalise(rows)
    if not rows:
        raise DatasetFormatError(f"文件没有任何数据行：{path}")

    width = _resolve_width(rows, has_header=has_header, expected_columns=expected_columns)
    rows, repaired = _align_rows(rows, width, skip_first=has_header)

    if has_header:
        header = [cell.strip() or f"column_{i}" for i, cell in enumerate(rows[0])]
        data = rows[1:]
    else:
        header = [f"column_{i}" for i in range(width)]
        data = rows

    return Table(
        header=header, rows=data, encoding=encoding, format="csv", repaired=repaired
    )


# --------------------------------------------------------------------------
# ARFF
# --------------------------------------------------------------------------

def _parse_attribute(line: str) -> tuple[str, str]:
    """解析 ``@attribute 'name' {a,b}`` / ``@attribute name numeric``。"""
    rest = line[len("@attribute"):].strip()
    if rest[:1] in ("'", '"'):
        quote = rest[0]
        end = rest.find(quote, 1)
        if end == -1:
            name, rest = rest[1:], ""
        else:
            name, rest = rest[1:end], rest[end + 1:].strip()
    else:
        parts = rest.split(None, 1)
        name = parts[0] if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
    return name, rest


def read_arff(path: str | Path, *, expected_columns: int | None = None) -> Table:
    path = Path(path)
    if not path.exists():
        raise DatasetFormatError(f"数据集文件不存在：{path}")
    text, encoding = decode_bytes(path.read_bytes())

    names: list[str] = []
    declared: list[str] = []
    data_rows: list[list[str]] = []
    in_data = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        lowered = stripped.lower()
        if lowered.startswith("@attribute"):
            name, spec = _parse_attribute(stripped)
            names.append(name)
            declared.append(spec)
        elif lowered.startswith("@data"):
            in_data = True
        elif lowered.startswith("@"):
            continue
        elif in_data:
            data_rows.append(next(csv.reader([stripped])))

    data = _normalise(data_rows)
    if not names:
        raise DatasetFormatError(f"ARFF 缺少 @attribute 声明：{path}")
    if not data:
        raise DatasetFormatError(f"ARFF 的 @data 段为空：{path}")

    # @attribute 的条数就是基准列数；声明数不足时以数据为准补齐列名
    width = expected_columns or len(names)
    if len(names) < width:
        names.extend(f"column_{i}" for i in range(len(names), width))
    names = names[:width]

    data, repaired = _align_rows(data, width)

    return Table(
        header=names, rows=data, encoding=encoding, format="arff", repaired=repaired
    )


# --------------------------------------------------------------------------
# Excel
# --------------------------------------------------------------------------

def _cell_to_text(value: object) -> str:
    """把 Excel 单元格转成字符串，且不让整数变成 ``1.0``。

    xlrd 把每个数字都读成 float，不这样处理的话 ``Concrete_Data.xls``
    的年龄列会变成 ``365.0`` 这类带小数点的值，污染类型推断。
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def read_excel(path: str | Path, *, has_header: bool = True, sheet: str | int | None = None) -> Table:
    path = Path(path)
    if not path.exists():
        raise DatasetFormatError(f"数据集文件不存在：{path}")
    suffix = path.suffix.lower()

    if suffix == ".xls":
        try:
            import xlrd  # noqa: PLC0415
        except ImportError as exc:
            raise DatasetFormatError(
                "读取 .xls 需要 xlrd：pip install xlrd（或把文件另存为 .xlsx）"
            ) from exc
        book = xlrd.open_workbook(str(path))
        sheet_obj = book.sheet_by_index(0) if sheet is None else book.sheet_by_name(str(sheet))
        raw = [
            [_cell_to_text(sheet_obj.cell_value(r, c)) for c in range(sheet_obj.ncols)]
            for r in range(sheet_obj.nrows)
        ]
    else:
        try:
            import openpyxl  # noqa: PLC0415
        except ImportError as exc:
            raise DatasetFormatError("读取 .xlsx 需要 openpyxl：pip install openpyxl") from exc
        workbook = openpyxl.load_workbook(str(path), data_only=True)
        worksheet = workbook[str(sheet)] if sheet is not None else workbook.active
        raw = [[_cell_to_text(value) for value in row] for row in worksheet.iter_rows(values_only=True)]

    rows = _normalise(raw)
    if not rows:
        raise DatasetFormatError(f"工作表没有任何数据：{path}")

    if has_header:
        header = [cell.strip() or f"column_{i}" for i, cell in enumerate(rows[0])]
        data = rows[1:]
    else:
        width = max(len(row) for row in rows)
        header = [f"column_{i}" for i in range(width)]
        data = rows

    # Excel 是二进制容器，没有「文件编码」这个概念
    return Table(header=header, rows=data, encoding="binary", format="excel")


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------

def read_table(
    path: str | Path,
    *,
    format: str,
    delimiter: str = ",",
    has_header: bool = False,
    comment_prefix: str | None = None,
    sheet: str | int | None = None,
    expected_columns: int | None = None,
) -> Table:
    """按格式分派到对应读取器。"""
    format = format.lower()
    if format in ("csv", "txt", "data", "delimited"):
        return read_delimited(
            path,
            delimiter=delimiter,
            has_header=has_header,
            comment_prefix=comment_prefix,
            expected_columns=expected_columns,
        )
    if format == "arff":
        return read_arff(path, expected_columns=expected_columns)
    if format in ("excel", "xls", "xlsx"):
        return read_excel(path, has_header=has_header, sheet=sheet)
    raise DatasetFormatError(f"不支持的格式：{format}（支持 {', '.join(SUPPORTED_FORMATS)}）")
