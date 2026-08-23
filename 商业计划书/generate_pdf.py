"""
将商业计划书 Markdown 转为 PDF
"""
import os
import sys
from pathlib import Path

import markdown
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether, Image
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


# ---------------------------------------------------------------------- #
# 读取 Markdown
# ---------------------------------------------------------------------- #
md_path = Path(__file__).parent / "智径-AdaptivePath-商业计划书.md"
md_text = md_path.read_text(encoding="utf-8")
html_text = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
# 把每行解析成段落


# ---------------------------------------------------------------------- #
# 配置
# ---------------------------------------------------------------------- #
PRIMARY = HexColor("#0075FF")
SECONDARY = HexColor("#10B981")
ACCENT = HexColor("#F59E0B")
DARK = HexColor("#1F2937")
GRAY = HexColor("#6B7280")
RED = HexColor("#EF4444")


# ---------------------------------------------------------------------- #
# 样式
# ---------------------------------------------------------------------- #
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "CustomTitle",
    parent=styles["Title"],
    fontSize=28,
    textColor=DARK,
    spaceAfter=20,
    alignment=TA_CENTER,
    fontName="Helvetica-Bold",
)
h1_style = ParagraphStyle(
    "CustomH1",
    parent=styles["Heading1"],
    fontSize=22,
    textColor=PRIMARY,
    spaceBefore=20,
    spaceAfter=12,
    fontName="Helvetica-Bold",
)
h2_style = ParagraphStyle(
    "CustomH2",
    parent=styles["Heading2"],
    fontSize=18,
    textColor=DARK,
    spaceBefore=15,
    spaceAfter=10,
    fontName="Helvetica-Bold",
)
h3_style = ParagraphStyle(
    "CustomH3",
    parent=styles["Heading3"],
    fontSize=14,
    textColor=PRIMARY,
    spaceBefore=10,
    spaceAfter=8,
    fontName="Helvetica-Bold",
)
body_style = ParagraphStyle(
    "CustomBody",
    parent=styles["Normal"],
    fontSize=11,
    textColor=DARK,
    leading=18,
    spaceAfter=6,
    fontName="Helvetica",
    alignment=TA_JUSTIFY,
)
bullet_style = ParagraphStyle(
    "CustomBullet",
    parent=body_style,
    leftIndent=20,
    bulletIndent=8,
)
quote_style = ParagraphStyle(
    "CustomQuote",
    parent=body_style,
    leftIndent=30,
    textColor=GRAY,
    fontName="Helvetica-Oblique",
)
code_style = ParagraphStyle(
    "CustomCode",
    parent=body_style,
    fontName="Courier",
    fontSize=9,
    textColor=DARK,
    backColor=HexColor("#F3F4F6"),
    leftIndent=15,
    spaceBefore=6,
    spaceAfter=6,
)


# ---------------------------------------------------------------------- #
# 解析 markdown 为 reportlab 元素
# ---------------------------------------------------------------------- #
import re

def parse_inline(text):
    """处理行内的 **bold**、*italic*、`code` 等"""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font name="Courier" color="#10B981">\1</font>', text)
    return text


def md_to_pdf_elements(md_text):
    elements = []
    lines = md_text.split("\n")
    i = 0
    in_table = False
    table_rows = []
    in_code = False
    code_lines = []
    in_list = False
    list_items = []

    def flush_list():
        nonlocal list_items, in_list
        if list_items:
            for item in list_items:
                elements.append(Paragraph(item, bullet_style))
            list_items.clear()
            in_list = False

    def flush_table():
        nonlocal table_rows, in_table
        if table_rows:
            t = Table(table_rows, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F9FAFB")]),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 0.3*cm))
            table_rows = []
            in_table = False

    while i < len(lines):
        line = lines[i].rstrip()

        # 代码块
        if line.startswith("```"):
            if in_code:
                elements.append(Paragraph("\n".join(code_lines).replace(" ", "&nbsp;").replace("\n", "<br/>"), code_style))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # 表格
        if "|" in line and i + 1 < len(lines) and "---" in lines[i+1]:
            in_table = True
            table_rows.append([parse_inline(c.strip()) for c in line.split("|") if c.strip()])
            i += 2
            continue
        if in_table and "|" in line:
            table_rows.append([parse_inline(c.strip()) for c in line.split("|") if c.strip()])
            i += 1
            continue
        if in_table:
            flush_table()

        # 标题
        if line.startswith("# "):
            flush_list()
            elements.append(Spacer(1, 0.5*cm))
            elements.append(Paragraph(parse_inline(line[2:]), title_style))
            elements.append(Spacer(1, 0.3*cm))
        elif line.startswith("## "):
            flush_list()
            elements.append(Paragraph(parse_inline(line[3:]), h1_style))
        elif line.startswith("### "):
            flush_list()
            elements.append(Paragraph(parse_inline(line[4:]), h2_style))
        elif line.startswith("#### "):
            flush_list()
            elements.append(Paragraph(parse_inline(line[5:]), h3_style))
        # 列表
        elif line.startswith("- ") or line.startswith("* "):
            in_list = True
            list_items.append("• " + parse_inline(line[2:]))
        # 分隔线
        elif line == "---":
            flush_list()
            elements.append(Spacer(1, 0.3*cm))
        # 空行
        elif not line:
            flush_list()
        # 普通段落
        else:
            flush_list()
            elements.append(Paragraph(parse_inline(line), body_style))
        i += 1

    flush_list()
    flush_table()
    return elements


# ---------------------------------------------------------------------- #
# 生成 PDF
# ---------------------------------------------------------------------- #
output_path = Path(__file__).parent / "智径-AdaptivePath-商业计划书.pdf"
doc = SimpleDocTemplate(
    str(output_path),
    pagesize=A4,
    leftMargin=2*cm,
    rightMargin=2*cm,
    topMargin=2*cm,
    bottomMargin=2*cm,
    title="智径 AdaptivePath 商业计划书",
    author="智径团队",
)

elements = md_to_pdf_elements(md_text)
doc.build(elements)

print(f"PDF saved: {output_path}")
print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")
