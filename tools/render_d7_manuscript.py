"""Render the editable Korean d7 paper and its frozen figures to an A4 PDF."""

import argparse
from html import escape
import os
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/thesis/PREDECODER_PAPER_KO.md'
FIGURE = re.compile(r'!\[(.*?)\]\((.*?)\)')
URL = re.compile(r'(?<!["=])https://[A-Za-z0-9./:_-]+')
WIDTHS = {
    '평가': (.31, .14, .32, .23),
    '거리': (.10, .19, .19, .36, .16),
    '비교': (.19, .15, .25, .15, .26),
    '잡음': (.09, .16, .16, .16, .43),
}


def fonts():
    directory = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'Fonts'
    pdfmetrics.registerFont(TTFont('Paper', str(directory / 'NotoSansKR-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('PaperBold', str(directory / 'NotoSansKR-Bold.ttf')))
    pdfmetrics.registerFontFamily('Paper', normal='Paper', bold='PaperBold',
                                  italic='Paper', boldItalic='PaperBold')


def styles():
    base = ParagraphStyle('Body', fontName='Paper', fontSize=10.1, leading=16.1,
                          spaceAfter=8, wordWrap='CJK', allowWidows=0, allowOrphans=0,
                          textColor=colors.HexColor('#182027'))
    return {
        'p': base,
        'title': ParagraphStyle('Title', parent=base, fontName='PaperBold',
                                fontSize=18.5, leading=25, spaceAfter=11, alignment=TA_CENTER,
                                textColor=colors.black),
        'author': ParagraphStyle('Author', parent=base, alignment=TA_CENTER, spaceAfter=14),
        'h2': ParagraphStyle('Section', parent=base, fontName='PaperBold', fontSize=13, leading=19,
                             spaceBefore=12, spaceAfter=5, keepWithNext=True, textColor=colors.black),
        'h3': ParagraphStyle('Subsection', parent=base, fontName='PaperBold', fontSize=11, leading=16,
                             spaceBefore=8, spaceAfter=4, keepWithNext=True, textColor=colors.black),
        'caption': ParagraphStyle('Caption', parent=base, fontSize=8.65, leading=12.8,
                                  spaceBefore=4, spaceAfter=10),
        'table_note': ParagraphStyle('TableNote', parent=base, fontSize=9, leading=13,
                                     spaceBefore=3, spaceAfter=5, keepWithNext=True),
        'cell': ParagraphStyle('Cell', parent=base, fontSize=8.8, leading=12.4,
                               spaceAfter=0, alignment=TA_RIGHT),
        'cell_left': ParagraphStyle('CellLeft', parent=base, fontSize=8.8, leading=12.4,
                                    spaceAfter=0, alignment=TA_LEFT),
        'reference': ParagraphStyle('Reference', parent=base, fontSize=8.5, leading=12.6,
                                    leftIndent=15, firstLineIndent=-15, spaceAfter=5),
    }


def format_text(text):
    text = escape(text, quote=False)
    text = text.replace('₀', '<sub>0</sub>').replace('₁', '<sub>1</sub>').replace('–', '-')
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\x60([^\x60]+)\x60', r'\1', text)
    return URL.sub(lambda match: f'<link href="{match[0]}" color="#205a78">{match[0]}</link>', text)


def table_for(lines, styles, width):
    rows = [[x.strip() for x in line.strip('|').split('|')] for line in lines]
    content = [rows[0], *rows[2:]]
    assert all(len(row) == len(content[0]) for row in content)
    weights = WIDTHS[rows[0][0]]
    assert len(weights) == len(rows[0])
    cells = [[Paragraph(format_text(f'**{v}**' if i == 0 else v),
                        styles['cell_left' if j == 0 or rows[0][0] == '평가' else 'cell'])
              for j, v in enumerate(row)] for i, row in enumerate(content)]
    table = Table(cells, colWidths=[x*width for x in weights], repeatRows=1, hAlign='LEFT',
                  spaceAfter=11)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf1f4')),
        ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#a9b6c0')),
        ('LINEBELOW', (0, -1), (-1, -1), .55, colors.HexColor('#a9b6c0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return table


def render(out):
    fonts()
    style = styles()
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    title = lines[0].removeprefix('# ')
    out.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(out), pagesize=A4, title=title, author='Jaemin Jeon',
                                 leftMargin=21*mm, rightMargin=21*mm, topMargin=19*mm, bottomMargin=19*mm)
    width = document.width-12
    content = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith('|'):
            rows = [line]
            while i < len(lines) and lines[i].startswith('|'):
                rows.append(lines[i]); i += 1
            content.append(table_for(rows, style, width))
        elif match := FIGURE.fullmatch(line):
            caption, rel = match.groups()
            diagram = Image(str((SOURCE.parent / rel).resolve()))
            ratio = min(width/diagram.imageWidth, 69*mm/diagram.imageHeight)
            diagram.drawWidth, diagram.drawHeight = ratio*diagram.imageWidth, ratio*diagram.imageHeight
            content.append(KeepTogether([diagram, Paragraph(format_text(caption), style['caption'])]))
        else:
            key = 'p'
            for prefix, name in [('### ', 'h3'), ('## ', 'h2'), ('# ', 'title')]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    key = name
                    break
            if line.startswith('**Jaemin Jeon**'):
                key = 'author'
            elif line.startswith('**표 '):
                key = 'table_note'
            elif re.match(r'^\[\d+\] ', line):
                key = 'reference'
            content.append(Paragraph(format_text(line), style[key]))

    def footer(canvas, doc):
        canvas.setFillColor(colors.HexColor('#4e5963'))
        canvas.setFont('Paper', 8)
        canvas.drawRightString(A4[0]-21*mm, 11*mm, str(doc.page))

    document.build(content, onFirstPage=footer, onLaterPages=footer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    render(parser.parse_args().out)
