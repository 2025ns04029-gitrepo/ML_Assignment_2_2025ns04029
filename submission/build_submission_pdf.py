"""Build the single assignment PDF using your own lab screenshot and URLs."""

import argparse
from pathlib import Path
import re
from urllib.parse import urlsplit
from xml.sax.saxutils import escape, quoteattr

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import CondPageBreak, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


HERE = Path(__file__).resolve().parent
NAVY = colors.HexColor('#102A43')
TEAL = colors.HexColor('#0F716B')
INK = colors.HexColor('#23384D')
PALE = colors.HexColor('#EDF5F4')
RULE = colors.HexColor('#D5E0E7')
pdfmetrics.registerFont(TTFont('Vera', 'Vera.ttf'))
pdfmetrics.registerFont(TTFont('Vera-Bold', 'VeraBd.ttf'))
pdfmetrics.registerFontFamily('Vera', normal='Vera', bold='Vera-Bold')
STYLES = {
    'title': ParagraphStyle('title', fontName='Vera-Bold', fontSize=23, leading=29,
                            textColor=NAVY, spaceAfter=12),
    'section': ParagraphStyle('section', fontName='Vera-Bold', fontSize=14, leading=19,
                              textColor=TEAL, spaceBefore=12, spaceAfter=6),
    'subsection': ParagraphStyle('subsection', fontName='Vera-Bold', fontSize=11, leading=15,
                                 textColor=NAVY, spaceBefore=10, spaceAfter=5),
    'body': ParagraphStyle('body', fontName='Vera', fontSize=10, leading=14,
                           textColor=INK, spaceAfter=7),
    'table': ParagraphStyle('table', fontName='Vera', fontSize=9, leading=12, textColor=INK),
    'table_head': ParagraphStyle('table_head', fontName='Vera-Bold', fontSize=8,
                                 leading=12, textColor=colors.white),
    'empty': ParagraphStyle('empty', fontName='Vera', fontSize=11, leading=16,
                            textColor=INK, alignment=TA_CENTER),
}


def url_markup(url):
    return f'<link href={quoteattr(url)} color="#0F716B"><u>{escape(url)}</u></link>'


def paragraph(text, style='body'):
    return Paragraph(escape(text), STYLES[style])


def readme_flowables(markdown, width):
    """Render the README headings, tables and paragraphs in their original order."""
    for block in re.split(r'\n\s*\n', markdown.strip()):
        lines = block.splitlines()
        if all(line.startswith('|') for line in lines):
            fields = [[entry.strip() for entry in line.strip('|').split('|')] for line in lines]
            values = [fields[0], *fields[2:]]
            items = [[paragraph(value, 'table_head' if i == 0 else 'table') for value in row]
                     for i, row in enumerate(values)]
            if len(fields[0]) == 7:
                widths = [width * 0.32] + [width * 0.68 / 6] * 6
            elif len(fields[0]) == 2:
                widths = [width * 0.32, width * 0.68]
            else:
                widths = [width / len(fields[0])] * len(fields[0])
            table = Table(items, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), NAVY),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                ('LINEBELOW', (0, -1), (-1, -1), 0.5, RULE),
            ]))
            yield table
            yield Spacer(1, 8)
        elif block.startswith('#'):
            marks, title = block.split(' ', 1)
            style = 'section' if len(marks) <= 2 else 'subsection'
            if title == 'Model observations':
                yield CondPageBreak(205)
            yield paragraph(title, style)
        else:
            body = escape(' '.join(lines))
            body = re.sub(r'`([^`]+)`', r'<font color="#0F716B">\1</font>', body)
            body = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', body)
            body = re.sub(r'https://[^\s<>]+', lambda m: url_markup(m.group().rstrip('.'))
                          + ('.' if m.group().endswith('.') else ''), body)
            yield Paragraph(body, STYLES['body'])


def footer(canvas, doc):
    width, _ = doc.pagesize
    canvas.setStrokeColor(RULE)
    canvas.line(doc.leftMargin, 38, width - doc.rightMargin, 38)
    canvas.setFillColor(INK)
    canvas.setFont('Vera', 8)
    canvas.drawString(doc.leftMargin, 24, 'BITS WILP | Machine Learning | Assignment 2')
    canvas.drawRightString(width - doc.rightMargin, 24, f'Page {doc.page}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', action='store_true', help='Generate a preview with clearly marked evidence placeholders.')
    parser.add_argument('--github-url')
    parser.add_argument('--app-url')
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--bits-id')
    parser.add_argument('--readme', type=Path, default=HERE.parent / 'project' / 'README.md')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    if not args.draft:
        for field in ('github_url', 'app_url', 'screenshot', 'name', 'bits_id'):
            if not getattr(args, field):
                parser.error(f'--{field.replace("_", "-")} is required for the final PDF.')
        for url in (args.github_url, args.app_url):
            parts = urlsplit(url)
            if parts.scheme != 'https' or not parts.hostname or parts.hostname == 'localhost':
                parser.error('Provide working public HTTPS links for GitHub and Streamlit.')
        if not args.screenshot.is_file():
            parser.error(f'Screenshot file does not exist: {args.screenshot}')

    output = args.output or HERE / ('Assignment_2_Draft.pdf' if args.draft else 'Assignment_2_Final.pdf')
    readme = args.readme.read_text(encoding='utf-8')
    github = args.github_url or '[Add your GitHub repository link]'
    app = args.app_url or '[Add your live Streamlit link]'
    readme = readme.replace('{GITHUB_URL}', github).replace('{STREAMLIT_URL}', app)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=47, leftMargin=47,
                            topMargin=48, bottomMargin=56, title='Machine Learning Assignment 2')
    story = [paragraph('Machine Learning Assignment 2', 'title'),
             paragraph('Card Campaign Booking Prediction | Five classification models'),
             paragraph('DRAFT - add your links and BITS Virtual Lab evidence' if args.draft else
                       f'{args.name} | BITS ID {args.bits_id}'),
             paragraph('1. GitHub repository link', 'section'),
             Paragraph(url_markup(github), STYLES['body']) if args.github_url else paragraph(github),
             paragraph('2. Live Streamlit app link', 'section'),
             Paragraph(url_markup(app), STYLES['body']) if args.app_url else paragraph(app),
             paragraph('3. BITS Virtual Lab execution screenshot', 'section')]
    if args.screenshot:
        image_width, image_height = ImageReader(str(args.screenshot)).getSize()
        scale = min(doc.width / image_width, 370 / image_height)
        story.append(Image(str(args.screenshot), width=image_width * scale, height=image_height * scale))
    else:
        placeholder = Table([[paragraph('Insert your BITS Virtual Lab screenshot showing the executed notebook and model comparison.', 'empty')]],
                            colWidths=[doc.width], rowHeights=[230])
        placeholder.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), PALE),
                                         ('BOX', (0, 0), (-1, -1), 0.7, RULE),
                                         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                         ('LEFTPADDING', (0, 0), (-1, -1), 35),
                                         ('RIGHTPADDING', (0, 0), (-1, -1), 35)]))
        story.append(placeholder)
    story.extend([PageBreak(), paragraph('4. GitHub README content', 'title')])
    story.extend(readme_flowables(readme, doc.width))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
