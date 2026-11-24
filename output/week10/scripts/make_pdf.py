"""Turn the two notes of output/week10 into PDFs (A4 landscape, formulas rendered, figures included).

    pip install latex2mathml          (once; a small pure-Python package that is not in requirements.txt)
    python output/week10/scripts/make_pdf.py

Writes homework1_summary.pdf and deep_dive/deep_dive_how_a_run_works.pdf next to their .md files.
The markdown becomes one HTML page, with every formula converted to MathML, and Chrome or Edge prints
that page to PDF without opening a window.
"""
import re
import subprocess
import tempfile
from pathlib import Path

import mistune
from latex2mathml.converter import convert as mathml

HERE = Path(__file__).resolve().parents[1]
NOTES = (HERE / "homework1_summary.md", HERE / "deep_dive" / "deep_dive_how_a_run_works.md")
BROWSERS = (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")

CSS = """
@page { size: A4 landscape; margin: 13mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 11pt; line-height: 1.45; color: #111; }
h1 { font-size: 21pt; margin: 0 0 4mm; }
h2 { font-size: 16pt; margin: 0 0 3mm; padding-bottom: 1.5mm; border-bottom: 1.5px solid #888; break-before: page; }
h2:first-of-type { break-before: auto; margin-top: 6mm; }
h3 { font-size: 12.5pt; margin: 5mm 0 2mm; break-after: avoid; }
p, ul, ol, blockquote, pre { max-width: 225mm; }
p, li { orphans: 3; widows: 3; }
p:has(> strong:only-child) { break-after: avoid; }
li { margin: 1mm 0; }
p:has(img) { max-width: none; }
img { display: block; max-width: 100%; max-height: 104mm; margin: 3mm auto; break-inside: avoid; }
table { border-collapse: collapse; margin: 3mm 0; font-size: 10pt; break-inside: avoid; }
th, td { border: 1px solid #999; padding: 1.4mm 2.5mm; text-align: left; vertical-align: top; }
th { background: #eee; }
code { font-family: Consolas, monospace; font-size: 9.5pt; background: #f2f2f2; padding: 0 1mm; border-radius: 2px; }
pre { background: #f2f2f2; padding: 2.5mm 3mm; break-inside: avoid; white-space: pre-wrap; }
pre code { padding: 0; background: none; }
blockquote { margin: 3mm 0; padding: 0.5mm 4mm; border-left: 3px solid #777; color: #333; background: #fafafa; }
math { font-family: "Cambria Math", "STIX Two Math", serif; font-size: 108%; }
math[display="block"] { margin: 3mm 0; break-inside: avoid; }
"""


class Renderer(mistune.HTMLRenderer):
    """HTML with the formulas as MathML, which Chrome and Edge draw themselves."""

    def inline_math(self, text):
        return mathml(text)

    def block_math(self, text):
        return mathml(text, display="block")


def to_html(note):
    text = note.read_text(encoding="utf-8")
    # mistune wants display formulas on their own lines: turn "$$x$$" on one line into three lines.
    text = re.sub(r"^([ >]*)\$\$(.+?)\$\$[ \t]*$", r"\1$$\n\1\2\n\1$$", text, flags=re.M)
    body = mistune.create_markdown(renderer=Renderer(escape=False), plugins=["table", "math"])(text)
    title = text.splitlines()[0].lstrip("# ").strip()
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
            f'<base href="{note.parent.as_uri()}/"><style>{CSS}</style></head><body>{body}</body></html>')


def main():
    exe = next((path for path in BROWSERS if Path(path).exists()), None)
    if exe is None:
        raise SystemExit("Chrome or Edge not found: add its path to BROWSERS in this script.")
    with tempfile.TemporaryDirectory() as tmp:
        for note in NOTES:
            page = Path(tmp) / (note.stem + ".html")
            page.write_text(to_html(note), encoding="utf-8")
            pdf = note.with_suffix(".pdf")
            subprocess.run([exe, "--headless=new", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", page.as_uri()],
                           check=True, timeout=180)
            print("saved", pdf, f"({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
