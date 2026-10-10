"""Build the final report PDF from the section drafts and the notebooks.

Fill in the team settings below, then run from the repo root:

    python report/build_report.py          # combined Markdown: report/Final-Project-Report.md
    python report/build_report.py --pdf    # final PDF with the notebooks as the appendix

The PDF needs Google Chrome installed (used to print the HTML to PDF) and an
internet connection (math is rendered with KaTeX loaded from a CDN).
"""

import base64
import html
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import mistune

# Team settings: fill these in before submitting
TEAM_NUMBER = "X"
TEAM_MEMBERS = ["Renee Dhanaraj", "Person B Name", "Anthony Candelas"]
INSTRUCTOR = "Instructor Name"
COURSE = "AAI 150"
PROGRAM = "M.S. Applied Artificial Intelligence, University of San Diego"
TITLE = "Modeling Wine Quality Using Physicochemical Attributes"
REPO_URL = "https://github.com/renjen/AAI150"

REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent
SECTIONS = [
    "01_introduction.md",
    "02_data_cleaning.md",
    "03_eda.md",
    "04_model_selection.md",
    "05_model_analysis.md",
    "06_conclusion.md",
    "07_references.md",
]
NOTEBOOKS = [
    ("A.1", "notebooks/01_data_cleaning.ipynb"),
    ("A.2", "notebooks/02_eda.ipynb"),
    ("A.3", "notebooks/03_modeling.ipynb"),
]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

markdown = mistune.create_markdown(plugins=["table"], escape=False)

CSS = """
@page { size: Letter; margin: 1in; }
body { font-family: "Times New Roman", Times, serif; font-size: 12pt; line-height: 1.5; color: #000; }
h1 { font-size: 18pt; margin-top: 0; break-before: page; }
h2 { font-size: 14pt; margin-top: 1.4em; }
h3 { font-size: 12pt; }
p, li { text-align: justify; }
hr { display: none; }
table { border-collapse: collapse; margin: 0.8em auto; font-size: 10.5pt; break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3px 8px; text-align: left; vertical-align: top; }
th { background: #eee; }
img { display: block; max-width: 100%; max-height: 7in; margin: 0.8em auto 0.2em; break-inside: avoid; }
code { font-family: Menlo, Consolas, monospace; font-size: 10pt; }
.keep { break-inside: avoid; }
.title-page { text-align: center; padding-top: 2in; }
.title-page h1 { font-size: 24pt; break-before: auto; margin-bottom: 1.5em; }
.title-page p { text-align: center; margin: 0.3em 0; }
.appendix-nb h1 { font-size: 14pt; break-before: auto; }
.appendix-nb h2 { font-size: 12.5pt; }
.appendix-nb h3 { font-size: 11.5pt; }
.appendix-nb p, .appendix-nb li { font-size: 10.5pt; }
.nb-title { break-before: page; font-size: 16pt; }
.appendix-intro + .nb-title { break-before: auto; }
pre { font-family: Menlo, Consolas, monospace; font-size: 8pt; line-height: 1.3;
      white-space: pre-wrap; word-break: break-word; margin: 0.4em 0; padding: 6px 8px; }
pre.code { background: #f4f4f4; border-left: 3px solid #888; }
pre.out { border-left: 3px solid #ccc; }
.appendix-nb table { font-size: 8pt; }
.appendix-nb th, .appendix-nb td { padding: 2px 5px; }
.appendix-nb img { max-height: 6.5in; }
"""


def render_markdown(text):
    """Render markdown, keeping $...$ and $$...$$ math untouched for KaTeX."""
    math = []

    def stash(match):
        math.append(match.group(0))
        return f"MATHPLACEHOLDER{len(math) - 1}X"

    text = re.sub(r"\$\$.+?\$\$|\$[^$\n]+?\$", stash, text, flags=re.S)
    rendered = markdown(text)
    return re.sub(r"MATHPLACEHOLDER(\d+)X",
                  lambda m: html.escape(math[int(m.group(1))], quote=False), rendered)


def alt_text_captions(text):
    """Turn images whose alt text is 'Figure N: caption' into an image plus a caption line."""
    return re.sub(
        r'<p><img src="([^"]+)" alt="Figure (\d+): ([^"]+)" ?/?></p>',
        r'<p><img src="\1" alt="Figure \2"></p>\n<p><strong>Figure \2.</strong> \3</p>',
        text,
    )


def keep_captions_together(text):
    """Keep each figure with the caption below it and each table with the caption above it."""
    text = re.sub(r"(<p><img[^>]*></p>\s*<p><strong>Figure \d+\.</strong>.*?</p>)",
                  r'<div class="keep">\1</div>', text, flags=re.S)
    text = re.sub(r"(<p><strong>Table \d+\.</strong>.*?</p>\s*<table>.*?</table>)",
                  r'<div class="keep">\1</div>', text, flags=re.S)
    return text


def embed_images(text, base_dir):
    """Turn relative image paths into data URIs so the HTML is self-contained."""
    def replace(match):
        data = base64.b64encode((base_dir / match.group(2)).read_bytes()).decode()
        return f'{match.group(1)}data:image/png;base64,{data}"'
    return re.sub(r'(<img[^>]*src=")([^"]+\.png)"', replace, text)


def title_page():
    members = "".join(f"<p>{html.escape(name)}</p>" for name in TEAM_MEMBERS)
    return f"""
<div class="title-page">
  <h1>{html.escape(TITLE)}</h1>
  <p><strong>Final Project Report, Team {html.escape(TEAM_NUMBER)}</strong></p>
  <br>{members}<br>
  <p>{html.escape(COURSE)}</p>
  <p>{html.escape(PROGRAM)}</p>
  <p>Instructor: {html.escape(INSTRUCTOR)}</p>
  <p>{date.today().strftime("%B %-d, %Y")}</p>
  <br><p>GitHub repository: {html.escape(REPO_URL)}</p>
</div>"""


def render_output(output):
    kind = output["output_type"]
    if kind == "stream":
        if output.get("name") == "stderr":
            return ""
        return f'<pre class="out">{html.escape("".join(output["text"]))}</pre>'
    if kind == "error":
        return f'<pre class="out">{html.escape(output["ename"] + ": " + output["evalue"])}</pre>'
    data = output.get("data", {})
    if "image/png" in data:
        return f'<img src="data:image/png;base64,{data["image/png"]}">'
    if "text/html" in data:
        return "".join(data["text/html"])
    if "text/plain" in data:
        return f'<pre class="out">{html.escape("".join(data["text/plain"]))}</pre>'
    return ""


def render_notebook(label, path):
    nb = json.loads((ROOT / path).read_text())
    parts = [f'<h1 class="nb-title">Appendix {label}: {html.escape(Path(path).name)}</h1>',
             '<div class="appendix-nb">']
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if not source.strip():
            continue
        if cell["cell_type"] == "markdown":
            parts.append(render_markdown(source))
        elif cell["cell_type"] == "code":
            parts.append(f'<pre class="code">{html.escape(source)}</pre>')
            parts.extend(render_output(o) for o in cell.get("outputs", []))
    parts.append("</div>")
    return "\n".join(parts)


def build_markdown():
    """Combine the title page and all sections into one Markdown file."""
    members = "  \n".join(TEAM_MEMBERS)
    parts = [
        f"# {TITLE}\n\n"
        f"**Final Project Report, Team {TEAM_NUMBER}**\n\n"
        f"{members}\n\n"
        f"{COURSE}  \n{PROGRAM}  \nInstructor: {INSTRUCTOR}  \n"
        f"{date.today().strftime('%B %-d, %Y')}\n\n"
        f"GitHub repository: {REPO_URL}"
    ]
    for name in SECTIONS:
        text = (REPORT_DIR / name).read_text().strip()
        # Images with the caption in the alt text get a caption line, like the other figures
        text = re.sub(r"!\[Figure (\d+): ([^\]]+)\]\(([^)]+)\)",
                      r"![Figure \1](\3)\n\n**Figure \1.** \2", text)
        parts.append(text)
    notebooks = "\n".join(
        f"- Appendix {label}: [{Path(path).name}](../{path})" for label, path in NOTEBOOKS
    )
    parts.append(
        "# Appendix: Technical Notebooks\n\n"
        "The code and output of our three Jupyter notebooks, in the order they are run. "
        "The PDF version of this report includes their full output.\n\n" + notebooks
    )
    md_path = REPORT_DIR / "Final-Project-Report.md"
    md_path.write_text("\n\n---\n\n".join(parts) + "\n")
    print(f"Wrote {md_path.relative_to(ROOT)}")


def build_pdf():
    body = [title_page()]
    for name in SECTIONS:
        section = render_markdown((REPORT_DIR / name).read_text())
        body.append(keep_captions_together(alt_text_captions(section)))
    body.append(
        "<div class='appendix-intro'><h1>Appendix: Technical Notebooks</h1>"
        "<p>This appendix contains the code and output of our three Jupyter notebooks, "
        "in the order they are run: data cleaning, exploratory data analysis, and modeling. "
        f"The notebooks and data are also available in our GitHub repository ({html.escape(REPO_URL)}).</p></div>"
    )
    body.extend(render_notebook(label, path) for label, path in NOTEBOOKS)

    head = (
        f"<meta charset='utf-8'><title>{html.escape(TITLE)}</title><style>{CSS}</style>"
        f"<link rel='stylesheet' href='{KATEX}/katex.min.css'>"
        f"<script defer src='{KATEX}/katex.min.js'></script>"
        f"<script defer src='{KATEX}/contrib/auto-render.min.js' onload=\"renderMathInElement(document.body,"
        "{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],"
        "ignoredTags:['script','noscript','style','textarea','pre','code']})\"></script>"
    )
    page = f"<!doctype html><html><head>{head}</head><body>{''.join(body)}</body></html>"
    page = embed_images(page, REPORT_DIR)

    html_path = REPORT_DIR / "final_report.html"
    pdf_path = REPORT_DIR / f"Final-Project-Report-Team-{TEAM_NUMBER}.pdf"
    html_path.write_text(page)

    result = subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=15000", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        capture_output=True, text=True,
    )
    if not pdf_path.exists():
        sys.exit(f"PDF export failed:\n{result.stderr}")
    print(f"Wrote {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    if "--pdf" in sys.argv:
        build_pdf()
    else:
        build_markdown()
