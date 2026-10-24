#!/usr/bin/env python3
"""Markdown -> styled PDF via pandoc + weasyprint. Shared by the batch
pipeline and the `supermeet.py pdf` subcommand, so there is one stylesheet and
one renderer."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PDF_CSS = """
@page {
    size: A4;
    margin: 2.2cm 2.5cm 2.5cm 2.5cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: 'Inter', sans-serif;
        font-size: 9pt;
        color: #999;
    }
}

body {
    font-family: 'Lora', Georgia, serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #2c2c2c;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
}

h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 22pt;
    color: #1a1a1a;
    line-height: 1.25;
    margin: 0 0 0.4em;
    padding-bottom: 0.3em;
    border-bottom: 2px solid #d4d4d4;
    letter-spacing: -0.01em;
}

h2 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 15pt;
    color: #1a1a1a;
    margin: 1.8em 0 0.6em;
    line-height: 1.3;
    letter-spacing: -0.005em;
}

h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 12pt;
    color: #2c2c2c;
    margin: 1.4em 0 0.4em;
}

p {
    margin: 0 0 0.8em;
    text-align: justify;
    hyphens: auto;
}

ul, ol {
    padding-left: 1.4em;
    margin: 0.5em 0 1em;
}

li {
    margin-bottom: 0.4em;
    line-height: 1.65;
}

li > p {
    margin-bottom: 0.3em;
}

ul ul, ol ol, ul ol, ol ul {
    margin: 0.3em 0;
}

strong {
    color: #1a1a1a;
    font-weight: 600;
}

em {
    color: #444;
    font-style: italic;
}

code {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    background: #f4f4f5;
    padding: 0.12em 0.4em;
    border-radius: 3px;
    font-size: 0.88em;
    color: #d63384;
    border: 0.5px solid #e4e4e7;
}

pre {
    background: #f8f8f8;
    border: 0.5px solid #e4e4e7;
    border-radius: 5px;
    padding: 0.8em 1em;
    overflow-x: auto;
    line-height: 1.5;
}

pre code {
    background: transparent;
    border: none;
    padding: 0;
    color: #2c2c2c;
}

blockquote {
    border-left: 3px solid #aaa;
    padding: 0.4em 1em;
    margin: 1em 0;
    color: #555;
    font-style: italic;
    background: #fafafa;
}

table {
    border-collapse: collapse;
    margin: 1em 0;
    width: 100%;
    font-family: 'Inter', sans-serif;
    font-size: 10pt;
}

th, td {
    border: 0.5px solid #d4d4d4;
    padding: 0.5em 0.8em;
    text-align: left;
    vertical-align: top;
}

th {
    background: #f4f4f5;
    font-weight: 600;
    color: #1a1a1a;
}

tr:nth-child(even) td {
    background: #fafafa;
}

hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 2em 0;
}

a {
    color: #0066cc;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

nav#TOC {
    background: #fafafa;
    border-left: 3px solid #d4d4d4;
    padding: 1em 1.5em;
    margin: 1.5em 0;
    font-family: 'Inter', sans-serif;
    font-size: 10pt;
}

nav#TOC ul {
    list-style: none;
    padding-left: 1em;
    margin: 0.3em 0;
}

nav#TOC > ul {
    padding-left: 0;
}

nav#TOC li {
    margin-bottom: 0.25em;
    line-height: 1.5;
}

nav#TOC a {
    color: #2c2c2c;
}

h2:has(+ ul) + ul li {
    margin-bottom: 0.6em;
}
"""


def _weasyprint_bin() -> str:
    """pandoc is a system binary, so it won't see weasyprint unless the venv is
    on PATH. Resolve it next to the running interpreter first."""
    local = Path(sys.executable).parent / "weasyprint"
    if local.exists():
        return str(local)
    found = shutil.which("weasyprint")
    if not found:
        raise RuntimeError(
            "weasyprint not found next to the interpreter or on PATH. "
            "Install it into the venv: .venv/bin/pip install weasyprint"
        )
    return found


def render_pdf(md_path: Path, pdf_path: Path, toc: bool = True,
               title: str = None) -> Path:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # A fixed /tmp path races when two runs overlap.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".css", encoding="utf-8", delete=False
    ) as fh:
        fh.write(PDF_CSS)
        css_file = Path(fh.name)

    cmd = [
        "pandoc", str(md_path),
        "-o", str(pdf_path),
        f"--pdf-engine={_weasyprint_bin()}",
        "--css", str(css_file),
        "-V", "lang=id-ID",
        "--standalone",
    ]
    if title:
        cmd.extend(["--metadata", f"title={title}"])
    if toc:
        cmd.extend(["--toc", "--toc-depth=2"])

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"pandoc failed ({e.returncode}):\n{stderr}") from None
    finally:
        css_file.unlink(missing_ok=True)

    return pdf_path
