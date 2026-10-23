"""Convert docs/paper_draft.md to the JAGI LaTeX template (docs/paper/paper.tex).

Deterministic: headings, lists, tables, display/inline math, figures and citations are
translated by rules; no text is rewritten. Citations ("Name et al. (2020)", "(Name et al.,
2020; ...)") are mapped to keys in docs/paper/refs.bib through CITES below; any citation-like
string that is not in the table is reported, never guessed.

Usage: python scripts/md2tex.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "PAPER_DRAFT.md"
OUT = ROOT / "docs" / "paper"

# (author string as written, year) -> bib key
CITES: dict[tuple[str, str], str] = {
    ("DeepSeek-AI", "2024"): "deepseek2024v2",
    ("Efron and Tibshirani", "1993"): "efron1993",
    ("Eliseev and Mazur", "2023"): "eliseev2023",
    ("Fang et al.", "2025"): "fang2025fate",
    ("Google", "2026"): "google2026gemma4",
    ("Holm", "1979"): "holm1979",
    ("Jiang et al.", "2024"): "jiang2024mixtral",
    ("Kwon et al.", "2023"): "kwon2023vllm",
    ("Li et al.", "2023"): "li2023starcoder",
    ("Liang et al.", "2025"): "liang2025offloading",
    ("Miller", "1955"): "miller1955",
    ("Muennighoff et al.", "2024"): "muennighoff2024olmoe",
    ("Nguyen et al.", "2024"): "nguyen2024culturax",
    ("NLLB Team", "2022"): "nllb2022",
    ("OpenAI", "2025"): "openai2025gptoss",
    ("Paster et al.", "2023"): "paster2023openwebmath",
    ("Penedo et al.", "2024"): "penedo2024fineweb",
    ("Qwen Team", "2025"): "qwen2025qwen3",
    ("Song et al.", "2024"): "song2024promoe",
}
NAME = r"[A-Z][A-Za-z\-]+(?: et al\.| and [A-Z][a-z]+| Team)?"
NARRATIVE = re.compile(rf"({NAME}) \((\d{{4}})([^)]*)\)")
PAREN = re.compile(r"\(([^()]*\d{4}[^()]*)\)")
PIECE = re.compile(rf"^(?P<pre>(?:e\.g\., |see )?)(?P<n>{NAME}), (?P<y>\d{{4}})(?P<post>.*)$")

TEXT_UNI = {
    "–": "--",
    "—": "---",
    "−": "$-$",
    "≤": "$\\le$",
    "≥": "$\\ge$",
    "×": "$\\times$",
    "±": "$\\pm$",
    "≈": "$\\approx$",
    "→": "$\\to$",
    "‑": "-",
    "“": "``",
    "”": "''",
    "’": "'",
    "‘": "`",
    "…": "\\ldots{}",
    "ε": "$\\varepsilon$",
    "μ": "$\\mu$",
    "σ": "$\\sigma$",
    "π": "$\\pi$",
    "α": "$\\alpha$",
    "Δ": "$\\Delta$",
    "≠": "$\\ne$",
    "∞": "$\\infty$",
    "·": "$\\cdot$",
    "½": "$\\frac12$",
    "š": "\\v{s}",
    "Š": "\\v{S}",
    "ö": '\\"o',
    "ä": '\\"a',
    "ü": '\\"u',
    "é": "\\'e",
    "è": "\\`e",
    "ć": "\\'c",
    "ˆ": "\\^{}",
    " ": "~",
    "↔": "$\\leftrightarrow$",
    "√": "$\\sqrt{}$",
    "²": "$^2$",
    "⁻": "$^-$",
    "∈": "$\\in$",
    "≪": "$\\ll$",
    "✓": "yes",
    "§": "\\S{}",
    "Î": "$\\hat{I}$",
    "₂": "$_2$",
    "ℓ": "$\\ell$",
    "³": "$^3$",
    "⁴": "$^4$",
    "⁶": "$^6$",
    "ρ": "$\\rho$",
    "τ": "$\\tau$",
    "λ": "$\\lambda$",
    "ẑ": "$\\hat{z}$",
    "Ŝ": "$\\hat{S}$",
    "∩": "$\\cap$",
    "∝": "$\\propto$",
}
MATH_UNI = {
    "−": "-",
    "≤": "\\le ",
    "≥": "\\ge ",
    "×": "\\times ",
    "±": "\\pm ",
    "≈": "\\approx ",
    "→": "\\to ",
    "ε": "\\varepsilon ",
    "μ": "\\mu ",
    "σ": "\\sigma ",
    "π": "\\pi ",
    "α": "\\alpha ",
    "·": "\\cdot ",
    " ": "~",
    "–": "-",
    "…": "\\ldots ",
}
SPECIAL = {
    "&": "\\&",
    "%": "\\%",
    "#": "\\#",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
    "$": "\\$",
}
unmapped: set[str] = set()


def esc_text(s: str) -> str:
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\textbackslash{}")
        elif ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in TEXT_UNI:
            out.append(TEXT_UNI[ch])
        elif ord(ch) > 127:
            unmapped.add(ch)
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def esc_math(s: str) -> str:
    out = []
    for ch in s:
        if ch in MATH_UNI:
            out.append(MATH_UNI[ch])
        else:
            if ord(ch) > 127:
                unmapped.add(ch)
            out.append(ch)
    return "".join(out)


def esc_code(s: str) -> str:
    s = s.replace("\\", "\\textbackslash{}")
    for k, v in SPECIAL.items():
        s = s.replace(k, v)
    return s.replace("\\_", "\\_\\allowbreak{}")


def protect(line: str, store: list[str]) -> str:
    """Replace code spans and $math$ by placeholders (index into store)."""

    def put(txt: str) -> str:
        store.append(txt)
        return f"\x01{len(store) - 1}\x01"

    line = re.sub(r"`([^`]+)`", lambda m: put(f"\\texttt{{{esc_code(m.group(1))}}}"), line)
    line = re.sub(r"\$([^$]+)\$", lambda m: put(f"${esc_math(m.group(1))}$"), line)
    return line


def cite_cmd(kind: str, keys: list[str], post: str = "") -> str:
    opt = f"[{esc_text(post.strip(' ,'))}]" if post.strip(" ,") else ""
    return f"\\{kind}{opt}{{{','.join(keys)}}}"


def cites(line: str, store: list[str]) -> str:
    def put(txt: str) -> str:
        store.append(txt)
        return f"\x01{len(store) - 1}\x01"


    def narrative(m: re.Match) -> str:
        key = CITES.get((m.group(1), m.group(2)))
        if key is None:
            return m.group(0)
        return put(cite_cmd("citet", [key], m.group(3)))

    line = NARRATIVE.sub(narrative, line)

    def paren(m: re.Match) -> str:
        pieces = [p.strip() for p in m.group(1).split(";")]
        keys, pre = [], ""
        for p in pieces:
            pm = PIECE.match(p)
            key = CITES.get((pm.group("n"), pm.group("y"))) if pm else None
            post = pm.group("post").strip(" ,") if pm else ""
            if key is None or (post and len(pieces) > 1):
                return m.group(0)
            if post:
                return put(f"({pm.group('pre')}" + cite_cmd("citealp", [key], post) + ")") if pm.group("pre") else put(cite_cmd("citep", [key], post))
            keys.append(key)
            pre = pre or pm.group("pre")
        cmd = cite_cmd("citep", keys)
        return put(f"({pre}{cmd})" if pre else cmd)

    line = PAREN.sub(paren, line)
    return line


def emphasis(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"\\emph{\1}", s)
    return s


def inline(line: str) -> str:
    store: list[str] = []
    line = protect(line, store)
    line = cites(line, store)
    line = esc_text(line.replace("\x01", "\x00")).replace("\x00", "\x01")
    line = emphasis(line)
    # placeholders may be nested one level (cites inside protect indices): resolve repeatedly
    return re.sub(r"\x01(\d+)\x01", lambda m: store[int(m.group(1))], line)


def split_row(row: str) -> list[str]:
    cells, cur, in_math, in_code = [], [], False, False
    for ch in row.strip().strip("|"):
        if ch == "$" and not in_code:
            in_math = not in_math
        elif ch == "`":
            in_code = not in_code
        if ch == "|" and not in_math and not in_code:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


def table(rows: list[str]) -> str:
    cells = [split_row(r) for r in rows if not re.match(r"^\|\s*:?-{3,}", r)]
    ncol = len(cells[0])
    widths = [
        max(len(re.sub(r"[$`*\\]", "", r[j])) if j < len(r) else 0 for r in cells)
        for j in range(ncol)
    ]
    wide = [w > 28 for w in widths]
    spec = "".join(">{\\raggedright\\arraybackslash}X" if w else "l" for w in wide)
    env, args = (
        ("tabularx", f"{{\\textwidth}}{{{spec}}}") if any(wide) else ("tabular", f"{{{spec}}}")
    )
    out = ["\\begin{center}", "\\small", f"\\begin{{{env}}}{args}", "\\hline"]
    for i, r in enumerate(cells):
        r = (r + [""] * ncol)[:ncol]
        out.append(" & ".join(inline(c) for c in r) + " \\\\")
        if i == 0:
            out.append("\\hline")
    out += ["\\hline", f"\\end{{{env}}}", "\\end{center}"]
    return "\n".join(out)


FIGCAP: dict[str, str] = {}


def figure_index(md: str) -> None:
    """Captions come from the Appendix D figure index (Figure | Section | Content)."""
    sec = md.split("## Appendix D", 1)[-1].split("\n## ", 1)[0]
    for r in sec.splitlines():
        c = split_row(r) if r.startswith("|") else []
        if len(c) == 3 and c[0].isdigit():
            FIGCAP[c[0]] = c[2] + "."


def convert(md: str) -> tuple[str, list[str]]:
    lines = md.split("\n")
    body: list[str] = []
    figures: list[str] = []
    abstract: list[str] = []
    i, n = 0, len(lines)
    stack: list[tuple[int, str]] = []  # (indent, env)
    in_refs = appendix = False
    sink = body

    def close_lists(to: int = -1) -> None:
        while stack and stack[-1][0] > to:
            sink.append(f"\\end{{{stack.pop()[1]}}}")

    para: list[str] = []

    def flush() -> None:
        if para:
            sink.append(inline(" ".join(para)))
            sink.append("")
            para.clear()

    while i < n:
        ln = lines[i]
        s = ln.strip()
        if i < 5 and (ln.startswith("# ") or ln.startswith("**Draft")):
            i += 1
            continue
        if s == "---" or s == "":
            flush()
            if not s:
                pass
            i += 1
            continue
        m = re.match(r"^(#{2,3}) (.*)$", ln)
        if m:
            flush()
            close_lists()
            title = m.group(2)
            if m.group(1) == "##":
                if title == "Abstract":
                    sink = abstract
                    i += 1
                    continue
                sink = body
                if title == "References":
                    in_refs = True
                    i += 1
                    continue
                if in_refs and not title.startswith("Appendix"):
                    in_refs = False
                if title.startswith("Appendix"):
                    in_refs = False
                    if not appendix:
                        sink.append("\\appendix")
                        appendix = True
                    title = re.sub(r"^Appendix [A-Z]\.\s*", "", title)
                else:
                    title = re.sub(r"^\d+\.\s*", "", title)
                sink.append(f"\\section{{{inline(title)}}}")
            else:
                if in_refs:
                    i += 1
                    continue
                title = re.sub(r"^\d+\.\d+\s*", "", title)
                sink.append(f"\\subsection{{{inline(title)}}}")
            i += 1
            continue
        if in_refs:
            i += 1
            continue
        if s.startswith("<!--") and s.endswith("-->"):
            i += 1
            continue
        if s.startswith("```"):
            # Code fences (the mermaid diagram) have no LaTeX rendering; drop them
            # instead of printing the raw source as a paragraph.
            flush()
            close_lists()
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                i += 1
            i += 1
            continue
        if s.startswith("$$"):
            flush()
            close_lists()
            buf = [s]
            while not (buf[-1].endswith("$$") and (len(buf) > 1 or len(s) > 2)):
                i += 1
                buf.append(lines[i].strip())
            math = esc_math(" ".join(buf)[2:-2].strip())
            sink.append("\\[\n" + math + "\n\\]")
            i += 1
            continue
        if s.startswith("|"):
            flush()
            close_lists()
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            sink.append(table(rows))
            continue
        fm = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if fm:
            flush()
            close_lists()
            path = (SRC.parent / fm.group(2)).resolve()
            figures.append(str(path))
            j = i + 1
            while lines[j].strip() == "":
                j += 1
            cap = re.match(r"^\*Figure \d+[a-z]?\.\s*(.*)\*$", lines[j].strip())
            num = re.sub(r"\D", "", fm.group(1))
            caption = inline(cap.group(1)) if cap else inline(FIGCAP.get(num, fm.group(1)))
            i = j + 1 if cap else i + 1
            sink.append(
                "\\begin{figure}[t]\n\\centering\n"
                f"\\includegraphics[width=0.9\\textwidth]{{figures/{path.name}}}\n"
                f"\\caption{{{caption}}}\n\\end{{figure}}"
            )
            continue
        lm = re.match(r"^(\s*)([-*]|\d+\.) (.*)$", ln)
        if lm:
            flush()
            ind = len(lm.group(1))
            env = "enumerate" if lm.group(2)[0].isdigit() else "itemize"
            while stack and stack[-1][0] > ind:
                sink.append(f"\\end{{{stack.pop()[1]}}}")
            if not stack or stack[-1][0] < ind:
                stack.append((ind, env))
                sink.append(f"\\begin{{{env}}}")
            item = [lm.group(3)]
            while (
                i + 1 < n
                and lines[i + 1].strip()
                and re.match(r"^\s{2,}(?![-*] |\d+\. )\S", lines[i + 1])
                and not lines[i + 1].strip().startswith(("|", "$$"))
            ):
                i += 1
                item.append(lines[i].strip())
            sink.append("\\item " + inline(" ".join(item)))
            i += 1
            continue
        if stack:
            close_lists()
        para.append(s)
        i += 1
    flush()
    close_lists()
    return "\n".join(abstract).strip(), body, figures  # type: ignore[return-value]


HEAD = r"""\documentclass[twoside,11pt]{article}
\usepackage{jagi}
\usepackage{times}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{tabularx}
\usepackage{booktabs}

% TODO(author): fill in volume/issue/pages/year/dates/DOI once assigned by the journal.
\jagiheading{1}{1}{1-40}{2026}{2026-10-23}{2026-10-23}{10.0000/pending}{Arora}
\ShortHeadings{How Predictable Is MoE Routing?}{Arora}
\firstpageno{1}

\begin{document}

\title{How Predictable Is Mixture-of-Experts Routing?\
       A Seven-Checkpoint Measurement Under One Definition}

% TODO(author): affiliation and e-mail.
\author{\name Kavish Arora \email TODO@example.org \
        \addr TODO affiliation}

\editor{TODO}

\maketitle

\begin{abstract}
@@ABSTRACT@@
\end{abstract}

\begin{keywords}
  mixture of experts, expert routing, predictability, mutual information, expert prefetching, inference systems
\end{keywords}

"""

TAIL = r"""
\vskip 0.2in
\nocite{miller1955,gerganov_llamacpp}
\bibliography{refs}

\end{document}
"""


def main() -> int:
    md = SRC.read_text(encoding="utf-8")
    figure_index(md)
    abstract, body, figures = convert(md)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    names = [Path(f).name for f in figures]
    assert len(names) == len(set(names)), "figure basename collision"
    for f in figures:
        shutil.copy(f, OUT / "figures" / Path(f).name)
    tex = HEAD.replace("@@ABSTRACT@@", abstract) + "\n".join(body) + TAIL
    (OUT / "paper.tex").write_text(tex, encoding="utf-8")
    cited = set(re.findall(r"\\cite(?:t|p|alp)(?:\[[^\]]*\])?\{([^}]*)\}", tex))
    used = {k for c in cited for k in c.split(",")}
    bib = set(re.findall(r"^@\w+\{([^,]+),", (OUT / "refs.bib").read_text(encoding="utf-8"), re.M))
    print(f"wrote {OUT / 'paper.tex'}: {len(tex.splitlines())} lines, {len(figures)} figures")
    print(f"cited keys: {len(used)}; bib entries: {len(bib)}")
    print("cited but missing in bib:", sorted(used - bib) or "none")
    print("in bib but never cited:", sorted(bib - used) or "none")
    src = md.split("## References")[0]
    left = re.findall(rf"({NAME})(?: \(|, )(\d{{4}})", src)
    uncited = [f"{n} {y}" for n, y in left if (n, y) not in CITES]
    print("citation-like strings with no key:", sorted(set(uncited)) or "none")
    print("non-ASCII characters left unmapped:", "".join(sorted(unmapped)) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
