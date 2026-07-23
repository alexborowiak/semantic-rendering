#!/usr/bin/env python3
"""
semantic_render.py
==================

Turn an *executed* Jupyter notebook into an interactive, figure-first
"semantic analysis" environment -- an HTML page that treats the notebook as
computational state and recovers the scientific structure underneath it:

        Dataset -> Transform -> Diagnostic -> Figure -> Interpretation

instead of rendering every cell with equal weight.

This is a *static* renderer: it reads the outputs already stored in the
notebook (run it once, normally, in Jupyter), so there is no kernel, no
backend and no re-execution. Open the resulting .html in any browser.

--------------------------------------------------------------------------
Authoring a notebook for this renderer
--------------------------------------------------------------------------
Add `#| key: value` directive lines to the TOP of a code cell. They are
parsed, then stripped from the displayed source. Everything is optional --
absent directives are inferred from the cell's outputs.

    #| section:    <name>      Group this cell under a top-level section.
    #| subsection: <name>      Optional nested group within a section.
    #| title:      <text>      Human title for the card (else inferred).
    #| display:    <type>      figure | dataset | transform | diagnostic
                               | metric | text | code | hidden
    #| code:       hidden|show Default code visibility for this card.
    #| id:         <slug>      Stable id, referenced by `depends`.
    #| depends:    a, b, c     ids this card derives from (provenance edges).
    #| caption:    <text>      Interpretation / what to look for.
    #| group:      <name>      Merge several cells into ONE card (alias: tag).
    #| order:      <int>       Sort this cell within its group.
    #| step:       <label>     Label this cell's chunk in the folded code.
    #| stack:      a, b        Fold the code of cells with these ids under
                               this card (reusable across figures).

Markdown cells: a leading `# / ## / ###` heading opens a section /
subsection; any prose beneath it becomes an interpretation note.

Grouping vs stacking, two ways to put several cells under one figure:
  * group (push): cells self-tag with `#| group:`; one group per cell; best
    for a few adjacent cells authored as a unit.
  * stack (pull): a figure names upstream cells by id with `#| stack:`; the
    named cells are folded in (and consumed, so they get no card of their
    own) and the SAME cell can be stacked under many figures. Use it for
    shared prep like opening data or regridding. `depends:` keeps a cell as
    its own graph node; `stack:` folds its code and collapses it.

Inference when `display` is absent:
    image output            -> figure
    xarray HTML repr        -> dataset
    only text / stdout       -> metric (if short) else text
    no output                -> code (collapsed by default)

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
App mode (the normal way to work) -- a local GUI in your browser with a
tab per notebook, an Open dialog, drag-and-drop, and project-level
presentations that can mix cards from every open notebook:

    python semantic_render.py                    # launch the app (cwd root)
    python semantic_render.py --app A.ipynb B.ipynb   # preload as tabs
    python semantic_render.py --app --root C:/work/proj --port 8765

Static export (shareable single .html, no server needed to view):

    python semantic_render.py NOTEBOOK.ipynb [-o OUT.html] [--title "..."]
    python semantic_render.py A.ipynb B.ipynb -o bundle.html   # tabbed
    python semantic_render.py NOTEBOOK.ipynb --deck DECK.json
    python semantic_render.py NOTEBOOK.ipynb --embed-deck DECK.json
    python semantic_render.py --self-test

The rendered page includes a Present mode (toolbar) with a slide builder;
decks persist in the notebook's metadata.semantic.presentations, in a
<notebook>.deck.json sidecar, via --embed-deck, or (app mode) in a
semantic_project.json next to where the app was started. Slides reference
cards by stable anchors (`#| id:` first, else the nbformat cell id);
multi-notebook decks namespace them as `<stem>::<anchor>`.
"""

from __future__ import annotations

import argparse
import ast
import base64
import html
import http.server
import io
import json
import keyword
import re
import secrets
import sys
import threading
import tokenize
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_URL = "https://github.com/alexborowiak/semantic-rendering"
_KOFI_URL = "https://ko-fi.com/plotline"

# --------------------------------------------------------------------------
# Directive parsing
# --------------------------------------------------------------------------

_DIRECTIVE_RE = re.compile(r"^\s*#\|\s*([A-Za-z_]+)\s*:\s*(.*?)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")

# display types we understand; anything else falls back to "code"
_DISPLAY_TYPES = {
    "figure", "dataset", "transform", "diagnostic",
    "metric", "text", "code", "hidden",
}


def split_directives(source: str) -> tuple[dict[str, str], str]:
    """Pull the leading `#| k: v` block off a code cell.

    Returns (directives, remaining_source). Directives may be preceded by
    blank lines; the block ends at the first non-directive, non-blank line.
    """
    lines = source.splitlines()
    directives: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = _DIRECTIVE_RE.match(line)
        if not m:
            break
        key, value = m.group(1).lower(), m.group(2)
        directives[key] = value
        i += 1
    remaining = "\n".join(lines[i:]).strip("\n")
    return directives, remaining


# --------------------------------------------------------------------------
# Python syntax highlighting (robust: real tokenizer, plain-text fallback)
# --------------------------------------------------------------------------

def highlight_python(src: str) -> str:
    """Return HTML for `src` with lightweight, safe Python highlighting."""
    if not src.strip():
        return ""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return html.escape(src)

    # absolute char offset for each (row, col)
    line_starts = [0]
    for ln in src.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(ln))

    def idx(row: int, col: int) -> int:
        if row - 1 >= len(line_starts):
            return len(src)
        return line_starts[row - 1] + col

    out: list[str] = []
    prev = 0
    builtins_set = set(dir(__builtins__)) if isinstance(__builtins__, dict) \
        else set(dir(__builtins__))
    for tok in toks:
        try:
            start = idx(*tok.start)
            end = idx(*tok.end)
        except Exception:
            continue
        if start < prev:
            start = prev
        # gap (whitespace / newlines) preserved verbatim
        if start > prev:
            out.append(html.escape(src[prev:start]))
        text = src[start:end]
        cls = None
        tname = tokenize.tok_name.get(tok.type, "")
        if tok.type == tokenize.NAME:
            if keyword.iskeyword(tok.string):
                cls = "kw"
            elif tok.string in builtins_set:
                cls = "bn"
        elif tok.type == tokenize.STRING or tname.startswith("FSTRING"):
            cls = "st"
        elif tok.type == tokenize.NUMBER:
            cls = "nu"
        elif tok.type == tokenize.COMMENT:
            cls = "co"
        elif tok.type == tokenize.OP:
            cls = "op"
        esc = html.escape(text)
        out.append(f'<span class="{cls}">{esc}</span>' if cls else esc)
        prev = end
    if prev < len(src):
        out.append(html.escape(src[prev:]))
    return "".join(out)


# --------------------------------------------------------------------------
# Output rendering
# --------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _as_text(value: Any) -> str:
    return "".join(value) if isinstance(value, list) else (value or "")


def _looks_like_xarray(htmltext: str) -> bool:
    return ("xr-" in htmltext) or ("xarray" in htmltext.lower())


@dataclass
class RenderedOutput:
    kind: str          # "image" | "xarray" | "html" | "text" | "error"
    payload: str       # html fragment ready to drop in
    has_image: bool = False
    has_xarray: bool = False


def render_outputs(outputs: list[dict]) -> list[RenderedOutput]:
    """Convert nbformat output dicts into ready-to-embed HTML fragments."""
    rendered: list[RenderedOutput] = []
    for out in outputs or []:
        otype = out.get("output_type")
        if otype == "stream":
            text = _strip_ansi(_as_text(out.get("text", "")))
            if text.strip():
                rendered.append(RenderedOutput(
                    "text", f'<pre class="stream">{html.escape(text)}</pre>'))
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "image/png" in data:
                b64 = data["image/png"]
                b64 = b64 if isinstance(b64, str) else "".join(b64)
                b64 = b64.strip().replace("\n", "")
                rendered.append(RenderedOutput(
                    "image",
                    f'<div class="figframe">'
                    f'<img loading="lazy" alt="figure output" '
                    f'src="data:image/png;base64,{b64}"></div>',
                    has_image=True))
            elif "image/svg+xml" in data:
                svg = _as_text(data["image/svg+xml"])
                rendered.append(RenderedOutput(
                    "image", f'<div class="figframe">{svg}</div>',
                    has_image=True))
            elif "text/html" in data:
                htmltext = _as_text(data["text/html"])
                if _looks_like_xarray(htmltext):
                    rendered.append(RenderedOutput(
                        "xarray",
                        f'<div class="xr-wrap">{htmltext}</div>',
                        has_xarray=True))
                else:
                    rendered.append(RenderedOutput(
                        "html", f'<div class="rich">{htmltext}</div>'))
            elif "text/plain" in data:
                text = _as_text(data["text/plain"])
                if text.strip():
                    rendered.append(RenderedOutput(
                        "text", f'<pre class="result">{html.escape(text)}</pre>'))
        elif otype == "error":
            tb = _strip_ansi("\n".join(out.get("traceback", [])))
            rendered.append(RenderedOutput(
                "error", f'<pre class="error">{html.escape(tb)}</pre>'))
    return rendered


# --------------------------------------------------------------------------
# Semantic model
# --------------------------------------------------------------------------

@dataclass
class CodeStep:
    label: str
    code: str
    outputs: list[RenderedOutput] = field(default_factory=list)
    is_primary: bool = False


@dataclass
class Item:
    kind: str                      # card display type
    title: str
    code: str = ""                 # kept for notes / simple use
    code_visible: bool = False
    outputs: list[RenderedOutput] = field(default_factory=list)  # face outputs
    caption: str = ""
    item_id: str = ""              # explicit slug or auto
    node_id: str = ""              # provenance node (only if user gave `id`)
    anchor: str = ""               # stable deck ref: node_id > nb cell id > slug
    chain: list[str] = field(default_factory=list)  # upstream card anchors
    depends: list[str] = field(default_factory=list)
    subsection: str = ""
    is_note: bool = False          # pure-markdown interpretation card
    title_echo: bool = False       # title merely repeats a code line
    steps: list[CodeStep] = field(default_factory=list)  # folded code chunks
    members: list = field(default_factory=list)          # transient, build-only

    @property
    def has_image(self) -> bool:
        return any(o.has_image for o in self.outputs)

    @property
    def has_xarray(self) -> bool:
        return any(o.has_xarray for o in self.outputs)


@dataclass
class Section:
    title: str
    section_id: str
    items: list[Item] = field(default_factory=list)


@dataclass
class Document:
    title: str
    sections: list[Section] = field(default_factory=list)
    presentations: list = field(default_factory=list)  # named slide decks
    source_name: str = ""          # notebook stem, names deck downloads
    raw_html: str = ""             # linear "raw notebook" view of the cells


def _slug(text: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "item"
    slug, n = base, 1
    while slug in used:
        n += 1
        slug = f"{base}-{n}"
    used.add(slug)
    return slug


def _infer_kind(item_outputs: list[RenderedOutput]) -> str:
    if any(o.has_image for o in item_outputs):
        return "figure"
    if any(o.has_xarray for o in item_outputs):
        return "dataset"
    text_like = [o for o in item_outputs if o.kind in ("text", "html", "error")]
    if text_like:
        total = sum(len(o.payload) for o in text_like)
        return "metric" if total < 900 else "text"
    return "code"


def _title_from_code(code: str) -> tuple[str, bool]:
    """Best-effort title. Returns (title, echo): echo=True when the title
    merely repeats a line of the cell's code — such titles still label the
    item in the nav but are not repeated as a heading on the card."""
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("#|")]
    if lines and lines[0].startswith("#"):
        return (lines[0].lstrip("#").strip() or "Code"), False
    funcs: list[tuple[str, bool]] = []      # (name, is_function)
    other = False
    try:
        for node in ast.parse(code).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append((node.name, True))
            elif isinstance(node, ast.ClassDef):
                funcs.append((node.name, False))
            else:
                other = True
    except SyntaxError:
        pass
    if funcs:
        if len(funcs) == 1:
            name, is_fn = funcs[0]
            base = name + ("()" if is_fn else "")
            return (base + (" + code" if other else "")), False
        if other:
            return f"{len(funcs)} functions + code", False
        names = ", ".join(n for n, _ in funcs[:3])
        if len(funcs) > 3:
            names += ", …"
        return f"{len(funcs)} functions ({names})", False
    for s in lines:
        if not s.startswith("#"):
            return ((s[:60] + "...") if len(s) > 60 else s), True
    return "Code", False


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _finalize_item(item: Item, used_slugs: set[str],
                   cell_by_id: dict[str, dict]) -> None:
    """Resolve a card from its grouped member cells plus any stacked cells."""
    members = sorted(item.members, key=lambda m: (m["order"], m["idx"]))
    multi = len(members) > 1

    def has_img(m):
        return any(o.has_image for o in m["outputs"])

    # the card's face: the cell that draws the figure (or the last with output)
    primary = next(
        (m for m in members
         if m["d"].get("display", "").lower() in ("figure", "diagnostic")), None)
    if primary is None:
        primary = next((m for m in reversed(members) if has_img(m)), None)
    if primary is None:
        primary = next((m for m in reversed(members) if m["outputs"]), None)
    if primary is None:
        primary = members[-1]

    # this card's own code chunks (from its grouped members)
    own_steps: list[CodeStep] = []
    for m in members:
        d = m["d"]
        label = (d.get("step") or d.get("label")
                 or (d.get("subsection", "") if multi else "")).strip()
        own_steps.append(CodeStep(label=label, code=m["code"],
                                 outputs=m["outputs"], is_primary=(m is primary)))

    # cells pulled in by `stack:` (referenced by id), folded in front
    stack_ids: list[str] = []
    for m in members:
        for sid in _csv(m["d"].get("stack", "")):
            if sid not in stack_ids:
                stack_ids.append(sid)
    own_idx = {m["idx"] for m in members}
    stacked_steps: list[CodeStep] = []
    for sid in stack_ids:
        cm = cell_by_id.get(sid)
        if cm is None or cm["idx"] in own_idx:
            continue
        label = (cm["d"].get("step") or cm["d"].get("label")
                 or cm["d"].get("title") or sid)
        stacked_steps.append(CodeStep(label=label.strip(), code=cm["code"],
                                      outputs=cm["outputs"], is_primary=False))

    item.steps = stacked_steps + own_steps
    item.outputs = primary["outputs"]

    # display kind: first explicit non-code display wins, else infer from face
    display = ""
    for m in members:
        cand = m["d"].get("display", "").lower()
        if cand == "hidden":
            display = "hidden"
            break
        if cand in _DISPLAY_TYPES and cand != "code":
            display = cand
            break
    item.kind = display or _infer_kind(primary["outputs"])

    # give the face a default step label once it shares the fold with others
    if len(item.steps) > 1:
        for s in item.steps:
            if s.is_primary and not s.label:
                s.label = {"figure": "plot", "dataset": "load data",
                           "transform": "transform",
                           "metric": "compute"}.get(item.kind, "")

    explicit = next(
        (m["d"]["title"] for m in members if m["d"].get("title")), "")
    if explicit:
        item.title = explicit
    else:
        item.title, item.title_echo = _title_from_code(primary["code"])
    item.caption = (primary["d"].get("caption")
                    or next((m["d"]["caption"] for m in members
                             if m["d"].get("caption")), ""))
    item.node_id = next(
        (m["d"]["id"].strip() for m in members if m["d"].get("id", "").strip()), "")

    member_ids = {m["d"].get("id", "").strip() for m in members}
    depends, seen = [], set()
    for m in members:
        for dep in _csv(m["d"].get("depends", "")):
            if dep not in seen and dep not in member_ids:
                seen.add(dep)
                depends.append(dep)
    item.depends = depends
    item.code_visible = any(
        m["d"].get("code", "").lower() in ("show", "shown", "visible", "true")
        for m in members)
    item.item_id = _slug(item.node_id or item.title or "item", used_slugs)
    cid = primary.get("cell_id", "")
    item.anchor = item.node_id or (f"cell:{cid}" if cid else "") or item.item_id


_LAYOUT_PANES = {"full": 1, "halves": 2, "rows": 2, "quarters": 4,
                 "title": 0, "blank": 0}


def _as_presentations(obj: Any) -> list:
    """Normalize saved presentation data to [{name, slides}, ...].

    Accepts the current schema (a list, or {"presentations": [...]}) plus
    the legacy single-deck schema ({"slides": [{kind, anchor, beside}]}),
    whose card slides are converted to pane layouts. Slides may carry
    free annotations (text boxes / arrows / rects) and title-slide text.
    """
    if isinstance(obj, list):
        pres = obj
    elif isinstance(obj, dict) and isinstance(obj.get("presentations"), list):
        pres = obj["presentations"]
    elif isinstance(obj, dict) and isinstance(obj.get("slides"), list):
        pres = [{"name": obj.get("name") or "deck", "slides": obj["slides"]}]
    else:
        return []
    out = []
    for p in pres:
        if not isinstance(p, dict) or not isinstance(p.get("slides"), list):
            continue
        slides = []
        for s in p["slides"]:
            if not isinstance(s, dict):
                continue
            if "panes" in s or s.get("layout") in _LAYOUT_PANES:
                lay = s.get("layout")
                raw_panes = [a if isinstance(a, str) and a else None
                             for a in (s.get("panes") or [])]
                if lay not in _LAYOUT_PANES:
                    lay = {1: "full", 2: "halves"}.get(
                        len(raw_panes) or 1, "quarters")
                n = _LAYOUT_PANES[lay]
                panes = (raw_panes + [None] * n)[:n]
                slide: dict = {"layout": lay, "panes": panes}
                if lay == "title":
                    slide["title"] = str(s.get("title") or "")
                    slide["sub"] = str(s.get("sub") or "")
                    for k in ("tprops", "sprops"):
                        if isinstance(s.get(k), dict):
                            slide[k] = s[k]
                if isinstance(s.get("annots"), list):
                    ann = [a for a in s["annots"] if isinstance(a, dict)]
                    if ann:
                        slide["annots"] = ann
                slides.append(slide)
            elif s.get("kind") == "card" and s.get("anchor"):   # legacy
                panes = [s["anchor"]] + [b for b in (s.get("beside") or [])
                                         if isinstance(b, str)][:3]
                lay = {1: "full", 2: "halves"}.get(len(panes), "quarters")
                slides.append({"layout": lay, "panes": panes})
        entry = {"name": str(p.get("name") or "deck"), "slides": slides}
        if isinstance(p.get("folder"), str) and p["folder"].strip():
            entry["folder"] = p["folder"].strip()
        out.append(entry)
    return out


def _cell_names(code: str) -> tuple[set[str], set[str]]:
    """Best-effort (defined, externally-read) names for one cell's code.

    A name counts as externally read when the cell uses it at or before its
    own first assignment (so `z = z + 1` reads the earlier z, but
    `x = 1; print(x)` does not read an external x). Function parameters are
    excluded; IPython magic/shell lines are stripped before parsing.
    """
    src = "\n".join(ln for ln in code.splitlines()
                    if not ln.lstrip().startswith(("%", "!")))
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set(), set()
    first_def: dict[str, int] = {}
    first_use: dict[str, int] = {}
    params: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            params.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            first_def.setdefault(node.name, node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                first_def.setdefault((a.asname or a.name).split(".")[0],
                                     node.lineno)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target,
                                                            ast.Name):
            first_use.setdefault(node.target.id, node.lineno)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                first_def.setdefault(node.id, node.lineno)
            elif isinstance(node.ctx, ast.Load):
                first_use.setdefault(node.id, node.lineno)
    uses = {n for n, ln in first_use.items()
            if n not in params
            and (n not in first_def or ln <= first_def[n])}
    return set(first_def), uses


def _build_chains(doc: Document) -> None:
    """Attach to every card the ordered chain of upstream cards feeding it.

    Edges come from two sources, unioned: automatic variable tracing (the
    card that last assigned each name this card reads) and declared
    `depends:` ids. The transitive closure, in document order, becomes
    `item.chain` -- the full "open data -> transform -> plot" story shown
    under a figure's Show code.
    """
    cards: list[tuple[int, Item]] = []
    for sec in doc.sections:
        for it in sec.items:
            if it.is_note or not it.members:
                continue
            cards.append((min(m["idx"] for m in it.members), it))
    cards.sort(key=lambda t: t[0])
    order = {id(it): i for i, (_, it) in enumerate(cards)}
    by_node = {it.node_id: it for _, it in cards if it.node_id}
    items_by_id = {id(it): it for _, it in cards}

    deps: dict[int, set[int]] = {id(it): set() for _, it in cards}
    last: dict[str, Item] = {}          # name -> card that last assigned it
    for _, it in cards:
        for m in sorted(it.members, key=lambda m: m["idx"]):
            defs, uses = _cell_names(m["code"])
            for n in uses:
                src = last.get(n)
                if src is not None and src is not it:
                    deps[id(it)].add(id(src))
            for n in defs:
                last[n] = it
        for d in it.depends:
            src = by_node.get(d)
            if src is not None and src is not it:
                deps[id(it)].add(id(src))

    def ancestors(iid: int, seen: set[int]) -> None:
        for p in deps.get(iid, ()):
            if p not in seen:
                seen.add(p)
                ancestors(p, seen)

    for _, it in cards:
        seen: set[int] = set()
        ancestors(id(it), seen)
        it.chain = [items_by_id[i].anchor or items_by_id[i].item_id
                    for i in sorted(seen, key=lambda i: order.get(i, 0))]


def parse_notebook(nb: dict, title: str | None = None) -> Document:
    used_slugs: set[str] = set()
    title_locked = title is not None          # explicit --title wins over H1
    nb_title = title or nb.get("metadata", {}).get("title")

    doc = Document(title=nb_title or "Untitled analysis")
    sem_meta = nb.get("metadata", {}).get("semantic", {})
    if isinstance(sem_meta, dict):
        doc.presentations = _as_presentations(
            sem_meta.get("presentations") or sem_meta.get("deck"))
    cur_section: Section | None = None
    cur_subsection = ""
    group_index: dict[str, Item] = {}
    cell_by_id: dict[str, dict] = {}   # id -> member cell (for `stack:` lookup)
    all_members: list[dict] = []       # every code cell, to find stacked ids

    def ensure_section() -> Section:
        nonlocal cur_section
        if cur_section is None:
            cur_section = Section("Overview", _slug("overview", used_slugs))
            doc.sections.append(cur_section)
        return cur_section

    for idx, cell in enumerate(nb.get("cells", [])):
        ctype = cell.get("cell_type")
        source = _as_text(cell.get("source", ""))

        if ctype == "markdown":
            handled_heading = False
            md_anchor = f"cell:{cell.get('id')}" if cell.get("id") else ""
            stripped = source.strip()
            m = _HEADING_RE.match(stripped.splitlines()[0]) if stripped else None
            if m:
                level, text = len(m.group(1)), m.group(2).strip()
                if level == 1:
                    if not title_locked:
                        doc.title = text          # document title; no section
                elif level == 2:
                    cur_section = Section(text, _slug(text, used_slugs))
                    doc.sections.append(cur_section)
                    cur_subsection = ""
                    handled_heading = True
                else:  # level >= 3 -> subsection marker
                    cur_subsection = text
                    handled_heading = True
                # prose after the heading becomes a note
                rest = "\n".join(stripped.splitlines()[1:]).strip()
                if rest:
                    sec = ensure_section()
                    nid = _slug("note", used_slugs)
                    sec.items.append(Item(
                        kind="text", title=text if handled_heading else "Note",
                        caption=rest, is_note=True, subsection=cur_subsection,
                        item_id=nid, anchor=md_anchor or nid))
            else:
                if stripped:
                    sec = ensure_section()
                    nid = _slug("note", used_slugs)
                    sec.items.append(Item(
                        kind="text", title="Note", caption=stripped,
                        is_note=True, subsection=cur_subsection,
                        item_id=nid, anchor=md_anchor or nid))
            continue

        if ctype != "code":
            continue

        directives, code = split_directives(source)
        group_key = (directives.get("group") or directives.get("tag") or "").strip()
        seen_before = bool(group_key) and group_key in group_index

        # only the first cell of a group steers section / subsection context
        if not seen_before:
            if "section" in directives:
                sec_name = directives["section"]
                existing = next(
                    (s for s in doc.sections if s.title == sec_name), None)
                if existing is None:
                    cur_section = Section(sec_name, _slug(sec_name, used_slugs))
                    doc.sections.append(cur_section)
                else:
                    cur_section = existing
                cur_subsection = ""
            if "subsection" in directives:
                cur_subsection = directives["subsection"]

        outputs = render_outputs(cell.get("outputs", []))
        try:
            order_val = float(directives.get("order", idx))
        except ValueError:
            order_val = float(idx)
        member = {"d": directives, "code": code, "outputs": outputs,
                  "order": order_val, "idx": idx,
                  "cell_id": str(cell.get("id") or "")}
        all_members.append(member)
        cell_id = directives.get("id", "").strip()
        if cell_id:
            cell_by_id.setdefault(cell_id, member)

        if seen_before:
            group_index[group_key].members.append(member)
            continue

        sec = ensure_section()
        item = Item(kind="", title="", subsection=cur_subsection, members=[member])
        sec.items.append(item)
        if group_key:
            group_index[group_key] = item

    # cells named in any `stack:` list are consumed (folded into figures,
    # not shown as their own card)
    consumed_ids: set[str] = set()
    for m in all_members:
        consumed_ids.update(_csv(m["d"].get("stack", "")))

    # resolve every code-derived card from its member cell(s) + stacked cells
    for sec in doc.sections:
        for item in sec.items:
            if not item.is_note and item.members:
                _finalize_item(item, used_slugs, cell_by_id)

    # drop consumed standalone cards, hidden cards, and empty sections
    for sec in doc.sections:
        sec.items = [
            it for it in sec.items
            if (it.is_note or it.kind not in ("", "hidden"))
            and not (it.node_id and it.node_id in consumed_ids)
        ]
    doc.sections = [s for s in doc.sections if s.items]
    _build_chains(doc)
    doc.raw_html = render_raw(nb)
    return doc


# --------------------------------------------------------------------------
# Provenance graph layout (layered top-down, fits a narrow rail)
# --------------------------------------------------------------------------

_NODE_FILL = {
    "dataset": "#2f6f9e",
    "transform": "#3b5566",
    "diagnostic": "#2f9bb0",
    "figure": "#2f9bb0",
    "metric": "#2c8c7d",
    "text": "#7a6a52",
    "code": "#4a5564",
}


def build_graph_svg(doc: Document, width: int = 268) -> str:
    """Return an SVG node-link diagram of items that declared an `id`."""
    nodes = [it for s in doc.sections for it in s.items if it.node_id]
    if len(nodes) < 2:
        return ""

    id_to_item = {it.node_id: it for it in nodes}
    # edges (dep -> item), ignoring references to unknown ids
    edges = [(d, it.node_id) for it in nodes for d in it.depends if d in id_to_item]

    # longest-path depth via memoised DFS over reverse edges
    parents: dict[str, list[str]] = {nid: [] for nid in id_to_item}
    for a, b in edges:
        parents[b].append(a)
    depth_cache: dict[str, int] = {}

    def depth(nid: str, stack: frozenset = frozenset()) -> int:
        if nid in depth_cache:
            return depth_cache[nid]
        if nid in stack or not parents[nid]:
            depth_cache[nid] = 0
            return 0
        d = 1 + max(depth(p, stack | {nid}) for p in parents[nid])
        depth_cache[nid] = d
        return d

    order = [it.node_id for it in nodes]  # document order
    layers: dict[int, list[str]] = {}
    for nid in order:
        layers.setdefault(depth(nid), []).append(nid)

    row_h, pad_top, pad_x = 64, 26, 16
    nh = 30
    max_depth = max(layers)
    height = pad_top * 2 + max_depth * row_h + nh
    pos: dict[str, tuple[float, float]] = {}
    for d, ids in layers.items():
        n = len(ids)
        usable = width - 2 * pad_x
        for i, nid in enumerate(ids):
            cx = pad_x + (usable * (i + 0.5) / n)
            cy = pad_top + d * row_h
            pos[nid] = (cx, cy)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" class="provsvg" role="img" '
        f'aria-label="Analysis provenance graph">']

    # edges first (under nodes), amber lineage curves
    for a, b in edges:
        ax, ay = pos[a]
        bx, by = pos[b]
        midy = (ay + nh / 2 + by - nh / 2) / 2
        parts.append(
            f'<path class="provedge" '
            f'd="M {ax:.1f} {ay + nh/2:.1f} '
            f'C {ax:.1f} {midy:.1f} {bx:.1f} {midy:.1f} '
            f'{bx:.1f} {by - nh/2:.1f}" '
            f'data-from="{a}" data-to="{b}"/>')

    # nodes
    for nid, it in id_to_item.items():
        cx, cy = pos[nid]
        fill = _NODE_FILL.get(it.kind, "#4a5564")
        label = it.node_id if len(it.node_id) <= 13 else it.node_id[:12] + "\u2026"
        bw = max(58, min(width - 2 * pad_x, len(label) * 7.2 + 16))
        x = cx - bw / 2
        y = cy - nh / 2
        parts.append(
            f'<g class="provnode" data-node="{it.node_id}" '
            f'data-target="{it.item_id}" tabindex="0" '
            f'role="button" aria-label="Go to {html.escape(it.title)}">'
            f'<rect x="{x:.1f}" y="{y:.1f}" rx="5" width="{bw:.1f}" '
            f'height="{nh}" fill="{fill}"/>'
            f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle">'
            f'{html.escape(label)}</text></g>')

    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Minimal Markdown for notes (bullets, bold/italic/code, paragraphs).
# Math ($...$ / $$...$$) is left as text for MathJax to typeset in-browser.
# --------------------------------------------------------------------------

_MD_CODE_RE = re.compile(r"`([^`]+)`")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_EM_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+")


def _md_with_headings(text: str) -> str:
    """md_to_html plus #-heading support, for the raw notebook view."""
    parts: list[str] = []
    plain: list[str] = []

    def flush() -> None:
        if plain:
            parts.append(md_to_html("\n".join(plain)))
            plain.clear()

    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            level = min(len(m.group(1)) + 1, 6)
            parts.append(f"<h{level}>{html.escape(m.group(2))}</h{level}>")
        else:
            plain.append(line)
    flush()
    return "".join(parts)


def render_raw(nb: dict) -> str:
    """Linear rendering of the notebook exactly as authored: every cell in
    order, code with its `#|` directives visible, outputs underneath.

    This is the transparency view -- it shows where the semantic page's
    titles, captions and sections come from.
    """
    parts: list[str] = []
    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        source = _as_text(cell.get("source", ""))
        if ctype == "markdown":
            parts.append(
                '<div class="rawcell md"><span class="rawtag">markdown</span>'
                f'<div class="rawmd">{_md_with_headings(source)}</div></div>')
        elif ctype == "code":
            n = cell.get("execution_count")
            label = f"In [{n if n is not None else ' '}]"
            outs = "".join(o.payload for o in
                           render_outputs(cell.get("outputs", [])))
            out_html = f'<div class="rawout">{outs}</div>' if outs else ""
            parts.append(
                f'<div class="rawcell code"><span class="rawtag">{label}'
                '</span><pre class="code"><code>'
                f'{highlight_python(source)}</code></pre>{out_html}</div>')
    return "".join(parts) or '<p class="rawempty">Empty notebook.</p>'


def md_to_html(text: str) -> str:
    def inline(s: str) -> str:
        s = _MD_CODE_RE.sub(r"<code>\1</code>", s)
        s = _MD_BOLD_RE.sub(r"<strong>\1</strong>", s)
        s = _MD_EM_RE.sub(r"<em>\1</em>", s)
        return s

    out: list[str] = []
    for block in re.split(r"\n\s*\n", html.escape(text)):
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        if all(_MD_BULLET_RE.match(ln) for ln in lines):
            lis = "".join(
                f"<li>{inline(_MD_BULLET_RE.sub('', ln))}</li>" for ln in lines)
            out.append(f"<ul>{lis}</ul>")
        else:
            out.append(f"<p>{inline('<br>'.join(lines))}</p>")
    return "".join(out)


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------

_BADGE = {
    "figure": "figure", "dataset": "dataset", "transform": "transform",
    "diagnostic": "diagnostic", "metric": "metric", "text": "note",
    "code": "code",
}


def _kind_class(kind: str) -> str:
    return {
        "figure": "k-figure", "diagnostic": "k-figure", "dataset": "k-dataset",
        "transform": "k-transform", "metric": "k-metric", "text": "k-note",
        "code": "k-code",
    }.get(kind, "k-code")


def _fig_pager(imgs) -> str:
    """Several figures from one cell: a pager, one figure at a time."""
    pages = "".join(
        f'<div class="figpage{" current" if i == 0 else ""}">{o.payload}'
        f'</div>' for i, o in enumerate(imgs))
    return (
        f'<div class="figpager" data-n="{len(imgs)}">{pages}'
        f'<div class="figpager-nav">'
        f'<button class="fp-btn fp-prev" title="Previous figure">'
        f'&#8249;</button>'
        f'<span class="fp-count">1 / {len(imgs)}</span>'
        f'<button class="fp-btn fp-next" title="Next figure">'
        f'&#8250;</button></div></div>')


def render_item(item: Item) -> str:
    badge = _BADGE.get(item.kind, item.kind)
    kclass = _kind_class(item.kind)
    # mixed outputs (e.g. a printed dataset THEN a plot): the figure is
    # the face; everything else folds into an "also printed" disclosure
    imgs = [o for o in item.outputs if o.has_image]
    others = [o for o in item.outputs if not o.has_image]
    fig_html = (_fig_pager(imgs) if len(imgs) > 1
                else "".join(o.payload for o in imgs))
    if imgs and others:
        out_html = fig_html + (
            '<details class="alsoprinted"><summary>also printed by '
            'this cell</summary><div class="alsoinner">'
            + "".join(o.payload for o in others)
            + "</div></details>")
    elif len(imgs) > 1:
        out_html = fig_html
    else:
        out_html = "".join(o.payload for o in item.outputs)

    # code: one or more labelled steps folded behind a single toggle
    code_block = ""
    steps = [s for s in item.steps if s.code.strip()]
    if steps and not item.is_note:
        multi = len(steps) > 1
        chunks = []
        for i, s in enumerate(steps, 1):
            label_html = ""
            if multi:
                lbl = html.escape(s.label)
                label_html = (
                    f'<div class="codestep-h"><span class="stepnum">{i}</span>'
                    f'<span class="steplabel">{lbl}</span></div>')
            extra_out = ""
            if multi and not s.is_primary and s.outputs:
                extra_out = ('<div class="codestep-out">'
                             + "".join(o.payload for o in s.outputs) + "</div>")
            chunks.append(
                f'<div class="codestep">{label_html}'
                f'<pre class="code"><code>{highlight_python(s.code)}</code></pre>'
                f"{extra_out}</div>")
        steps_count = (f'<span class="ct-steps">\u00b7 {len(steps)} steps</span>'
                       if multi else "")
        # a card with no output face IS its code: expanded, no toggle
        bare = not item.outputs
        is_open = item.code_visible or bare
        open_attr = " data-open='1'" if is_open else ""
        code_block = (
            f'<div class="codewrap{" bare" if bare else ""}"{open_attr}>'
            f'<button class="codetoggle" aria-expanded='
            f'"{"true" if is_open else "false"}">'
            f'<span class="chev">\u203a</span>'
            f'<span class="ct-show">Show code</span>'
            f'<span class="ct-hide">Hide code</span>{steps_count}</button>'
            f'<div class="codebody"><div class="codeinner">'
            f'{"".join(chunks)}</div></div></div>')

    caption = ""
    if item.caption:
        cap_html = html.escape(item.caption).replace("\n", "<br>")
        caption = f'<p class="caption">{cap_html}</p>'

    prov = ""
    if item.depends:
        chips = "".join(
            f'<a class="depchip" href="#" data-dep="{html.escape(d)}">{html.escape(d)}</a>'
            for d in item.depends)
        prov = f'<div class="prov"><span class="prov-l">derives from</span>{chips}</div>'

    id_tag = ""
    if item.node_id:
        id_tag = f'<span class="nodeid">{html.escape(item.node_id)}</span>'

    body = out_html
    if item.is_note:
        body = f'<div class="note">{md_to_html(item.caption)}</div>'
        caption = ""

    return (
        f'<article class="card {kclass}" id="card-{item.item_id}" '
        f'data-kind="{item.kind}" data-node="{item.node_id}" '
        f'data-note="{"1" if item.is_note else "0"}" '
        f'data-anchor="{html.escape(item.anchor or item.item_id)}" tabindex="-1">'
        f'<header class="cardhead">'
        f'<span class="badge">{badge}</span>'
        f'<h3 class="cardtitle{" echo" if item.title_echo else ""}">'
        f'{html.escape(item.title)}</h3>'
        f'{id_tag}</header>'
        f'<div class="cardbody">{body}</div>'
        f'{caption}{prov}{code_block}</article>')


def render_nav(doc: Document) -> str:
    parts = ['<nav class="nav" aria-label="Analysis sections">']
    for s in doc.sections:
        figs = sum(1 for it in s.items if it.kind in ("figure", "diagnostic"))
        parts.append(
            f'<a class="navsec" href="#sec-{s.section_id}" '
            f'data-sec="{s.section_id}">'
            f'<span class="navsec-t">{html.escape(s.title)}</span>'
            f'<span class="navsec-c">{figs or ""}</span></a>')
        parts.append('<div class="navitems">')
        last_sub = None
        for it in s.items:
            if it.subsection and it.subsection != last_sub:
                parts.append(
                    f'<div class="navsub">{html.escape(it.subsection)}</div>')
                last_sub = it.subsection
            dot = _kind_class(it.kind)
            parts.append(
                f'<a class="navitem {dot}" href="#card-{it.item_id}" '
                f'data-item="{it.item_id}">'
                f'<span class="dot"></span>'
                f'<span class="navitem-t">{html.escape(it.title)}</span></a>')
        parts.append('</div>')
    # key: one entry per item kind that actually occurs in this notebook
    labels = {"k-figure": "figure", "k-dataset": "dataset",
              "k-transform": "transform", "k-metric": "metric",
              "k-note": "note", "k-code": "code"}
    seen: list[str] = []
    for s in doc.sections:
        for it in s.items:
            kc = _kind_class(it.kind)
            if kc not in seen:
                seen.append(kc)
    if seen:
        parts.append('<div class="navkey"><span class="navkey-h">key</span>')
        for kc in seen:
            parts.append(f'<span class="nk {kc}"><span class="dot"></span>'
                         f'{labels.get(kc, kc)}</span>')
        parts.append('</div>')
    parts.append('</nav>')
    return "".join(parts)


def render_sections(doc: Document) -> str:
    """The stage content: every section with its cards. Reused by the widget."""
    sections_html: list[str] = []
    for s in doc.sections:
        cards = "".join(render_item(it) for it in s.items)
        sections_html.append(
            f'<section class="section" id="sec-{s.section_id}" '
            f'data-sec="{s.section_id}">'
            f'<div class="sectionhead"><span class="eyebrow">section</span>'
            f'<h2>{html.escape(s.title)}</h2></div>{cards}</section>')
    return "".join(sections_html)


def doc_meta(doc: Document) -> str:
    n_fig = sum(1 for s in doc.sections for it in s.items
                if it.kind in ("figure", "diagnostic"))
    n_data = sum(1 for s in doc.sections for it in s.items if it.kind == "dataset")
    return (f"{n_fig} figures \u00b7 {n_data} datasets "
            f"\u00b7 {len(doc.sections)} sections")


def deck_payload(doc: Document) -> str:
    """JSON blob embedded in the page: the card index + any saved deck.

    Slide payloads are NOT duplicated here -- the deck JS clones card DOM
    nodes (figures, notes, code) already present on the page.
    """
    items = []
    for s in doc.sections:
        for it in s.items:
            items.append({
                "anchor": it.anchor or it.item_id,
                "card": it.item_id,
                "title": it.title,
                "kind": "note" if it.is_note else it.kind,
                "section": s.section_id,
                "hasCode": any(st.code.strip() for st in it.steps),
                "chain": it.chain,
            })
    payload = {
        "title": doc.title,
        "meta": doc_meta(doc),
        "stem": doc.source_name,
        "sections": [{"id": s.section_id, "title": s.title}
                     for s in doc.sections],
        "items": items,
        "presentations": doc.presentations,
    }
    # "</" would terminate the inline <script> block early
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def render_graph_panel(doc: Document) -> str:
    graph_svg = build_graph_svg(doc)
    if not graph_svg:
        return ""
    return (
        '<div class="railgraph">'
        '<div class="railgraph-h"><span class="eyebrow">analysis graph</span>'
        '<button class="rg-collapse" aria-expanded="true" '
        'title="Collapse graph">\u2013</button></div>'
        f'<div class="railgraph-b">{graph_svg}</div></div>')


def render_shell(doc: Document, path: str = "") -> str:
    """One notebook's complete document view (rail + toolbar + cards).

    Several of these mount side by side as tabs; the embedded `nb-data`
    JSON is the card index the tab/deck JS consumes.
    """
    stem = doc.source_name or "notebook"
    path_attr = f' data-path="{html.escape(path)}"' if path else ""
    return _SHELL_TEMPLATE.format(
        stem=html.escape(stem),
        path_attr=path_attr,
        title=html.escape(doc.title),
        meta=html.escape(doc_meta(doc)),
        nav=render_nav(doc),
        graph_panel=render_graph_panel(doc),
        sections=render_sections(doc),
        rawview=doc.raw_html or "",
        nb_data=deck_payload(doc),
    )


def render_page(docs: list[Document], mode: str = "static",
                app_cfg: dict | None = None) -> str:
    """The full HTML page: tab strip, one shell per notebook, deck, app UI.

    mode "static": fixed tabs, shareable file (tab strip hidden when only
    one notebook). mode "app": served by the local server; tabs can be
    opened / closed / reloaded and presentations save to the project file.
    """
    cfg = app_cfg or {}
    paths = cfg.get("paths", {})
    shells = "".join(render_shell(d, path=paths.get(d.source_name, ""))
                     for d in docs)
    app_data = {
        "mode": mode,
        "token": cfg.get("token", ""),
        "root": cfg.get("root", ""),
        "project": {
            "presentations": cfg.get("presentations", []),
            "recent": cfg.get("recent", []),
        },
    }
    if len(docs) == 1:
        title = docs[0].title
    elif docs:
        title = f"{docs[0].title} (+{len(docs) - 1})"
    else:
        title = "PlotLine"
    return _TEMPLATE.format(
        title=html.escape(title),
        shells=shells,
        css=_CSS,
        app_css=_APP_CSS,
        js=_JS,
        mathjax=_MATHJAX,
        deck_shell=_DECK_HTML,
        app_data=json.dumps(app_data, ensure_ascii=False).replace("</", "<\\/"),
        deck_css=_DECK_CSS,
        deck_js=_DECK_JS,
        repo=_REPO_URL,
        kofi=_KOFI_URL,
        help_html=_HELP_HTML,
    )


def render_html(doc: Document, source_name: str | None = None) -> str:
    """Single-notebook page (kept for the widget and simple exports)."""
    if source_name:
        doc.source_name = source_name
    return render_page([doc])


# --------------------------------------------------------------------------
# Static assets
# --------------------------------------------------------------------------

_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Serif:ital,wght@0,400;1,400&display=swap');

:root{
  --ink:#16202b; --ink-2:#33414f; --ink-3:#69788a;
  --paper:#fbfcfd; --paper-2:#eef2f6; --paper-3:#e2e8ee;
  --line:#d8e0e8;
  --chrome:#11202c; --chrome-2:#16273544; --chrome-line:#ffffff14;
  --chrome-ink:#cdd9e3; --chrome-ink-2:#7e93a4;
  --cyan:#39a9c0; --cyan-deep:#1f7e93;
  --amber:#cf9a4e; --amber-soft:#caa06a66;
  --sans:'IBM Plex Sans',system-ui,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;
  --serif:'IBM Plex Serif',Georgia,serif;
  --rad:6px; --rail-w:300px;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;font-family:var(--sans);color:var(--ink);
  background:var(--paper-2);line-height:1.5;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;}

/* ---------- layout ---------- */
.shell{display:grid;grid-template-columns:var(--rail-w) 1fr;min-height:100vh;}
.rail{position:sticky;top:0;height:100vh;overflow-y:auto;
  background:var(--chrome);color:var(--chrome-ink);
  border-right:1px solid var(--chrome-line);
  display:flex;flex-direction:column;}
.stage{min-width:0;}

/* ---------- rail header ---------- */
.railhead{padding:22px 22px 16px;border-bottom:1px solid var(--chrome-line);}
.brand{font-family:var(--mono);font-size:10.5px;letter-spacing:.22em;
  text-transform:uppercase;color:var(--cyan);margin:0 0 12px;display:flex;
  align-items:center;gap:8px;}
.brand::before{content:"";width:7px;height:7px;border-radius:50%;
  background:var(--cyan);box-shadow:0 0 0 3px #39a9c029;}
.railtitle{font-size:18px;font-weight:600;line-height:1.25;margin:0;
  color:#eef4f8;letter-spacing:-.01em;}
.railmeta{font-family:var(--mono);font-size:10.5px;color:var(--chrome-ink-2);
  margin-top:8px;letter-spacing:.02em;}

/* ---------- nav ---------- */
.nav{padding:14px 12px 8px;flex:1 0 auto;}
.navsec{display:flex;justify-content:space-between;align-items:center;
  gap:8px;padding:9px 10px;margin-top:6px;border-radius:var(--rad);
  text-decoration:none;color:var(--chrome-ink);font-weight:600;font-size:13.5px;
  letter-spacing:-.005em;transition:background .15s,color .15s;}
.navsec:hover{background:#ffffff0c;}
.navsec.active{background:#39a9c014;color:#eef4f8;}
.navsec-t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.navsec-c{font-family:var(--mono);font-size:10px;color:var(--cyan);
  background:#39a9c016;border-radius:20px;padding:1px 7px;min-width:18px;
  text-align:center;}
.navsec-c:empty{display:none;}
.navitems{margin:2px 0 4px 4px;padding-left:8px;
  border-left:1px solid var(--chrome-line);}
.navsub{font-family:var(--mono);font-size:9.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--chrome-ink-2);
  padding:9px 10px 3px;}
.navitem{display:flex;align-items:center;gap:9px;padding:5px 10px;
  border-radius:5px;text-decoration:none;color:var(--chrome-ink-2);
  font-size:12.5px;transition:color .15s,background .15s;}
.navitem:hover{color:var(--chrome-ink);background:#ffffff08;}
.navitem.active{color:#eef4f8;}
.navitem-t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.navitem .dot{width:6px;height:6px;border-radius:2px;flex:none;
  background:var(--chrome-ink-2);}
.navitem.k-figure .dot,.nk.k-figure .dot{background:var(--cyan);}
.navitem.k-dataset .dot,.nk.k-dataset .dot{background:#4d90c0;}
.navitem.k-transform .dot,.nk.k-transform .dot{background:#5b7589;}
.navitem.k-metric .dot,.nk.k-metric .dot{background:#46a892;}
.navitem.k-note .dot,.nk.k-note .dot{background:var(--amber);
  border-radius:50%;}
.navitem.k-code .dot,.nk.k-code .dot{background:#56627033;
  border:1px solid #ffffff22;}

/* ---------- nav key (what the dot colours mean) ---------- */
.navkey{display:flex;flex-wrap:wrap;gap:4px 12px;align-items:center;
  padding:10px 12px 12px;margin:8px 10px 0;
  border-top:1px solid var(--chrome-line);}
.navkey-h{font-family:var(--mono);font-size:9.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--chrome-ink-2);flex:0 0 100%;}
.nk{display:inline-flex;align-items:center;gap:6px;font-size:11px;
  font-family:var(--mono);color:var(--chrome-ink-2);}
.nk .dot{width:6px;height:6px;border-radius:2px;flex:none;
  background:var(--chrome-ink-2);}

/* ---------- rail graph (signature) ---------- */
.railgraph{border-top:1px solid var(--chrome-line);padding:14px 14px 20px;
  margin-top:auto;background:#0c1822;}
.railgraph-h{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:8px;}
.railgraph .eyebrow{color:var(--amber);}
.rg-collapse{background:none;border:1px solid var(--chrome-line);
  color:var(--chrome-ink-2);width:22px;height:22px;border-radius:5px;
  cursor:pointer;font-size:14px;line-height:1;}
.rg-collapse:hover{color:var(--chrome-ink);border-color:#ffffff33;}
.railgraph-b{overflow:auto;max-height:46vh;transition:max-height .3s ease;}
.railgraph.collapsed .railgraph-b{max-height:0;overflow:hidden;}
.provsvg text{font-family:var(--mono);font-size:9.5px;fill:#dfeaf1;
  pointer-events:none;}
.provedge{fill:none;stroke:var(--amber-soft);stroke-width:1.4;
  transition:stroke .2s,stroke-width .2s;}
.provedge.lit{stroke:var(--amber);stroke-width:2.2;}
.provnode{cursor:pointer;}
.provnode rect{transition:filter .2s,stroke .2s;stroke:transparent;stroke-width:2;}
.provnode:hover rect{filter:brightness(1.18);}
.provnode.active rect{stroke:var(--cyan);filter:brightness(1.12)
  drop-shadow(0 0 6px #39a9c077);}
.provnode:focus-visible rect{stroke:var(--cyan);outline:none;}

/* ---------- toolbar ---------- */
.toolbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;
  gap:12px;padding:12px 28px;background:#fbfcfdf2;
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line);}
.menubtn{display:none;}
.tb-title{font-family:var(--mono);font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink-3);margin-right:auto;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.tb-actions{display:flex;gap:8px;flex:none;}
.toggle{font-family:var(--mono);font-size:11px;letter-spacing:.04em;
  border:1px solid var(--line);background:#fff;color:var(--ink-2);
  padding:7px 12px;border-radius:var(--rad);cursor:pointer;
  transition:all .15s;display:inline-flex;align-items:center;gap:7px;}
.toggle:hover{border-color:var(--cyan);color:var(--ink);}
.toggle[aria-pressed="true"]{background:var(--ink);color:#eef4f8;
  border-color:var(--ink);}
.toggle .tdot{width:6px;height:6px;border-radius:50%;background:currentColor;
  opacity:.4;}
.toggle[aria-pressed="true"] .tdot{opacity:1;background:var(--cyan);}
/* view-mode buttons (Docs / Present): pressed = the view you are in */
.toggle.mode[aria-pressed="true"]{background:var(--ink);color:#eef4f8;
  border-color:var(--ink);}
.tb-sep{width:1px;height:22px;background:var(--line);margin:0 4px;
  flex:none;}
/* per-type Show/Hide buttons: cyan dot = shown, dim dot = hidden */
.toggle.tv .tdot{opacity:1;background:var(--cyan);}
.toggle.tv.off .tdot{opacity:.3;background:currentColor;}
.toggle.tv.off{color:var(--ink-3);}

/* ---------- content ---------- */
.content{max-width:920px;margin:0 auto;padding:30px 28px 30vh;}
.section{margin-bottom:14px;scroll-margin-top:70px;}
.sectionhead{padding:24px 0 6px;margin-bottom:8px;
  border-bottom:1px solid var(--line);}
.eyebrow{font-family:var(--mono);font-size:10px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--cyan-deep);display:block;
  margin-bottom:6px;}
.sectionhead h2{font-size:26px;font-weight:600;margin:0;letter-spacing:-.02em;
  color:var(--ink);}

/* ---------- cards ---------- */
.card{background:var(--paper);border:1px solid var(--line);
  border-radius:10px;padding:18px 18px 16px;margin:14px 0;
  scroll-margin-top:78px;position:relative;
  box-shadow:0 1px 2px #1a26340a;
  opacity:0;transform:translateY(10px);
  transition:opacity .5s ease,transform .5s ease,box-shadow .2s,border-color .2s;}
.card.in{opacity:1;transform:none;}
.card:hover{box-shadow:0 6px 22px #1a26341a;}
.card::before{content:"";position:absolute;left:0;top:16px;bottom:16px;
  width:3px;border-radius:3px;background:var(--line);
  transition:background .2s;}
.card.k-figure::before{background:var(--cyan);}
.card.k-dataset::before{background:#4d90c0;}
.card.k-transform::before{background:#5b7589;}
.card.k-metric::before{background:#46a892;}
.card.k-note::before{background:var(--amber);}
.card.k-code::before{background:var(--paper-3);}
.card.target-flash{border-color:var(--cyan);box-shadow:0 0 0 3px #39a9c033;}

/* filtered view: non-matching cards collapse to expandable stubs */
.card.is-stub{padding:7px 14px;margin:7px 0;background:var(--paper);
  border-style:dashed;}
.card.is-stub .cardhead{margin-bottom:0;cursor:pointer;user-select:none;}
.card.is-stub.stub-open{padding:18px 18px 16px;border-style:solid;}
.card.is-stub.stub-open .cardhead{margin-bottom:12px;}
.card.is-stub:not(.stub-open) .cardbody,
.card.is-stub:not(.stub-open) .caption,
.card.is-stub:not(.stub-open) .prov,
.card.is-stub:not(.stub-open) .codewrap{display:none;}
.card.is-stub:not(.stub-open) .cardtitle{font-size:13px;font-weight:500;
  color:var(--ink-3);}
.card.is-stub:not(.stub-open)::before{top:9px;bottom:9px;}
.card.is-stub .cardhead::after{content:"\203A";margin-left:auto;flex:none;
  color:var(--ink-3);font-size:15px;line-height:1;transition:transform .2s;}
.card.is-stub.stub-open .cardhead::after{transform:rotate(90deg);}
.card.is-stub .cardhead:hover .cardtitle{color:var(--ink);}

.cardhead{display:flex;align-items:center;gap:10px;margin-bottom:12px;
  padding-left:6px;}
.badge{font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;
  text-transform:uppercase;padding:3px 8px;border-radius:4px;
  background:var(--paper-2);color:var(--ink-3);flex:none;}
.k-figure .badge{background:#39a9c014;color:var(--cyan-deep);}
.k-dataset .badge{background:#4d90c014;color:#2f6f9e;}
.k-transform .badge{background:#5b758914;color:#41566a;}
.k-metric .badge{background:#46a89214;color:#2c8c7d;}
.k-note .badge{background:#cf9a4e1f;color:#8a6326;}
.cardtitle{font-size:16px;font-weight:600;margin:0;letter-spacing:-.01em;
  flex:1;min-width:0;}
/* titles that merely echo the first code line label the item in the
   nav, but are not repeated as a heading on the card */
.card:not(.is-stub) .cardtitle.echo{display:none;}
.nodeid{font-family:var(--mono);font-size:10px;color:var(--ink-3);
  background:var(--paper-2);padding:2px 7px;border-radius:4px;flex:none;}

.cardbody{padding-left:6px;}
.figframe{background:#fff;border:1px solid var(--paper-3);border-radius:8px;
  padding:8px;overflow:auto;text-align:center;}
.figframe img{max-width:100%;height:auto;display:block;margin:0 auto;}
.figframe svg{max-width:100%;height:auto;}

/* several figures from one cell: pager with prev/next arrows */
.figpager .figpage{display:none;}
.figpager .figpage.current{display:block;}
.figpager-nav{display:flex;align-items:center;justify-content:center;
  gap:10px;margin-top:6px;}
.fp-btn{font-family:var(--mono);font-size:15px;line-height:1;
  border:1px solid var(--paper-3);background:#fff;color:var(--ink-2);
  border-radius:6px;width:28px;height:22px;cursor:pointer;padding:0;}
.fp-btn:hover{border-color:var(--cyan);color:var(--ink);}
.fp-count{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);}

/* huge markdown notes: clamped with a Show more toggle */
.cardbody.mdclamp{max-height:440px;overflow:hidden;position:relative;}
.cardbody.mdclamp::after{content:"";position:absolute;left:0;right:0;
  bottom:0;height:64px;pointer-events:none;
  background:linear-gradient(#fbfcfd00,var(--paper));}
.cardbody.mdclamp.mdopen{max-height:none;}
.cardbody.mdclamp.mdopen::after{display:none;}
.mdmore{display:block;margin:8px 0 0 6px;font-family:var(--mono);
  font-size:10.5px;border:1px solid var(--line);background:#fff;
  color:var(--cyan-deep);border-radius:6px;padding:4px 12px;
  cursor:pointer;}
.mdmore:hover{border-color:var(--cyan);color:var(--ink);}

pre.result,pre.stream,pre.error{font-family:var(--mono);font-size:12px;
  background:var(--paper-2);border:1px solid var(--paper-3);
  border-radius:7px;padding:11px 13px;overflow:auto;margin:0;line-height:1.45;}
pre.error{background:#fbf0ee;border-color:#f0d2cc;color:#8a3221;}
.card.k-metric .cardbody pre.result{font-size:14px;
  background:#46a8920d;border-color:#46a89233;color:#1f5f54;
  font-weight:500;}

.note{font-family:var(--serif);font-size:15px;line-height:1.65;
  color:var(--ink-2);}
.note .caption{font-family:var(--serif);font-style:normal;color:var(--ink-2);
  margin:0;padding:0;border:none;font-size:15px;}

.caption{font-family:var(--serif);font-size:14px;
  color:var(--ink-2);margin:13px 0 0;padding-left:6px;line-height:1.6;}

details.alsoprinted{margin-top:10px;border:1px dashed var(--paper-3);
  border-radius:7px;}
details.alsoprinted>summary{cursor:pointer;font-family:var(--mono);
  font-size:10.5px;letter-spacing:.08em;color:var(--ink-3);
  padding:7px 11px;user-select:none;}
details.alsoprinted[open]>summary{
  border-bottom:1px solid var(--paper-3);}
details.alsoprinted .alsoinner{padding:10px 11px;}

.prov{display:flex;align-items:center;gap:7px;flex-wrap:wrap;
  margin:13px 0 0;padding-left:6px;}
.prov-l{font-family:var(--mono);font-size:9.5px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--amber);}
.depchip{font-family:var(--mono);font-size:11px;color:var(--ink-2);
  text-decoration:none;background:#cf9a4e14;border:1px solid #cf9a4e33;
  padding:2px 9px;border-radius:20px;transition:all .15s;}
.depchip:hover{background:var(--amber);color:#fff;border-color:var(--amber);}

/* ---------- code ---------- */
.codewrap{margin:13px 0 0;border-top:1px solid var(--line);padding-top:11px;}
.codewrap.bare{border-top:none;padding-top:0;margin-top:10px;}
.codewrap.bare .codetoggle{display:none;}
.codetoggle{font-family:var(--mono);font-size:11px;letter-spacing:.05em;
  color:var(--ink-3);background:none;border:none;cursor:pointer;padding:2px 6px;
  display:inline-flex;align-items:center;gap:7px;border-radius:5px;
  transition:color .15s;}
.codetoggle:hover{color:var(--cyan-deep);}
.codetoggle .chev{display:inline-block;transition:transform .2s;font-size:14px;}
.ct-hide{display:none;}
.codewrap[data-open] .codetoggle .chev{transform:rotate(90deg);}
.codewrap[data-open] .ct-show{display:none;}
.codewrap[data-open] .ct-hide{display:inline;}
.codebody{display:grid;grid-template-rows:0fr;transition:grid-template-rows .28s ease;}
.codewrap[data-open] .codebody{grid-template-rows:1fr;}
.codebody>.codeinner,.codeinner{overflow:hidden;min-height:0;}
.codestep{margin-top:14px;}
.codestep:first-child{margin-top:0;}
.codestep-h{display:flex;align-items:center;gap:9px;margin:0 0 7px;}
.stepnum{font-family:var(--mono);font-size:10px;font-weight:600;
  width:19px;height:19px;border-radius:5px;display:inline-flex;
  align-items:center;justify-content:center;background:#cf9a4e1f;
  color:#8a6326;flex:none;}
.steplabel{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);}
.codestep-out{margin-top:9px;}
.codestep pre.code{margin:0;}
.ct-steps{margin-left:9px;color:var(--ink-3);}
pre.code{font-family:var(--mono);font-size:12.5px;line-height:1.55;
  background:#0e1b25;color:#c9d6e0;border-radius:8px;padding:14px 16px;
  margin:9px 0 2px;overflow:auto;}
pre.code .kw{color:#6bb8d6;}
pre.code .bn{color:#86c5a8;}
pre.code .st{color:#d8a36a;}
pre.code .nu{color:#c98fd0;}
pre.code .co{color:#5d7185;}
pre.code .op{color:#9fb1c0;}

/* ---------- xarray repr ---------- */
.xr-wrap{font-size:13px;overflow:auto;border:1px solid var(--paper-3);
  border-radius:8px;padding:4px 8px;background:#fff;}
.xr-wrap .xr-array-wrap,.xr-wrap .xr-var-list{font-family:var(--mono);}

/* ---------- empty / fallback ---------- */
.rich{overflow:auto;}
.rich table{border-collapse:collapse;font-size:13px;}
.rich th,.rich td{border:1px solid var(--line);padding:4px 8px;}

/* ---------- focus ---------- */
a:focus-visible,button:focus-visible,.toggle:focus-visible{
  outline:2px solid var(--cyan);outline-offset:2px;}

/* ---------- responsive ---------- */
@media (max-width:860px){
  .shell{grid-template-columns:1fr;}
  .rail{position:fixed;left:0;top:0;width:min(86vw,330px);z-index:60;
    transform:translateX(-102%);transition:transform .3s ease;
    box-shadow:0 0 40px #00000055;}
  .rail.open{transform:none;}
  .menubtn{display:inline-flex;align-items:center;justify-content:center;
    width:36px;height:36px;border:1px solid var(--line);background:#fff;
    border-radius:var(--rad);cursor:pointer;flex:none;}
  .menubtn span,.menubtn span::before,.menubtn span::after{content:"";
    display:block;width:16px;height:2px;background:var(--ink);position:relative;}
  .menubtn span::before{position:absolute;top:-5px;}
  .menubtn span::after{position:absolute;top:5px;}
  .scrim{position:fixed;inset:0;background:#0a131b66;z-index:55;display:none;}
  .scrim.show{display:block;}
  .content{padding:22px 18px 30vh;}
  .sectionhead h2{font-size:22px;}
}

@media (prefers-reduced-motion:reduce){
  *{transition:none!important;scroll-behavior:auto!important;}
  .card{opacity:1;transform:none;}
}
"""

# --------------------------------------------------------------------------
# Help overlay -- "how to use / what it can do", shown in every mode
# --------------------------------------------------------------------------

_HELP_HTML = r"""
<h3>What this is</h3>
<p>A <b>figure-first view of executed Jupyter notebooks</b>. Instead of a
wall of cells, you get the scientific structure: figures, datasets and
notes as cards, code folded underneath, sections in a sidebar, and a
provenance graph of how each result derives from the data. On top of
that sits a <b>presentation builder</b>: turn any notebooks into slides
and present straight from the browser.</p>
<p>Everything runs locally &mdash; in the web version, notebooks are
processed <i>in your browser</i> and never uploaded anywhere.</p>

<h3>Open notebooks</h3>
<ul>
<li><b>Drag &amp; drop</b> one or more <code>.ipynb</code> files
anywhere onto the window.</li>
<li><b>+ Open</b> (top left) &mdash; a file picker, or paste a
<b>URL</b> to a notebook (GitHub links are converted automatically).</li>
<li>Notebooks must be <b>executed</b> (run once in Jupyter so outputs
are saved) &mdash; nothing is re-run here.</li>
<li>Every notebook is a <b>tab</b>: click to switch, <b>&#8635;</b>
re-reads it after you re-run it in Jupyter, <b>&#10005;</b> closes.</li>
</ul>

<h3>Reading a notebook</h3>
<ul>
<li><b>Hide figures / markup / code</b> (top bar) filter every tab at
once; <i>Show code</i> unfolds ALL code, including the code tucked
under each figure. Hidden cards become slim stubs that expand when
clicked.</li>
<li><b>Raw notebook</b> flips the current tab to the notebook exactly
as authored &mdash; cells in order, directives visible &mdash; so you
can always see where a title or caption came from.</li>
<li>The left sidebar navigates sections; the <b>analysis graph</b> at
its foot jumps to any node. <i>derives from</i> chips under a card jump
to its inputs.</li>
<li><b>Show code</b> under a figure tells the whole story: every
upstream cell (load &rarr; transform &rarr; plot) in execution order,
traced automatically from the code's variables.</li>
</ul>

<h3>Make notebooks render better (optional)</h3>
<p>Add <code>#|</code> directive lines to the top of a code cell; they
are parsed and hidden from display. Everything works without them
&mdash; they are how you take control:</p>
<table>
<tr><td><code>#| title:</code></td><td>card title (else inferred from
the first comment)</td></tr>
<tr><td><code>#| caption:</code></td><td>interpretation text under the
output</td></tr>
<tr><td><code>#| section:</code></td><td>start a section (or use a
markdown <code>##</code> heading)</td></tr>
<tr><td><code>#| id:</code></td><td>stable name; makes the cell a node
in the graph and a reliable slide anchor</td></tr>
<tr><td><code>#| depends: a, b</code></td><td>declare inputs; draws
the graph edges</td></tr>
<tr><td><code>#| display:</code></td><td>force a card type: figure,
dataset, metric, text, code, hidden</td></tr>
<tr><td><code>#| group:</code> / <code>#| stack:</code></td><td>fold
several cells under one figure (see the README for details)</td></tr>
</table>

<h3>Presentations</h3>
<ul>
<li>The <b>left rail</b> lists presentations under a <b>Documents</b>
button &mdash; exactly one is active, so that button is always the way
back. <b>New</b> starts one; <b>&#171;</b> shrinks or hides the
rail.</li>
<li>In the <b>builder</b>, pick a slide layout (diagrams: full, halves,
rows, quarters, a <b>title slide</b>, or a <b>blank canvas</b>), click
a pane, then <b>click any card in the document</b> to place it &mdash;
from <i>any</i> open tab, so one deck can mix several notebooks.</li>
<li><b>&#9998; Edit slide</b> opens a PowerPoint-style editor:
<b>+ Text</b>, <b>+ Arrow</b>, <b>+ Box</b>, and <b>+ Cell</b> &mdash;
a draggable, resizable frame that holds any notebook card (click it,
pick a card, swap later with &#8644; Replace). Select anything for
colours, text size, line thickness, dash and fill.</li>
<li><b>&#9654; Present</b> plays full screen. Arrow keys &larr;/&rarr;
move through the story; on slides with code, &darr; descends the
<b>code trail</b> &mdash; every cell that made the figure, one per
screen, in execution order &mdash; and &uarr; climbs back out.</li>
</ul>

<h3>Saving</h3>
<ul>
<li>Edits <b>autosave as drafts</b> in your browser as you work.</li>
<li>Desktop app: presentations autosave to
<code>plotline_project.json</code> next to where you launched it,
along with your open tabs.</li>
<li>Anywhere: <i>File &rarr; Download JSON</i> saves a deck as a
file on your machine; <i>File &rarr; Load deck JSON</i> brings it back
&mdash; later, or on another computer.</li>
<li>Decks are robust to notebook edits: slides reference cells by
<b>stable ids</b>, never position &mdash; re-upload an edited notebook
and everything still resolves; a deleted cell just leaves an empty
frame you can refill.</li>
</ul>

<h3>Run it locally</h3>
<p>The whole tool is one Python file with no dependencies. For daily
use &mdash; local file browsing, project files, session restore:
<code>pip install</code> the repo and run <code>plotline</code>,
or just download <code>semantic_render.py</code> and run
<code>python semantic_render.py</code>.</p>
"""

# App chrome (controls bar + tab rows), welcome screen, open dialog,
# drag-drop hint
_APP_CSS = r"""
:root{--appbar-h:44px;--tabsrow-h:44px;--chrome-h:88px;--dc-w:430px;
  --presrail-w:176px;}
body.presrail-min{--presrail-w:46px;}

/* ---------- row 1: global controls; row 2: notebook + presentation tabs */
.apptop{position:fixed;top:0;left:0;right:0;height:var(--chrome-h);
  z-index:90;display:flex;flex-direction:column;background:#0a141d;
  border-bottom:1px solid #ffffff14;}
.appbar{display:flex;align-items:center;gap:8px;height:var(--appbar-h);
  padding:0 12px 0 0;border-bottom:1px solid #ffffff0d;}
.apptop-brand{font-family:var(--mono);font-size:9.5px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--cyan);display:flex;align-items:center;
  align-self:stretch;padding:0 14px;border-right:1px solid #ffffff10;
  flex:none;}
.appbar-spring{flex:1;}
/* dark variants of the show/hide toggles */
.appbar .toggle{border-color:#ffffff22;background:#ffffff0a;color:#cdd9e3;}
.appbar-link{text-decoration:none;display:inline-flex;
  align-items:center;}
.appbar .toggle:hover{border-color:var(--cyan);color:#fff;}
.appbar .toggle.tv.off{color:#69788a;}
.appbar .menubtn{display:inline-flex;align-items:center;
  justify-content:center;width:34px;height:34px;
  border:1px solid #ffffff22;background:none;
  border-radius:var(--rad);cursor:pointer;flex:none;}
.appbar .menubtn span,.appbar .menubtn span::before,
.appbar .menubtn span::after{content:"";display:block;width:15px;
  height:2px;background:#cdd9e3;position:relative;}
.appbar .menubtn span::before{position:absolute;top:-5px;}
.appbar .menubtn span::after{position:absolute;top:5px;}
.appbar .menubtn[aria-pressed="true"]{background:#39a9c022;
  border-color:#39a9c088;}
.tab-openbtn{font-family:var(--mono);font-size:11px;background:none;
  border:1px solid #ffffff22;border-radius:var(--rad);color:var(--cyan);
  padding:7px 14px;cursor:pointer;white-space:nowrap;flex:none;}
.tab-openbtn:hover{background:#39a9c01a;border-color:var(--cyan);}
.tab-openbtn[hidden]{display:none;}

.tabsrow{display:flex;align-items:stretch;height:var(--tabsrow-h);
  background:#0d1a26;}
.tabstrip{display:flex;align-items:stretch;overflow-x:auto;
  min-width:0;scrollbar-width:thin;flex:0 1 auto;
  gap:5px;padding:6px 8px 0;}
.tab{display:flex;align-items:center;gap:8px;padding:0 10px 0 15px;
  max-width:260px;min-width:0;cursor:pointer;user-select:none;
  font-size:13px;color:#96a9ba;background:#ffffff08;
  border:1px solid #ffffff14;border-bottom:none;
  border-radius:9px 9px 0 0;
  white-space:nowrap;transition:background .12s,color .12s;}
.tab:hover{background:#ffffff12;color:#cdd9e3;}
.tab.current{background:#0b141d;color:#e6edf3;font-weight:600;
  border-color:#ffffff1f;}
.tab-t{overflow:hidden;text-overflow:ellipsis;max-width:200px;}
.tab-b{background:none;border:none;color:inherit;opacity:.55;
  cursor:pointer;font-size:13px;padding:4px 6px;border-radius:5px;
  line-height:1;flex:none;}
.tab-b:hover{opacity:1;background:#00000033;}
.tabs-label{font-family:var(--mono);font-size:8.5px;letter-spacing:.18em;
  text-transform:uppercase;color:#54677a;display:flex;align-items:center;
  padding:0 10px 0 14px;flex:none;user-select:none;}

/* ---------- presentations rail: vertical stack on the left edge.
   Exactly ONE item is active at a time: "Documents" (no builder) or a
   presentation (builder open) — so the way out is always visible. */
.presrail{position:fixed;left:0;top:0;bottom:0;width:var(--presrail-w);
  z-index:95;background:#0a141d;border-right:1px solid #ffffff1f;
  display:flex;flex-direction:column;padding:8px 6px;gap:2px;}
.pr-item{display:flex;align-items:center;gap:9px;width:100%;
  background:none;border:none;border-radius:7px;padding:9px 10px;
  font-family:var(--sans);font-size:12.5px;color:#8ba0b2;cursor:pointer;
  text-align:left;min-width:0;transition:background .12s,color .12s;}
.pr-item:hover{background:#ffffff0c;color:#cdd9e3;}
.pr-item.current{background:#39a9c022;color:#eef4f8;font-weight:600;}
.pr-item.editing{background:var(--cyan-deep);color:#fff;font-weight:600;}
.pr-ico{font-size:11px;flex:none;width:16px;text-align:center;
  opacity:.85;}
.pr-item.ptab .pr-ico{font-size:8.5px;color:var(--cyan);}
.pr-item.editing .pr-ico{color:#fff;}
.pr-t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;}
.pr-docs{margin-bottom:6px;}
.pr-label{font-family:var(--mono);font-size:8.5px;letter-spacing:.18em;
  text-transform:uppercase;color:#4e93a6;padding:10px 10px 6px;
  user-select:none;white-space:nowrap;overflow:hidden;}
.pr-list{display:flex;flex-direction:column;gap:2px;overflow-y:auto;
  min-height:0;flex:0 1 auto;}
/* real buttons for the create actions */
.pr-btn{display:flex;align-items:center;justify-content:center;gap:7px;
  width:100%;background:#ffffff08;border:1px solid #ffffff22;
  border-radius:7px;padding:8px 10px;font-family:var(--mono);
  font-size:10.5px;letter-spacing:.03em;color:#9fb2c2;cursor:pointer;
  margin-top:6px;transition:border-color .15s,color .15s,
  background .15s;}
.pr-btn:hover{border-color:var(--cyan);color:#fff;
  background:#39a9c014;}
.pr-btn .pr-ico{display:none;}
body.presrail-min .pr-btn .pr-t{display:none;}
body.presrail-min .pr-btn .pr-ico{display:flex;align-items:center;
  justify-content:center;}
/* folder icon next to folder names */
.pr-fico{display:flex;align-items:center;color:#7590a5;flex:none;}
.pr-folder:hover .pr-fico{color:#9fb2c2;}
body.presrail-min .pr-fico{display:none;}
.pr-collapse{margin-top:auto;background:none;border:1px solid #ffffff1f;
  border-radius:7px;color:#69788a;font-size:13px;padding:5px 0;
  cursor:pointer;}
.pr-collapse:hover{color:#cdd9e3;border-color:#ffffff40;}
/* collapsed: icons only */
body.presrail-min .pr-t,body.presrail-min .pr-label{display:none;}
body.presrail-min .pr-item{justify-content:center;padding:9px 0;}
body.presrail-min .pr-ico{width:auto;}
/* fully hidden: a small edge handle brings it back */
body.presrail-hidden{--presrail-w:0px;}
body.presrail-hidden .presrail{display:none;}
.presrail-show{position:fixed;left:0;bottom:20px;z-index:96;width:22px;
  height:46px;border:1px solid #ffffff22;border-left:none;
  border-radius:0 8px 8px 0;background:#0a141d;color:#7fb6c6;
  cursor:pointer;display:none;font-size:12px;padding:0;}
body.presrail-hidden .presrail-show{display:block;}
.presrail-show:hover{color:#fff;border-color:var(--cyan);}
/* draft-only presentations get an unsaved dot */
.pr-item.draftonly .pr-t::after{content:" \2022";color:var(--amber);}
/* presentation folders: real folders — drag items in/out, collapsible */
.pr-folder{display:flex;align-items:center;gap:7px;width:100%;
  border:1px solid transparent;background:none;border-radius:7px;
  padding:7px 10px;margin-top:5px;font-family:var(--mono);font-size:10px;
  letter-spacing:.1em;text-transform:uppercase;color:#5e7488;
  cursor:pointer;text-align:left;min-width:0;user-select:none;}
.pr-folder:hover{background:#ffffff0a;color:#9fb2c2;}
.pr-folder.dropping{border-color:var(--cyan);background:#39a9c01c;
  color:#aadbe8;}
.pr-folder .pr-t{flex:1;}
.pr-fchev{flex:none;font-size:9px;}
.pr-fcount{flex:none;font-size:9px;background:#ffffff10;
  border-radius:8px;padding:1px 6px;color:#69788a;}
.pr-fctrl{display:none;gap:2px;flex:none;}
.pr-folder:hover .pr-fctrl{display:flex;}
.pr-fctrl button{background:none;border:none;color:#8ba0b2;
  cursor:pointer;font-size:10px;padding:1px 4px;border-radius:4px;}
.pr-fctrl button:hover{color:#fff;background:#ffffff14;}
.pr-frename{width:100%;background:#16273a;border:1px solid var(--cyan);
  color:#dce6ee;font-family:var(--sans);font-size:12px;padding:3px 7px;
  border-radius:5px;min-width:0;}
.pr-frename:focus{outline:none;}
.pr-item.infolder{padding-left:26px;}
.pr-item.ptab[draggable="true"]{cursor:grab;}
.pr-item.ptab.dragging{opacity:.45;}
.presrail.dropping-root{outline:2px dashed #39a9c066;
  outline-offset:-4px;}
body.presrail-min .pr-folder .pr-t,
body.presrail-min .pr-fcount,body.presrail-min .pr-fctrl{display:none;}
body.presrail-min .pr-folder{justify-content:center;padding:8px 0;}
body.presrail-min .pr-item.infolder{padding-left:0;}

.nbshell[hidden]{display:none;}
body{padding-top:var(--chrome-h);padding-left:var(--presrail-w);}
.apptop{left:var(--presrail-w);}
.welcome{left:var(--presrail-w);}
.rail{top:var(--chrome-h);height:calc(100vh - var(--chrome-h));}
.section{scroll-margin-top:calc(var(--chrome-h) + 12px);}
.card{scroll-margin-top:calc(var(--chrome-h) + 18px);}
/* builder docked: full-height panel right of the rail; tab / controls
   chrome shifts right so it sits above the DOCUMENT (IDE-style) */
.deck.creating{left:var(--presrail-w);}
body.creating-docs .apptop{
  left:calc(var(--presrail-w) + min(var(--dc-w),94vw));}
@media (max-width:860px){
  .deck.creating{top:var(--chrome-h);}
  body.creating-docs .apptop{left:var(--presrail-w);}
}

/* ---------- welcome (app mode, nothing open) ---------- */
.welcome{position:fixed;left:0;right:0;top:var(--chrome-h);bottom:0;
  display:flex;align-items:center;justify-content:center;
  background:var(--paper-2);z-index:5;overflow:auto;}
.welcome[hidden]{display:none;}
.welcome-box{text-align:center;max-width:460px;padding:40px 30px;}
.welcome-box .brand{justify-content:center;}
.welcome-box h1{font-size:26px;letter-spacing:-.02em;margin:12px 0 8px;
  color:var(--ink);}
.welcome-hint{color:var(--ink-3);font-size:13.5px;line-height:1.6;
  margin:0 0 18px;}
.recent{margin-top:26px;display:flex;flex-direction:column;gap:6px;
  text-align:left;}
.recent-h{font-family:var(--mono);font-size:9.5px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--ink-3);margin-bottom:2px;}
.recent-i{font-family:var(--mono);font-size:11.5px;color:var(--cyan-deep);
  background:#fff;border:1px solid var(--line);padding:8px 11px;
  border-radius:6px;cursor:pointer;text-align:left;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;direction:rtl;}
.recent-i:hover{border-color:var(--cyan);}
.welcome-btns{display:flex;gap:8px;justify-content:center;
  flex-wrap:wrap;}
.welcome-btns .dbtn{border-color:var(--line);background:#fff;
  color:var(--ink-2);}
.welcome-btns .dbtn:hover{border-color:var(--cyan);color:var(--ink);}
.welcome-btns .dbtn.primary{background:var(--cyan-deep);
  border-color:var(--cyan-deep);color:#fff;}
.welcome-btns .dbtn.primary:hover{background:var(--cyan);}
.welcome-links{margin-top:16px;font-size:12.5px;color:var(--ink-3);}
.welcome-links a{color:var(--cyan-deep);text-decoration:none;}
.welcome-links a:hover{text-decoration:underline;}
.wl-sep{margin:0 7px;}

/* ---------- help overlay: how to use / what it can do ---------- */
.helpdlg{position:fixed;inset:0;z-index:135;background:#0a131b88;
  display:flex;align-items:center;justify-content:center;padding:24px;}
.helpdlg[hidden]{display:none;}
.help-box{width:min(760px,94vw);height:min(720px,90vh);
  background:var(--paper);border-radius:12px;display:flex;
  flex-direction:column;overflow:hidden;
  box-shadow:0 24px 80px #00000066;}
.help-head{display:flex;align-items:center;gap:12px;
  padding:13px 18px;border-bottom:1px solid var(--line);}
.help-title{font-family:var(--mono);font-size:11px;
  letter-spacing:.18em;text-transform:uppercase;
  color:var(--cyan-deep);font-weight:600;}
.help-gh{font-size:12.5px;color:var(--cyan-deep);text-decoration:none;}
.help-gh:hover{text-decoration:underline;}
.help-head .dbtn{border-color:var(--line);background:#fff;
  color:var(--ink-2);}
.help-head .dbtn:hover{border-color:var(--cyan);color:var(--ink);}
.help-body{flex:1;overflow-y:auto;padding:6px 26px 30px;
  color:var(--ink-2);font-size:13.5px;line-height:1.65;}
.help-body h3{font-size:15px;color:var(--ink);letter-spacing:-.01em;
  margin:24px 0 8px;padding-top:14px;
  border-top:1px solid var(--paper-3);}
.help-body h3:first-child{border-top:none;margin-top:8px;}
.help-body ul{margin:6px 0;padding-left:20px;}
.help-body li{margin:5px 0;}
.help-body a{color:var(--cyan-deep);}
.help-body code{font-family:var(--mono);font-size:12px;
  background:var(--paper-2);border:1px solid var(--paper-3);
  border-radius:4px;padding:1px 5px;}
.help-body table{border-collapse:collapse;margin:8px 0;width:100%;}
.help-body td{border:1px solid var(--paper-3);padding:6px 10px;
  vertical-align:top;}
.help-body td:first-child{white-space:nowrap;}

/* ---------- open dialog (app mode file browser) ---------- */
.opendlg{position:fixed;inset:0;z-index:130;background:#0a131b88;
  display:flex;align-items:center;justify-content:center;padding:24px;}
.opendlg[hidden]{display:none;}
.odlg-box{width:min(580px,94vw);height:min(620px,86vh);
  background:var(--paper);border-radius:12px;display:flex;
  flex-direction:column;overflow:hidden;box-shadow:0 24px 80px #00000066;}
.odlg-head{display:flex;align-items:center;gap:10px;padding:12px 14px;
  border-bottom:1px solid var(--line);}
.odlg-head .dbtn{border-color:var(--line);background:#fff;
  color:var(--ink-2);}
.odlg-head .dbtn:hover{border-color:var(--cyan);color:var(--ink);}
.odlg-path{flex:1;font-family:var(--mono);font-size:11px;color:var(--ink-3);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  direction:rtl;text-align:left;}
.odlg-list{flex:1;overflow-y:auto;padding:8px;}
.odlg-i{display:flex;align-items:center;gap:10px;width:100%;
  background:none;border:none;font-family:var(--sans);font-size:13px;
  color:var(--ink-2);padding:8px 10px;border-radius:6px;cursor:pointer;
  text-align:left;}
.odlg-i:hover{background:var(--paper-2);color:var(--ink);}
.odlg-i .ic{font-size:13px;flex:none;width:20px;text-align:center;}
.odlg-i.nb{color:var(--cyan-deep);font-weight:500;}
.odlg-i .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  flex:1;min-width:0;}
.odlg-i .sz{font-family:var(--mono);font-size:10px;color:var(--ink-3);
  flex:none;}
.odlg-empty{padding:26px;text-align:center;color:var(--ink-3);
  font-size:12.5px;}
.odlg-foot{border-top:1px solid var(--line);padding:10px 12px;}
#odlg-input{width:100%;box-sizing:border-box;font-family:var(--mono);
  font-size:11.5px;border:1px solid var(--line);border-radius:6px;
  padding:8px 10px;background:#fff;color:var(--ink);}
#odlg-input:focus{outline:none;border-color:var(--cyan);}
.odlg-inrow{display:flex;gap:8px;align-items:stretch;}
.odlg-inrow #odlg-input{flex:1;min-width:0;}
#odlg-go{flex:none;font-weight:600;padding:8px 16px;
  border-color:var(--cyan);color:var(--cyan-deep);background:#fff;}
#odlg-go:hover:not(:disabled){background:var(--cyan);color:#fff;}
#odlg-go:disabled,#odlg-input:disabled{opacity:.55;cursor:default;}
.odlg-load{height:3px;margin-top:8px;border-radius:2px;overflow:hidden;
  background:#39a9c022;position:relative;}
.odlg-load[hidden]{display:none;}
.odlg-load span{position:absolute;left:-40%;top:0;width:40%;height:100%;
  background:var(--cyan);border-radius:2px;
  animation:odlg-slide 1.1s ease-in-out infinite;}
@keyframes odlg-slide{to{left:100%;}}

/* ---------- section sidebar (TOC): hidden until ☰ toggles it ------- */
@media(min-width:861px){
  body:not(.tocshow) .shell{grid-template-columns:1fr;}
  body:not(.tocshow) .nbshell .rail{display:none;}
}

/* ---------- instant tooltips (replaces slow native titles) -------- */
.apptip{position:fixed;z-index:300;background:#0e1926;color:#dce6ee;
  font-family:var(--sans);font-size:11.5px;line-height:1.45;
  padding:6px 10px;border-radius:7px;border:1px solid #39a9c055;
  box-shadow:0 6px 24px #00000066;pointer-events:none;max-width:290px;
  white-space:pre-line;display:none;}

/* ---------- light theme: the app chrome flips, the presentation
   canvas stays dark (decks look identical on every machine) -------- */
body.light .apptop{background:#f4f7fa;border-color:var(--line);}
body.light .appbar{border-bottom-color:var(--line);}
body.light .apptop-brand{color:var(--cyan-deep);
  border-right-color:var(--line);}
body.light .appbar .toggle{border-color:var(--line);background:#fff;
  color:var(--ink-2);}
body.light .appbar .toggle:hover{border-color:var(--cyan);
  color:var(--ink);}
body.light .appbar .toggle.tv.off{color:var(--ink-3);}
body.light .appbar .menubtn{border-color:var(--line);}
body.light .appbar .menubtn span,
body.light .appbar .menubtn span::before,
body.light .appbar .menubtn span::after{background:var(--ink-2);}
body.light .tab-openbtn{border-color:var(--line);
  color:var(--cyan-deep);}
body.light .tabsrow{background:#e9eef3;}
body.light .tab{border-color:var(--line);background:#00000006;
  color:var(--ink-3);}
body.light .tab:hover{background:#00000010;color:var(--ink);}
body.light .tab.current{background:var(--paper);color:var(--ink);}
body.light .tabs-label{color:var(--ink-3);}
body.light .presrail{background:#f4f7fa;
  border-right-color:var(--line);}
body.light .pr-item{color:var(--ink-3);}
body.light .pr-item:hover{background:#00000008;color:var(--ink);}
body.light .pr-item.current{background:#39a9c01f;color:var(--ink);}
body.light .pr-item.editing{background:var(--cyan-deep);color:#fff;}
body.light .pr-item.ptab .pr-ico{color:var(--cyan-deep);}
body.light .pr-item.editing .pr-ico{color:#fff;}
body.light .pr-label{color:var(--cyan-deep);}
body.light .pr-folder{color:var(--ink-3);}
body.light .pr-folder:hover{background:#00000008;
  color:var(--ink-2);}
body.light .pr-fico{color:var(--ink-3);}
body.light .pr-fcount{background:#00000012;color:var(--ink-3);}
body.light .pr-fctrl button{color:var(--ink-3);}
body.light .pr-fctrl button:hover{color:var(--ink);
  background:#00000012;}
body.light .pr-frename{background:#fff;color:var(--ink);}
body.light .pr-btn{background:#fff;border-color:var(--line);
  color:var(--ink-2);}
body.light .pr-btn:hover{border-color:var(--cyan);color:var(--ink);
  background:#39a9c012;}
body.light .pr-collapse{border-color:var(--line);
  color:var(--ink-3);}
body.light .pr-collapse:hover{color:var(--ink);
  border-color:var(--ink-3);}
body.light .presrail-show{background:#f4f7fa;
  border-color:var(--line);color:var(--ink-3);}
body.light .deck-create{background:#f4f7fa;}
body.light .dc-head{background:#eef2f6;
  border-bottom-color:var(--line);}
body.light .dc-block{border-bottom-color:var(--line);}
body.light .dc-label{color:var(--ink-3);}
body.light .deck-create .dbtn{border-color:var(--line);
  background:#fff;color:var(--ink-2);}
body.light .deck-create .dbtn:hover{border-color:var(--cyan);
  color:var(--ink);}
body.light .deck-create .dbtn.primary{background:var(--cyan-deep);
  border-color:var(--cyan-deep);color:#fff;}
body.light .deck-create .dbtn.lay .layico i{background:#8ba0b2;}
body.light .deck-create .dbtn.lay[aria-pressed="true"]{
  background:var(--cyan-deep);border-color:var(--cyan-deep);}
body.light .dc-presname{color:var(--ink);}
body.light #pres-name,body.light .title-editor input{
  background:#fff;border-color:var(--line);color:var(--ink);}
body.light .dc-hint{color:var(--ink-3);}
body.light .dc-menu{background:#fff;border-color:var(--line);}
body.light .dc-mi{color:var(--ink-2);}
body.light .dc-mi:hover{background:#39a9c026;}
body.light .dc-msep{background:var(--line);}
body.light .deck-status{background:#00000010;color:var(--ink-3);}
body.light .deck-status.draft{background:#b5731a22;color:#8a5410;}
body.light .deck-status.saved{background:#2e8a7222;color:#1e6f5a;}
body.light .film-label{color:var(--ink-2);}
body.light .film-row.current{background:#39a9c022;
  outline-color:#39a9c066;}
body.light .film-mini{color:var(--ink-3);}
body.light .film-mini:hover{background:#00000012;color:var(--ink);}
body.light .film-label .film-n{color:var(--ink-3);}
/* document rail (section nav aka Overview + analysis graph) */
body.light .rail{background:#f2f5f8;color:var(--ink-2);
  border-right-color:var(--line);}
body.light .railhead{border-bottom-color:var(--line);}
body.light .railtitle{color:var(--ink);}
body.light .railmeta{color:var(--ink-3);}
body.light .brand{color:var(--cyan-deep);}
body.light .navsec{color:var(--ink-2);}
body.light .navsec:hover{background:#00000008;}
body.light .navsec.active{background:#39a9c01c;color:var(--ink);}
body.light .navitems{border-left-color:var(--line);}
body.light .navsub{color:var(--ink-3);}
body.light .navitem{color:var(--ink-3);}
body.light .navitem:hover{color:var(--ink);background:#00000006;}
body.light .navitem.active{color:var(--ink);}
body.light .navitem.k-code .dot,body.light .nk.k-code .dot{
  background:#56627022;border-color:#00000026;}
body.light .navkey{border-top-color:var(--line);}
body.light .navkey-h,body.light .nk{color:var(--ink-3);}
body.light .nk .dot{background:var(--ink-3);}
body.light .navitem .dot{background:var(--ink-3);}
body.light .navitem.k-figure .dot,body.light .nk.k-figure .dot{
  background:var(--cyan);}
body.light .navitem.k-dataset .dot,body.light .nk.k-dataset .dot{
  background:#4d90c0;}
body.light .navitem.k-transform .dot,body.light .nk.k-transform .dot{
  background:#5b7589;}
body.light .navitem.k-metric .dot,body.light .nk.k-metric .dot{
  background:#46a892;}
body.light .navitem.k-note .dot,body.light .nk.k-note .dot{
  background:var(--amber);}
body.light .railgraph{background:#e9eef3;border-top-color:var(--line);}
body.light .rg-collapse{border-color:var(--line);color:var(--ink-3);}
body.light .rg-collapse:hover{color:var(--ink);
  border-color:#00000033;}

/* ---------- dark document (default theme; body.light keeps paper) --- */
body:not(.light){background:#0b141d;}
body:not(.light) .stage{background:#0b141d;}
body:not(.light) .sectionhead{border-bottom-color:#ffffff14;}
body:not(.light) .sectionhead h2{color:#e6edf3;}
body:not(.light) .card{background:#101c28;border-color:#ffffff14;
  box-shadow:0 1px 2px #00000040;}
body:not(.light) .card:hover{box-shadow:0 6px 22px #00000055;}
body:not(.light) .card.k-code::before{background:#2c3c4c;}
body:not(.light) .card.is-stub{background:#0e1824;
  border-color:#ffffff1f;}
body:not(.light) .card.is-stub:not(.stub-open) .cardtitle{
  color:#8ba0b2;}
body:not(.light) .card.is-stub .cardhead:hover .cardtitle{
  color:#e6edf3;}
body:not(.light) .cardtitle{color:#e6edf3;}
body:not(.light) .badge{background:#ffffff0d;color:#8ba0b2;}
body:not(.light) .k-figure .badge{background:#39a9c022;color:#5fc3d8;}
body:not(.light) .k-dataset .badge{background:#4d90c022;color:#7fb3d8;}
body:not(.light) .k-transform .badge{background:#5b758922;
  color:#93a7b8;}
body:not(.light) .k-metric .badge{background:#46a89222;color:#6fcab4;}
body:not(.light) .k-note .badge{background:#cf9a4e26;color:#dfb277;}
body:not(.light) .nodeid{background:#ffffff0f;color:#8ba0b2;}
body:not(.light) .note{color:#c3cfda;}
body:not(.light) .note .caption{color:#c3cfda;}
body:not(.light) .caption{color:#9fb0bf;}
body:not(.light) pre.result,body:not(.light) pre.stream{
  background:#0d1926;border-color:#ffffff14;color:#c9d6e2;}
body:not(.light) pre.error{background:#38180f;border-color:#6b352a;
  color:#f2b3a6;}
body:not(.light) .card.k-metric .cardbody pre.result{
  background:#46a89216;border-color:#46a89240;color:#7fd0bd;}
body:not(.light) .figframe{border-color:#ffffff1f;}
body:not(.light) .xr-wrap,body:not(.light) .rich{background:#fbfcfd;
  border:1px solid #ffffff1f;border-radius:8px;padding:8px;
  color:var(--ink);}
body:not(.light) .codewrap{border-top-color:#ffffff14;}
body:not(.light) .codetoggle{color:#8ba0b2;}
body:not(.light) .codetoggle:hover{color:#5fc3d8;}
body:not(.light) .steplabel,body:not(.light) .ct-steps{color:#8ba0b2;}
body:not(.light) details.alsoprinted{border-color:#ffffff1f;}
body:not(.light) details.alsoprinted>summary{color:#8ba0b2;}
body:not(.light) details.alsoprinted[open]>summary{
  border-bottom-color:#ffffff1f;}
body:not(.light) .depchip{color:#dfc49a;}
body:not(.light) .depchip:hover{color:#fff;}
body:not(.light) .mdmore{background:#101c28;border-color:#ffffff22;
  color:#5fc3d8;}
body:not(.light) .mdmore:hover{border-color:var(--cyan);color:#fff;}
body:not(.light) .cardbody.mdclamp::after{
  background:linear-gradient(#101c2800,#101c28);}
body:not(.light) .fp-btn{background:#101c28;border-color:#ffffff22;
  color:#c9d6e2;}
body:not(.light) .fp-btn:hover{border-color:var(--cyan);color:#fff;}
body:not(.light) .fp-count{color:#8ba0b2;}
body:not(.light) .rawcell{background:#101c28;border-color:#ffffff14;}
body:not(.light) .rawmd{color:#c3cfda;}
body:not(.light) .rawmd h1,body:not(.light) .rawmd h2,
body:not(.light) .rawmd h3,body:not(.light) .rawmd h4,
body:not(.light) .rawmd h5,body:not(.light) .rawmd h6{color:#e6edf3;}
body:not(.light) .welcome{background:#0b141d;}
body:not(.light) .welcome-box h1{color:#e6edf3;}
body:not(.light) .recent-i{background:#101c28;
  border-color:#ffffff22;color:#5fc3d8;}
body:not(.light) .welcome-btns .dbtn{background:#101c28;
  border-color:#ffffff22;color:#c9d6e2;}
body:not(.light) .welcome-btns .dbtn:hover{border-color:var(--cyan);
  color:#fff;}
body:not(.light) .welcome-btns .dbtn.primary{
  background:var(--cyan-deep);border-color:var(--cyan-deep);
  color:#fff;}
body:not(.light) .welcome-links a{color:#5fc3d8;}

/* ---------- drag-drop hint ---------- */
.drophint{position:fixed;inset:10px;z-index:140;border:2px dashed var(--cyan);
  border-radius:14px;background:#39a9c018;display:flex;align-items:center;
  justify-content:center;font-family:var(--mono);font-size:14px;
  color:var(--cyan-deep);pointer-events:none;letter-spacing:.08em;}
.drophint[hidden]{display:none;}

/* ---------- notebook source chips on slides / panes ---------- */
.spane-nb,.slide-nb{font-family:var(--mono);font-size:9px;
  letter-spacing:.1em;text-transform:uppercase;color:#5fc3d8;
  background:#39a9c01f;border-radius:4px;padding:2px 7px;flex:none;}
.slide-head{display:flex;align-items:baseline;gap:10px;}
.slide-head .slide-nb{position:relative;top:-2px;}
.spane-h{display:flex;align-items:center;gap:8px;margin:0 0 8px;flex:none;}
.spane-h .spane-t{margin:0;flex:1;min-width:0;}
.pane-nbtag{position:absolute;left:3px;bottom:2px;z-index:1;
  font-family:var(--mono);font-size:8px;letter-spacing:.08em;
  text-transform:uppercase;color:#5fc3d8;}

/* ---------- raw notebook view (transparency: cells as authored) ------ */
.rawview{display:none;max-width:920px;margin:0 auto;
  padding:30px 28px 30vh;}
.nbshell.raw .content{display:none;}
.nbshell.raw .rawview{display:block;}
#view-raw[aria-pressed="true"]{background:var(--cyan-deep);
  border-color:var(--cyan-deep);color:#fff;}
.rawcell{position:relative;background:var(--paper);
  border:1px solid var(--line);border-radius:10px;
  padding:14px 16px 14px 16px;margin:12px 0;}
.rawtag{font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);display:inline-block;
  margin-bottom:8px;}
.rawcell.code .rawtag{color:var(--cyan-deep);}
.rawcell pre.code{margin:0;}
.rawout{margin-top:10px;}
.rawmd{font-family:var(--serif);font-size:15px;line-height:1.65;
  color:var(--ink-2);}
.rawmd h2,.rawmd h3,.rawmd h4,.rawmd h5,.rawmd h6{font-family:var(--sans);
  color:var(--ink);margin:4px 0 8px;letter-spacing:-.01em;}
.rawmd h2{font-size:24px;}
.rawmd h3{font-size:19px;}
.rawmd h4{font-size:16px;}
.rawempty{color:var(--ink-3);text-align:center;padding:40px;}
"""

_JS = r"""
(function(){
  var $=function(s,r){return (r||document).querySelector(s);};
  var $$=function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s));};

  /* ================= app state ================= */
  var APP={mode:'static',token:'',root:'',project:{presentations:[],recent:[]}};
  var appEl=document.getElementById('app-data');
  if(appEl){try{APP=JSON.parse(appEl.textContent);}catch(e){}}
  APP.project=APP.project||{presentations:[],recent:[]};
  APP.shells={};          /* stem -> {el, data, path, title} */
  APP.order=[];           /* stems in tab order */
  APP.active=null;
  window.SemApp=APP;

  function api(path,body){
    var url=path+(path.indexOf('?')<0?'?':'&')
      +'t='+encodeURIComponent(APP.token||'');
    var opt=body===undefined?{method:'GET'}
      :{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify(body)};
    return fetch(url,opt).then(function(r){
      return r.json().catch(function(){throw new Error('HTTP '+r.status);})
        .then(function(j){
          if(!r.ok||(j&&j.error))
            throw new Error((j&&j.error)||('HTTP '+r.status));
          return j;
        });
    });
  }
  APP.api=api;

  /* ================= tab strip ================= */
  var tabstrip=$('#tabstrip'), openBtn=$('#tab-open');
  function refreshChrome(){
    var canOpen=APP.mode==='app'||APP.mode==='web';
    if(openBtn) openBtn.hidden=!canOpen;
    var wel=$('#welcome');
    if(wel) wel.hidden=!(canOpen&&!APP.order.length);
    var demo=$('#welcome-demo');
    if(demo) demo.hidden=(APP.mode!=='web');
    renderRecent();
  }
  function renderTabs(){
    if(!tabstrip){refreshChrome();return;}
    tabstrip.innerHTML='';
    APP.order.forEach(function(stem){
      var sh=APP.shells[stem];
      var t=document.createElement('div');
      t.className='tab'+(stem===APP.active?' current':'');
      t.setAttribute('role','tab');
      t.title=sh.path||sh.title||stem;
      var lbl=document.createElement('span');lbl.className='tab-t';
      lbl.textContent=stem;t.appendChild(lbl);
      if(APP.mode==='app'||APP.mode==='web'){
        if(sh.path){
          var r=document.createElement('button');r.className='tab-b';
          r.innerHTML='&#8635;';
          r.title=/^https?:/.test(sh.path)
            ?'Reload from URL':'Reload from disk';
          r.addEventListener('click',function(e){e.stopPropagation();
            openPath(sh.path);});
          t.appendChild(r);
        }
        var x=document.createElement('button');x.className='tab-b';
        x.innerHTML='&#10005;';x.title='Close tab';
        x.addEventListener('click',function(e){e.stopPropagation();
          closeNotebook(stem);});
        t.appendChild(x);
      }
      t.addEventListener('click',function(){activate(stem);});
      tabstrip.appendChild(t);
    });
    refreshChrome();
  }
  function activate(stem){
    if(!APP.shells[stem]) return;
    APP.active=stem;
    APP.order.forEach(function(s){APP.shells[s].el.hidden=(s!==stem);});
    renderTabs();
    renderRawBtn();
    document.dispatchEvent(new CustomEvent('sem:activate',
      {detail:{stem:stem}}));
  }
  APP.activate=activate;

  /* ================= per-notebook document behaviors ================= */
  var scrim=$('#scrim');
  if(scrim) scrim.addEventListener('click',function(){
    $$('.rail.open').forEach(function(r){r.classList.remove('open');});
    scrim.classList.remove('show');
  });

  /* ---- global show/hide filters (top bar; apply to every tab) ---- */
  var vis={figs:true,markup:true,code:true};
  var TYPES=[['figs','figures'],['markup','markup'],['code','code']];
  function renderTypeButtons(){
    TYPES.forEach(function(p){
      var b=$('#tv-'+p[0]); if(!b) return;
      b.innerHTML='<span class="tdot"></span>'
        +(vis[p[0]]?'Hide ':'Show ')+p[1];
      b.classList.toggle('off',!vis[p[0]]);
    });
  }
  function applyFilters(){
    $$('.nbshell .card').forEach(function(c){
      var kind=c.dataset.kind,note=c.dataset.note==='1';
      var show=note?vis.markup
        :(kind==='figure'||kind==='diagnostic')?vis.figs:vis.code;
      c.classList.toggle('is-stub',!show);
      if(show) c.classList.remove('stub-open');
    });
    renderTypeButtons();
  }
  /* "Show code" means ALL code: code cards AND the blocks folded under
     every figure / dataset card */
  var codeAllOpen=null;   /* null until the user toggles */
  function setAllCode(open,root){
    $$('.codewrap',root||document).forEach(function(w){
      if(open) w.setAttribute('data-open','');
      else w.removeAttribute('data-open');
      var btn=$('.codetoggle',w);
      if(btn) btn.setAttribute('aria-expanded',open?'true':'false');
    });
  }
  TYPES.forEach(function(p){
    var b=$('#tv-'+p[0]);
    if(b) b.addEventListener('click',function(){
      vis[p[0]]=!vis[p[0]];applyFilters();
      if(p[0]==='code'){codeAllOpen=vis.code;setAllCode(vis.code);}
    });
  });
  renderTypeButtons();

  /* ---- raw notebook toggle (applies to the ACTIVE tab) ---- */
  var rawBtn=$('#view-raw');
  function renderRawBtn(){
    if(!rawBtn) return;
    var sh=APP.active&&APP.shells[APP.active];
    var on=!!(sh&&sh.el.classList.contains('raw'));
    rawBtn.setAttribute('aria-pressed',on.toString());
    rawBtn.textContent=on?'Formatted view':'Raw notebook';
    rawBtn.disabled=!sh;
  }
  if(rawBtn) rawBtn.addEventListener('click',function(){
    var sh=APP.active&&APP.shells[APP.active];
    if(!sh) return;
    var on=sh.el.classList.toggle('raw');
    if(on&&!sh.el.dataset.rawTypeset){
      sh.el.dataset.rawTypeset='1';
      var rv=$('.rawview',sh.el);
      if(rv&&window.MathJax&&MathJax.typesetPromise)
        MathJax.typesetPromise([rv]).catch(function(){});
    }
    renderRawBtn();
  });

  /* ---- theme toggle (chrome only; the slide canvas stays dark) --- */
  var themeBtn=$('#theme-btn');
  function applyTheme(light){
    document.body.classList.toggle('light',light);
    if(themeBtn){
      themeBtn.innerHTML=light?'&#9789; Dark':'&#9788; Light';
      themeBtn.setAttribute('data-tip',light
        ?'Switch to the dark theme':'Switch to the light theme');
      themeBtn.removeAttribute('title');
    }
    try{localStorage.setItem('plotline-theme',
      light?'light':'dark');}catch(e){}
  }
  var themePref=null;
  try{themePref=localStorage.getItem('plotline-theme');}catch(e){}
  applyTheme(themePref==='light');
  if(themeBtn) themeBtn.addEventListener('click',function(){
    applyTheme(!document.body.classList.contains('light'));
  });

  /* ---- builder panel width: draggable right edge, persisted ------- */
  var dcR=$('#dc-resize');
  var dcwPref=null;
  try{dcwPref=parseInt(localStorage.getItem('plotline-dcw'),10);}
  catch(e){}
  if(dcwPref&&dcwPref>=300&&dcwPref<=760)
    document.documentElement.style.setProperty('--dc-w',dcwPref+'px');
  if(dcR) dcR.addEventListener('mousedown',function(e){
    e.preventDefault();
    dcR.classList.add('on');
    var host=$('#deck-create');
    var left=host?host.getBoundingClientRect().left:0;
    var w=0;
    function mv(ev){
      w=Math.max(300,Math.min(760,ev.clientX-left));
      document.documentElement.style.setProperty('--dc-w',w+'px');
    }
    function up(){
      dcR.classList.remove('on');
      document.removeEventListener('mousemove',mv);
      document.removeEventListener('mouseup',up);
      if(w) try{localStorage.setItem('plotline-dcw',w);}catch(e){}
    }
    document.addEventListener('mousemove',mv);
    document.addEventListener('mouseup',up);
  });

  /* ---- instant tooltips: every [title] becomes a styled tip ------- */
  var tipEl=document.createElement('div');
  tipEl.className='apptip';
  document.body.appendChild(tipEl);
  var tipTimer=null,tipTarget=null;
  function hideTip(){
    clearTimeout(tipTimer);tipTimer=null;
    tipTarget=null;tipEl.style.display='none';
  }
  document.addEventListener('mouseover',function(e){
    var t=e.target.closest&&e.target.closest('[title],[data-tip]');
    if(!t){hideTip();return;}
    if(t===tipTarget) return;
    if(t.hasAttribute&&t.hasAttribute('title')){
      var tt=t.getAttribute('title');
      if(tt) t.setAttribute('data-tip',tt);
      t.removeAttribute('title');
    }
    var tip=t.getAttribute&&t.getAttribute('data-tip');
    if(!tip){hideTip();return;}
    tipTarget=t;
    clearTimeout(tipTimer);
    tipTimer=setTimeout(function(){
      if(tipTarget!==t||!document.contains(t)){return;}
      tipEl.textContent=tip;
      tipEl.style.display='block';
      var r=t.getBoundingClientRect();
      var tw=tipEl.offsetWidth,th=tipEl.offsetHeight;
      var x=r.left+r.width/2-tw/2;
      x=Math.max(6,Math.min(window.innerWidth-tw-6,x));
      var y=r.bottom+8;
      if(y+th>window.innerHeight-6) y=r.top-th-8;
      tipEl.style.left=x+'px';
      tipEl.style.top=Math.max(6,y)+'px';
    },220);
  });
  document.addEventListener('mouseout',function(e){
    if(tipTarget&&!tipTarget.contains(e.relatedTarget)) hideTip();
  });
  document.addEventListener('mousedown',hideTip,true);
  document.addEventListener('scroll',hideTip,true);

  /* ---- figure pager: ‹ › flips between figures of one cell -------- */
  /* delegated so it works in cloned slide frames too */
  document.addEventListener('click',function(e){
    var b=e.target.closest&&e.target.closest('.fp-btn');
    if(!b) return;
    var pg=b.closest('.figpager'); if(!pg) return;
    e.preventDefault();e.stopPropagation();
    var pages=[].slice.call(pg.querySelectorAll(':scope > .figpage'));
    if(!pages.length) return;
    var cur=0;
    pages.forEach(function(p,i){
      if(p.classList.contains('current')) cur=i;});
    var nx=(cur+(b.classList.contains('fp-next')?1:-1)
      +pages.length)%pages.length;
    pages[cur].classList.remove('current');
    pages[nx].classList.add('current');
    var ct=pg.querySelector('.fp-count');
    if(ct) ct.textContent=(nx+1)+' / '+pages.length;
  },true);

  /* ---- presentations rail: full -> icons -> hidden (edge handle
     brings it back) ---- */
  var prCollapse=$('#pr-collapse'), prShow=$('#presrail-show');
  function railState(){
    return document.body.classList.contains('presrail-hidden')?'hidden'
      :document.body.classList.contains('presrail-min')?'min':'full';
  }
  function setRailState(st){
    document.body.classList.toggle('presrail-min',st==='min');
    document.body.classList.toggle('presrail-hidden',st==='hidden');
    if(prCollapse)
      prCollapse.title=st==='full'
        ?'Collapse to icons (click again to hide)':'Hide this panel';
    try{localStorage.setItem('sempresrail2',st);}catch(e){}
  }
  var railPref=null;
  try{railPref=localStorage.getItem('sempresrail2');}catch(e){}
  setRailState(railPref==='min'||railPref==='hidden'||railPref==='full'
    ?railPref:(window.innerWidth<1100?'min':'full'));
  if(prCollapse) prCollapse.addEventListener('click',function(){
    setRailState(railState()==='full'?'min':'hidden');
  });
  if(prShow) prShow.addEventListener('click',function(){
    setRailState('full');
  });

  /* ---- help overlay ---- */
  var helpDlg=$('#helpdlg');
  function showHelp(){if(helpDlg) helpDlg.hidden=false;}
  function hideHelp(){if(helpDlg) helpDlg.hidden=true;}
  var helpBtn=$('#help-btn');
  if(helpBtn) helpBtn.addEventListener('click',showHelp);
  var wHelp=$('#welcome-help');
  if(wHelp) wHelp.addEventListener('click',function(e){
    e.preventDefault();showHelp();});
  var helpClose=$('#help-close');
  if(helpClose) helpClose.addEventListener('click',hideHelp);
  if(helpDlg) helpDlg.addEventListener('click',function(e){
    if(e.target===helpDlg) hideHelp();});
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape'&&helpDlg&&!helpDlg.hidden){
      e.stopPropagation();hideHelp();
    }
  },true);

  /* ---- ☰ toggles the section sidebar (TOC). Desktop: body.tocshow
     (hidden by default, pref persisted); mobile keeps the slide-in. */
  var menuBtn=$('#menubtn');
  function applyToc(show){
    document.body.classList.toggle('tocshow',show);
    if(menuBtn) menuBtn.setAttribute('aria-pressed',
      show?'true':'false');
    try{localStorage.setItem('plotline-toc',
      show?'open':'hidden');}catch(e){}
  }
  var tocPref=null;
  try{tocPref=localStorage.getItem('plotline-toc');}catch(e){}
  applyToc(tocPref==='open');
  if(menuBtn) menuBtn.addEventListener('click',function(){
    if(window.matchMedia
       &&window.matchMedia('(max-width:860px)').matches){
      var sh=APP.active&&APP.shells[APP.active];
      if(!sh) return;
      var rail=$('.rail',sh.el);
      if(rail){rail.classList.toggle('open');
        if(scrim) scrim.classList.toggle('show');}
      return;
    }
    applyToc(!document.body.classList.contains('tocshow'));
  });

  /* huge markdown notes: clamp with a Show more toggle */
  function mdClampScan(shell){
    $$('.card[data-note="1"] .cardbody',shell).forEach(function(bd){
      if(bd.dataset.mdclamp) return;
      var nt=$('.note',bd); if(!nt) return;
      if(nt.scrollHeight<=460) return;
      bd.dataset.mdclamp='1';
      bd.classList.add('mdclamp');
      var btn=document.createElement('button');
      btn.className='mdmore';
      btn.textContent='Show more';
      btn.title='This note is long — expand it to full length';
      btn.addEventListener('click',function(){
        var open=bd.classList.toggle('mdopen');
        btn.textContent=open?'Show less':'Show more';
      });
      bd.parentNode.insertBefore(btn,bd.nextSibling);
    });
  }
  APP.mdscan=mdClampScan;
  function initShell(shell){
    var data={};
    var de=$('.nb-data',shell);
    if(de){try{data=JSON.parse(de.textContent);}catch(e){}}
    var stem=shell.dataset.nb||data.stem||('nb-'+(APP.order.length+1));
    mdClampScan(shell);

    /* ---- reveal on scroll ---- */
    var cards=$$('.card',shell);
    if('IntersectionObserver' in window){
      var io=new IntersectionObserver(function(es){
        es.forEach(function(e){if(e.isIntersecting){
          e.target.classList.add('in');io.unobserve(e.target);}});
      },{rootMargin:'0px 0px -8% 0px',threshold:0.04});
      cards.forEach(function(c){io.observe(c);});
    } else cards.forEach(function(c){c.classList.add('in');});

    /* ---- scroll-spy: active section + item + graph node ---- */
    var navSecs={},navItems={},graphNodes={};
    $$('.navsec',shell).forEach(function(a){navSecs[a.dataset.sec]=a;});
    $$('.navitem',shell).forEach(function(a){navItems[a.dataset.item]=a;});
    $$('.provnode',shell).forEach(function(g){graphNodes[g.dataset.node]=g;});
    function setActiveSection(id){
      $$('.navsec.active',shell).forEach(function(a){a.classList.remove('active');});
      if(navSecs[id]) navSecs[id].classList.add('active');
    }
    function setActiveItem(item){
      $$('.navitem.active',shell).forEach(function(a){a.classList.remove('active');});
      if(navItems[item]) navItems[item].classList.add('active');
      var node=$('.card[id="card-'+item+'"]',shell);
      var nodeId=node?node.dataset.node:'';
      $$('.provnode.active',shell).forEach(function(g){g.classList.remove('active');});
      $$('.provedge.lit',shell).forEach(function(p){p.classList.remove('lit');});
      if(nodeId&&graphNodes[nodeId]){
        graphNodes[nodeId].classList.add('active');
        $$('.provedge',shell).forEach(function(p){
          if(p.dataset.to===nodeId||p.dataset.from===nodeId)
            p.classList.add('lit');
        });
      }
    }
    if('IntersectionObserver' in window){
      var visible={};
      var spy=new IntersectionObserver(function(es){
        es.forEach(function(e){
          if(e.isIntersecting) visible[e.target.id]=e.intersectionRatio;
          else delete visible[e.target.id];
        });
        var bestC=null,bc=0;
        Object.keys(visible).forEach(function(k){
          if(k.indexOf('card-')===0&&visible[k]>=bc){bc=visible[k];bestC=k;}
        });
        if(bestC){
          var item=bestC.slice(5);
          setActiveItem(item);
          var card=$('.card[id="'+bestC+'"]',shell);
          var sec=card?card.closest('.section'):null;
          if(sec) setActiveSection(sec.dataset.sec);
        }
      },{rootMargin:'-12% 0px -55% 0px',threshold:[0,0.25,0.6,1]});
      cards.forEach(function(c){spy.observe(c);});
    }

    /* ---- nav links: resolve inside THIS shell (ids repeat across tabs) */
    var rail=$('.rail',shell);
    function closeRail(){
      if(rail) rail.classList.remove('open');
      if(scrim) scrim.classList.remove('show');
    }
    $$('.navsec,.navitem',shell).forEach(function(a){
      a.addEventListener('click',function(e){
        e.preventDefault();
        if(shell.classList.contains('raw')){
          shell.classList.remove('raw');renderRawBtn();
        }
        var id=(a.getAttribute('href')||'').slice(1);
        var el=id?$('[id="'+id+'"]',shell):null;
        if(el) el.scrollIntoView({behavior:'smooth',block:'start'});
        if(window.innerWidth<=860) closeRail();
      });
    });

    /* ---- code toggles ---- */
    $$('.codetoggle',shell).forEach(function(btn){
      btn.addEventListener('click',function(){
        var wrap=btn.closest('.codewrap');
        var open=wrap.hasAttribute('data-open');
        if(open){wrap.removeAttribute('data-open');
          btn.setAttribute('aria-expanded','false');}
        else{wrap.setAttribute('data-open','');
          btn.setAttribute('aria-expanded','true');}
      });
    });

    /* ---- filtered-out cards expand in place when clicked ---- */
    cards.forEach(function(c){
      var head=$('.cardhead',c);
      if(head) head.addEventListener('click',function(){
        if(c.classList.contains('is-stub')) c.classList.toggle('stub-open');
      });
    });

    /* ---- graph node / dep chip -> scroll to card ---- */
    function gotoItem(itemId){
      var card=$('.card[id="card-'+itemId+'"]',shell);
      if(!card) return;
      card.scrollIntoView({behavior:'smooth',block:'center'});
      card.classList.add('target-flash');
      setTimeout(function(){card.classList.remove('target-flash');},1400);
    }
    $$('.provnode',shell).forEach(function(g){
      function act(){gotoItem(g.dataset.target);}
      g.addEventListener('click',act);
      g.addEventListener('keydown',function(e){
        if(e.key==='Enter'||e.key===' '){e.preventDefault();act();}});
    });
    $$('.depchip',shell).forEach(function(a){
      a.addEventListener('click',function(e){
        e.preventDefault();
        var src=$('.card[data-node="'+a.dataset.dep+'"]',shell);
        if(src){src.scrollIntoView({behavior:'smooth',block:'center'});
          src.classList.add('target-flash');
          setTimeout(function(){src.classList.remove('target-flash');},1400);}
      });
    });
    var rgBtn=$('.rg-collapse',shell);
    if(rgBtn) rgBtn.addEventListener('click',function(){
      var rg=rgBtn.closest('.railgraph');
      var c=rg.classList.toggle('collapsed');
      rgBtn.setAttribute('aria-expanded',(!c).toString());
      rgBtn.textContent=c?'+':'\u2013';
    });

    /* ---- register ---- */
    var replaced=!!APP.shells[stem];
    APP.shells[stem]={el:shell,data:data,path:shell.dataset.path||'',
      title:data.title||stem};
    if(APP.order.indexOf(stem)<0) APP.order.push(stem);
    applyFilters();
    if(codeAllOpen!==null) setAllCode(codeAllOpen,shell);
    document.dispatchEvent(new CustomEvent('sem:shell',
      {detail:{stem:stem,el:shell,data:data,replaced:replaced}}));
    renderTabs();
    return stem;
  }

  /* ================= app mode: open / close / reload ================= */
  function mountShellHTML(htmlStr,path){
    var host=$('#docs');
    var tmp=document.createElement('div');
    tmp.innerHTML=htmlStr;
    var shell=tmp.querySelector('.nbshell');
    if(!shell){alert('Open failed: bad response');return;}
    if(path) shell.dataset.path=path;
    var stem=shell.dataset.nb;
    var old=APP.shells[stem];
    if(old&&old.el.parentNode) host.replaceChild(shell,old.el);
    else host.appendChild(shell);
    initShell(shell);
    activate(stem);
    if(window.MathJax&&MathJax.typesetPromise)
      MathJax.typesetPromise([shell]).catch(function(){});
  }
  /* one open per source at a time: repeated Enter/clicks are ignored
     while the fetch runs, and the dialog shows a loading bar */
  var OPENBUSY={},dlgBusyN=0;
  function setDlgBusy(b){
    /* counter, not flag: several sources can load at once and the
       dialog stays locked until the last one settles */
    dlgBusyN=Math.max(0,dlgBusyN+(b?1:-1));
    var on=dlgBusyN>0;
    var go=$('#odlg-go'),inp=$('#odlg-input'),ld=$('#odlg-load');
    if(go){go.disabled=on;go.textContent=on?'Opening…':'Open';}
    if(inp) inp.disabled=on;
    if(ld) ld.hidden=!on;
  }
  function openPath(path){
    if(APP.mode==='web'){
      if(isUrl(path)) webOpenUrl(path,false);
      return;
    }
    if(OPENBUSY[path]) return;
    OPENBUSY[path]=1;setDlgBusy(true);
    api('/api/open',{path:path}).then(function(j){
      delete OPENBUSY[path];setDlgBusy(false);
      mountShellHTML(j.shell,j.path||path);
      hideDlg();
    }).catch(function(e){
      delete OPENBUSY[path];setDlgBusy(false);
      alert('Open failed: '+e.message);});
  }
  APP.openPath=openPath;
  function closeNotebook(stem){
    var sh=APP.shells[stem]; if(!sh) return;
    if(sh.el.parentNode) sh.el.parentNode.removeChild(sh.el);
    delete APP.shells[stem];
    var i=APP.order.indexOf(stem);
    if(i>=0) APP.order.splice(i,1);
    if(APP.active===stem){
      APP.active=null;
      if(APP.order.length)
        activate(APP.order[Math.min(Math.max(i,0),APP.order.length-1)]);
      else renderRawBtn();
    }
    renderTabs();
    document.dispatchEvent(new CustomEvent('sem:shellclosed',
      {detail:{stem:stem}}));
    if(APP.mode==='app'&&sh.path)
      api('/api/close',{path:sh.path}).catch(function(){});
    if(APP.mode==='web'&&sh.path) webUnnote(sh.path);
  }

  /* ================= web mode (Pyodide, fully client-side) ============ */
  var WEBKEY='semweb:'+location.pathname;
  function isUrl(s){return /^https?:\/\//i.test(String(s||''));}
  function normNbUrl(u){
    u=String(u||'').trim();
    var m=u.match(
      /^https?:\/\/github\.com\/([^\/]+)\/([^\/]+)\/(?:blob|raw)\/(.+)$/);
    if(m) return 'https://raw.githubusercontent.com/'
      +m[1]+'/'+m[2]+'/'+m[3];
    return u;
  }
  function webReady(){return !!window.semPy;}
  function webParseText(name,text){
    if(!webReady()){
      alert('Python is still loading — try again in a moment.');
      return;
    }
    try{
      var shell=window.semPy.parse(name,text,APP.order);
      mountShellHTML(shell,'');
      hideDlg();
    }catch(e){
      alert('Could not open '+name+': '+((e&&e.message)||e));
    }
  }
  function webOpenFiles(files){
    Array.prototype.slice.call(files||[])
      .filter(function(f){return /\.ipynb$/i.test(f.name);})
      .forEach(function(f){
        f.text().then(function(txt){webParseText(f.name,txt);});
      });
  }
  function webOpenUrl(url,silent){
    url=normNbUrl(url);
    var pend=OPENBUSY[url];
    if(pend){
      /* already loading; a real click on a silently-restoring URL
         surfaces the busy UI instead of dying quietly */
      if(!silent&&pend.s){pend.s=false;setDlgBusy(true);}
      return;
    }
    pend=OPENBUSY[url]={s:silent};
    if(!silent) setDlgBusy(true);
    function done(){delete OPENBUSY[url];if(!pend.s) setDlgBusy(false);}
    fetch(url).then(function(r){
      if(!r.ok) throw new Error('HTTP '+r.status);
      return r.text();
    }).then(function(txt){
      if(!webReady()) throw new Error('Python is still loading');
      var name=decodeURIComponent(
        url.split('?')[0].split('/').pop()||'notebook.ipynb');
      var shell=window.semPy.parse(name,txt,APP.order);
      mountShellHTML(shell,url);
      webNote(url);
      done();
      hideDlg();
    }).catch(function(e){
      var wasSilent=pend.s;
      done();
      if(wasSilent){
        webUnnote(url);
        return;
      }
      alert('Could not fetch '+url+'\n'+((e&&e.message)||e)
        +'\nIf that host blocks cross-site requests, download the '
        +'file and drop it here instead.');
    });
  }
  function webNote(url){
    try{
      var rec=JSON.parse(localStorage.getItem(WEBKEY+':recent')||'[]');
      rec=[url].concat(rec.filter(function(r){return r!==url;}))
        .slice(0,6);
      localStorage.setItem(WEBKEY+':recent',JSON.stringify(rec));
      var open=JSON.parse(localStorage.getItem(WEBKEY+':open')||'[]');
      if(open.indexOf(url)<0) open.push(url);
      localStorage.setItem(WEBKEY+':open',JSON.stringify(open));
      APP.project.recent=rec;
      renderRecent();
    }catch(e){}
  }
  function webUnnote(url){
    try{
      var open=JSON.parse(localStorage.getItem(WEBKEY+':open')||'[]');
      localStorage.setItem(WEBKEY+':open',
        JSON.stringify(open.filter(function(u){return u!==url;})));
    }catch(e){}
  }

  /* ================= open dialog (server file browser) ================= */
  var dlg=$('#opendlg'), dlgList=$('#odlg-list'), dlgPath=$('#odlg-path');
  var dlgDir='';
  function hideDlg(){if(dlg) dlg.hidden=true;}
  function showDlg(){
    if(!dlg) return;
    dlg.hidden=false;
    var inp=$('#odlg-input'); if(inp) inp.value='';
    var up=$('#odlg-up'), fb=$('#odlg-files');
    if(APP.mode==='web'){
      if(up) up.hidden=true;
      if(fb) fb.hidden=false;
      if(dlgPath) dlgPath.textContent='Open notebooks';
      if(inp) inp.placeholder='…or paste a notebook URL '
        +'(GitHub links work) and hit Open';
      dlgList.innerHTML='<div class="odlg-empty">Drop .ipynb files '
        +'anywhere in the window, use &#8220;Choose files&#8230;&#8221;, '
        +'or paste a URL below.<br><br>Everything runs in your browser '
        +'&#8212; notebooks are never uploaded anywhere.</div>';
      return;
    }
    if(inp) inp.placeholder='…or paste a folder, .ipynb path or URL '
      +'and hit Open';
    listDir(dlgDir||APP.root||'');
  }
  function listDir(dir){
    api('/api/list?dir='+encodeURIComponent(dir||'')).then(function(j){
      dlgDir=j.dir;
      dlgPath.textContent=j.dir;dlgPath.title=j.dir;
      var up=$('#odlg-up');
      up.disabled=!j.parent;up.dataset.parent=j.parent||'';
      dlgList.innerHTML='';
      if(!j.dirs.length&&!j.notebooks.length)
        dlgList.innerHTML='<div class="odlg-empty">'
          +'No folders or notebooks here.</div>';
      j.dirs.forEach(function(d){
        var b=document.createElement('button');b.className='odlg-i';
        b.innerHTML='<span class="ic">&#128193;</span>';
        var nm=document.createElement('span');nm.className='nm';
        nm.textContent=d.name;b.appendChild(nm);
        b.addEventListener('click',function(){listDir(d.path);});
        dlgList.appendChild(b);
      });
      j.notebooks.forEach(function(n){
        var b=document.createElement('button');b.className='odlg-i nb';
        b.innerHTML='<span class="ic">&#128209;</span>';
        var nm=document.createElement('span');nm.className='nm';
        nm.textContent=n.name;b.appendChild(nm);
        var sz=document.createElement('span');sz.className='sz';
        sz.textContent=n.size||'';b.appendChild(sz);
        b.addEventListener('click',function(){openPath(n.path);});
        dlgList.appendChild(b);
      });
    }).catch(function(e){
      dlgList.innerHTML='';
      var d=document.createElement('div');d.className='odlg-empty';
      d.textContent=String(e.message);
      dlgList.appendChild(d);
    });
  }
  function renderRecent(){
    var host=$('#welcome-recent'); if(!host) return;
    host.innerHTML='';
    var rec=(APP.project&&APP.project.recent)||[];
    if(!rec.length) return;
    var h=document.createElement('div');h.className='recent-h';
    h.textContent='recent';host.appendChild(h);
    rec.slice(0,6).forEach(function(p){
      var b=document.createElement('button');b.className='recent-i';
      b.textContent=p;b.title=p;
      b.addEventListener('click',function(){openPath(p);});
      host.appendChild(b);
    });
  }

  if(APP.mode==='app'||APP.mode==='web'){
    var isWeb=(APP.mode==='web');
    if(openBtn) openBtn.addEventListener('click',showDlg);
    var wOpen=$('#welcome-open');
    if(wOpen) wOpen.addEventListener('click',showDlg);
    var up=$('#odlg-up');
    if(up&&!isWeb) up.addEventListener('click',function(){
      if(up.dataset.parent) listDir(up.dataset.parent);});
    var filesBtn=$('#odlg-files'), fileInput=$('#fileinput');
    if(filesBtn) filesBtn.addEventListener('click',function(){
      if(fileInput) fileInput.click();});
    if(fileInput) fileInput.addEventListener('change',function(){
      webOpenFiles(this.files);this.value='';});
    var cl=$('#odlg-close');
    if(cl) cl.addEventListener('click',hideDlg);
    if(dlg) dlg.addEventListener('click',function(e){
      if(e.target===dlg) hideDlg();});
    var inp=$('#odlg-input');
    function submitOpenInput(){
      if(!inp||inp.disabled) return;
      var v=inp.value.trim(); if(!v) return;
      if(isWeb){
        if(isUrl(v)) webOpenUrl(v,false);
        else alert('Paste an http(s) link to a .ipynb file, or use '
          +'Choose files / drag-and-drop.');
        return;
      }
      if(isUrl(v)||/\.ipynb$/i.test(v)) openPath(v);
      else listDir(v);
    }
    if(inp) inp.addEventListener('keydown',function(e){
      if(e.key!=='Enter') return;
      submitOpenInput();
    });
    var goBtn=$('#odlg-go');
    if(goBtn) goBtn.addEventListener('click',submitOpenInput);
    document.addEventListener('keydown',function(e){
      if(e.key==='Escape'&&dlg&&!dlg.hidden) hideDlg();
    });

    /* ---- drag & drop .ipynb anywhere on the window ---- */
    var hint=$('#drophint'), dragDepth=0;
    window.addEventListener('dragover',function(e){e.preventDefault();});
    window.addEventListener('dragenter',function(e){
      e.preventDefault();dragDepth++;
      if(hint) hint.hidden=false;
    });
    window.addEventListener('dragleave',function(){
      dragDepth=Math.max(0,dragDepth-1);
      if(!dragDepth&&hint) hint.hidden=true;
    });
    window.addEventListener('drop',function(e){
      e.preventDefault();dragDepth=0;
      if(hint) hint.hidden=true;
      var files=Array.prototype.slice.call(
        (e.dataTransfer||{}).files||[]);
      files.filter(function(f){return /\.ipynb$/i.test(f.name);})
        .forEach(function(f){
          if(isWeb){
            f.text().then(function(txt){webParseText(f.name,txt);});
            return;
          }
          f.text().then(function(txt){
            return api('/api/parse',{name:f.name,nb:txt});
          }).then(function(j){mountShellHTML(j.shell,j.path||'');})
          .catch(function(err){
            alert('Could not open '+f.name+': '+err.message);});
        });
    });
  }
  if(APP.mode==='web'){
    var demoBtn=$('#welcome-demo');
    if(demoBtn) demoBtn.addEventListener('click',function(){
      webOpenUrl('example_climate_analysis.ipynb',false);
    });
    try{
      APP.project.recent=JSON.parse(
        localStorage.getItem(WEBKEY+':recent')||'[]');
    }catch(e){}
    /* reopen last session's URL notebooks once Python is up */
    document.addEventListener('sem:pyready',function(){
      var open=[];
      try{open=JSON.parse(
        localStorage.getItem(WEBKEY+':open')||'[]');}catch(e){}
      open.forEach(function(u){webOpenUrl(u,true);});
    });
  }

  /* ================= boot: mount shells already on the page ============ */
  $$('.nbshell').forEach(function(sh){initShell(sh);});
  if(APP.order.length) activate(APP.order[0]);
  else renderTabs();
  renderRawBtn();
})();
"""

# --------------------------------------------------------------------------
# Presentation deck (Present mode + PowerPoint-style builder)
# --------------------------------------------------------------------------

_MATHJAX = r"""<script>
window.MathJax = {
  tex: {inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]},
  options: {skipHtmlTags: ['script','noscript','style','textarea','pre','code']}
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>"""

_DECK_HTML = """
<div class="deck" id="deck" hidden>
  <div class="deck-top">
    <button class="dbtn" id="deck-docs"
      title="Back to the document view">Docs</button>
    <span class="deck-spring"></span>
    <button class="dbtn" id="deck-exit"
      title="Exit playback, back to the builder">&#10005; Exit</button>
  </div>
  <div class="deck-main">
    <aside class="deck-create" id="deck-create" hidden>
      <div class="dc-resize" id="dc-resize"
        title="Drag to resize the builder panel"></div>
      <div class="dc-head">
        <button class="dbtn primary" id="dc-play"
          title="Play the presentation fullscreen">&#9654; Present</button>
        <div class="dc-menuwrap">
          <button class="dbtn" id="dc-file" aria-haspopup="true"
            aria-expanded="false">File &#9662;</button>
          <div class="dc-menu" id="dc-menu" hidden>
            <button class="dc-mi" id="mi-new">New presentation</button>
            <button class="dc-mi" id="mi-rename">Rename&#8230;</button>
            <div class="dc-msep"></div>
            <button class="dc-mi" id="mi-auto-figs">Auto-build: figures</button>
            <button class="dc-mi" id="mi-auto-figdocs">Auto-build: figures + docs</button>
            <div class="dc-msep"></div>
            <button class="dc-mi" id="mi-save">Save to notebook&#8230;</button>
            <button class="dc-mi" id="mi-autosave" hidden></button>
            <button class="dc-mi" id="mi-dl">Download JSON</button>
            <button class="dc-mi" id="mi-load">Load deck
              JSON&#8230;</button>
            <button class="dc-mi" id="mi-discard">Discard changes</button>
            <button class="dc-mi" id="mi-del">Delete presentation</button>
          </div>
        </div>
        <button class="dbtn" id="dc-save">Save</button>
        <span class="deck-status" id="deck-status"></span>
        <span class="dc-spring"></span>
        <button class="dbtn" id="dc-close"
          title="Close the builder, back to the documents (Esc)">&#10005;
          Close</button>
      </div>
      <div class="dc-block">
        <span class="dc-label">Presentation</span>
        <div class="dc-presname" id="pres-current"
          title="Click to rename"></div>
        <input id="pres-name" type="text" placeholder="presentation name"
          spellcheck="false" autocomplete="off" hidden>
      </div>
      <div class="dc-block">
        <span class="dc-label">Slide layout</span>
        <div class="dc-row" id="layout-row">
          <button class="dbtn lay" data-lay="full" title="One pane">
            <span class="layico full"><i></i></span></button>
          <button class="dbtn lay" data-lay="halves"
            title="Two panes, side by side">
            <span class="layico halves"><i></i><i></i></span></button>
          <button class="dbtn lay" data-lay="rows"
            title="Two panes, stacked">
            <span class="layico rows"><i></i><i></i></span></button>
          <button class="dbtn lay" data-lay="quarters" title="Four panes">
            <span class="layico quarters"><i></i><i></i><i></i><i></i>
            </span></button>
          <button class="dbtn lay" data-lay="title"
            title="Title slide (free text)">
            <span class="layico title"><i class="tl1"></i>
            <i class="tl2"></i></span></button>
          <button class="dbtn lay" data-lay="blank"
            title="Blank canvas — build it with the slide editor">
            <span class="layico blank"></span></button>
        </div>
        <div class="pane-editor" id="pane-editor"></div>
        <div class="title-editor" id="title-editor" hidden>
          <input id="ts-title" type="text" placeholder="Slide title"
            spellcheck="false" autocomplete="off">
          <input id="ts-sub" type="text" placeholder="Subtitle (optional)"
            spellcheck="false" autocomplete="off">
        </div>
        <p class="dc-hint" id="dc-hint">Pick a pane, then click a card in
        the document to place it there.</p>
        <button class="dbtn" id="dc-edit"
          title="Open this slide full-screen; add text, arrows and boxes">
          &#9998; Edit slide</button>
      </div>
      <div class="dc-block dc-film">
        <span class="dc-label">Slides</span>
        <div class="film-list" id="film-list"></div>
        <button class="dbtn addslide" id="film-add">+ Add slide</button>
      </div>
    </aside>
    <div class="deck-stagewrap" id="deck-stagewrap">
      <div class="edit-tools" id="edit-tools" hidden>
        <span class="et-label">edit slide</span>
        <button class="dbtn et" data-tool="select"
          aria-pressed="true">Select</button>
        <button class="dbtn et" data-tool="text" aria-pressed="false">
          + Text</button>
        <button class="dbtn et" data-tool="arrow" aria-pressed="false">
          + Arrow</button>
        <button class="dbtn et" data-tool="rect" aria-pressed="false">
          + Box</button>
        <button class="dbtn et" data-tool="cell" aria-pressed="false"
          title="A resizable frame holding any notebook card">
          + Cell</button>
        <span class="et-fmt" id="et-fmt" hidden>
          <span class="fmt-lab" id="fmt-txlab" hidden>T</span>
          <button class="sw" data-c="#ff6b57"
            style="background:#ff6b57" title="Coral"></button>
          <button class="sw" data-c="#f0a848"
            style="background:#f0a848" title="Amber"></button>
          <button class="sw" data-c="#39a9c0"
            style="background:#39a9c0" title="Cyan"></button>
          <button class="sw" data-c="#46a892"
            style="background:#46a892" title="Green"></button>
          <button class="sw" data-c="#ffffff"
            style="background:#ffffff" title="White"></button>
          <button class="sw" data-c="#16202b"
            style="background:#16202b" title="Ink"></button>
          <span class="fmt-lab" id="fmt-bglab" hidden>box</span>
          <button class="sw swbg trans" data-c="none" hidden
            title="Transparent box"></button>
          <button class="sw swbg" data-c="#0e1926" hidden
            style="background:#0e1926" title="Dark box"></button>
          <button class="sw swbg" data-c="#ffffff" hidden
            style="background:#ffffff" title="White box"></button>
          <button class="sw swbg" data-c="#ff6b57" hidden
            style="background:#ff6b57" title="Coral box"></button>
          <button class="sw swbg" data-c="#f0a848" hidden
            style="background:#f0a848" title="Amber box"></button>
          <button class="sw swbg" data-c="#39a9c0" hidden
            style="background:#39a9c0" title="Cyan box"></button>
          <button class="dbtn etm" id="fmt-smaller"
            title="Smaller text">A&#8722;</button>
          <button class="dbtn etm" id="fmt-bigger"
            title="Bigger text">A+</button>
          <select class="etm" id="fmt-font" hidden
            title="Text font">
            <option value="sans">Sans</option>
            <option value="serif">Serif</option>
            <option value="mono">Mono</option>
            <option value="system">System</option>
            <option value="hand">Hand</option>
          </select>
          <button class="dbtn etm" id="fmt-bold"
            title="Bold"><b>B</b></button>
          <button class="dbtn etm" id="fmt-ital"
            title="Italic"><i>I</i></button>
          <button class="dbtn etm" id="fmt-list"
            title="Bullet list (Enter adds a point)">&#8226; List</button>
          <button class="dbtn etm" id="fmt-line"
            title="Cycle line thickness">Line</button>
          <button class="dbtn etm" id="fmt-dash"
            title="Dashed on/off">Dash</button>
          <button class="dbtn etm" id="fmt-fill"
            title="Fill on/off">Fill</button>
          <button class="dbtn etm" id="fmt-shape"
            title="Box or ellipse">&#9711;</button>
          <button class="dbtn etm" id="fmt-op"
            title="Cycle transparency">Op</button>
          <button class="dbtn etm" id="fmt-rotl"
            title="Rotate left 15&#176;">&#10226;</button>
          <button class="dbtn etm" id="fmt-rotr"
            title="Rotate right 15&#176;">&#10227;</button>
          <button class="dbtn etm" id="fmt-dup"
            title="Duplicate (Ctrl+D)">&#10697;</button>
          <button class="dbtn etm" id="fmt-front"
            title="Bring to front">&#8613;</button>
          <button class="dbtn etm" id="fmt-back"
            title="Send to back">&#8615;</button>
          <button class="dbtn etm" id="fmt-replace"
            title="Swap in a different notebook card">&#8644;
            Replace</button>
        </span>
        <span class="et-hint" id="et-hint"></span>
        <span class="deck-spring"></span>
        <button class="dbtn" id="et-del" disabled
          title="Delete the selected item (Del)">Delete</button>
        <button class="dbtn primary" id="et-done"
          title="Back to the builder (Esc)">Done</button>
      </div>
      <button class="deck-arrow prev" id="deck-prev"
        title="Previous slide (&#8592;)"
        aria-label="Previous slide">&#8249;</button>
      <div class="deck-stage" id="deck-stage"></div>
      <button class="deck-arrow next" id="deck-next"
        title="Next slide (&#8594;)"
        aria-label="Next slide">&#8250;</button>
      <button class="deck-arrow up" id="deck-up" hidden
        title="Back to the slide (&#8593;)"
        aria-label="Back up to the slide">&#8593;</button>
      <button class="deck-codepill" id="deck-down" hidden
        title="Scroll down to the code trace (&#8595;)"
        aria-label="Show the code trace that made this slide">
        <span class="cp-arr">&#8595;</span> Show code</button>
      <span class="deck-count" id="deck-count"></span>
    </div>
  </div>
  <div class="vfull" id="vfull" hidden>
    <div class="vfull-head">
      <span class="chain-badge" id="vfull-badge"></span>
      <span class="vfull-t" id="vfull-t"></span>
      <button class="dbtn" id="vfull-close"
        title="Close (Esc)">&#10005; Close</button>
    </div>
    <div class="vfull-body" id="vfull-body"></div>
  </div>
  <div class="deck-toast" id="deck-toast" hidden></div>
</div>
<div class="pickbar" id="pickbar" hidden>
  <span>&#128204; Click a card in the notebook to place it in the
  slide</span>
  <span class="deck-spring"></span>
  <button class="dbtn" id="pick-cancel">Cancel (Esc)</button>
</div>
"""

_DECK_CSS = r"""
.deck{position:fixed;inset:0;z-index:100;background:#0b141d;color:#dce6ee;
  display:flex;flex-direction:column;font-family:var(--sans);}
.deck[hidden]{display:none!important;}
.deck [hidden]{display:none!important;}
body.deck-open{overflow:hidden;}

.deck-top{display:flex;align-items:center;gap:9px;padding:10px 18px;
  border-bottom:1px solid #ffffff14;background:#0e1926;flex:none;}
.deck-brand{font-family:var(--mono);font-size:10.5px;letter-spacing:.22em;
  text-transform:uppercase;color:var(--cyan);}
.deck-status{font-family:var(--mono);font-size:10px;padding:3px 10px;
  border-radius:20px;background:#ffffff12;color:#9fb2c2;letter-spacing:.06em;}
.deck-status.draft{background:#cf9a4e26;color:#e6b877;}
.deck-status.saved{background:#46a89226;color:#7fd0bd;}
.deck-status:empty{display:none;}
.deck-spring{flex:1;}
.dbtn{font-family:var(--mono);font-size:11px;border:1px solid #ffffff22;
  background:#ffffff0a;color:#cdd9e3;padding:6px 11px;border-radius:6px;
  cursor:pointer;transition:all .15s;}
.dbtn:hover{border-color:var(--cyan);color:#fff;}
.dbtn.primary{background:var(--cyan-deep);border-color:var(--cyan-deep);color:#fff;}
.dbtn.primary:hover{background:var(--cyan);}
.dbtn[aria-pressed="true"]{background:var(--cyan-deep);
  border-color:var(--cyan-deep);color:#fff;}
.deck-save{display:flex;gap:7px;align-items:center;}

.deck-main{flex:1;display:flex;min-height:0;}
.deck-stagewrap{flex:1;display:flex;flex-direction:column;min-width:0;
  position:relative;}
.deck-stage{flex:1;min-height:0;display:flex;padding:26px 78px 6px;
  overflow:hidden;}

/* editing: the slide is a real bounded 16:9 surface, so you can see
   exactly where things will sit when presented */
.deck.editing .deck-stage{align-items:center;justify-content:center;
  padding:18px 26px 10px;}
.deck.editing .slide{flex:none;width:100%;max-height:100%;
  aspect-ratio:16/9;margin:auto;background:#0b141d;
  border:2px solid #ffffff2b;border-radius:12px;
  box-shadow:0 14px 60px #00000066,inset 0 0 0 1px #00000055;}

/* vertical "code trail": each slide can descend into the cells that
   made it (down arrow / ArrowDown), one step per screen */
.vstack{flex:1;min-width:0;display:flex;flex-direction:column;
  transition:transform .35s ease;}
.vslide{flex:none;height:100%;display:flex;min-height:0;min-width:0;}
.vslide.vstep{padding:10px 0 4px;}
.vstep-in{flex:1;display:flex;flex-direction:column;min-height:0;
  min-width:0;background:#0e1926;border:1px solid #ffffff10;
  border-radius:12px;padding:16px 20px;}
.vstep-head{display:flex;align-items:center;gap:10px;flex:none;
  margin-bottom:10px;min-width:0;}
.vstep-t{font-size:15px;font-weight:600;color:#dbe7ef;flex:1;
  min-width:0;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;}
.vstep-n{font-family:var(--mono);font-size:10.5px;color:#7e93a4;
  flex:none;}
.vstep-body{flex:1;min-height:0;overflow:auto;}
.vstep-none{color:#7e93a4;font-size:13px;}
.vstep-thumb{flex:none;height:34px;max-width:56px;object-fit:contain;
  border-radius:5px;background:#fff;padding:2px;}

/* the trace map: minimised, numbered steps grouped per plot, arranged
   like the plots on the slide; colours tie group <-> plot */
.vslide.voverview{padding:10px 0 4px;}
.vo-in{flex:1;display:flex;flex-direction:column;min-height:0;
  min-width:0;gap:12px;}
.vo-title{flex:none;font-family:var(--mono);font-size:10.5px;
  letter-spacing:.18em;text-transform:uppercase;color:#7e93a4;
  text-align:center;display:flex;gap:10px;align-items:center;
  justify-content:center;flex-wrap:wrap;}
.vo-xall{font-family:var(--mono);font-size:10px;letter-spacing:.06em;
  text-transform:none;background:#ffffff0a;border:1px solid #ffffff22;
  color:#cdd9e3;border-radius:5px;padding:3px 9px;cursor:pointer;}
.vo-xall:hover{border-color:var(--cyan);color:#fff;}
.vo-plots{flex:none;display:flex;gap:16px;justify-content:center;
  flex-wrap:wrap;}
.vo-plot{display:flex;flex-direction:column;align-items:center;gap:6px;
  padding:9px 11px;border-radius:10px;background:#0e1926;
  border:2px solid #ffffff22;}
.vo-plot img{max-height:11vh;max-width:16vw;width:auto;height:auto;
  object-fit:contain;border-radius:6px;background:#fff;padding:2px;}
.vo-plot-t{font-size:11px;color:#dbe7ef;max-width:16vw;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.vo-groups{flex:1;min-height:0;display:flex;gap:14px;}
.vo-col{flex:1;min-width:0;display:flex;flex-direction:column;gap:8px;
  border:1.5px solid #ffffff1f;border-radius:12px;padding:11px;
  overflow:auto;background:#0e1926;}
.vo-col-h{flex:none;font-size:12.5px;font-weight:600;color:#dbe7ef;
  display:flex;align-items:center;gap:8px;min-width:0;}
.vo-col-h span{overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;}
.vo-step{display:flex;flex-direction:column;background:#12202e;
  border:1px solid #ffffff14;border-radius:8px;overflow:hidden;
  flex:none;min-width:0;transition:border-color .15s;}
.vo-step-h{display:flex;align-items:center;gap:9px;width:100%;
  padding:9px 11px;background:none;border:none;cursor:pointer;
  text-align:left;font-family:var(--sans);color:#c3d2df;min-width:0;}
.vo-step-h:hover{background:#1a2c3d;}
.vo-num{font-family:var(--mono);font-size:11px;font-weight:600;
  width:22px;height:22px;border-radius:6px;display:flex;
  align-items:center;justify-content:center;flex:none;
  background:#39a9c022;color:#5fc3d8;}
.vo-step-t{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font-size:12.5px;}
.vo-chev{flex:none;color:#8ba0b2;font-size:13px;
  transition:transform .2s;}
.vo-step.open .vo-chev{transform:rotate(90deg);}
.vo-full{background:none;border:none;color:#8ba0b2;cursor:pointer;
  font-size:13px;flex:none;padding:2px 5px;border-radius:4px;}
.vo-full:hover{color:#fff;background:#ffffff14;}
.vo-step-b{display:none;padding:2px 10px 10px;}
.vo-step.open .vo-step-b{display:block;}

/* scrollable playback: the slide fills the screen, the trace flows
   beneath it — scroll (or ArrowDown) between them */
.deck-stage.scrolly{display:block;overflow-y:auto;
  scroll-snap-type:y proximity;}
.vpage{height:100%;display:flex;flex-direction:column;min-width:0;
  scroll-snap-align:start;}
.vtrace{scroll-snap-align:start;display:flex;flex-direction:column;
  gap:14px;padding:26px 0 60px;min-height:70%;}
.vtrace .vo-groups{flex:none;align-items:flex-start;}
.vtrace .vo-col{overflow:visible;}
.deck-codepill{position:absolute;left:50%;transform:translateX(-50%);
  bottom:16px;z-index:7;display:flex;align-items:center;gap:8px;
  background:#16273ae0;border:1px solid #ffffff2e;border-radius:22px;
  color:#cdd9e3;font-family:var(--mono);font-size:11.5px;
  padding:9px 17px;cursor:pointer;backdrop-filter:blur(4px);
  transition:border-color .15s,color .15s;}
.deck-codepill:hover{border-color:var(--cyan);color:#fff;}
.deck-codepill[hidden]{display:none;}
.cp-arr{font-size:14px;line-height:1;}
.deck-arrow.up{left:50%;top:12px;transform:translateX(-50%);
  width:44px;height:44px;font-size:24px;}
.deck-count{position:absolute;right:18px;bottom:12px;z-index:7;
  font-family:var(--mono);font-size:11.5px;color:#7e93a4;}
.deck.creating .deck-count,.deck.editing .deck-count{display:none;}

/* one step, full screen */
.vfull{position:fixed;inset:0;z-index:135;background:#0b141dfa;
  display:flex;flex-direction:column;padding:22px 44px 30px;}
.vfull[hidden]{display:none;}
.vfull-head{display:flex;align-items:center;gap:12px;flex:none;
  margin-bottom:14px;}
.vfull-t{font-size:17px;font-weight:600;color:#eef4f8;flex:1;
  min-width:0;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;}
.vfull-body{flex:1;min-height:0;overflow:auto;}

.slide{flex:1;display:flex;flex-direction:column;min-width:0;min-height:0;
  animation:slidein .28s ease;}
@keyframes slidein{from{opacity:0;transform:translateY(8px);}
  to{opacity:1;transform:none;}}
.slide-titlecard{align-items:center;justify-content:center;text-align:center;
  gap:12px;}
.slide-eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.24em;
  text-transform:uppercase;color:var(--cyan);margin:0;}
.slide-titlecard h2{font-size:clamp(28px,4.5vw,54px);font-weight:600;
  letter-spacing:-.02em;color:#f0f6fa;margin:0;max-width:82%;line-height:1.15;}
.slide-meta{font-family:var(--mono);font-size:12px;color:#7e93a4;margin:0;}
.slide-empty{align-items:center;justify-content:center;color:#7e93a4;
  font-size:14px;text-align:center;}

.slide-head h3{font-size:clamp(18px,2.2vw,28px);font-weight:600;color:#eef4f8;
  margin:0 0 14px;letter-spacing:-.015em;}
.slide-body{flex:1;display:flex;gap:26px;min-height:0;}
.slide-fig{flex:1;min-width:0;display:flex;flex-direction:column;min-height:0;}
.slide-fig .cardbody{flex:1;min-height:0;display:flex;flex-direction:column;
  padding-left:0;}
.slide-fig .figframe{flex:1;min-height:0;display:flex;align-items:center;
  justify-content:center;border:none;border-radius:10px;padding:14px;
  overflow:hidden;box-shadow:0 10px 40px #00000055;}
.slide-fig .figframe img{max-width:100%;max-height:100%;width:auto;height:auto;
  object-fit:contain;margin:0;}
.slide-fig .note{background:#f7fafc;color:var(--ink-2);border-radius:12px;
  padding:26px 32px;overflow:auto;font-size:16.5px;line-height:1.7;}
.slide-fig .caption{flex:none;color:#a9bccb;margin-top:12px;padding-left:2px;
  font-size:14.5px;}
.slide-fig pre.result,.slide-fig pre.stream{flex:none;overflow:auto;}
.slide-fig .xr-wrap{flex:1;min-height:0;overflow:auto;}

/* halves / quarters slide layouts */
.slide-grid{flex:1;display:grid;gap:16px;min-height:0;}
.slide-grid.halves{grid-template-columns:1fr 1fr;}
.slide-grid.quarters{grid-template-columns:1fr 1fr;
  grid-template-rows:1fr 1fr;}
.spane{display:flex;flex-direction:column;min-width:0;min-height:0;
  background:#0e1926;border:1px solid #ffffff10;border-radius:10px;
  padding:12px 14px;}
.spane-t{font-size:13.5px;font-weight:600;color:#dbe7ef;margin:0 0 8px;
  letter-spacing:-.01em;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;flex:none;}
.spane .cardbody{flex:1;min-height:0;display:flex;flex-direction:column;
  padding-left:0;}
.spane .figframe{flex:1;min-height:0;display:flex;align-items:center;
  justify-content:center;border:none;border-radius:8px;padding:8px;
  overflow:hidden;}
.spane .figframe img{max-width:100%;max-height:100%;width:auto;height:auto;
  object-fit:contain;margin:0;}
.spane .note{flex:1;min-height:0;background:#f7fafc;color:var(--ink-2);
  border-radius:8px;padding:14px 18px;overflow:auto;font-size:13.5px;
  line-height:1.6;}
.spane .xr-wrap,.spane pre.result,.spane pre.stream{overflow:auto;
  min-height:0;}
.spane.empty{align-items:center;justify-content:center;color:#54677a;
  font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;
  text-transform:uppercase;}

.chain-sec{border-top:1px solid #ffffff10;}
.chain-sec:first-child{border-top:none;}
.chain-h{display:flex;align-items:center;gap:10px;width:100%;
  padding:9px 4px;margin:0;background:none;border:none;cursor:pointer;
  font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;
  text-transform:uppercase;color:#9fb2c2;text-align:left;border-radius:6px;
  transition:color .15s,background .15s;}
.chain-h:hover{color:#e6eef4;background:#ffffff08;}
.chain-chev{display:inline-block;font-size:14px;line-height:1;flex:none;
  transition:transform .2s;}
.chain-h[aria-expanded="true"] .chain-chev{transform:rotate(90deg);}
.chain-badge{font-size:9px;padding:2px 7px;border-radius:4px;
  background:#39a9c01f;color:#5fc3d8;letter-spacing:.1em;flex:none;}
.chain-t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.chain-b{padding:0 4px 10px;}

.deck-arrow{position:absolute;top:50%;transform:translateY(-50%);width:52px;
  height:52px;border-radius:50%;border:1px solid #ffffff22;background:#ffffff0a;
  color:#cdd9e3;font-size:30px;line-height:1;cursor:pointer;z-index:5;
  transition:all .15s;}
.deck-arrow:hover{border-color:var(--cyan);color:#fff;background:#39a9c022;}
.deck-arrow:disabled{opacity:.22;cursor:default;}
.deck-arrow.prev{left:13px;}
.deck-arrow.next{right:13px;}

.deck-foot{display:flex;align-items:center;justify-content:center;gap:16px;
  padding:9px 18px 13px;flex:none;}
.deck-count{font-family:var(--mono);font-size:11.5px;color:#7e93a4;}
.deck-drawer{max-height:44vh;overflow-y:auto;background:#0e1b25;
  border-top:1px solid #ffffff14;padding:14px 78px 22px;flex:none;}
.deck-drawer .steplabel{color:#8fa3b4;}
.deck-drawer pre.result,.deck-drawer pre.stream{background:#13222f;
  border-color:#ffffff14;color:#b6c6d3;}

/* ---------- create mode: deck docks left, document stays interactive */
.deck.creating{width:min(var(--dc-w),94vw);right:auto;
  border-right:1px solid #ffffff22;box-shadow:8px 0 40px #00000055;}
.deck.creating .deck-stagewrap{display:none;}
.deck.creating .deck-top{display:none;}

/* create panel header: File menu + status */
.dc-head{display:flex;align-items:center;gap:10px;padding:10px 14px;
  border-bottom:1px solid #ffffff14;background:#0b141d;}
.dc-menuwrap{position:relative;}
.dc-menu{position:absolute;left:0;top:calc(100% + 6px);z-index:30;
  background:#16273a;border:1px solid #ffffff22;border-radius:8px;
  padding:5px;min-width:214px;display:flex;flex-direction:column;
  box-shadow:0 12px 34px #00000066;}
.dc-mi{text-align:left;background:none;border:none;color:#dce6ee;
  font-size:12.5px;font-family:var(--sans);padding:8px 11px;
  border-radius:5px;cursor:pointer;transition:background .12s;}
.dc-mi:hover{background:#39a9c026;}
.dc-msep{height:1px;background:#ffffff14;margin:4px 6px;}
body.creating-docs .docs{margin-left:min(var(--dc-w),94vw);}
body.creating-docs .card{cursor:copy;}
body.creating-docs .card:hover{outline:2px solid var(--cyan);
  outline-offset:2px;}

.deck-create{flex:1;overflow-y:auto;display:flex;flex-direction:column;
  min-height:0;background:#0e1926;}
/* fixed so it hugs the panel's right edge in both creating (docked
   deck) and editing (flex column) modes, and survives panel scroll */
.dc-resize{position:fixed;top:0;bottom:0;width:6px;z-index:130;
  cursor:col-resize;
  left:calc(var(--presrail-w) + min(var(--dc-w),94vw) - 1px);}
.dc-resize:hover,.dc-resize.on{background:#39a9c066;}
@media(max-width:860px){.dc-resize{display:none;}}
.dc-block{padding:14px 14px 12px;border-bottom:1px solid #ffffff14;}
.dc-block.dc-film{flex:1;display:flex;flex-direction:column;min-height:120px;
  border-bottom:none;padding-bottom:8px;}
.dc-label{display:block;font-family:var(--mono);font-size:9.5px;
  letter-spacing:.16em;text-transform:uppercase;color:#7e93a4;
  margin-bottom:8px;}
.dc-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px;}
.dc-hint{font-size:11.5px;color:#7e93a4;line-height:1.5;margin:9px 0 0;}
.dc-spring{flex:1;}
.dc-presname{font-size:13px;font-weight:600;color:#dce6ee;padding:2px 1px;
  cursor:text;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.dc-presname:hover{color:#fff;text-decoration:underline dotted #39a9c088;}
#pres-name{width:100%;background:#16273a;border:1px solid #ffffff22;
  color:#dce6ee;font-family:var(--sans);font-size:12.5px;padding:7px 9px;
  border-radius:6px;margin-top:7px;box-sizing:border-box;}
#pres-name:focus{outline:none;border-color:var(--cyan);}
.dbtn.lay[aria-pressed="true"]{background:var(--cyan-deep);
  border-color:var(--cyan-deep);color:#fff;}
/* layout picker: little diagrams instead of words */
.dbtn.lay{padding:5px 7px;line-height:0;}
.layico{display:grid;gap:2px;width:34px;height:22px;}
.layico.full{grid-template-columns:1fr;}
.layico.halves{grid-template-columns:1fr 1fr;}
.layico.rows{grid-template-rows:1fr 1fr;}
.layico.quarters{grid-template-columns:1fr 1fr;
  grid-template-rows:1fr 1fr;}
.layico i{background:#8ba0b2;border-radius:2px;display:block;}
.dbtn.lay[aria-pressed="true"] .layico i{background:#fff;}
.layico.title{display:block;position:relative;}
.layico.title i{position:absolute;border-radius:2px;}
.layico.title .tl1{left:15%;right:15%;top:26%;height:24%;}
.layico.title .tl2{left:28%;right:28%;top:62%;height:12%;opacity:.5;}
.layico.blank{display:block;border:1.5px dashed #8ba0b2;
  border-radius:3px;}
.dbtn.lay[aria-pressed="true"] .layico.blank{border-color:#fff;}
/* title-slide text inputs */
.title-editor{display:flex;flex-direction:column;gap:7px;margin-top:9px;}
.title-editor[hidden]{display:none;}
.title-editor input{background:#16273a;border:1px solid #ffffff22;
  color:#dce6ee;font-family:var(--sans);font-size:12.5px;padding:8px 9px;
  border-radius:6px;}
.title-editor input:focus{outline:none;border-color:var(--cyan);}
.pane-editor[hidden]{display:none;}
#dc-edit{margin-top:9px;width:100%;}
/* freeform slot editor + film thumbnails: boxes at frame positions */
.pane-editor.freeform{display:block;position:relative;}
.pane-editor.freeform .pane.slot{position:absolute;padding:4px 16px
  4px 6px;}
.pane-editor.freeform .pane.slot .pane-t{font-size:9.5px;
  -webkit-line-clamp:2;}
.mini-diagram.free{display:block;position:relative;}
.mini-diagram.free .mini-pane{position:absolute;}
.mini-diagram.title{grid-template-columns:1fr;}
.mini-pane.is-title{background:#12202e;}
.mini-pane.is-title::before{content:"";position:absolute;left:18%;
  right:18%;top:32%;height:18%;background:#4d90c0;border-radius:1px;}
.mini-pane.is-title::after{content:"";position:absolute;left:30%;
  right:30%;top:60%;height:9%;background:#4d90c066;}

/* ---------- slide editor ----------
   Editing docks like an IDE: the builder panel stays on the left and
   the slide canvas takes the document area. The document chrome (tabs,
   filters) is hidden while editing — it acts on the hidden documents —
   and comes back for cell-picking / on Done. */
.deck.editing{left:var(--presrail-w);top:0;}
body.slide-editing .apptop{display:none;}
.deck.editing .deck-create{flex:0 0 var(--dc-w);
  border-right:1px solid #ffffff22;}
.edit-tools{display:flex;align-items:center;gap:7px;flex-wrap:wrap;
  padding:9px 16px;border-bottom:1px solid #ffffff14;
  background:#0e1926;flex:none;}
/* the format bar keeps its row reserved (visibility, not display) and
   scrolls instead of wrapping — the canvas below must NEVER shift when
   a selection appears */
.et-fmt{flex-basis:100%;display:flex;align-items:center;gap:7px;
  flex-wrap:nowrap;overflow-x:auto;min-height:30px;
  scrollbar-width:thin;}
.et-fmt>*{flex:none;}
.et-fmt[hidden]{display:flex;visibility:hidden;}
.et-label{font-family:var(--mono);font-size:10px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--amber);}
.et-hint{font-size:11px;color:#7e93a4;}
.dbtn.et[aria-pressed="true"]{background:var(--cyan-deep);
  border-color:var(--cyan-deep);color:#fff;}
.dbtn.etm{padding:5px 9px;}
select#fmt-font{background:#16273a;border:1px solid #ffffff22;
  color:#cdd9e3;font-family:var(--mono);font-size:11px;
  padding:5px 6px;border-radius:6px;}
select#fmt-font[hidden]{display:none;}
.dbtn.etm[aria-pressed="true"]{background:var(--cyan-deep);
  border-color:var(--cyan-deep);color:#fff;}
.et-fmt{display:flex;align-items:center;gap:6px;padding-left:10px;
  margin-left:3px;border-left:1px solid #ffffff1c;flex-wrap:wrap;}
.et-fmt[hidden]{display:none!important;}
.sw{width:18px;height:18px;border-radius:50%;padding:0;cursor:pointer;
  border:2px solid #ffffff30;}
.sw[aria-pressed="true"]{border-color:#fff;
  box-shadow:0 0 0 2px #39a9c0aa;}
.fmt-lab{font-family:var(--mono);font-size:9px;letter-spacing:.1em;
  text-transform:uppercase;color:#7e93a4;}
.sw.trans{background:#16273a;position:relative;overflow:hidden;}
.sw.trans::after{content:"";position:absolute;left:-2px;right:-2px;
  top:50%;height:2px;background:#ff6b57;
  transform:rotate(-45deg);}
.deck.editing .deck-arrow,.deck.editing .deck-foot{display:none;}
.slide{position:relative;}

/* annotation layer */
.annot-layer{position:absolute;inset:0;z-index:6;pointer-events:none;}
.deck.editing .annot-layer{pointer-events:auto;cursor:crosshair;}
.deck.editing .annot-layer.tool-select{cursor:default;}
.deck.editing .annot-layer:not(.tool-select) .an-item,
.deck.editing .annot-layer:not(.tool-select) .an-item *{
  cursor:crosshair!important;}
.annot-layer>svg{position:absolute;inset:0;width:100%;height:100%;
  overflow:visible;pointer-events:none;}
.deck.editing .annot-layer>svg{pointer-events:auto;}
.annot-layer>svg.an-svgtop{pointer-events:none!important;z-index:5;}
.an-item{pointer-events:none;}
.deck.editing .an-item{pointer-events:auto;}
.an-rect{position:absolute;border:3px solid #ff6b57;border-radius:4px;}
.deck.editing .an-rect{cursor:move;}
.an-rect.sel,.an-text.sel,.an-title.sel,.an-cell.sel{
  outline:2px dashed var(--cyan);outline-offset:3px;}
.an-text{position:absolute;max-width:60%;font-family:var(--sans);
  line-height:1.35;color:#fff;background:#0e1926d9;
  border:1px solid #ffffff2e;border-radius:8px;padding:.35em .55em;
  display:flex;align-items:flex-start;gap:.35em;}
.an-text.nobg{background:none;border:none;
  text-shadow:0 1px 4px #000d,0 0 10px #0009;}
.an-tx{white-space:pre-wrap;min-width:14px;outline:none;}
ul.an-ul{margin:0;padding-left:1.15em;list-style:disc;}
ul.an-ul li{margin:.18em 0;white-space:pre-wrap;}
.an-handle{cursor:move;color:#8ba0b2;font-size:.65em;flex:none;
  user-select:none;margin-top:.3em;}
.an-handle:hover{color:#fff;}
.an-arrow-line{fill:none;}
.an-arrow-line.sel{filter:drop-shadow(0 0 5px #39a9c0cc);}
.an-arrow-hit{stroke:transparent;stroke-width:16;fill:none;}
.deck.editing .an-arrow-hit{cursor:move;}
.an-resize{position:absolute;right:-7px;bottom:-7px;width:15px;
  height:15px;border-radius:4px;background:var(--cyan);
  border:2px solid #0b141d;cursor:nwse-resize;display:none;z-index:3;}
.an-item.sel .an-resize{display:block;}
.an-endpt{position:absolute;width:15px;height:15px;
  margin:-7.5px 0 0 -7.5px;border-radius:50%;background:var(--cyan);
  border:2px solid #0b141d;display:none;z-index:6;
  pointer-events:none;}
.deck.editing .an-endpt.sel{display:block;pointer-events:auto;
  cursor:grab;}

/* movable title / subtitle on title slides */
.an-title{position:absolute;transform:translate(-50%,-50%);
  max-width:88%;text-align:center;display:flex;gap:.4em;
  align-items:flex-start;justify-content:center;
  font-family:var(--sans);line-height:1.2;}
.an-title.t-main .an-tx{font-weight:600;letter-spacing:-.018em;}
.slide-titlefree .ttl-eyebrow{position:absolute;top:7%;left:0;right:0;
  text-align:center;font-family:var(--mono);font-size:11px;
  letter-spacing:.24em;text-transform:uppercase;color:var(--cyan);}

/* notebook-cell frames */
.an-cell{position:absolute;background:#0e1926;
  border:1.5px solid #39a9c05c;border-radius:10px;overflow:hidden;
  display:flex;flex-direction:column;}
.deck.editing .an-cell{cursor:move;}
.deck:not(.editing) .an-cell.empty{display:none;}
.an-cellhead{flex:none;display:flex;align-items:center;gap:8px;
  padding:8px 12px 0;min-width:0;}
.an-cellhead-t{font-size:13px;font-weight:600;color:#dbe7ef;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;
  min-width:0;}
.an-cellcap{flex:none;font-family:var(--serif);font-size:12.5px;
  color:#a9bccb;padding:0 12px 9px;margin:0;overflow:hidden;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;}
.slide-emptyhint{position:absolute;inset:0;display:flex;
  align-items:center;justify-content:center;color:#54677a;
  font-size:14px;margin:0;}
.an-cell .cardbody{flex:1;min-height:0;display:flex;
  flex-direction:column;padding:8px;}
.an-cell .figframe{flex:1;min-height:0;display:flex;
  align-items:center;justify-content:center;overflow:hidden;
  border:none;padding:6px;}
.an-cell .figframe+.figframe{border-top:1px solid #ffffff10;}
.an-cell .figpager{flex:1;min-height:0;display:flex;
  flex-direction:column;}
.an-cell .figpager .figpage{display:none;}
.an-cell .figpager .figpage.current{flex:1;min-height:0;display:flex;
  flex-direction:column;}
.an-cell .fp-btn{background:transparent;border-color:#ffffff22;
  color:#cdd9e3;}
.an-cell .fp-count{color:#7e93a4;}
.an-cell .cardbody.mdclamp,.spane .cardbody.mdclamp{max-height:none;}
.an-cell .cardbody.mdclamp::after,
.spane .cardbody.mdclamp::after{display:none;}
.spane .figpager{flex:1;min-height:0;display:flex;
  flex-direction:column;}
.spane .figpager .figpage.current{flex:1;min-height:0;display:flex;
  flex-direction:column;}
.an-cell .figframe img{max-width:100%;max-height:100%;width:auto;
  height:auto;object-fit:contain;margin:0;}
.an-cell .note{flex:1;min-height:0;overflow:auto;background:#f7fafc;
  color:var(--ink-2);border-radius:6px;padding:10px 14px;
  font-size:13px;}
.an-cell .xr-wrap,.an-cell pre.result,.an-cell pre.stream{
  overflow:auto;min-height:0;}
.an-cell.empty{align-items:center;justify-content:center;
  border-style:dashed;background:#0e192699;}
.an-cellpick{background:none;border:none;color:#7fb6c6;
  font-family:var(--mono);font-size:11px;letter-spacing:.05em;
  cursor:pointer;padding:14px;text-align:center;line-height:1.5;}
.an-cellpick:hover{color:#fff;}
.an-cellbtn{position:absolute;top:5px;right:5px;z-index:3;display:none;
  background:#0e1926ee;border:1px solid #39a9c066;border-radius:6px;
  color:#7fd0e0;font-family:var(--mono);font-size:10px;padding:4px 9px;
  cursor:pointer;}
.deck.editing .an-cell:hover .an-cellbtn,
.deck.editing .an-cell.sel .an-cellbtn{display:block;}
.an-cellbtn:hover{color:#fff;border-color:var(--cyan);}

/* picking a card for a cell frame */
.pickbar{position:fixed;top:var(--chrome-h);left:var(--presrail-w);
  right:0;z-index:99;background:var(--cyan-deep);color:#fff;
  padding:9px 16px;font-size:13px;display:flex;gap:12px;
  align-items:center;box-shadow:0 6px 24px #00000055;}
.pickbar[hidden]{display:none;}
.pickbar .dbtn{border-color:#ffffff55;color:#fff;}
body.picking .card{cursor:copy;}
body.picking .card:hover{outline:2px solid #fff;outline-offset:2px;}

/* pane editor: the current slide as clickable regions */
.pane-editor{aspect-ratio:16/9;display:grid;gap:6px;background:#0b141d;
  border:1px solid #ffffff22;border-radius:8px;padding:6px;margin-top:9px;}
.pane-editor.full{grid-template-columns:1fr;grid-template-rows:1fr;}
.pane-editor.halves{grid-template-columns:1fr 1fr;grid-template-rows:1fr;}
.pane-editor.quarters{grid-template-columns:1fr 1fr;
  grid-template-rows:1fr 1fr;}
.pane{position:relative;background:#12202e;border:1px dashed #ffffff26;
  border-radius:6px;cursor:pointer;display:flex;align-items:center;
  justify-content:center;padding:6px 18px 6px 8px;overflow:hidden;
  transition:border-color .15s,background .15s;}
.pane.filled{border-style:solid;background:#1b3247;}
.pane.active{border:2px solid var(--cyan);}
.pane-t{font-size:10.5px;line-height:1.35;color:#c3d2df;text-align:center;
  overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;
  -webkit-box-orient:vertical;}
.pane.empty .pane-t{color:#54677a;font-family:var(--mono);font-size:9.5px;
  letter-spacing:.1em;text-transform:uppercase;}
.pane-x{position:absolute;top:1px;right:3px;background:none;border:none;
  color:#8ba0b2;cursor:pointer;font-size:11px;padding:2px 4px;}
.pane-x:hover{color:#fff;}

/* filmstrip: mini slide thumbnails */
.film-list{flex:1;overflow-y:auto;min-height:60px;margin:0 -4px;
  padding:0 4px;}
.film-row{display:flex;align-items:center;gap:4px;border-radius:7px;
  margin-bottom:3px;cursor:grab;}
.film-row.current{background:#39a9c01c;outline:1px solid #39a9c055;}
.film-row.dragging{opacity:.45;}
.film-row.drop-above{box-shadow:0 -2px 0 var(--cyan);}
.film-row.drop-below{box-shadow:0 2px 0 var(--cyan);}
.film-label{flex:1;display:flex;align-items:center;gap:9px;background:none;
  border:none;color:#c3d2df;font-size:11.5px;padding:5px 6px;cursor:pointer;
  text-align:left;min-width:0;font-family:var(--sans);}
.film-label .film-t{overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;}
.film-label .film-n{font-family:var(--mono);font-size:9.5px;color:#6c8093;
  width:15px;flex:none;text-align:right;}
.mini-diagram{width:92px;height:52px;flex:none;display:grid;gap:2px;
  background:#0b141d;border:1px solid #ffffff22;border-radius:4px;
  padding:2px;}
.mini-diagram.full{grid-template-columns:1fr;}
.mini-diagram.halves{grid-template-columns:1fr 1fr;}
.mini-diagram.quarters{grid-template-columns:1fr 1fr;
  grid-template-rows:1fr 1fr;}
.mini-pane{position:relative;overflow:hidden;border-radius:2px;
  background:#1b2c3e;display:flex;align-items:center;
  justify-content:center;}
.mini-pane img{width:100%;height:100%;object-fit:cover;display:block;
  background:#fff;}
.mini-pane.is-note{background:#eef2f6 repeating-linear-gradient(180deg,
  #eef2f6 0,#eef2f6 4px,#b9c8d4 4px,#b9c8d4 5px);
  background-clip:padding-box;border:2px solid #eef2f6;}
.mini-pane.is-code{font-family:var(--mono);font-size:8.5px;
  color:#6f8ba3;background:#101d2a;}
.mini-pane.is-fig{background:#2a4761;}
.mini-pane.empty{background:#12202e;}

/* pane editor: faint live preview behind the title */
.pane-img{position:absolute;inset:0;width:100%;height:100%;
  object-fit:cover;opacity:.4;}
.pane.filled .pane-t{position:relative;z-index:1;color:#eef4f8;
  text-shadow:0 1px 3px #000c,0 0 8px #0008;}
.film-ctr{display:none;gap:1px;padding-right:5px;flex:none;}
.film-row:hover .film-ctr,.film-row.current .film-ctr{display:flex;}
.film-mini{background:none;border:none;color:#8ba0b2;cursor:pointer;
  font-size:11px;padding:2px 4px;border-radius:4px;}
.film-mini:hover{background:#ffffff14;color:#fff;}
.addslide{margin-top:8px;}

.deck-toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);
  background:#16273a;border:1px solid var(--cyan);color:#e6eef4;
  font-size:12.5px;font-family:var(--mono);padding:9px 16px;border-radius:8px;
  z-index:120;box-shadow:0 8px 30px #00000066;max-width:80vw;}

@media (max-width:860px){
  .deck-stage{padding:16px 52px 4px;}
  .deck-drawer{padding:12px 20px 18px;}
  .slide-grid.halves,.slide-grid.quarters{grid-template-columns:1fr;
    grid-template-rows:none;grid-auto-rows:1fr;}
}
@media (prefers-reduced-motion:reduce){.slide{animation:none;}}
"""

_DECK_JS = r"""
(function(){
  var deckEl=document.getElementById('deck');
  if(!deckEl) return;
  var APP=window.SemApp||{mode:'static',shells:{},order:[],
    project:{presentations:[],recent:[]}};

  var $=function(s,r){return (r||document).querySelector(s);};
  var $$=function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s));};
  function esc(t){var d=document.createElement('div');d.textContent=(t==null?'':String(t));return d.innerHTML;}
  function deep(o){return JSON.parse(JSON.stringify(o));}

  var stage=$('#deck-stage');
  /* layouts are just preset ARRANGEMENTS of cell frames (percent rects);
     every box on a slide is a "+ Cell" frame — movable and resizable */
  var PRESETS={
    full:[[3,4,94,91]],
    halves:[[2,7,47.5,86],[50.5,7,47.5,86]],
    rows:[[6,2,88,47],[6,51,88,47]],
    quarters:[[2,2,47.5,47],[50.5,2,47.5,47],
              [2,51,47.5,47],[50.5,51,47.5,47]]
  };
  function slideCells(s){
    return (s&&s.annots||[]).map(function(a,i){return {a:a,i:i};})
      .filter(function(p){return p.a.k==='cell';});
  }

  /* ---------- registry: every open notebook's cards ----------
     Refs are namespaced "stem::anchor" so one deck can mix cards from
     every open tab; plain legacy anchors still resolve. */
  var ITEMS={};        /* ns -> item {..., nb, ns} */
  var SHELLITEMS={};   /* stem -> [ns, ...] in document order */
  var nbPres=[];       /* presentations embedded in notebooks (namespaced) */
  function nsKey(stem,anchor){return stem+'::'+anchor;}
  function splitRef(ref){
    var i=String(ref).indexOf('::');
    return i<0?[null,String(ref)]:[String(ref).slice(0,i),String(ref).slice(i+2)];
  }
  function resolveRef(ref){
    if(!ref) return null;
    if(ITEMS[ref]) return ITEMS[ref];
    if(String(ref).indexOf('::')>=0) return null;
    for(var s=0;s<APP.order.length;s++){
      var k=nsKey(APP.order[s],ref);
      if(ITEMS[k]) return ITEMS[k];
    }
    return null;
  }
  function normRef(ref){
    if(!ref) return null;
    var it=resolveRef(ref);
    return it?it.ns:String(ref);
  }
  function normPres(p,stem){
    /* deep-copy a presentation, namespacing plain anchors (against
       `stem` when it came from one notebook, else best-effort);
       folder, title-slide text and free annotations ride along.
       Legacy grid-pane slides convert to preset cell-frame layouts. */
    function ns(a){
      if(!a) return null;
      if(String(a).indexOf('::')>=0) return a;
      return stem?nsKey(stem,a):(normRef(a)||a);
    }
    var out={name:String(p.name||'presentation'),
      slides:(p.slides||[]).map(function(s){
        var o={layout:s.layout,
          panes:(s.panes||[]).map(ns)};
        if(s.layout==='title'){
          o.title=String(s.title||'');o.sub=String(s.sub||'');
          if(s.tprops) o.tprops=JSON.parse(JSON.stringify(s.tprops));
          if(s.sprops) o.sprops=JSON.parse(JSON.stringify(s.sprops));
        }
        if(Array.isArray(s.annots)&&s.annots.length)
          o.annots=JSON.parse(JSON.stringify(s.annots));
        (o.annots||[]).forEach(function(a){
          if(a.k==='cell'&&a.ref) a.ref=ns(a.ref);
        });
        /* legacy pane layouts -> cell frames at the preset rects */
        if(o.layout!=='title'){
          if(PRESETS[o.layout]){
            var rects=PRESETS[o.layout];
            o.annots=o.annots||[];
            for(var i=0;i<rects.length;i++){
              o.annots.push({k:'cell',x:rects[i][0],y:rects[i][1],
                w:rects[i][2],h:rects[i][3],
                ref:o.panes[i]||null});
            }
          }
          o.layout='blank';
        }
        o.panes=[];
        return o;
      })};
    if(typeof p.folder==='string'&&p.folder) out.folder=p.folder;
    return out;
  }
  function registerShell(stem,data){
    Object.keys(ITEMS).forEach(function(k){
      if(ITEMS[k].nb===stem) delete ITEMS[k];});
    SHELLITEMS[stem]=[];
    (data.items||[]).forEach(function(it){
      var o={};for(var k in it) o[k]=it[k];
      o.nb=stem;o.ns=nsKey(stem,it.anchor);
      ITEMS[o.ns]=o;SHELLITEMS[stem].push(o.ns);
    });
    nbPres=nbPres.filter(function(p){return p.origin!==stem;});
    (data.presentations||[]).forEach(function(p){
      var cp=normPres(p,stem);cp.origin=stem;nbPres.push(cp);
    });
  }
  function unregisterShell(stem){
    Object.keys(ITEMS).forEach(function(k){
      if(ITEMS[k].nb===stem) delete ITEMS[k];});
    delete SHELLITEMS[stem];
    nbPres=nbPres.filter(function(p){return p.origin!==stem;});
  }
  APP.order.forEach(function(stem){
    registerShell(stem,APP.shells[stem].data||{});});

  /* ---------- saved presentations: project file + notebook-embedded --- */
  var projectPres=(APP.project&&Array.isArray(APP.project.presentations))
    ?deep(APP.project.presentations).map(function(p){return normPres(p);})
    :[];
  function allSaved(){
    var out=[],seen={};
    projectPres.forEach(function(p){out.push(p);seen[p.name]=1;});
    nbPres.forEach(function(p){
      var n=p.name;
      if(seen[n]) n=p.name+' ('+p.origin+')';
      if(seen[n]) return;
      var cp=deep(p);cp.name=n;out.push(cp);seen[n]=1;
    });
    return out;
  }
  function savedByName(name){
    return allSaved().filter(function(p){return p.name===name;})[0]||null;
  }

  /* ---------- draft persistence scope ---------- */
  var SCOPE=APP.mode==='app'?'proj:'+(APP.root||'')
    :APP.mode==='web'?'web:'+location.pathname
    :(APP.order.length>1
      ?'bundle:'+APP.order.slice().sort().join('+')
      :(APP.order[0]||document.title));
  var PFX='sempres:'+SCOPE+':';
  function lsGet(k){try{return localStorage.getItem(k);}catch(e){return null;}}
  function lsSet(k,v){try{localStorage.setItem(k,v);}catch(e){}}
  function lsDel(k){try{localStorage.removeItem(k);}catch(e){}}
  function loadDraft(name){
    var raw=lsGet(PFX+name); if(!raw) return null;
    try{var d=JSON.parse(raw);
      return (d&&Array.isArray(d.slides))?normPres(d):null;
    }catch(e){return null;}
  }
  function draftNames(){
    var out=[];
    try{
      for(var i=0;i<localStorage.length;i++){
        var k=localStorage.key(i);
        if(k&&k.indexOf(PFX)===0){
          var nm=k.slice(PFX.length);
          if(nm&&nm!=='last'&&out.indexOf(nm)<0) out.push(nm);
        }
      }
    }catch(e){}
    return out.sort();
  }
  function fullFrame(ref){
    var r=PRESETS.full[0];
    return {k:'cell',x:r[0],y:r[1],w:r[2],h:r[3],ref:ref||null};
  }
  function emptySlide(){
    return {layout:'blank',panes:[],annots:[fullFrame(null)]};
  }
  function autoSlides(withDocs){
    var out=[];
    APP.order.forEach(function(stem){
      (SHELLITEMS[stem]||[]).forEach(function(ns){
        var it=ITEMS[ns];
        var fig=it.kind==='figure'||it.kind==='diagnostic';
        if(fig||(withDocs&&it.kind==='note'))
          out.push({layout:'blank',panes:[],
            annots:[fullFrame(ns)]});
      });
    });
    return out;
  }
  function defaultPres(){return {name:'presentation',slides:autoSlides(false)};}

  var pres=null, source='auto', mode='view', cur=0, activePane=0;
  function loadPresentation(name){
    var d=loadDraft(name);
    if(d){pres=d;source='draft';return;}
    var s=savedByName(name);
    if(s){pres=normPres(deep(s));source='saved';return;}
    pres=defaultPres();source='auto';
  }
  var last=lsGet(PFX+'last');
  if(last&&(loadDraft(last)||savedByName(last))) loadPresentation(last);
  else if(allSaved().length) loadPresentation(allSaved()[0].name);
  else {pres=defaultPres();source='auto';}

  var saveStamp=null,saveKind='';
  function fmtT(d){
    var h=d.getHours(),m=d.getMinutes();
    return (h<10?'0':'')+h+':'+(m<10?'0':'')+m;
  }
  function status(){
    var el=$('#deck-status');
    var auto=APP.mode==='app'
      &&(typeof autosaveOn==='undefined'||autosaveOn);
    if(source==='draft'){
      if(APP.mode!=='app'&&saveKind==='manual'&&saveStamp){
        el.textContent='in this browser · '+fmtT(saveStamp);
        el.className='deck-status saved';
        return;
      }
      el.textContent=auto?'unsaved — autosaving…'
        :(APP.mode==='app'?'unsaved draft'
          :'draft — kept in this browser');
    } else if(source==='saved'){
      el.textContent=saveStamp
        ?((saveKind==='auto'?'autosaved ':'saved ')+fmtT(saveStamp))
        :'saved';
    } else el.textContent='auto';
    el.className='deck-status '+source;
  }
  function markDirty(){
    source='draft';
    saveKind='';
    lsSet(PFX+(pres.name||'untitled'),JSON.stringify(pres));
    lsSet(PFX+'last',pres.name||'untitled');
    status();
    scheduleAutosave();
  }

  /* ---------- DOM cloning from the cards already on the page ---------- */
  function cardEl(ref){
    var it=resolveRef(ref); if(!it) return null;
    var sh=APP.shells[it.nb]; if(!sh) return null;
    return sh.el.querySelector(
      '.card[data-anchor="'+String(it.anchor).replace(/"/g,'\\"')+'"]');
  }
  function stripIds(node){
    if(node.removeAttribute) node.removeAttribute('id');
    $$('[id]',node).forEach(function(n){n.removeAttribute('id');});
    return node;
  }
  function cloneBody(ref){
    var c=cardEl(ref); if(!c) return null;
    var b=$('.cardbody',c); if(!b) return null;
    return stripIds(b.cloneNode(true));
  }
  function cloneCode(ref){
    var c=cardEl(ref); if(!c) return null;
    var inner=$('.codeinner',c); if(!inner) return null;
    return stripIds(inner.cloneNode(true));
  }
  function typeset(el){
    if(window.MathJax&&MathJax.typesetPromise){
      MathJax.typesetPromise([el]).catch(function(){});}
  }
  function multiNb(){return APP.order.length>1;}
  function nbChip(cls,stem){
    var c=document.createElement('span');c.className=cls;
    c.textContent=stem;return c;
  }
  /* ---------- view mode: slide rendering + vertical code trail ------
     Horizontal = the story; vertical = how each slide was made. Every
     framed card contributes its full upstream chain (open data ->
     transforms -> plot), deduped, in execution order — one cell per
     screen below the slide. */
  var vGroups=[];
  var TRACE_COLORS=['#39a9c0','#ff6b57','#f0a848','#46a892',
    '#c98fd0','#5b8dd6'];
  function lineageFor(s){
    /* one group per framed card, ordered like the frames sit on the
       slide (row by row, left to right); each group = that card's full
       chain + its own code */
    var frames=[],seen={};
    (s.annots||[]).forEach(function(a){
      if(a.k!=='cell'||!a.ref) return;
      var it=resolveRef(a.ref);
      if(it&&!seen[it.ns]){seen[it.ns]=1;frames.push({a:a,it:it});}
    });
    frames.sort(function(p,q){
      var ry=Math.round((p.a.y||0)/12)-Math.round((q.a.y||0)/12);
      return ry!==0?ry:((p.a.x||0)-(q.a.x||0));
    });
    var groups=[];
    frames.forEach(function(f){
      var steps=[],seen2={};
      (f.it.chain||[]).forEach(function(anchor){
        var ns=nsKey(f.it.nb,anchor);
        var up=ITEMS[ns];
        if(up&&up.hasCode&&!seen2[ns]){seen2[ns]=1;steps.push(up);}
      });
      if(f.it.hasCode&&!seen2[f.it.ns]) steps.push(f.it);
      if(steps.length)
        groups.push({it:f.it,steps:steps,
          color:TRACE_COLORS[groups.length%TRACE_COLORS.length]});
    });
    var flat=[];
    groups.forEach(function(g){
      g.steps.forEach(function(st,k){
        flat.push({it:st,g:g,num:k+1});
      });
    });
    return {groups:groups,flat:flat};
  }
  function plotThumb(g,glow){
    var w=document.createElement('div');w.className='vo-plot';
    if(glow){
      w.style.borderColor=g.color;
      w.style.boxShadow='0 0 16px '+g.color+'66';
    }
    var src=paneImgSrc(g.it.ns);
    if(src){
      var im=document.createElement('img');
      im.src=src;im.alt='';w.appendChild(im);
    }
    var tl=document.createElement('span');tl.className='vo-plot-t';
    tl.textContent=g.it.title;w.appendChild(tl);
    return w;
  }
  function openVFull(st){
    var vf=$('#vfull'); if(!vf) return;
    var b=$('#vfull-badge'); if(b) b.textContent=st.kind;
    var t=$('#vfull-t'); if(t) t.textContent=st.title;
    var body=$('#vfull-body');
    if(body){
      body.innerHTML='';
      var c=cloneCode(st.ns);
      if(c) body.appendChild(c);
    }
    vf.hidden=false;
  }
  function closeVFull(){
    var vf=$('#vfull'); if(vf) vf.hidden=true;
  }
  function traceStep(st,k,g,multi){
    var box=document.createElement('div');box.className='vo-step';
    var h=document.createElement('button');h.className='vo-step-h';
    h.title='Expand this cell';
    var n=document.createElement('span');n.className='vo-num';
    n.textContent=(k+1);
    if(multi){n.style.background=g.color+'26';n.style.color=g.color;}
    h.appendChild(n);
    var bd=document.createElement('span');
    bd.className='chain-badge';bd.textContent=st.kind;
    h.appendChild(bd);
    var bt=document.createElement('span');bt.className='vo-step-t';
    bt.textContent=st.title;h.appendChild(bt);
    if(multiNb()) h.appendChild(nbChip('spane-nb',st.nb));
    var fb=document.createElement('span');fb.className='vo-full';
    fb.innerHTML='&#x26F6;';fb.title='View this cell full screen';
    fb.addEventListener('click',function(e){
      e.stopPropagation();openVFull(st);});
    h.appendChild(fb);
    var ch=document.createElement('span');ch.className='vo-chev';
    ch.innerHTML='&#8250;';
    h.appendChild(ch);
    var body=document.createElement('div');body.className='vo-step-b';
    h.addEventListener('click',function(){
      var open=box.classList.toggle('open');
      if(open&&!body.firstChild){
        var c=cloneCode(st.ns);
        if(c) body.appendChild(c);
        else{
          var no=document.createElement('p');no.className='vstep-none';
          no.textContent='(no code on this card)';
          body.appendChild(no);
        }
        typeset(body);
      }
    });
    box.appendChild(h);box.appendChild(body);
    return box;
  }
  function setAllSteps(v,open){
    $$('.vo-step',v).forEach(function(box){
      if(open===box.classList.contains('open')) return;
      if(open) box.querySelector('.vo-step-h').click();
      else box.classList.remove('open');
    });
  }
  function buildTrace(){
    var multi=vGroups.length>1;
    var v=document.createElement('div');v.className='vtrace';
    var tl=document.createElement('div');tl.className='vo-title';
    var ts=document.createElement('span');
    ts.textContent='code trace — click a step to expand it';
    tl.appendChild(ts);
    var xa=document.createElement('button');xa.className='vo-xall';
    xa.textContent='Expand all';
    xa.title='Open the code of every step';
    xa.addEventListener('click',function(){setAllSteps(v,true);});
    var ca=document.createElement('button');ca.className='vo-xall';
    ca.textContent='Collapse all';
    ca.title='Fold every step back down';
    ca.addEventListener('click',function(){setAllSteps(v,false);});
    tl.appendChild(xa);tl.appendChild(ca);
    v.appendChild(tl);
    if(multi){
      var strip=document.createElement('div');strip.className='vo-plots';
      vGroups.forEach(function(g){strip.appendChild(plotThumb(g,true));});
      v.appendChild(strip);
    }
    var cols=document.createElement('div');cols.className='vo-groups';
    vGroups.forEach(function(g){
      var col=document.createElement('div');col.className='vo-col';
      if(multi){
        col.style.borderColor=g.color;
        col.style.boxShadow='0 0 16px '+g.color+'44';
      }
      var h=document.createElement('div');h.className='vo-col-h';
      if(multi) h.style.color=g.color;
      var hs=document.createElement('span');
      hs.textContent=g.it.title;h.appendChild(hs);
      col.appendChild(h);
      g.steps.forEach(function(st,k){
        col.appendChild(traceStep(st,k,g,multi));
      });
      cols.appendChild(col);
    });
    v.appendChild(cols);
    return v;
  }
  function updateVNav(){
    var down=$('#deck-down'),up=$('#deck-up');
    var inView=(mode==='view');
    var hasTrace=inView&&!!stage.querySelector('.vtrace');
    var atTop=(stage.scrollTop||0)<60;
    if(down) down.hidden=!(hasTrace&&atTop);
    if(up) up.hidden=!(hasTrace&&!atTop);
    var c=$('#deck-count');
    if(c) c.textContent=pres.slides.length
      ?((cur+1)+' / '+pres.slides.length):'0 / 0';
  }
  function scrollToTrace(){
    var tr=stage.querySelector('.vtrace');
    if(tr) tr.scrollIntoView({behavior:'smooth',block:'start'});
  }
  function scrollToSlide(){
    stage.scrollTo({top:0,behavior:'smooth'});
  }
  stage.addEventListener('scroll',function(){
    if(mode==='view') updateVNav();
  });
  function renderSlide(){
    var s=pres.slides[cur];
    stage.innerHTML='';
    vGroups=[];
    closeVFull();
    if(!s){
      stage.innerHTML='<div class="slide slide-empty"><p>No slides yet.'
        +'<br>Use <b>Create</b> to build some.</p></div>';
    } else if(s.layout==='title'){
      /* title + sub are movable items drawn by the annotation layer */
      var ts=document.createElement('div');
      ts.className='slide slide-titlefree';
      ts.innerHTML='<p class="ttl-eyebrow">'+esc(pres.name||'')+'</p>';
      stage.appendChild(ts);
    } else {
      var bs=document.createElement('div');
      bs.className='slide slide-blank';
      if(mode==='view'&&!(s.annots||[]).length){
        bs.innerHTML='<p class="slide-emptyhint">Empty slide — pick a '
          +'layout or use ✎ Edit slide.</p>';
      }
      stage.appendChild(bs);
    }
    var slideEl=stage.firstElementChild;
    if(s&&slideEl){
      attachAnnots(slideEl,s);
      typeset(slideEl);
    }
    /* playback: the code trace flows beneath the slide — scroll (or
       ArrowDown) between them; steps expand in place */
    stage.classList.remove('scrolly');
    if(mode==='view'&&s){
      var lin=lineageFor(s);
      vGroups=lin.groups;
      if(vGroups.length){
        var page=document.createElement('div');
        page.className='vpage';
        while(stage.firstChild) page.appendChild(stage.firstChild);
        stage.appendChild(page);
        stage.appendChild(buildTrace());
        stage.classList.add('scrolly');
      }
    }
    stage.scrollTop=0;
    updateVNav();
    $('#deck-prev').disabled=cur<=0;
    $('#deck-next').disabled=cur>=pres.slides.length-1;
  }

  /* ---------- free annotations: text, arrows, boxes, cell frames -----
     Stored per slide as s.annots, coordinates in % of the slide box so
     they scale with the screen; text size is % of slide height. Title
     slides also carry movable title/sub text (s.tprops / s.sprops,
     addressed with the special indices 't' / 's'). */
  var AN_NS='http://www.w3.org/2000/svg';
  var FONTMAP={sans:'var(--sans)',serif:'var(--serif)',
    mono:'var(--mono)',system:'system-ui,sans-serif',
    hand:"'Segoe Print','Comic Sans MS',cursive"};
  var tool='select', selAnnot=null, picking=-1;
  function titleProps(s,which){
    var key=which==='t'?'tprops':'sprops';
    if(!s[key]) s[key]=(which==='t')
      ?{x:50,y:42,size:6,color:'#f0f6fa'}
      :{x:50,y:58,size:2.6,color:'#7e93a4'};
    return s[key];
  }
  function annotByIdx(s,idx){
    if(idx==='t'||idx==='s') return titleProps(s,idx);
    if(typeof idx==='number') return (s.annots||[])[idx];
    return null;
  }
  function fontPx(layer,size){
    var h=layer.getBoundingClientRect().height||600;
    return Math.max(9,h*(size||2.6)/100)+'px';
  }
  function applyCommon(el,a,extraTransform){
    if(a.op!=null&&a.op<1) el.style.opacity=a.op;
    var tr=extraTransform||'';
    if(a.rot) tr+=(tr?' ':'')+'rotate('+a.rot+'deg)';
    if(tr) el.style.transform=tr;
  }
  function mkHandle(){
    var h=document.createElement('span');h.className='an-handle';
    h.title='Drag to move';h.textContent='⠿';
    return h;
  }
  function mkResize(){
    var r=document.createElement('span');r.className='an-resize';
    r.title='Drag to resize';
    return r;
  }
  function attachAnnots(slideEl,s){
    var layer=document.createElement('div');
    layer.className='annot-layer tool-'+tool;
    slideEl.appendChild(layer);
    renderAnnots(layer,s);
    if(mode==='edit') wireEditor(layer,s);
  }
  function editableText(layer,el,getVal,setVal,idx){
    try{
      el.contentEditable=(el.tagName==='UL')?'true':'plaintext-only';
      if(el.contentEditable!=='plaintext-only'&&el.tagName!=='UL')
        el.contentEditable='true';
    }catch(e){el.contentEditable='true';}
    el.spellcheck=false;
    el.addEventListener('focus',function(){
      if(tool!=='select') el.blur();
    });
    el.addEventListener('focus',function(){
      if(!getVal()) el.textContent='';
    });
    el.addEventListener('blur',function(){
      var v=(el.innerText||'').replace(/\r/g,'')
        .replace(/\n+$/,'');
      setVal(v);
      markDirty();
    });
    el.addEventListener('mousedown',function(e){
      if(tool!=='select') return;   /* placing mode: draw over me */
      e.stopPropagation();
      selectAnnot(layer,idx);
    });
  }
  function renderAnnots(layer,s){
    var editing=(mode==='edit');
    layer.innerHTML='';
    /* two svg layers: fat invisible hit-lines UNDER the items (so
       frames stay clickable), visible strokes ON TOP of everything
       (click-transparent) so arrows are never hidden behind frames */
    var svg=document.createElementNS(AN_NS,'svg');
    layer.appendChild(svg);
    var svgTop=document.createElementNS(AN_NS,'svg');
    svgTop.setAttribute('class','an-svgtop');
    var defs=document.createElementNS(AN_NS,'defs');
    svgTop.appendChild(defs);

    if(s.layout==='title'){
      ['t','s'].forEach(function(which){
        var p=titleProps(s,which);
        var d=document.createElement('div');
        d.className='an-item an-title'+(which==='t'?' t-main':'')
          +(selAnnot===which?' sel':'');
        d.style.left=p.x+'%';d.style.top=p.y+'%';
        d.style.fontSize=fontPx(layer,p.size);
        d.style.color=p.color||'#f0f6fa';
        if(p.b) d.style.fontWeight='700';
        if(p.i) d.style.fontStyle='italic';
        if(p.font&&FONTMAP[p.font])
          d.style.fontFamily=FONTMAP[p.font];
        applyCommon(d,p,'translate(-50%,-50%)');
        d.setAttribute('data-idx',which);
        if(editing) d.appendChild(mkHandle());
        var tx=document.createElement('span');tx.className='an-tx';
        var val=which==='t'?s.title:s.sub;
        tx.textContent=val
          ||(editing?(which==='t'?'Click to edit title':'subtitle'):'');
        if(editing){
          editableText(layer,tx,
            function(){return which==='t'?s.title:s.sub;},
            function(v){
              if(which==='t') s.title=v.trim();
              else s.sub=v.trim();
              renderFilm();renderPaneEditor();
            },which);
        }
        d.appendChild(tx);
        layer.appendChild(d);
      });
    }

    (s.annots||[]).forEach(function(a,i){
      if(a.k==='arrow'){
        var col=a.color||'#ff6b57';
        var mk=document.createElementNS(AN_NS,'marker');
        mk.setAttribute('id','an-head-'+i);
        mk.setAttribute('viewBox','0 0 10 10');
        mk.setAttribute('refX','8');mk.setAttribute('refY','5');
        mk.setAttribute('markerWidth','6.5');
        mk.setAttribute('markerHeight','6.5');
        mk.setAttribute('orient','auto-start-reverse');
        var mp=document.createElementNS(AN_NS,'path');
        mp.setAttribute('d','M 0 0 L 10 5 L 0 10 z');
        mp.setAttribute('fill',col);
        mk.appendChild(mp);defs.appendChild(mk);
        var ln=document.createElementNS(AN_NS,'line');
        ln.setAttribute('x1',a.x1+'%');ln.setAttribute('y1',a.y1+'%');
        ln.setAttribute('x2',a.x2+'%');ln.setAttribute('y2',a.y2+'%');
        ln.setAttribute('class','an-arrow-line'
          +(selAnnot===i?' sel':''));
        ln.setAttribute('data-idx',i);
        ln.setAttribute('stroke',col);
        ln.setAttribute('stroke-width',a.sw||3);
        if(a.dash) ln.setAttribute('stroke-dasharray','9 7');
        if(a.op!=null&&a.op<1) ln.style.opacity=a.op;
        ln.setAttribute('marker-end','url(#an-head-'+i+')');
        svgTop.appendChild(ln);
        var hit=document.createElementNS(AN_NS,'line');
        hit.setAttribute('x1',a.x1+'%');hit.setAttribute('y1',a.y1+'%');
        hit.setAttribute('x2',a.x2+'%');hit.setAttribute('y2',a.y2+'%');
        hit.setAttribute('class','an-arrow-hit an-item');
        hit.setAttribute('data-idx',i);
        svg.appendChild(hit);
        if(editing){
          ['1','2'].forEach(function(which){
            var ep=document.createElement('span');
            ep.className='an-endpt an-endpt-'+which
              +(selAnnot===i?' sel':'');
            ep.style.left=a['x'+which]+'%';
            ep.style.top=a['y'+which]+'%';
            ep.setAttribute('data-idx',i);
            ep.setAttribute('data-ep',which);
            ep.title='Drag to redirect the arrow';
            layer.appendChild(ep);
          });
        }
      } else if(a.k==='rect'){
        var r=document.createElement('div');
        r.className='an-item an-rect'+(selAnnot===i?' sel':'');
        r.style.left=a.x+'%';r.style.top=a.y+'%';
        r.style.width=(a.w||10)+'%';r.style.height=(a.h||10)+'%';
        r.style.borderColor=a.color||'#ff6b57';
        r.style.borderWidth=(a.sw||3)+'px';
        r.style.borderStyle=a.dash?'dashed':'solid';
        r.style.background=a.fill?((a.color||'#ff6b57')+'26'):'transparent';
        if(a.shape==='ellipse') r.style.borderRadius='50%';
        applyCommon(r,a);
        r.setAttribute('data-idx',i);
        if(editing) r.appendChild(mkResize());
        layer.appendChild(r);
      } else if(a.k==='cell'){
        var c=document.createElement('div');
        var it=a.ref?resolveRef(a.ref):null;
        c.className='an-item an-cell'+(it?'':' empty')
          +(selAnnot===i?' sel':'');
        c.style.left=a.x+'%';c.style.top=a.y+'%';
        c.style.width=(a.w||34)+'%';c.style.height=(a.h||30)+'%';
        applyCommon(c,a);
        c.setAttribute('data-idx',i);
        if(it){
          c.title=it.nb+' — '+it.title;
          var ch=document.createElement('div');
          ch.className='an-cellhead';
          var chT=document.createElement('span');
          chT.className='an-cellhead-t';
          chT.textContent=it.title;
          ch.appendChild(chT);
          if(multiNb()) ch.appendChild(nbChip('spane-nb',it.nb));
          c.appendChild(ch);
          var b=cloneBody(it.ns);
          if(b){
            if(b.querySelector('.figframe')){
              /* a slide frame wants the plot(s), not the printed
                 reprs that came with them */
              $$('.xr-wrap,pre.result,pre.stream,.rich,'
                +'details.alsoprinted',b).forEach(function(n){
                if(n.parentNode) n.parentNode.removeChild(n);
              });
            }
            if(a.ts) b.style.zoom=a.ts;
            c.appendChild(b);
          }
          if((a.h||30)>=55){
            var card=cardEl(it.ns);
            var cap=card?card.querySelector('.caption'):null;
            if(cap){
              var capc=stripIds(cap.cloneNode(true));
              capc.classList.add('an-cellcap');
              c.appendChild(capc);
            }
          }
          if(editing){
            var rb=document.createElement('button');
            rb.className='an-cellbtn';
            rb.innerHTML='&#8644; Replace';
            rb.title='Swap in a different notebook card';
            rb.addEventListener('mousedown',function(e){
              if(tool==='select') e.stopPropagation();});
            rb.addEventListener('click',function(e){
              if(tool!=='select') return;
              e.stopPropagation();startPick(i);});
            c.appendChild(rb);
          }
        } else if(editing){
          var pb=document.createElement('button');
          pb.className='an-cellpick';
          pb.textContent=a.ref?('missing: '+a.ref)
            :'Click to add from notebook';
          pb.addEventListener('mousedown',function(e){
            if(tool==='select') e.stopPropagation();});
          pb.addEventListener('click',function(e){
            if(tool!=='select') return;
            e.stopPropagation();startPick(i);});
          c.appendChild(pb);
        }
        if(editing) c.appendChild(mkResize());
        layer.appendChild(c);
      } else if(a.k==='text'){
        var d2=document.createElement('div');
        d2.className='an-item an-text'+(a.bg===0?' nobg':'')
          +(selAnnot===i?' sel':'');
        d2.style.left=a.x+'%';d2.style.top=a.y+'%';
        d2.style.fontSize=fontPx(layer,a.size);
        d2.style.color=a.color||'#ffffff';
        if(a.b) d2.style.fontWeight='700';
        if(a.i) d2.style.fontStyle='italic';
        if(a.font&&FONTMAP[a.font])
          d2.style.fontFamily=FONTMAP[a.font];
        if(a.bg!==0&&a.bgc){
          d2.style.background=a.bgc;
          d2.style.borderColor='transparent';
        }
        if(a.w){d2.style.width=a.w+'%';d2.style.maxWidth='none';}
        applyCommon(d2,a);
        d2.setAttribute('data-idx',i);
        if(editing) d2.appendChild(mkHandle());
        if(editing) d2.appendChild(mkResize());
        var tx2;
        if(a.list){
          tx2=document.createElement('ul');
          tx2.className='an-tx an-ul';
          String(a.text||'').split('\n').forEach(function(line){
            var li=document.createElement('li');
            li.textContent=line;
            tx2.appendChild(li);
          });
        } else {
          tx2=document.createElement('span');
          tx2.className='an-tx';
          tx2.textContent=a.text||'';
        }
        if(editing){
          editableText(layer,tx2,
            function(){return a.text;},
            function(v){a.text=v;},i);
        }
        d2.appendChild(tx2);
        layer.appendChild(d2);
      }
    });
    layer.appendChild(svgTop);
  }
  function selectAnnot(layer,idx){
    selAnnot=idx;
    $$('[data-idx]',layer).forEach(function(el){
      el.classList.toggle('sel',
        idx!==null&&el.getAttribute('data-idx')===String(idx));
    });
    var d=$('#et-del');
    if(d) d.disabled=(typeof idx!=='number');
    showFmt();
  }
  function defaultColor(kind){
    return kind==='text'?'#ffffff':'#ff6b57';
  }
  function showFmt(){
    var bar=$('#et-fmt'); if(!bar) return;
    var s=pres.slides[cur];
    var a=(s&&selAnnot!==null)?annotByIdx(s,selAnnot):null;
    if(!a){bar.hidden=true;return;}
    var kind=(selAnnot==='t'||selAnnot==='s')?'text':a.k;
    bar.hidden=false;
    function show(id,on,pressed){
      var el=$(id); if(!el) return;
      el.hidden=!on;
      if(on&&pressed!==undefined)
        el.setAttribute('aria-pressed',pressed.toString());
    }
    $$('.sw:not(.swbg)',bar).forEach(function(sw){
      sw.hidden=(kind==='cell');
      sw.setAttribute('aria-pressed',
        ((a.color||defaultColor(kind))===sw.dataset.c).toString());
    });
    var isText=(kind==='text');
    var isNum=(typeof selAnnot==='number');
    var cellText=false;
    if(kind==='cell'&&a.ref){
      var ci=resolveRef(a.ref);
      cellText=!!ci&&ci.kind!=='figure'&&ci.kind!=='diagnostic';
    }
    show('#fmt-smaller',isText||cellText);
    show('#fmt-bigger',isText||cellText);
    var fontSel=$('#fmt-font');
    if(fontSel){
      fontSel.hidden=!isText;
      if(isText) fontSel.value=a.font||'sans';
    }
    show('#fmt-bold',isText,!!a.b);
    show('#fmt-ital',isText,!!a.i);
    show('#fmt-list',isText&&isNum,!!a.list);
    show('#fmt-line',kind==='arrow'||kind==='rect');
    show('#fmt-dash',kind==='arrow'||kind==='rect',!!a.dash);
    show('#fmt-fill',kind==='rect',!!a.fill);
    show('#fmt-shape',kind==='rect',a.shape==='ellipse');
    show('#fmt-op',true);
    var opBtn=$('#fmt-op');
    if(opBtn) opBtn.textContent='Op '
      +Math.round((a.op==null?1:a.op)*100)+'%';
    show('#fmt-rotl',kind!=='arrow');
    show('#fmt-rotr',kind!=='arrow');
    show('#fmt-dup',isNum);
    show('#fmt-front',isNum&&kind!=='arrow');
    show('#fmt-back',isNum&&kind!=='arrow');
    var plainText=isText&&typeof selAnnot==='number';
    show('#fmt-txlab',isText&&kind!=='cell');
    show('#fmt-bglab',plainText);
    $$('.swbg',bar).forEach(function(sw){
      sw.hidden=!plainText;
      var cur_=(a.bg===0)?'none':(a.bgc||'#0e1926');
      sw.setAttribute('aria-pressed',(cur_===sw.dataset.c).toString());
    });
    show('#fmt-replace',kind==='cell');
  }
  function fmtApply(fn){
    var s=pres.slides[cur]; if(!s) return;
    var a=annotByIdx(s,selAnnot); if(!a) return;
    fn(a);
    markDirty();
    var l=stage.querySelector('.annot-layer');
    if(l){renderAnnots(l,s);selectAnnot(l,selAnnot);}
  }
  function pctPoint(layer,ev){
    var r=layer.getBoundingClientRect();
    return {x:Math.max(0,Math.min(100,(ev.clientX-r.left)/r.width*100)),
            y:Math.max(0,Math.min(100,(ev.clientY-r.top)/r.height*100))};
  }
  function startMove(layer,s,idx,ev0){
    ev0.preventDefault();
    var a=annotByIdx(s,idx); if(!a) return;
    var start=pctPoint(layer,ev0);
    var orig=JSON.parse(JSON.stringify(a));
    function mm(ev){
      var p=pctPoint(layer,ev);
      var dx=p.x-start.x,dy=p.y-start.y;
      if(a.k==='arrow'){
        a.x1=orig.x1+dx;a.y1=orig.y1+dy;
        a.x2=orig.x2+dx;a.y2=orig.y2+dy;
      } else {a.x=orig.x+dx;a.y=orig.y+dy;}
      renderAnnots(layer,s);selectAnnot(layer,idx);
    }
    function mu(){
      document.removeEventListener('mousemove',mm);
      document.removeEventListener('mouseup',mu);
      markDirty();
    }
    document.addEventListener('mousemove',mm);
    document.addEventListener('mouseup',mu);
  }
  function startResize(layer,s,idx,ev0){
    ev0.preventDefault();ev0.stopPropagation();
    var a=annotByIdx(s,idx);
    if(!a||typeof idx!=='number') return;
    var start=pctPoint(layer,ev0);
    var el=layer.querySelector('.an-item[data-idx="'+idx+'"]');
    var lr=layer.getBoundingClientRect();
    var er=el?el.getBoundingClientRect():null;
    var ow=a.w||(er?er.width/lr.width*100:10);
    var oh=a.h||(er?er.height/lr.height*100:10);
    function mm(ev){
      var p=pctPoint(layer,ev);
      a.w=Math.max(4,ow+p.x-start.x);
      if(a.k!=='text') a.h=Math.max(4,oh+p.y-start.y);
      renderAnnots(layer,s);selectAnnot(layer,idx);
    }
    function mu(){
      document.removeEventListener('mousemove',mm);
      document.removeEventListener('mouseup',mu);
      markDirty();
    }
    document.addEventListener('mousemove',mm);
    document.addEventListener('mouseup',mu);
  }
  function startDraw(layer,s,kind,p0){
    var a=(kind==='rect')
      ?{k:'rect',x:p0.x,y:p0.y,w:0,h:0,color:'#ff6b57',sw:3}
      :{k:'arrow',x1:p0.x,y1:p0.y,x2:p0.x,y2:p0.y,
        color:'#ff6b57',sw:3};
    s.annots=s.annots||[];
    s.annots.push(a);
    var idx=s.annots.length-1;
    function mm(ev){
      var p=pctPoint(layer,ev);
      if(a.k==='rect'){
        a.x=Math.min(p0.x,p.x);a.y=Math.min(p0.y,p.y);
        a.w=Math.abs(p.x-p0.x);a.h=Math.abs(p.y-p0.y);
      } else {a.x2=p.x;a.y2=p.y;}
      renderAnnots(layer,s);
    }
    function mu(){
      document.removeEventListener('mousemove',mm);
      document.removeEventListener('mouseup',mu);
      var tiny=(a.k==='rect')?(a.w<1.5&&a.h<1.5)
        :(Math.abs(a.x2-a.x1)<1.5&&Math.abs(a.y2-a.y1)<1.5);
      if(tiny) s.annots.splice(idx,1);
      markDirty();setTool('select');
      renderAnnots(layer,s);
      if(!tiny) selectAnnot(layer,idx);
    }
    document.addEventListener('mousemove',mm);
    document.addEventListener('mouseup',mu);
  }
  function distToSeg(px,py,x1,y1,x2,y2){
    var dx=x2-x1,dy=y2-y1;
    var L2=dx*dx+dy*dy;
    var u=L2?((px-x1)*dx+(py-y1)*dy)/L2:0;
    u=Math.max(0,Math.min(1,u));
    return Math.hypot(px-(x1+u*dx),py-(y1+u*dy));
  }
  function startEndpoint(layer,s,idx,ep,ev0){
    ev0.preventDefault();
    var a=(s.annots||[])[idx];
    if(!a||a.k!=='arrow') return;
    function mm(ev){
      var p=pctPoint(layer,ev);
      a['x'+ep]=p.x;a['y'+ep]=p.y;
      renderAnnots(layer,s);selectAnnot(layer,idx);
    }
    function mu(){
      document.removeEventListener('mousemove',mm);
      document.removeEventListener('mouseup',mu);
      markDirty();
    }
    document.addEventListener('mousemove',mm);
    document.addEventListener('mouseup',mu);
  }
  function arrowAt(layer,s,ev){
    if(!s.annots) return -1;
    var r=layer.getBoundingClientRect();
    var px=ev.clientX-r.left,py=ev.clientY-r.top;
    var best=-1,bestD=12;
    s.annots.forEach(function(a,i){
      if(a.k!=='arrow') return;
      var d=distToSeg(px,py,
        a.x1/100*r.width,a.y1/100*r.height,
        a.x2/100*r.width,a.y2/100*r.height);
      if(d<bestD){bestD=d;best=i;}
    });
    return best;
  }
  function wireEditor(layer,s){
    layer.addEventListener('mousedown',function(ev){
      if(mode!=='edit') return;
      var t=ev.target;
      var item=(t.closest&&t.closest('.an-item'))
        ||(t.getAttribute&&t.classList
           &&t.classList.contains('an-item')?t:null);
      if(tool==='select'){
        /* endpoint handles first, then resize handles, then arrows
           (they render on top, so they win the click even over a
           frame), then the item */
        if(t.classList&&t.classList.contains('an-endpt')){
          var idxE=+t.getAttribute('data-idx');
          selectAnnot(layer,idxE);
          startEndpoint(layer,s,idxE,
            t.getAttribute('data-ep'),ev);
          return;
        }
        if(item&&t.classList&&t.classList.contains('an-resize')){
          var rawR=item.getAttribute('data-idx');
          var idxR=(rawR==='t'||rawR==='s')?rawR:+rawR;
          selectAnnot(layer,idxR);
          startResize(layer,s,idxR,ev);
          return;
        }
        var ai=arrowAt(layer,s,ev);
        if(ai>=0){
          selectAnnot(layer,ai);
          startMove(layer,s,ai,ev);
          return;
        }
        if(item){
          var raw=item.getAttribute('data-idx');
          var idx=(raw==='t'||raw==='s')?raw:+raw;
          selectAnnot(layer,idx);
          var handleOnly=item.classList.contains('an-text')
            ||item.classList.contains('an-title');
          if(!handleOnly
             ||(t.classList&&t.classList.contains('an-handle')))
            startMove(layer,s,idx,ev);
        } else selectAnnot(layer,null);
        return;
      }
      ev.preventDefault();
      var p=pctPoint(layer,ev);
      if(tool==='text'){
        s.annots=s.annots||[];
        s.annots.push({k:'text',x:p.x,y:p.y,text:'Text',
          size:2.6,color:'#ffffff',bg:1});
        var idx2=s.annots.length-1;
        markDirty();setTool('select');
        renderAnnots(layer,s);selectAnnot(layer,idx2);
        var tx=layer.querySelector(
          '.an-item[data-idx="'+idx2+'"] .an-tx');
        if(tx){
          tx.focus();
          try{
            var rng=document.createRange();
            rng.selectNodeContents(tx);
            var sl=window.getSelection();
            sl.removeAllRanges();sl.addRange(rng);
          }catch(e){}
        }
      } else if(tool==='cell'){
        s.annots=s.annots||[];
        s.annots.push({k:'cell',x:Math.min(p.x,64),
          y:Math.min(p.y,64),w:34,h:30,ref:null});
        markDirty();setTool('select');
        renderAnnots(layer,s);
        selectAnnot(layer,s.annots.length-1);
      } else if(tool==='rect'||tool==='arrow'){
        startDraw(layer,s,tool,p);
      }
    });
  }
  function setTool(t){
    tool=t;
    $$('#edit-tools .et').forEach(function(b){
      b.setAttribute('aria-pressed',(b.dataset.tool===t).toString());});
    var l=stage.querySelector('.annot-layer');
    if(l) l.className='annot-layer tool-'+t;
    var hint=$('#et-hint');
    if(hint) hint.textContent=
      t==='text'?'Click on the slide to place a text box'
      :t==='arrow'?'Drag on the slide to draw an arrow'
      :t==='rect'?'Drag on the slide to draw a box'
      :t==='cell'?'Click on the slide to place a notebook-cell frame'
      :'Click an item to select; drag to move; Del removes';
  }
  function deleteSel(){
    var s=pres.slides[cur];
    if(!s||typeof selAnnot!=='number'||!s.annots
       ||selAnnot>=s.annots.length) return;
    s.annots.splice(selAnnot,1);
    if(!s.annots.length) delete s.annots;
    selAnnot=null;markDirty();
    var l=stage.querySelector('.annot-layer');
    if(l) renderAnnots(l,s);
    var d=$('#et-del'); if(d) d.disabled=true;
    showFmt();
  }

  /* ---------- picking: click a notebook card into a cell frame ------- */
  function startPick(idx){
    if(typeof idx!=='number') return;
    picking=idx;
    deckEl.hidden=true;
    document.body.classList.remove('deck-open');
    document.body.classList.remove('creating-docs');
    document.body.classList.remove('slide-editing');
    document.body.classList.add('picking');
    var pb=$('#pickbar'); if(pb) pb.hidden=false;
  }
  function endPick(ref){
    var idx=picking; picking=-1;
    document.body.classList.remove('picking');
    var pb=$('#pickbar'); if(pb) pb.hidden=true;
    if(ref!==undefined&&idx>=0){
      var s=pres.slides[cur];
      var a=s&&(s.annots||[])[idx];
      if(a&&a.k==='cell'){a.ref=ref;markDirty();}
    }
    openDeck('edit');
    var l=stage.querySelector('.annot-layer');
    if(l&&idx>=0) selectAnnot(l,idx);
  }
  document.addEventListener('click',function(e){
    if(picking<0) return;
    var t=e.target;
    if(!t||!t.closest) return;
    if(t.closest('.pickbar')) return;
    var shellEl=t.closest('.nbshell');
    if(!shellEl) return;
    var card=t.closest('.card');
    if(!card) return;
    if(t.closest('.codetoggle,.depchip,a')) return;
    e.preventDefault();e.stopPropagation();
    endPick(nsKey(shellEl.dataset.nb,card.dataset.anchor));
  },true);

  /* ---------- format bar wiring ---------- */
  $$('#et-fmt .sw').forEach(function(sw){
    sw.addEventListener('click',function(){
      fmtApply(function(a){a.color=sw.dataset.c;});
    });
  });
  function onFmt(id,fn){
    var b=$(id);
    if(b) b.addEventListener('click',function(){fmtApply(fn);});
  }
  onFmt('#fmt-smaller',function(a){
    if(a.k==='cell') a.ts=Math.max(0.5,
      Math.round((a.ts||1)/1.15*100)/100);
    else a.size=Math.max(1.2,(a.size||2.6)/1.25);});
  onFmt('#fmt-bigger',function(a){
    if(a.k==='cell') a.ts=Math.min(3,
      Math.round((a.ts||1)*1.15*100)/100);
    else a.size=Math.min(20,(a.size||2.6)*1.25);});
  onFmt('#fmt-line',function(a){
    var cur_=a.sw||3;
    a.sw=cur_>=5?2:(cur_>=3.5?5:3.5);});
  onFmt('#fmt-dash',function(a){a.dash=a.dash?0:1;});
  onFmt('#fmt-fill',function(a){a.fill=a.fill?0:1;});
  $$('#et-fmt .swbg').forEach(function(sw){
    sw.addEventListener('click',function(){
      fmtApply(function(a){
        if(sw.dataset.c==='none'){a.bg=0;}
        else{a.bg=1;a.bgc=sw.dataset.c;}
      });
    });
  });
  var fontSelEl=$('#fmt-font');
  if(fontSelEl) fontSelEl.addEventListener('change',function(){
    var v=this.value;
    fmtApply(function(a){
      if(v==='sans') delete a.font; else a.font=v;
    });
  });
  onFmt('#fmt-bold',function(a){a.b=a.b?0:1;});
  onFmt('#fmt-ital',function(a){a.i=a.i?0:1;});
  onFmt('#fmt-list',function(a){a.list=a.list?0:1;});
  onFmt('#fmt-shape',function(a){
    a.shape=(a.shape==='ellipse')?undefined:'ellipse';
    if(a.shape===undefined) delete a.shape;});
  onFmt('#fmt-op',function(a){
    var steps=[1,0.75,0.5,0.25];
    var cur_=a.op==null?1:a.op;
    var k=steps.indexOf(cur_);
    a.op=steps[(k+1)%steps.length];
    if(a.op===1) delete a.op;});
  onFmt('#fmt-rotl',function(a){
    a.rot=(((a.rot||0)-15)%360+360)%360;
    if(!a.rot) delete a.rot;});
  onFmt('#fmt-rotr',function(a){
    a.rot=(((a.rot||0)+15)%360+360)%360;
    if(!a.rot) delete a.rot;});
  function duplicateSel(){
    var s=pres.slides[cur];
    if(!s||typeof selAnnot!=='number'||!s.annots) return;
    var cp=JSON.parse(JSON.stringify(s.annots[selAnnot]));
    if(cp.k==='arrow'){
      cp.x1+=3;cp.y1+=3;cp.x2+=3;cp.y2+=3;
    } else {cp.x=(cp.x||0)+3;cp.y=(cp.y||0)+3;}
    s.annots.push(cp);
    markDirty();
    var l=stage.querySelector('.annot-layer');
    if(l){renderAnnots(l,s);selectAnnot(l,s.annots.length-1);}
  }
  var dupBtn=$('#fmt-dup');
  if(dupBtn) dupBtn.addEventListener('click',duplicateSel);
  function zMove(front){
    var s=pres.slides[cur];
    if(!s||typeof selAnnot!=='number'||!s.annots) return;
    var a=s.annots.splice(selAnnot,1)[0];
    var idx;
    if(front){s.annots.push(a);idx=s.annots.length-1;}
    else{s.annots.unshift(a);idx=0;}
    markDirty();
    var l=stage.querySelector('.annot-layer');
    if(l){renderAnnots(l,s);selectAnnot(l,idx);}
  }
  var frontBtn=$('#fmt-front');
  if(frontBtn) frontBtn.addEventListener('click',function(){
    zMove(true);});
  var backBtn=$('#fmt-back');
  if(backBtn) backBtn.addEventListener('click',function(){
    zMove(false);});
  var repBtn=$('#fmt-replace');
  if(repBtn) repBtn.addEventListener('click',function(){
    if(typeof selAnnot==='number') startPick(selAnnot);
  });
  var pickCancel=$('#pick-cancel');
  if(pickCancel) pickCancel.addEventListener('click',function(){
    endPick();
  });
  window.addEventListener('resize',function(){
    if(deckEl.hidden) return;
    var s=pres.slides[cur];
    var l=stage.querySelector('.annot-layer');
    if(s&&l) renderAnnots(l,s);
  });
  function go(n){
    cur=Math.max(0,Math.min(pres.slides.length-1,n));
    refresh();
  }

  /* ---------- create mode: sidebar UI ---------- */
  function firstEmpty(s){
    /* annot index of the first cell frame without a card, else -1 */
    if(!s) return -1;
    var cells=slideCells(s);
    for(var i=0;i<cells.length;i++)
      if(!cells[i].a.ref) return cells[i].i;
    return -1;
  }
  /* ---------- presentations rail (vertical, left edge) ----------
     One item is active at any time: the "Documents" button (builder
     closed) or a presentation (builder open editing it). */
  var presstrip=document.getElementById('presstrip');
  var FOLDKEY='sempresfold:'+SCOPE;
  var FOLDERSKEY='sempresfolders:'+SCOPE;
  function foldState(){
    try{return JSON.parse(lsGet(FOLDKEY)||'{}');}catch(e){return {};}
  }
  function toggleFold(f){
    var s=foldState();
    if(s[f]) delete s[f]; else s[f]=1;
    lsSet(FOLDKEY,JSON.stringify(s));
    renderPresTabs();
  }
  /* folders exist on their own (created empty, dragged into) */
  function explicitFolders(){
    try{
      var l=JSON.parse(lsGet(FOLDERSKEY)||'[]');
      return Array.isArray(l)?l:[];
    }catch(e){return [];}
  }
  function saveFolders(list){lsSet(FOLDERSKEY,JSON.stringify(list));}
  /* move ANY presentation (current, saved, draft, embedded) */
  function setPresFolder(nm,folder){
    var f=(folder||'').trim();
    function apply(p){
      if(f) p.folder=f; else delete p.folder;
    }
    if(nm===pres.name){apply(pres);markDirty();renderPresRow();return;}
    var hit=false;
    projectPres.forEach(function(p){
      if(p.name===nm){apply(p);hit=true;}});
    nbPres.forEach(function(p){
      if(p.name===nm){apply(p);hit=true;}});
    var raw=lsGet(PFX+nm);
    if(raw){
      try{
        var d=JSON.parse(raw);apply(d);
        lsSet(PFX+nm,JSON.stringify(d));hit=true;
      }catch(e){}
    }
    if(hit&&APP.mode==='app') scheduleAutosave();
    renderPresTabs();
  }
  function newFolder(){
    var list=explicitFolders();
    var n=1,name='folder';
    function taken(x){
      return list.indexOf(x)>=0
        ||allSaved().some(function(p){return p.folder===x;});
    }
    while(taken(name)){n++;name='folder-'+n;}
    list.push(name);saveFolders(list);
    renderPresTabs();
    var h=presstrip.querySelector(
      '.pr-folder[data-folder="'+name+'"]');
    if(h) startFolderRename(h,name);
  }
  function renameFolder(oldName,newName){
    newName=(newName||'').trim();
    if(!newName||newName===oldName) return;
    var list=explicitFolders().map(function(x){
      return x===oldName?newName:x;});
    if(list.indexOf(newName)<0) list.push(newName);
    saveFolders(list.filter(function(x,i){
      return list.indexOf(x)===i;}));
    var st=foldState();
    if(st[oldName]){delete st[oldName];st[newName]=1;
      lsSet(FOLDKEY,JSON.stringify(st));}
    allSaved().concat([pres]).forEach(function(p){
      if(p.folder===oldName) setPresFolder(p.name,newName);
    });
    draftNames().forEach(function(nm){
      var d=loadDraft(nm);
      if(d&&d.folder===oldName) setPresFolder(nm,newName);
    });
    renderPresTabs();
  }
  function deleteFolder(f){
    saveFolders(explicitFolders().filter(function(x){return x!==f;}));
    allSaved().concat([pres]).forEach(function(p){
      if(p.folder===f) setPresFolder(p.name,'');
    });
    draftNames().forEach(function(nm){
      var d=loadDraft(nm);
      if(d&&d.folder===f) setPresFolder(nm,'');
    });
    renderPresTabs();
  }
  function startFolderRename(header,f){
    var t=header.querySelector('.pr-t');
    if(!t) return;
    var inp=document.createElement('input');
    inp.className='pr-frename';
    inp.value=f;inp.spellcheck=false;
    t.replaceWith(inp);
    inp.focus();inp.select();
    function commit(){
      var v=inp.value.trim();
      if(v&&v!==f) renameFolder(f,v);
      else renderPresTabs();
    }
    inp.addEventListener('keydown',function(e){
      e.stopPropagation();
      if(e.key==='Enter') this.blur();
      if(e.key==='Escape'){this.value=f;this.blur();}
    });
    inp.addEventListener('blur',commit);
    inp.addEventListener('click',function(e){e.stopPropagation();});
  }
  function renderPresTabs(){
    if(!presstrip) return;
    presstrip.innerHTML='';
    var savedList=allSaved();
    var savedNames=savedList.map(function(p){return p.name;});
    var byName={};
    savedList.forEach(function(p){byName[p.name]=p;});
    var names=savedNames.slice();
    /* drafts stay listed even while another presentation is open */
    draftNames().forEach(function(n){
      if(names.indexOf(n)<0){
        names.push(n);
        byName[n]=loadDraft(n)||{name:n};
      }
    });
    if(names.indexOf(pres.name)<0) names.unshift(pres.name);
    byName[pres.name]=pres;   /* in-memory version wins (live folder) */
    var editing=!deckEl.hidden;

    function presItem(nm,folder){
      var isCur=nm===pres.name;
      var t=document.createElement('button');
      t.className='pr-item ptab'+(isCur?' current':'')
        +(isCur&&editing?' editing':'')
        +(savedNames.indexOf(nm)<0?' draftonly':'');
      t.setAttribute('role','tab');
      t.dataset.pres=nm;
      t.dataset.folder=folder||'';
      t.title=(isCur&&editing
        ?('Editing "'+nm+'" — click Documents (top left) to go back')
        :('Open presentation "'+nm+'" in the builder'))
        +'\nDrag onto a folder to file it';
      t.innerHTML='<span class="pr-ico">&#9654;</span>';
      var lbl=document.createElement('span');lbl.className='pr-t';
      lbl.textContent=nm||'(unnamed)';
      t.appendChild(lbl);
      t.draggable=true;
      t.addEventListener('dragstart',function(e){
        draggingPres=nm;
        t.classList.add('dragging');
        try{e.dataTransfer.setData('text/plain',nm);}catch(err){}
        e.dataTransfer.effectAllowed='move';
      });
      t.addEventListener('dragend',function(){
        draggingPres=null;
        t.classList.remove('dragging');
        clearDropMarks();
      });
      t.addEventListener('click',function(){
        if(isCur&&!deckEl.hidden) return;
        choosePresentation(nm);
      });
      return t;
    }

    /* group by folder; loose items first, then collapsible folders
       (explicitly created folders show even while empty) */
    var rootNames=[],folders={},folderOrder=[];
    explicitFolders().forEach(function(f){
      folders[f]=[];folderOrder.push(f);
    });
    names.forEach(function(nm){
      var f=(byName[nm]&&byName[nm].folder)||'';
      if(!f){rootNames.push(nm);return;}
      if(!folders[f]){folders[f]=[];folderOrder.push(f);}
      folders[f].push(nm);
    });
    rootNames.forEach(function(nm){
      presstrip.appendChild(presItem(nm,''));});
    folderOrder.sort().forEach(function(f){
      var collapsed=!!foldState()[f]
        &&!(editing&&folders[f].indexOf(pres.name)>=0);
      var h=document.createElement('div');
      h.className='pr-folder';
      h.dataset.folder=f;
      h.title='Folder "'+f+'" — click to '
        +(collapsed?'expand':'collapse')
        +'; drag presentations onto it';
      h.innerHTML='<span class="pr-fchev">'
        +(collapsed?'&#9656;':'&#9662;')+'</span>'
        +'<span class="pr-fico"><svg viewBox="0 0 16 14" width="13" '
        +'height="12" fill="currentColor"><path d="M1 3.2C1 2.5 1.5 2 '
        +'2.2 2h3.4l1.5 1.6h6.7c.7 0 1.2.5 1.2 1.2v6c0 .7-.5 1.2-1.2 '
        +'1.2H2.2C1.5 12 1 11.5 1 10.8z"/></svg></span>';
      var ft=document.createElement('span');ft.className='pr-t';
      ft.textContent=f;h.appendChild(ft);
      var fc=document.createElement('span');fc.className='pr-fcount';
      fc.textContent=folders[f].length;h.appendChild(fc);
      var ctr=document.createElement('span');ctr.className='pr-fctrl';
      [['✎','Rename folder',function(){startFolderRename(h,f);}],
       ['✕','Delete folder (contents move out)',
        function(){deleteFolder(f);}]].forEach(function(b){
        var btn=document.createElement('button');
        btn.textContent=b[0];btn.title=b[1];
        btn.addEventListener('click',function(e){
          e.stopPropagation();b[2]();});
        ctr.appendChild(btn);
      });
      h.appendChild(ctr);
      h.addEventListener('click',function(){toggleFold(f);});
      presstrip.appendChild(h);
      if(!collapsed) folders[f].forEach(function(nm){
        var it=presItem(nm,f);
        it.classList.add('infolder');
        presstrip.appendChild(it);
      });
    });
    var docsBtn=document.getElementById('pr-docs');
    if(docsBtn) docsBtn.classList.toggle('current',!editing);
  }
  /* drag & drop filing: onto a folder header (or an item inside one)
     files it; onto empty rail space moves it back to the top level */
  var draggingPres=null;
  function clearDropMarks(){
    $$('.pr-folder.dropping',presstrip).forEach(function(el){
      el.classList.remove('dropping');});
    var rail=document.getElementById('presrail');
    if(rail) rail.classList.remove('dropping-root');
  }
  (function(){
    var rail=document.getElementById('presrail');
    if(!rail) return;
    rail.addEventListener('dragover',function(e){
      if(!draggingPres) return;
      e.preventDefault();
      e.dataTransfer.dropEffect='move';
      clearDropMarks();
      var h=e.target.closest&&e.target.closest('.pr-folder');
      if(!h){
        var it=e.target.closest&&e.target.closest('.pr-item.ptab');
        if(it&&it.dataset.folder)
          h=presstrip.querySelector(
            '.pr-folder[data-folder="'+it.dataset.folder+'"]');
      }
      if(h) h.classList.add('dropping');
      else rail.classList.add('dropping-root');
    });
    rail.addEventListener('dragleave',function(e){
      if(e.target===rail) clearDropMarks();
    });
    rail.addEventListener('drop',function(e){
      if(!draggingPres) return;
      e.preventDefault();
      var f='';
      var h=e.target.closest&&e.target.closest('.pr-folder');
      if(h) f=h.dataset.folder;
      else{
        var it=e.target.closest&&e.target.closest('.pr-item.ptab');
        if(it) f=it.dataset.folder||'';
      }
      var nm=draggingPres;
      draggingPres=null;
      clearDropMarks();
      setPresFolder(nm,f);
    });
  })();
  var newFoldBtn=document.getElementById('pr-newfold');
  if(newFoldBtn) newFoldBtn.addEventListener('click',newFolder);
  function choosePresentation(nm){
    if(nm!==pres.name){
      lsSet(PFX+'last',nm);
      loadPresentation(nm);
      cur=0;activePane=firstEmpty(pres.slides[0]);
    }
    openDeck('create');
  }
  function newPresentation(){
    var n2=1,name='presentation';
    while(savedByName(name)||loadDraft(name)){
      n2++;name='presentation-'+n2;}
    /* deliberately NOT persisted yet: a new presentation only starts
       saving (draft + autosave) once you actually edit it, so clicking
       "New" never litters the project with empty decks */
    pres={name:name,slides:[emptySlide()]};
    source='auto';
    cur=0;activePane=0;
    openDeck('create');
  }

  function renderPresRow(){
    var lbl=$('#pres-current');
    if(lbl) lbl.textContent=pres.name||'(unnamed)';
    var inp=$('#pres-name');
    if(document.activeElement!==inp&&inp.value!==pres.name)
      inp.value=pres.name;
    renderPresTabs();
  }
  function renderLayoutRow(){
    var s=pres.slides[cur];
    $$('#layout-row .lay').forEach(function(b){
      /* layouts are arrangement COMMANDS now; only the title slide is
         a persistent state worth showing as pressed */
      b.setAttribute('aria-pressed',
        (!!s&&s.layout==='title'&&b.dataset.lay==='title').toString());
      b.disabled=!s;
    });
  }
  function renderPaneEditor(){
    var ed=$('#pane-editor');ed.innerHTML='';
    var s=pres.slides[cur];
    var te=$('#title-editor'), hint=$('#dc-hint'), eb=$('#dc-edit');
    var isTitle=!!s&&s.layout==='title';
    if(te){
      te.hidden=!isTitle;
      if(isTitle){
        var ti=$('#ts-title'),su=$('#ts-sub');
        if(ti&&document.activeElement!==ti) ti.value=s.title||'';
        if(su&&document.activeElement!==su) su.value=s.sub||'';
      }
    }
    if(hint) hint.textContent=isTitle
      ?'Type the slide text (or edit it right on the slide in ✎ Edit).'
      :'Every box is a frame: pick one, then click a card in the '
        +'document. ✎ Edit slide moves and resizes them.';
    if(eb){
      eb.disabled=!s||mode==='edit';
      eb.innerHTML=(mode==='edit')
        ?'&#10003; Editing this slide':'&#9998; Edit slide';
    }
    ed.hidden=isTitle;
    ed.className='pane-editor freeform';
    if(isTitle) return;
    if(!s){
      ed.innerHTML='<div class="pane empty">'
        +'<span class="pane-t">no slide</span></div>';
      return;
    }
    var cells=slideCells(s);
    if(!cells.length){
      ed.innerHTML='<div class="pane empty">'
        +'<span class="pane-t">no frames — pick a layout, or just '
        +'click a card</span></div>';
      return;
    }
    cells.forEach(function(pair){
      var a=pair.a, ai=pair.i;
      var it=a.ref?resolveRef(a.ref):null;
      var p=document.createElement('div');
      p.className='pane slot'+(a.ref?' filled':' empty')
        +(ai===activePane?' active':'');
      p.style.left=a.x+'%';p.style.top=a.y+'%';
      p.style.width=(a.w||10)+'%';p.style.height=(a.h||10)+'%';
      var src=a.ref?paneImgSrc(a.ref):null;
      if(src){
        var pim=document.createElement('img');
        pim.className='pane-img';pim.src=src;pim.alt='';
        p.appendChild(pim);
      }
      var t=document.createElement('span');t.className='pane-t';
      t.textContent=it?it.title:(a.ref?('missing: '+a.ref):'empty');
      if(it) p.title=it.nb+' — '+it.title;
      p.appendChild(t);
      if(it&&multiNb()){
        var tag=document.createElement('span');tag.className='pane-nbtag';
        tag.textContent=it.nb;p.appendChild(tag);
      }
      if(a.ref){
        var x=document.createElement('button');x.className='pane-x';
        x.textContent='✕';x.title='Clear this frame';
        x.addEventListener('click',function(e){e.stopPropagation();
          a.ref=null;activePane=ai;markDirty();refresh();});
        p.appendChild(x);
      }
      p.addEventListener('click',function(){activePane=ai;refresh();});
      ed.appendChild(p);
    });
  }
  function paneImgSrc(ref){
    var card=ref?cardEl(ref):null;
    var img=card?$('.figframe img',card):null;
    return img?img.getAttribute('src'):null;
  }
  function paneThumb(ref){
    var w=document.createElement('span');w.className='mini-pane';
    var it=ref?resolveRef(ref):null;
    if(!it){w.className+=' empty';return w;}
    var src=paneImgSrc(ref);
    if(src){
      var m=document.createElement('img');
      m.src=src;m.alt='';m.loading='lazy';
      w.appendChild(m);
    } else if(it.kind==='note'){
      w.className+=' is-note';
    } else if(it.kind==='figure'||it.kind==='diagnostic'){
      w.className+=' is-fig';
    } else {
      w.className+=' is-code';
      w.textContent='</>';
    }
    return w;
  }
  function miniDiagram(s){
    var d=document.createElement('span');
    d.className='mini-diagram free';
    if(s.layout==='title'){
      var w=document.createElement('span');
      w.className='mini-pane is-title';
      d.appendChild(w);
      return d;
    }
    var cells=slideCells(s);
    if(!cells.length){
      var e=document.createElement('span');
      e.className='mini-pane empty';
      d.appendChild(e);
      return d;
    }
    cells.forEach(function(pair){
      var a=pair.a;
      var w2=paneThumb(a.ref);
      w2.style.position='absolute';
      w2.style.left=a.x+'%';w2.style.top=a.y+'%';
      w2.style.width=(a.w||10)+'%';w2.style.height=(a.h||10)+'%';
      d.appendChild(w2);
    });
    return d;
  }
  function slideTitle(s){
    if(s.layout==='title') return s.title||'title slide';
    var cells=slideCells(s);
    for(var i=0;i<cells.length;i++){
      var it=cells[i].a.ref&&resolveRef(cells[i].a.ref);
      if(it) return it.title;
    }
    var tx=(s.annots||[]).filter(function(a){
      return a.k==='text'&&a.text;})[0];
    return tx?tx.text:'empty slide';
  }
  var draggingSlide=-1;
  function renderFilm(){
    var list=$('#film-list');list.innerHTML='';
    pres.slides.forEach(function(s,i){
      var row=document.createElement('div');
      row.className='film-row'+(i===cur?' current':'');
      row.dataset.idx=i;
      row.draggable=true;
      row.title='Drag to reorder';
      row.addEventListener('dragstart',function(e){
        draggingSlide=i;
        row.classList.add('dragging');
        try{e.dataTransfer.setData('text/plain','slide-'+i);}
        catch(err){}
        e.dataTransfer.effectAllowed='move';
      });
      row.addEventListener('dragend',function(){
        draggingSlide=-1;
        row.classList.remove('dragging');
        clearFilmMarks();
      });
      var lbl=document.createElement('button');lbl.className='film-label';
      var num=document.createElement('span');num.className='film-n';
      num.textContent=(i+1);lbl.appendChild(num);
      lbl.appendChild(miniDiagram(s));
      var tt=document.createElement('span');tt.className='film-t';
      tt.textContent=slideTitle(s);lbl.appendChild(tt);
      lbl.addEventListener('click',function(){
        cur=i;activePane=firstEmpty(s);refresh();});
      row.appendChild(lbl);
      var ctr=document.createElement('span');ctr.className='film-ctr';
      [['↑',function(){moveSlide(i,-1);},'Move slide up'],
       ['↓',function(){moveSlide(i,1);},'Move slide down'],
       ['✕',function(){delSlide(i);},'Delete slide']]
        .forEach(function(p){
        var b=document.createElement('button');b.className='film-mini';
        b.textContent=p[0];
        b.title=p[2];
        b.addEventListener('click',function(ev){
          ev.stopPropagation();p[1]();});
        ctr.appendChild(b);
      });
      row.appendChild(ctr);
      list.appendChild(row);
    });
  }
  function clearFilmMarks(){
    $$('#film-list .film-row.drop-above,#film-list .film-row.drop-below')
      .forEach(function(r){
        r.classList.remove('drop-above');
        r.classList.remove('drop-below');
      });
  }
  (function(){
    var list=$('#film-list'); if(!list) return;
    list.addEventListener('dragover',function(e){
      if(draggingSlide<0) return;
      e.preventDefault();
      e.dataTransfer.dropEffect='move';
      clearFilmMarks();
      var row=e.target.closest&&e.target.closest('.film-row');
      if(!row) return;
      var r=row.getBoundingClientRect();
      row.classList.add(
        e.clientY>r.top+r.height/2?'drop-below':'drop-above');
    });
    list.addEventListener('dragleave',function(e){
      if(e.target===list) clearFilmMarks();
    });
    list.addEventListener('drop',function(e){
      if(draggingSlide<0) return;
      e.preventDefault();
      var from=draggingSlide;
      draggingSlide=-1;
      clearFilmMarks();
      var row=e.target.closest&&e.target.closest('.film-row');
      if(!row) return;
      var to=+row.dataset.idx;
      var r=row.getBoundingClientRect();
      if(e.clientY>r.top+r.height/2) to++;
      if(to>from) to--;
      if(to===from) return;
      var moved=pres.slides.splice(from,1)[0];
      pres.slides.splice(to,0,moved);
      if(cur===from) cur=to;
      else if(from<cur&&to>=cur) cur--;
      else if(from>cur&&to<=cur) cur++;
      markDirty();refresh();
    });
  })();
  function renderCreate(){
    renderPresRow();renderLayoutRow();renderPaneEditor();renderFilm();
  }
  function moveSlide(i,d){
    var j=i+d; if(j<0||j>=pres.slides.length) return;
    var t=pres.slides[i];pres.slides[i]=pres.slides[j];pres.slides[j]=t;
    if(cur===i)cur=j; else if(cur===j)cur=i;
    markDirty();refresh();
  }
  function delSlide(i){
    pres.slides.splice(i,1);
    if(cur>=pres.slides.length) cur=Math.max(0,pres.slides.length-1);
    activePane=firstEmpty(pres.slides[cur]);
    markDirty();refresh();
  }

  /* ---------- mode switching ---------- */
  function setUIMode(m){
    mode=m;
    var creating=(m==='create'), editing=(m==='edit');
    deckEl.classList.toggle('creating',creating);
    deckEl.classList.toggle('editing',editing);
    /* the builder panel stays visible while editing a slide */
    $('#deck-create').hidden=!(creating||editing);
    var et=$('#edit-tools'); if(et) et.hidden=!editing;
    var dt=$('.deck-top',deckEl); if(dt) dt.hidden=editing;
    document.body.classList.toggle('creating-docs',
      (creating||editing)&&!deckEl.hidden);
    document.body.classList.toggle('slide-editing',
      editing&&!deckEl.hidden);
    document.body.classList.toggle('deck-open',
      !creating&&!deckEl.hidden);
    selAnnot=null;
    var db=$('#et-del'); if(db) db.disabled=true;
    var fb=$('#et-fmt'); if(fb) fb.hidden=true;
    if(editing) setTool('select');
    /* real full screen while presenting (browser chrome gone) */
    try{
      if(m==='view'&&!deckEl.hidden&&deckEl.requestFullscreen
         &&!document.fullscreenElement)
        deckEl.requestFullscreen().catch(function(){});
      else if(m!=='view'&&document.fullscreenElement)
        document.exitFullscreen().catch(function(){});
    }catch(err){}
    if(creating||editing){
      activePane=firstEmpty(pres.slides[cur]);
      renderCreate();
    }
    if(!creating) renderSlide();
  }
  function refresh(){
    if(mode==='create'){renderCreate();}
    else if(mode==='edit'){renderCreate();renderSlide();}
    else renderSlide();
  }
  function openDeck(m){
    deckEl.hidden=false;
    status();
    setUIMode(m||'view');
  }
  function closeDeck(){
    try{
      if(document.fullscreenElement)
        document.exitFullscreen().catch(function(){});
    }catch(err){}
    closeVFull();
    deckEl.hidden=true;
    document.body.classList.remove('deck-open');
    document.body.classList.remove('creating-docs');
    document.body.classList.remove('slide-editing');
    deckEl.classList.remove('creating');
    deckEl.classList.remove('editing');
    renderPresTabs();
  }
  $('#deck-docs').addEventListener('click',closeDeck);
  $('#dc-close').addEventListener('click',closeDeck);
  var prDocs=document.getElementById('pr-docs');
  if(prDocs) prDocs.addEventListener('click',closeDeck);
  var prNew=document.getElementById('pr-new');
  if(prNew) prNew.addEventListener('click',newPresentation);
  $('#pres-current').addEventListener('click',function(){
    var inp=$('#pres-name');
    this.hidden=true;
    inp.hidden=false;inp.value=pres.name;
    inp.focus();inp.select();
  });
  $('#dc-play').addEventListener('click',function(){setUIMode('view');});
  $('#deck-exit').addEventListener('click',function(){
    setUIMode('create');});
  $('#deck-prev').addEventListener('click',function(){go(cur-1);});
  $('#deck-next').addEventListener('click',function(){go(cur+1);});
  var editBtn=$('#dc-edit');
  if(editBtn) editBtn.addEventListener('click',function(){
    if(!pres.slides[cur]) return;
    setUIMode('edit');
  });
  var doneBtn=$('#et-done');
  if(doneBtn) doneBtn.addEventListener('click',function(){
    setUIMode('create');
  });
  var delBtn=$('#et-del');
  if(delBtn) delBtn.addEventListener('click',deleteSel);
  $$('#edit-tools .et').forEach(function(b){
    b.addEventListener('click',function(){setTool(b.dataset.tool);});
  });
  var downBtn=$('#deck-down');
  if(downBtn) downBtn.addEventListener('click',scrollToTrace);
  var upBtn=$('#deck-up');
  if(upBtn) upBtn.addEventListener('click',scrollToSlide);
  var vfClose=$('#vfull-close');
  if(vfClose) vfClose.addEventListener('click',closeVFull);
  document.addEventListener('fullscreenchange',function(){
    /* Esc always exits browser fullscreen (the page cannot prevent
       it), so Esc while presenting leaves the presentation entirely —
       never a windowed half-presentation state. Inner layers (the code
       overlay, the trace) close via their own ✕ / scroll instead. */
    if(document.fullscreenElement) return;
    if(mode!=='view'||deckEl.hidden) return;
    closeVFull();
    setUIMode('create');
  });
  document.addEventListener('keydown',function(e){
    if(picking>=0){
      if(e.key==='Escape'){e.preventDefault();endPick();}
      return;
    }
    if(deckEl.hidden) return;
    var tag=(e.target.tagName||'').toLowerCase();
    if(tag==='input'||tag==='select'||tag==='textarea') return;
    if(e.target.isContentEditable) return;
    if(e.key==='Escape'){
      var vf=$('#vfull');
      if(vf&&!vf.hidden) closeVFull();
      else if(mode==='view'&&(stage.scrollTop||0)>50) scrollToSlide();
      else if(mode==='view'||mode==='edit') setUIMode('create');
      else closeDeck();
    }
    else if(mode==='edit'){
      if(e.key==='Delete'||e.key==='Backspace'){
        e.preventDefault();deleteSel();
      }
      else if((e.ctrlKey||e.metaKey)&&(e.key==='d'||e.key==='D')){
        e.preventDefault();duplicateSel();
      }
    }
    else if(mode==='view'){
      if(e.key==='ArrowRight'||e.key==='PageDown'
         ||(e.key===' '&&tag!=='button')){e.preventDefault();go(cur+1);}
      else if(e.key==='ArrowLeft'||e.key==='PageUp'){
        e.preventDefault();go(cur-1);}
      else if(e.key==='ArrowDown'){
        e.preventDefault();
        if((stage.scrollTop||0)<60) scrollToTrace();
        else stage.scrollBy({top:stage.clientHeight*0.7,
          behavior:'smooth'});
      }
      else if(e.key==='ArrowUp'){
        e.preventDefault();
        if((stage.scrollTop||0)<=stage.clientHeight*0.8) scrollToSlide();
        else stage.scrollBy({top:-stage.clientHeight*0.7,
          behavior:'smooth'});
      }
    }
  });

  /* ---------- create mode: click a card in ANY open tab to place it */
  document.addEventListener('click',function(e){
    if(deckEl.hidden||mode!=='create') return;
    var t=e.target;
    if(!t||!t.closest) return;
    if(deckEl.contains(t)) return;
    if(t.closest('.apptop,.opendlg,.welcome')) return;
    var shellEl=t.closest('.nbshell');
    if(!shellEl) return;
    var card=t.closest('.card');
    if(!card) return;
    if(t.closest('.codetoggle,.depchip,a')) return;
    e.preventDefault();e.stopPropagation();
    if(!pres.slides.length){
      pres.slides.push(emptySlide());cur=0;activePane=0;
    }
    var s=pres.slides[cur];
    if(s.layout==='title'){
      toast('This is a title slide — pick a layout to add card frames');
      return;
    }
    var ref=nsKey(shellEl.dataset.nb,card.dataset.anchor);
    var target=annotByIdx(s,activePane);
    if(!target||target.k!=='cell') activePane=firstEmpty(s);
    if(activePane<0){
      /* no empty frame: add one, cascading from the last */
      var cells=slideCells(s);
      var k2=cells.length;
      s.annots=s.annots||[];
      s.annots.push({k:'cell',
        x:Math.min(6+k2*4,42),y:Math.min(6+k2*4,36),
        w:47,h:56,ref:null});
      activePane=s.annots.length-1;
    }
    annotByIdx(s,activePane).ref=ref;
    activePane=firstEmpty(s);
    markDirty();refresh();
    card.classList.add('target-flash');
    setTimeout(function(){card.classList.remove('target-flash');},700);
  },true);

  /* ---------- create mode: slide + presentation operations ---------- */
  $('#film-add').addEventListener('click',function(){
    var at=pres.slides.length?cur+1:0;
    pres.slides.splice(at,0,emptySlide());
    cur=at;activePane=firstEmpty(pres.slides[at]);
    markDirty();refresh();
  });
  $$('#layout-row .lay').forEach(function(b){
    b.addEventListener('click',function(){
      var s=pres.slides[cur]; if(!s) return;
      var lay=b.dataset.lay;
      if(lay==='title'){
        s.layout='title';
        if(s.title===undefined) s.title='';
        if(s.sub===undefined) s.sub='';
      } else if(lay==='blank'){
        /* blank = clear the empty frames, keep everything placed */
        s.layout='blank';
        s.annots=(s.annots||[]).filter(function(a){
          return a.k!=='cell'||a.ref;});
        if(!s.annots.length) delete s.annots;
      } else {
        /* arrange: reposition existing frames into the preset rects,
           add empty frames for any leftover preset slots */
        s.layout='blank';
        var rects=PRESETS[lay]||[];
        var cells=slideCells(s);
        s.annots=s.annots||[];
        rects.forEach(function(r,i){
          if(cells[i]){
            cells[i].a.x=r[0];cells[i].a.y=r[1];
            cells[i].a.w=r[2];cells[i].a.h=r[3];
          } else {
            s.annots.push({k:'cell',x:r[0],y:r[1],w:r[2],h:r[3],
              ref:null});
          }
        });
      }
      activePane=firstEmpty(s);
      markDirty();refresh();
    });
  });
  /* title-slide text fields (panel); the slide canvas mirrors them */
  [['#ts-title','title'],['#ts-sub','sub']].forEach(function(p){
    var inp=$(p[0]); if(!inp) return;
    inp.addEventListener('input',function(){
      var s=pres.slides[cur];
      if(!s||s.layout!=='title') return;
      s[p[1]]=this.value;
      markDirty();renderFilm();
      if(mode==='edit') renderSlide();
    });
  });
  /* ---- File menu ---- */
  var fileBtn=$('#dc-file'), fileMenu=$('#dc-menu');
  function closeMenu(){
    if(fileMenu&&!fileMenu.hidden){
      fileMenu.hidden=true;
      fileBtn.setAttribute('aria-expanded','false');
    }
  }
  if(fileBtn){
    fileBtn.addEventListener('click',function(e){
      e.stopPropagation();
      var open=!fileMenu.hidden;
      fileMenu.hidden=open;
      fileBtn.setAttribute('aria-expanded',(!open).toString());
    });
    document.addEventListener('click',function(e){
      if(!fileMenu.hidden&&!fileMenu.contains(e.target)) closeMenu();
    });
  }
  function menuAction(id,fn){
    var b=$(id);
    if(b) b.addEventListener('click',function(){closeMenu();fn();});
  }
  menuAction('#mi-new',newPresentation);
  menuAction('#mi-rename',function(){
    var lbl=$('#pres-current'), inp=$('#pres-name');
    if(lbl) lbl.hidden=true;
    inp.hidden=false;inp.value=pres.name;
    inp.focus();inp.select();
  });
  menuAction('#mi-auto-figs',function(){
    pres.slides=autoSlides(false);cur=0;activePane=0;
    markDirty();refresh();
    toast(pres.slides.length+' slides: one per figure, in order');
  });
  menuAction('#mi-auto-figdocs',function(){
    pres.slides=autoSlides(true);cur=0;activePane=0;
    markDirty();refresh();
    toast(pres.slides.length+' slides: figures + docs, in order');
  });
  $('#pres-name').addEventListener('input',function(){
    var old=pres.name;
    pres.name=this.value.trim();
    if(old&&old!==pres.name) lsDel(PFX+old);
    markDirty();
  });
  $('#pres-name').addEventListener('keydown',function(e){
    if(e.key==='Enter'||e.key==='Escape') this.blur();
    e.stopPropagation();
  });
  $('#pres-name').addEventListener('blur',function(){
    this.hidden=true;
    var lbl=$('#pres-current');
    if(lbl) lbl.hidden=false;
    renderPresRow();
  });

  /* ---------- persistence ---------- */
  var toastTimer;
  function toast(msg){
    var t=$('#deck-toast');t.textContent=msg;t.hidden=false;
    clearTimeout(toastTimer);
    toastTimer=setTimeout(function(){t.hidden=true;},3600);
  }
  function mergedPresentations(){
    var out=allSaved().filter(function(p){return p.name!==pres.name;})
      .map(function(p){var c=deep(p);delete c.origin;return c;});
    var cp=deep(pres);delete cp.origin;out.push(cp);
    return out;
  }
  /* strip "stem::" when only one notebook is open, so decks saved from a
     single tab stay compatible with sidecars and --embed-deck */
  function plainIfSingle(list){
    if(APP.order.length!==1) return list;
    var pfx=APP.order[0]+'::';
    function strip(a){
      return (a&&String(a).indexOf(pfx)===0)
        ?String(a).slice(pfx.length):a;
    }
    return list.map(function(p){
      var c=deep(p);
      c.slides=(c.slides||[]).map(function(s){
        s.panes=(s.panes||[]).map(strip);
        (s.annots||[]).forEach(function(a){
          if(a.k==='cell'&&a.ref) a.ref=strip(a.ref);
        });
        return s;});
      return c;});
  }
  function requireName(){
    if(pres.name) return true;
    toast('Give the presentation a name first');
    var ni=$('#pres-name');ni.hidden=false;ni.focus();
    return false;
  }
  /* ---------- app mode: save to project + autosave ---------- */
  function saveToProject(silent){
    var merged=mergedPresentations();
    return APP.api('/api/save',{presentations:merged})
      .then(function(){
        projectPres=merged;
        lsDel(PFX+(pres.name||'untitled'));
        saveStamp=new Date();saveKind=silent?'auto':'manual';
        source='saved';status();renderPresRow();
        if(!silent)
          toast('Saved "'+pres.name+'" to plotline_project.json');
      }).catch(function(e){
        if(!silent)
          toast('Save failed: '+(e&&e.message?e.message:e));
      });
  }
  var AUTOKEY='semopts:'+SCOPE+':autosave';
  var autosaveOn=(APP.mode==='app')&&lsGet(AUTOKEY)!=='0';
  var autoTimer=null;
  function scheduleAutosave(){
    if(!autosaveOn||APP.mode!=='app') return;
    clearTimeout(autoTimer);
    autoTimer=setTimeout(function(){saveToProject(true);},1200);
  }
  function renderAutosaveItem(){
    var mi=$('#mi-autosave'); if(!mi) return;
    mi.hidden=(APP.mode!=='app');
    mi.textContent='Autosave: '+(autosaveOn?'on':'off');
  }
  var miAuto=$('#mi-autosave');
  if(miAuto) miAuto.addEventListener('click',function(){
    closeMenu();
    autosaveOn=!autosaveOn;
    lsSet(AUTOKEY,autosaveOn?'1':'0');
    renderAutosaveItem();renderSaveBtn();status();
    if(autosaveOn){scheduleAutosave();toast('Autosave on');}
    else toast('Autosave off — use the Save button');
  });
  renderAutosaveItem();

  /* always-visible Save button; the File menu keeps the rest */
  var saveBtn=$('#dc-save');
  function renderSaveBtn(){
    if(!saveBtn) return;
    if(APP.mode==='app'){
      saveBtn.setAttribute('data-tip','Save now to '
        +'plotline_project.json'
        +(autosaveOn
          ?' — autosave is ON: every change saves itself about a '
            +'second later'
          :' — autosave is OFF, only this button saves'));
    } else {
      saveBtn.setAttribute('data-tip','Presentations are kept in '
        +'this browser automatically as you edit — Save confirms '
        +'it; use File › Download JSON for a shareable file');
    }
    saveBtn.removeAttribute('title');
  }
  if(saveBtn) saveBtn.addEventListener('click',function(){
    if(APP.mode==='app'){
      if(!requireName()) return;
      saveToProject(false);
      return;
    }
    lsSet(PFX+(pres.name||'untitled'),JSON.stringify(pres));
    lsSet(PFX+'last',pres.name||'untitled');
    saveStamp=new Date();saveKind='manual';
    status();
    toast('Kept in this browser — it also autosaves as you edit. '
      +'File › Download JSON gives you a file.');
  });
  renderSaveBtn();

  /* direct save-into-.ipynb is parked for now (kept for later) */
  var ENABLE_SAVE_TO_IPYNB=false;
  var writeBtn=$('#mi-save');
  if(APP.mode==='app'){
    writeBtn.textContent='Save to project';
    writeBtn.addEventListener('click',function(){
      closeMenu();
      if(!requireName()) return;
      saveToProject(false);
    });
  } else if(ENABLE_SAVE_TO_IPYNB
      &&APP.order.length===1&&window.showOpenFilePicker){
    writeBtn.addEventListener('click',function(){
      closeMenu();
      if(!requireName()) return;
      (async function(){
        try{
          var picks=await window.showOpenFilePicker({types:[{
            description:'Jupyter notebook',
            accept:{'application/json':['.ipynb']}}]});
          var h=picks[0];
          var f=await h.getFile();
          var nb=JSON.parse(await f.text());
          nb.metadata=nb.metadata||{};
          nb.metadata.semantic=nb.metadata.semantic||{};
          nb.metadata.semantic.presentations=
            plainIfSingle(mergedPresentations());
          delete nb.metadata.semantic.deck;
          var w=await h.createWritable();
          await w.write(JSON.stringify(nb,null,1));
          await w.close();
          var stem0=APP.order[0];
          nbPres=mergedPresentations().map(function(p){
            var c=normPres(p,null);c.origin=stem0;return c;});
          lsDel(PFX+(pres.name||'untitled'));
          saveStamp=new Date();saveKind='manual';
          source='saved';status();renderPresRow();
          toast('Saved "'+pres.name+'" into '+f.name);
        }catch(e){
          if(!e||e.name!=='AbortError')
            toast('Save failed: '+(e&&e.message?e.message:e));
        }
      })();
    });
  } else {
    writeBtn.hidden=true;
  }
  menuAction('#mi-dl',function(){
    var blob=new Blob(
      [JSON.stringify({presentations:plainIfSingle(mergedPresentations())},
        null,2)],
      {type:'application/json'});
    var a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download=(APP.order.length===1?APP.order[0]:'project')+'.deck.json';
    a.click();
    setTimeout(function(){URL.revokeObjectURL(a.href);},2000);
    toast(APP.order.length===1
      ?'Downloaded. Keep it next to the .ipynb (auto-loads) '
        +'or bake in with --embed-deck.'
      :'Downloaded. Load it with --deck, or save to the project instead.');
  });
  menuAction('#mi-load',function(){
    var fi=document.getElementById('deckfile');
    if(fi) fi.click();
  });
  (function(){
    var fi=document.getElementById('deckfile');
    if(!fi) return;
    fi.addEventListener('change',function(){
      var f=this.files&&this.files[0];
      this.value='';
      if(!f) return;
      f.text().then(function(txt){
        var obj=JSON.parse(txt);
        var list=(obj&&Array.isArray(obj.presentations))
          ?obj.presentations
          :Array.isArray(obj)?obj
          :(obj&&Array.isArray(obj.slides))?[obj]:null;
        if(!list||!list.length){
          toast('That file does not look like a saved deck');
          return;
        }
        var imported=0,firstName=null;
        list.forEach(function(pr){
          if(!pr||!Array.isArray(pr.slides)) return;
          var np=normPres(pr);
          var base=np.name||'imported',nm=base,k=1;
          while(savedByName(nm)||lsGet(PFX+nm)){
            k++;nm=base+'-'+k;
          }
          np.name=nm;
          lsSet(PFX+nm,JSON.stringify(np));
          if(!firstName) firstName=nm;
          imported++;
        });
        if(!imported){
          toast('No presentations found in that file');
          return;
        }
        lsSet(PFX+'last',firstName);
        loadPresentation(firstName);
        cur=0;activePane=firstEmpty(pres.slides[0]);
        status();refresh();
        toast('Imported '+imported+' presentation'
          +(imported>1?'s':'')+' (as drafts)');
      }).catch(function(e){
        toast('Import failed: '+((e&&e.message)||e));
      });
    });
  })();
  menuAction('#mi-discard',function(){
    lsDel(PFX+(pres.name||'untitled'));
    loadPresentation(pres.name);
    cur=0;activePane=firstEmpty(pres.slides[0]);
    status();
    refresh();
  });
  menuAction('#mi-del',function(){
    var nm=pres.name;
    lsDel(PFX+nm);
    var wasEmbedded=nbPres.some(function(p){return p.name===nm;});
    projectPres=projectPres.filter(function(p){return p.name!==nm;});
    nbPres=nbPres.filter(function(p){return p.name!==nm;});
    if(APP.mode==='app')
      APP.api('/api/save',{presentations:deep(projectPres)})
        .catch(function(){});
    var names=allSaved().map(function(p){return p.name;})
      .concat(draftNames());
    if(names.length) loadPresentation(names[0]);
    else {pres=defaultPres();source='auto';}
    cur=0;activePane=firstEmpty(pres.slides[0]);
    status();refresh();
    toast(wasEmbedded
      ?('Deleted "'+nm+'" (it will return if it is embedded in a '
        +'notebook’s metadata)')
      :('Deleted "'+nm+'"'));
  });

  /* ---------- tabs opened / closed while the page lives ---------- */
  document.addEventListener('sem:shell',function(e){
    registerShell(e.detail.stem,e.detail.data||{});
    if(source==='auto'&&(!pres.slides||!pres.slides.length))
      pres=defaultPres();
    if(!deckEl.hidden) refresh();
    else renderPresTabs();
  });
  document.addEventListener('sem:shellclosed',function(e){
    unregisterShell(e.detail.stem);
    if(!deckEl.hidden) refresh();
    else renderPresTabs();
  });

  status();
  renderPresTabs();
})();
"""

_SHELL_TEMPLATE = """<div class="shell nbshell" data-nb="{stem}"{path_attr}>
  <aside class="rail">
    <div class="railhead">
      <p class="brand">PlotLine</p>
      <h1 class="railtitle">{title}</h1>
      <div class="railmeta">{meta}</div>
    </div>
    {nav}
    {graph_panel}
  </aside>
  <main class="stage">
    <div class="content">
      {sections}
    </div>
    <div class="rawview">
      {rawview}
    </div>
  </main>
  <script type="application/json" class="nb-data">{nb_data}</script>
</div>
"""

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
<style>{app_css}</style>
<style>{deck_css}</style>
{mathjax}
</head>
<body>
<div class="scrim" id="scrim"></div>
<header class="apptop" id="apptop">
  <div class="appbar">
    <span class="apptop-brand">PlotLine</span>
    <button class="menubtn" id="menubtn" aria-label="Toggle sections"
      title="Show or hide the section sidebar (table of contents)">
      <span></span></button>
    <button class="tab-openbtn" id="tab-open" hidden
      title="Open notebooks">+ Open</button>
    <button class="toggle tv" id="tv-figs"
      title="Show or hide figure cards"></button>
    <button class="toggle tv" id="tv-markup"
      title="Show or hide markdown/equation cards"></button>
    <button class="toggle tv" id="tv-code"
      title="Show or hide ALL code: code cards and the code folded under
 every figure and dataset"></button>
    <button class="toggle" id="view-raw"
      title="Toggle between the semantic view and the raw notebook
 (cells in order, directives visible)">Raw notebook</button>
    <span class="appbar-spring"></span>
    <button class="toggle" id="theme-btn"
      title="Switch between dark and light theme">&#9788;</button>
    <a class="toggle appbar-link" id="support-btn" href="{kofi}"
      target="_blank" rel="noopener"
      title="Support PlotLine on Ko-fi">Support &#9829;</a>
    <button class="toggle" id="help-btn"
      title="How to use, and everything this tool can do">Help</button>
  </div>
  <div class="tabsrow">
    <span class="tabs-label">docs</span>
    <div class="tabstrip" id="tabstrip" role="tablist"
      aria-label="Open notebooks"></div>
  </div>
</header>
<nav class="presrail" id="presrail" aria-label="Presentations">
  <button class="pr-item pr-docs current" id="pr-docs"
    title="Document view — closes the presentation builder">
    <span class="pr-ico">&#9636;</span>
    <span class="pr-t">Documents</span></button>
  <div class="pr-label">presentations</div>
  <div class="pr-list" id="presstrip" role="tablist"></div>
  <button class="pr-btn" id="pr-new"
    title="Create a new presentation">
    <span class="pr-ico">+</span>
    <span class="pr-t">+ New presentation</span></button>
  <button class="pr-btn" id="pr-newfold"
    title="New folder &#8212; drag presentations into it">
    <span class="pr-ico"><svg viewBox="0 0 16 14" width="13"
      height="12" fill="currentColor"><path d="M1 3.2C1 2.5 1.5 2
      2.2 2h3.4l1.5 1.6h6.7c.7 0 1.2.5 1.2 1.2v6c0 .7-.5 1.2-1.2
      1.2H2.2C1.5 12 1 11.5 1 10.8z"/></svg></span>
    <span class="pr-t">+ New folder</span></button>
  <button class="pr-collapse" id="pr-collapse"
    title="Collapse this panel">&#171;</button>
</nav>
<button class="presrail-show" id="presrail-show"
  title="Show presentations">&#187;</button>
<div class="docs" id="docs">
{shells}
</div>
<div class="welcome" id="welcome" hidden>
  <div class="welcome-box">
    <p class="brand">PlotLine</p>
    <h1>Open a notebook</h1>
    <p class="welcome-hint">Drop .ipynb files anywhere in this window,
    or browse for them. Each notebook opens as a tab; presentations can
    mix cards from every open tab.</p>
    <div class="welcome-btns">
      <button class="dbtn primary" id="welcome-open">Browse
        files&#8230;</button>
      <button class="dbtn" id="welcome-demo" hidden>Try the example
        notebook</button>
    </div>
    <div class="welcome-links">
      <a href="#" id="welcome-help">How to use</a>
      <span class="wl-sep">&middot;</span>
      <a href="{repo}" target="_blank" rel="noopener">GitHub</a>
      <span class="wl-sep">&middot;</span>
      <a href="{kofi}" target="_blank"
        rel="noopener">Support &#9829;</a>
    </div>
    <div class="recent" id="welcome-recent"></div>
  </div>
</div>
<div class="helpdlg" id="helpdlg" hidden>
  <div class="help-box">
    <div class="help-head">
      <span class="help-title">How to use</span>
      <span class="deck-spring"></span>
      <a class="help-gh" href="{repo}" target="_blank"
        rel="noopener">GitHub &#8599;</a>
      <button class="dbtn" id="help-close" title="Close">&#10005;</button>
    </div>
    <div class="help-body">
      {help_html}
    </div>
  </div>
</div>
<div class="opendlg" id="opendlg" hidden>
  <div class="odlg-box">
    <div class="odlg-head">
      <button class="dbtn" id="odlg-up" title="Parent folder">&#8593; Up</button>
      <span class="odlg-path" id="odlg-path"></span>
      <button class="dbtn" id="odlg-files" hidden>Choose
        files&#8230;</button>
      <button class="dbtn" id="odlg-close" title="Close">&#10005;</button>
    </div>
    <div class="odlg-list" id="odlg-list"></div>
    <div class="odlg-foot">
      <div class="odlg-inrow">
        <input id="odlg-input" type="text" spellcheck="false"
          autocomplete="off"
          placeholder="&#8230;or paste a folder or .ipynb path">
        <button class="dbtn" id="odlg-go"
          title="Open the path or URL typed on the left">Open</button>
      </div>
      <div class="odlg-load" id="odlg-load" hidden><span></span></div>
    </div>
  </div>
</div>
<div class="drophint" id="drophint" hidden>Drop .ipynb files to open</div>
<input type="file" id="fileinput" accept=".ipynb" multiple hidden>
<input type="file" id="deckfile" accept=".json" hidden>
{deck_shell}
<script type="application/json" id="app-data">{app_data}</script>
<script>{js}</script>
<script>{deck_js}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_doc(path: Path, title: str | None = None,
             deck_path: Path | None = None) -> Document:
    """Parse one notebook file into a Document, with its presentations.

    Deck priority: explicit deck_path > <notebook>.deck.json sidecar >
    embedded metadata (parse_notebook already loaded that).
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    doc = parse_notebook(nb, title=title)
    doc.source_name = path.stem
    if deck_path is None:
        sidecar = path.with_suffix(".deck.json")
        if sidecar.exists():
            deck_path = sidecar
    if deck_path is not None:
        pres = _as_presentations(
            json.loads(Path(deck_path).read_text(encoding="utf-8")))
        if pres:
            doc.presentations = pres
    return doc


def render_notebook_file(path: Path, title: str | None = None,
                         deck_path: Path | None = None) -> str:
    return render_html(load_doc(path, title=title, deck_path=deck_path))


def embed_deck(nb_path: Path, deck_path: Path) -> None:
    """Write presentations JSON into metadata.semantic.presentations."""
    pres = _as_presentations(
        json.loads(deck_path.read_text(encoding="utf-8")))
    if not pres:
        raise SystemExit(f"error: {deck_path} does not look like saved "
                         "presentations (expected {'presentations': [...]})")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    sem = nb.setdefault("metadata", {}).setdefault("semantic", {})
    sem["presentations"] = pres
    sem.pop("deck", None)
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")


# --------------------------------------------------------------------------
# Notebooks by URL (GitHub links are normalized to their raw form)
# --------------------------------------------------------------------------

_GH_BLOB_RE = re.compile(
    r"^https?://github\.com/([^/]+)/([^/]+)/(?:blob|raw)/(.+)$")


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _normalize_nb_url(url: str) -> str:
    url = url.strip()
    m = _GH_BLOB_RE.match(url)
    if m:
        return ("https://raw.githubusercontent.com/"
                f"{m.group(1)}/{m.group(2)}/{m.group(3)}")
    return url


def _fetch_notebook_url(url: str) -> tuple[str, dict]:
    """Download a notebook from a URL; returns (filename, nb dict)."""
    url = _normalize_nb_url(url)
    req = urllib.request.Request(
        url, headers={"User-Agent": "semantic-render"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    nb = json.loads(data.decode("utf-8"))
    if not isinstance(nb, dict) or "cells" not in nb:
        raise ValueError(f"{url} does not look like a notebook")
    name = urllib.parse.unquote(
        urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]) \
        or "notebook.ipynb"
    return name, nb


def doc_from_url(url: str) -> Document:
    name, nb = _fetch_notebook_url(url)
    doc = parse_notebook(nb)
    doc.source_name = re.sub(r"\.ipynb$", "", name, flags=re.I) \
        or "notebook"
    return doc


# --------------------------------------------------------------------------
# Web build -- the same tool as a static, fully client-side page (Python
# runs in the browser via Pyodide). Safe to publish: no server, notebooks
# never leave the visitor's machine.
# --------------------------------------------------------------------------

def web_parse(name: str, text: str, taken_json: str = "[]") -> str:
    """Bridge for the Pyodide build: notebook JSON text -> shell HTML."""
    nb = json.loads(text)
    doc = parse_notebook(nb)
    base = re.sub(r"\.ipynb$", "", str(name), flags=re.I) or "notebook"
    taken = set(json.loads(taken_json))
    stem, n = base, 1
    while stem in taken:
        n += 1
        stem = f"{base}-{n}"
    doc.source_name = stem
    return render_shell(doc)


def build_web(outdir: Path) -> None:
    """Write a deployable static web app (index.html + this module)."""
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "index.html").write_text(_WEB_LOADER, encoding="utf-8")
    (outdir / "semantic_render.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (outdir / ".nojekyll").write_text("", encoding="utf-8")
    # bundle the example so "Try the example notebook" works same-origin
    example = Path(__file__).parent / "example_climate_analysis.ipynb"
    if example.exists():
        (outdir / "example_climate_analysis.ipynb").write_bytes(
            example.read_bytes())


_WEB_LOADER = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PlotLine &mdash; presentations from Jupyter</title>
<meta name="description" content="Streamline presentations from
 Jupyter. Display your plots and documentation - figure-first notebook
 viewing and slide decks, entirely in your browser.">
<meta property="og:title" content="PlotLine">
<meta property="og:description" content="Streamline presentations from
 Jupyter. Display your plots and documentation.">
<meta property="og:type" content="website">
<style>
  body{margin:0;background:#0a141d;color:#cdd9e3;
    font-family:ui-monospace,Menlo,Consolas,monospace;display:flex;
    align-items:center;justify-content:center;min-height:100vh;}
  .boot{text-align:center;max-width:420px;padding:30px;}
  .boot h1{font-size:14px;letter-spacing:.22em;text-transform:uppercase;
    color:#39a9c0;font-weight:600;}
  .boot p{font-size:12.5px;line-height:1.7;color:#7e93a4;}
  .bar{height:3px;background:#16273a;border-radius:3px;overflow:hidden;
    margin-top:18px;}
  .bar i{display:block;height:100%;width:30%;background:#39a9c0;
    border-radius:3px;animation:sl 1.2s ease-in-out infinite alternate;}
  @keyframes sl{from{margin-left:0}to{margin-left:70%}}
</style>
</head>
<body>
<div class="boot" id="boot">
  <h1>PlotLine</h1>
  <p id="bootmsg">Loading the Python runtime (first visit only takes a
  few seconds)&#8230;</p>
  <div class="bar"><i></i></div>
</div>
<script src="https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js"></script>
<script>
(async function(){
  var msg=document.getElementById('bootmsg');
  function say(t){if(msg) msg.textContent=t;}
  try{
    var py=await loadPyodide();
    say('Loading the renderer…');
    var src=await (await fetch('semantic_render.py')).text();
    py.FS.writeFile('semantic_render.py',src);
    py.runPython('import semantic_render as sr');
    var page=py.runPython('sr.render_page([], mode="web")');
    document.open();document.write(page);document.close();
    window.semPy={
      parse:function(name,text,taken){
        py.globals.set('_wname',String(name));
        py.globals.set('_wtext',text);
        py.globals.set('_wtaken',JSON.stringify(taken||[]));
        return py.runPython('sr.web_parse(_wname,_wtext,_wtaken)');
      }
    };
    document.dispatchEvent(new Event('sem:pyready'));
  }catch(e){
    say('Failed to start: '+(e&&e.message?e.message:e)
      +' — check your connection and reload.');
  }
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Local app server -- the GUI: open notebooks as browser tabs, build
# cross-notebook presentations, everything saved in semantic_project.json
# --------------------------------------------------------------------------

_PROJECT_FILE = "plotline_project.json"


def _stem_for(path: Path, taken: set[str]) -> str:
    base = path.stem or "notebook"
    stem, n = base, 1
    while stem in taken:
        n += 1
        stem = f"{base}-{n}"
    return stem


class _AppState:
    """Project file + open-tab session, shared across requests."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.token = secrets.token_hex(8)
        self.lock = threading.Lock()
        self.presentations: list = []
        self.open: list[str] = []
        self.recent: list[str] = []
        self._load()

    @property
    def project_path(self) -> Path:
        return self.root / _PROJECT_FILE

    def _load(self) -> None:
        path = self.project_path
        if not path.exists():
            legacy = self.root / "semantic_project.json"
            if legacy.exists():        # migrate on next save
                path = legacy
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(d, dict):
            return
        self.presentations = _as_presentations(d.get("presentations"))
        for name in ("open", "recent"):
            v = d.get(name)
            setattr(self, name,
                    [str(x) for x in v if isinstance(x, str)]
                    if isinstance(v, list) else [])

    def _write(self) -> None:
        self.project_path.write_text(
            json.dumps({"presentations": self.presentations,
                        "open": self.open, "recent": self.recent},
                       indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")

    def note_open(self, path: "Path | str") -> None:
        with self.lock:
            s = str(path)
            if s not in self.open:
                self.open.append(s)
            self.recent = ([s] + [r for r in self.recent if r != s])[:10]
            self._write()

    def note_close(self, path: str) -> None:
        with self.lock:
            self.open = [p for p in self.open if p != path]
            self._write()

    def save_presentations(self, pres: list) -> None:
        with self.lock:
            self.presentations = pres
            self._write()

    def stems_taken(self, skip: Path | None = None,
                    skip_str: str | None = None) -> set[str]:
        """Deduped stems of the open tabs (mirrors the page-build order)."""
        taken: set[str] = set()
        for p in self.open:
            if skip is not None and not _is_url(p) and Path(p) == skip:
                continue
            if skip_str is not None and p == skip_str:
                continue
            taken.add(_stem_for(Path(p), taken))
        return taken


def _list_dir(raw: str) -> dict:
    d = Path(raw).expanduser()
    if not d.is_dir():
        raise FileNotFoundError(f"{d} is not a folder")
    d = d.resolve()
    dirs, nbs = [], []
    try:
        entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        entries = []
    for p in entries:
        name = p.name
        if name.startswith(".") or name == "__pycache__":
            continue
        try:
            if p.is_dir():
                dirs.append({"name": name, "path": str(p)})
            elif p.suffix.lower() == ".ipynb":
                kb = max(1, p.stat().st_size // 1024)
                nbs.append({"name": name, "path": str(p), "size": f"{kb} KB"})
        except OSError:
            continue
    parent = str(d.parent) if d.parent != d else ""
    return {"dir": str(d), "parent": parent, "dirs": dirs, "notebooks": nbs}


def _app_page(state: _AppState) -> str:
    """Rebuild the whole app page from the session's open notebooks."""
    docs, paths, taken, pruned = [], {}, set(), []
    for p in list(state.open):
        if _is_url(p):
            try:
                doc = doc_from_url(p)
            except Exception:       # noqa: BLE001 -- likely transient
                continue            # keep the URL in the session
            doc.source_name = _stem_for(
                Path(doc.source_name + ".ipynb"), taken)
            taken.add(doc.source_name)
            paths[doc.source_name] = p
            docs.append(doc)
            continue
        f = Path(p)
        try:
            doc = load_doc(f)
        except (OSError, ValueError):
            pruned.append(p)
            continue
        doc.source_name = _stem_for(f, taken)
        taken.add(doc.source_name)
        paths[doc.source_name] = str(f)
        docs.append(doc)
    if pruned:                      # notebooks meanwhile deleted / moved
        with state.lock:
            state.open = [p for p in state.open if p not in pruned]
            state._write()
    return render_page(docs, mode="app", app_cfg={
        "token": state.token,
        "root": str(state.root),
        "presentations": state.presentations,
        "recent": state.recent,
        "paths": paths,
    })


def _make_handler(state: _AppState):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):       # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code,
                       json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _html(self, text: str, code: int = 200) -> None:
            self._send(code, text.encode("utf-8"),
                       "text/html; charset=utf-8")

        def _authed(self, query: dict) -> bool:
            tok = (query.get("t") or [""])[0]
            return secrets.compare_digest(tok, state.token)

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            query = urllib.parse.parse_qs(url.query)
            if url.path == "/":
                if not self._authed(query):
                    self._html("<h1>PlotLine</h1>"
                               "<p>Open the exact URL printed in the "
                               "terminal (it carries a session token).</p>",
                               403)
                    return
                self._html(_app_page(state))
                return
            if not self._authed(query):
                self._json({"error": "bad token"}, 403)
                return
            try:
                if url.path == "/api/list":
                    raw = (query.get("dir") or [""])[0] or str(state.root)
                    self._json(_list_dir(raw))
                else:
                    self._json({"error": "not found"}, 404)
            except FileNotFoundError as e:
                self._json({"error": str(e)}, 404)
            except Exception as e:          # noqa: BLE001 -- surfaced in UI
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)

        def do_POST(self):
            url = urllib.parse.urlsplit(self.path)
            query = urllib.parse.parse_qs(url.query)
            if not self._authed(query):
                self._json({"error": "bad token"}, 403)
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
            except ValueError:
                self._json({"error": "bad JSON body"}, 400)
                return
            try:
                if url.path == "/api/open":
                    self._json(self._open_nb(body))
                elif url.path == "/api/parse":
                    self._json(self._parse_nb(body))
                elif url.path == "/api/save":
                    state.save_presentations(
                        _as_presentations(body.get("presentations")))
                    self._json({"ok": True})
                elif url.path == "/api/close":
                    state.note_close(str(body.get("path") or ""))
                    self._json({"ok": True})
                else:
                    self._json({"error": "not found"}, 404)
            except FileNotFoundError as e:
                self._json({"error": str(e)}, 404)
            except Exception as e:          # noqa: BLE001 -- surfaced in UI
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)

        def _open_nb(self, body: dict) -> dict:
            raw = str(body.get("path") or "").strip().strip('"')
            if not raw:
                raise ValueError("no path given")
            if _is_url(raw):
                url = _normalize_nb_url(raw)
                doc = doc_from_url(url)
                doc.source_name = _stem_for(
                    Path(doc.source_name + ".ipynb"),
                    state.stems_taken(skip_str=url))
                state.note_open(url)
                return {"stem": doc.source_name, "path": url,
                        "shell": render_shell(doc, path=url)}
            f = Path(raw).expanduser()
            if not f.is_absolute():
                f = state.root / f
            f = f.resolve()
            if not f.exists():
                raise FileNotFoundError(f"{f} not found")
            if f.suffix.lower() != ".ipynb":
                raise ValueError(f"{f.name} is not a .ipynb file")
            doc = load_doc(f)
            doc.source_name = _stem_for(f, state.stems_taken(skip=f))
            state.note_open(f)
            return {"stem": doc.source_name, "path": str(f),
                    "shell": render_shell(doc, path=str(f))}

        def _parse_nb(self, body: dict) -> dict:
            nb = body.get("nb")
            if isinstance(nb, str):
                nb = json.loads(nb)
            if not isinstance(nb, dict):
                raise ValueError("nb must be notebook JSON")
            name = str(body.get("name") or "notebook.ipynb")
            base = re.sub(r"\.ipynb$", "", name, flags=re.I) or "notebook"
            doc = parse_notebook(nb)
            doc.source_name = _stem_for(Path(base + ".ipynb"),
                                        state.stems_taken())
            return {"stem": doc.source_name, "path": "",
                    "shell": render_shell(doc)}

    return Handler


def run_app(root: Path, notebooks: list, port: int = 8765,
            open_browser: bool = True) -> int:
    state = _AppState(root)
    for nb in notebooks:
        if isinstance(nb, str) and _is_url(nb):
            state.note_open(_normalize_nb_url(nb))
            continue
        f = Path(nb).expanduser().resolve()
        if f.exists():
            state.note_open(f)
        else:
            print(f"warning: {nb} not found, skipping", file=sys.stderr)
    handler = _make_handler(state)
    try:
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:                 # port busy -> any free port
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/?t={state.token}"
    print("PlotLine")
    print(f"  url:     {url}")
    print(f"  project: {state.project_path}")
    print("  Open notebooks with '+ Open' or drop .ipynb files onto the "
          "page. Ctrl+C stops the app.")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


def _self_test() -> None:
    """Tiny built-in notebook so the renderer can be checked with no input."""
    nb = {
        "metadata": {"semantic": {"presentations": [
            {"name": "demo", "slides": [
                {"layout": "halves", "panes": ["clim", "cell:md1"]},
            ]},
        ]}},
        "cells": [
            {"cell_type": "markdown", "source": "# Demo analysis"},
            {"cell_type": "markdown", "source": "## Dataset"},
            {"cell_type": "markdown", "id": "md1",
             "source": "The anomaly is $z' = z - \\bar{z}$.\n\n- point one\n- point **two**"},
            {"cell_type": "code", "id": "c-load",
             "source": "#| display: metric\n#| id: load\n#| title: Load grid\nprint('shape (40, 80)')",
             "outputs": [{"output_type": "stream", "name": "stdout",
                          "text": "shape (40, 80)\n"}]},
            {"cell_type": "code", "id": "c-clim",
             "source": "#| display: figure\n#| id: clim\n#| depends: load\n#| title: Climatology\n#| caption: Note the ridge.\nplot()",
             "outputs": []},
            {"cell_type": "code", "id": "c-mixed",
             "source": "#| id: mixed\n#| title: Repr then plot\nds",
             "outputs": [
                 {"output_type": "display_data",
                  "data": {"text/html": "<div class='xr-a'>xarray</div>"}},
                 {"output_type": "display_data",
                  "data": {"image/png": "aGk="}}]},
            {"cell_type": "code", "id": "c-prep",
             "source": "#| title: Open dataset\nds = open_thing()",
             "outputs": []},
            {"cell_type": "code", "id": "c-fig2",
             "source": "#| display: figure\n#| id: fig2\n#| title: Second figure\nplot(ds)",
             "outputs": []},
            {"cell_type": "code", "id": "c-two",
             "source": "#| display: figure\n#| id: two\n#| title: Two panels\nplot(); plot()",
             "outputs": [
                 {"output_type": "display_data",
                  "data": {"image/png": "aGk="}},
                 {"output_type": "display_data",
                  "data": {"image/png": "aGk="}}]},
            {"cell_type": "code", "id": "c-fn",
             "source": "def rescale(arr):\n    return arr / arr.max()",
             "outputs": []},
            {"cell_type": "code", "id": "c-one",
             "source": "result = rescale(data)",
             "outputs": []},
        ]
    }
    doc = parse_notebook(nb)
    out = render_html(doc, source_name="demo")
    assert "Demo analysis" in out and "Climatology" in out and "provsvg" in out
    # presentation plumbing, incl. legacy single-deck conversion
    assert doc.presentations and doc.presentations[0]["name"] == "demo"
    assert doc.presentations[0]["slides"][0]["panes"] == ["clim", "cell:md1"]
    legacy = _as_presentations({"slides": [
        {"kind": "card", "anchor": "a", "beside": ["b"]}]})
    assert legacy[0]["slides"][0] == {"layout": "halves", "panes": ["a", "b"]}
    assert '"panes": ["clim", "cell:md1"]' in out
    assert 'class="nb-data"' in out and 'id="app-data"' in out
    assert 'id="presstrip"' in out and 'id="tv-markup"' in out
    assert 'id="pr-docs"' in out and 'id="pr-new"' in out
    assert 'id="deck-docs"' in out and 'id="dc-close"' in out
    assert 'id="dc-play"' in out and 'id="pane-editor"' in out
    assert 'data-anchor="clim"' in out and 'data-anchor="cell:md1"' in out
    assert '"stem": "demo"' in out or '"stem":"demo"' in out
    # markdown notes: bullets + bold survive, math left for MathJax
    assert "<li>point one</li>" in out and "<strong>two</strong>" in out
    assert "\\bar{z}$" in out  # ' is escaped to &#x27;; DOM text is intact
    # anchors fall back to node id / cell id
    items = [it for s in doc.sections for it in s.items]
    assert any(it.anchor == "clim" for it in items)
    assert any(it.anchor == "cell:md1" for it in items)
    # code chains: declared depends (clim <- load) and AST-traced variables
    # (fig2 reads ds, which cell:c-prep assigned)
    by_anchor = {it.anchor: it for it in items}
    assert by_anchor["clim"].chain == ["load"]
    assert by_anchor["fig2"].chain == ["cell:c-prep"]
    assert '"chain": ["cell:c-prep"]' in out

    # mixed-output cell: figure face first, repr behind a disclosure
    mixed = [it for s in doc.sections for it in s.items
             if it.anchor == "mixed"][0]
    assert mixed.kind == "figure"
    assert "alsoprinted" in out and "also printed by this cell" in out

    # several figures from one cell -> pager, one figure at a time
    assert 'class="figpager" data-n="2"' in out
    assert 'class="figpage current"' in out and "fp-next" in out
    assert "1 / 2" in out
    # nav key legend + long-markdown clamp plumbing shipped
    assert 'class="navkey"' in out and 'class="nk k-figure"' in out
    assert "mdClampScan" in out and "mdclamp" in out
    assert "vo-xall" in out and "fullscreenchange" in out

    # untitled code cells: function names become titles; a bare code
    # line labels the nav but is not repeated as a card heading
    all_items = [it for s in doc.sections for it in s.items]
    assert any(it.title == "rescale()" and not it.title_echo
               for it in all_items)
    assert any(it.title == "result = rescale(data)" and it.title_echo
               for it in all_items)
    assert 'cardtitle echo' in out
    assert _title_from_code("def a():\n    pass\n\ndef b():\n    pass") \
        == ("2 functions (a, b)", False)
    assert _title_from_code(
        "import x\n\ndef a():\n    pass\n\ndef b():\n    pass") \
        == ("2 functions + code", False)
    # chrome: TOC toggle, resizable builder, dark document, tab refresh
    assert 'id="menubtn"' in out and "tocshow" in out
    assert 'id="dc-resize"' in out and "--dc-w" in out
    assert 'id="dc-save"' in out
    assert "body:not(.light) .card" in out
    assert 'id="refresh-btn"' not in out

    # new slide layouts, title slides and annotations survive normalizing
    pres2 = _as_presentations([{"name": "n", "slides": [
        {"layout": "rows", "panes": ["a"]},
        {"layout": "title", "title": "Hi", "sub": "there",
         "annots": [{"k": "text", "x": 5, "y": 5, "text": "note"}]},
    ]}])
    assert pres2[0]["slides"][0] == {"layout": "rows", "panes": ["a", None]}
    t_slide = pres2[0]["slides"][1]
    assert t_slide["layout"] == "title" and t_slide["title"] == "Hi"
    assert t_slide["panes"] == [] and t_slide["annots"][0]["text"] == "note"
    assert 'data-lay="rows"' in out and 'data-lay="title"' in out
    assert 'id="edit-tools"' in out and 'id="dc-edit"' in out
    assert 'id="et-fmt"' in out and 'data-tool="cell"' in out
    assert 'id="fmt-op"' in out and 'id="fmt-rotl"' in out
    assert 'id="theme-btn"' in out
    assert 'id="fmt-font"' in out and "body.light .apptop" in out
    assert "apptip" in out
    assert 'id="fmt-list"' in out and 'id="fmt-shape"' in out
    assert 'id="fmt-dup"' in out and 'id="fmt-front"' in out
    assert 'id="pickbar"' in out and 'id="fmt-replace"' in out
    pres_f = _as_presentations([{"name": "a", "folder": "paper 1",
                                 "slides": []}])
    assert pres_f[0]["folder"] == "paper 1"
    assert 'id="pr-newfold"' in out
    pres3 = _as_presentations([{"name": "x", "slides": [
        {"layout": "title", "title": "T",
         "tprops": {"x": 30, "y": 20, "size": 5}},
        {"layout": "blank",
         "annots": [{"k": "cell", "x": 1, "y": 1, "w": 40, "h": 40,
                     "ref": "demo::clim"}]},
    ]}])
    assert pres3[0]["slides"][0]["tprops"]["x"] == 30
    blank = pres3[0]["slides"][1]
    assert blank["panes"] == [] and blank["annots"][0]["ref"] == "demo::clim"
    assert 'data-lay="blank"' in out and "an-cellbtn" in out

    # raw notebook view: cells as authored, directives visible
    assert 'id="view-raw"' in out and 'class="rawview"' in out
    assert 'class="rawcell code"' in out and "#| display: metric" in out
    assert 'class="rawcell md"' in out

    # multi-notebook page: two tabs, per-shell data, cross-notebook deck
    nb2 = {"cells": [
        {"cell_type": "markdown", "source": "# Second notebook"},
        {"cell_type": "code", "id": "x1",
         "source": "#| display: figure\n#| id: sst\n#| title: SST map\nplot()",
         "outputs": []},
    ]}
    doc_a = parse_notebook(nb)
    doc_a.source_name = "demo"
    doc_b = parse_notebook(nb2)
    doc_b.source_name = "other"
    page = render_page([doc_a, doc_b], app_cfg={
        "presentations": [{"name": "combo", "slides": [
            {"layout": "halves", "panes": ["demo::clim", "other::sst"]}]}],
    })
    assert page.count('class="shell nbshell"') == 2
    assert 'data-nb="demo"' in page and 'data-nb="other"' in page
    assert 'id="apptop"' in page and 'id="tabstrip"' in page
    assert '"mode": "static"' in page and "demo::clim" in page

    # app-mode page carries the session token and root for the GUI
    app_page = render_page([doc_a], mode="app", app_cfg={
        "token": "tok123", "root": "C:/proj", "paths": {"demo": "x.ipynb"},
        "recent": ["a.ipynb"],
    })
    assert '"mode": "app"' in app_page and "tok123" in app_page
    assert 'data-path="x.ipynb"' in app_page

    # decks survive notebook edits: anchors are ids, never positions —
    # reordering cells and editing text must keep every anchor alive
    import copy
    nb_edit = copy.deepcopy(nb)
    nb_edit["cells"] = list(reversed(nb_edit["cells"]))
    for c in nb_edit["cells"]:
        if c.get("id") == "md1":
            c["source"] = "EDITED prose, same cell id"
    doc_e = parse_notebook(nb_edit)
    anchors_e = {it.anchor for s in doc_e.sections for it in s.items}
    assert "clim" in anchors_e and "cell:md1" in anchors_e
    assert "fig2" in anchors_e

    # server helpers: directory listing shape + stem dedupe
    listing = _list_dir(str(Path(__file__).parent))
    assert {"dir", "parent", "dirs", "notebooks"} <= set(listing)
    assert _stem_for(Path("a/nb.ipynb"), {"nb"}) == "nb-2"

    # URLs: GitHub normalization + the client-side web build
    assert _normalize_nb_url(
        "https://github.com/u/r/blob/main/d/nb.ipynb") \
        == "https://raw.githubusercontent.com/u/r/main/d/nb.ipynb"
    assert _is_url("https://x/y.ipynb") and not _is_url("C:/y.ipynb")
    shell = web_parse("demo.ipynb", json.dumps(nb), '["demo"]')
    assert 'data-nb="demo-2"' in shell
    web_page = render_page([], mode="web")
    assert '"mode": "web"' in web_page and 'id="fileinput"' in web_page
    assert 'id="helpdlg"' in web_page and 'id="help-btn"' in web_page
    assert "ko-fi.com/plotline" in web_page
    assert 'id="support-btn"' in web_page
    assert 'id="welcome-demo"' in web_page and _REPO_URL in web_page
    assert "#| title:" in web_page          # directives documented in help
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        build_web(Path(td))
        idx = (Path(td) / "index.html").read_text(encoding="utf-8")
        assert "pyodide" in idx and "sem:pyready" in idx
        assert (Path(td) / "semantic_render.py").exists()

    print("self-test ok:", len(out), "bytes;",
          sum(len(s.items) for s in doc.sections), "items;",
          "presentations:", len(doc.presentations))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Semantic notebook environment: run with no arguments "
        "to launch the local GUI app (open .ipynb files as tabs), or pass "
        "notebook path(s) to export a static HTML page.")
    p.add_argument("notebooks", nargs="*",
                   help="path(s) to executed .ipynb notebooks; several "
                   "render as tabs in one page")
    p.add_argument("-o", "--output",
                   help="output .html (default: alongside the notebook, or "
                   "semantic_view.html for a multi-notebook bundle)")
    p.add_argument("--title", help="override the analysis title "
                   "(single-notebook export only)")
    p.add_argument("--deck", help="presentation deck JSON to use "
                   "(default: <notebook>.deck.json sidecar, then embedded "
                   "metadata)")
    p.add_argument("--embed-deck", metavar="DECK_JSON",
                   help="write DECK_JSON into the notebook's "
                   "metadata.semantic.presentations (modifies the .ipynb) "
                   "and exit")
    p.add_argument("--app", action="store_true",
                   help="launch the local GUI app (implied when no "
                   "notebooks are given); listed notebooks preload as tabs")
    p.add_argument("--root", help="app mode: folder for the file browser "
                   "and semantic_project.json (default: the first "
                   "notebook's folder, else the current folder)")
    p.add_argument("--port", type=int, default=8765,
                   help="app mode: port to serve on (default 8765; falls "
                   "back to a free port when busy)")
    p.add_argument("--no-browser", action="store_true",
                   help="app mode: don't auto-open the browser")
    p.add_argument("--build-web", metavar="DIR",
                   help="write the deployable client-side web app "
                   "(index.html + this module, runs Python in the "
                   "browser via Pyodide) into DIR and exit")
    p.add_argument("--self-test", action="store_true",
                   help="run a built-in sanity check and exit")
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0

    if args.build_web:
        build_web(Path(args.build_web))
        print(f"wrote web app to {args.build_web}\\index.html")
        print("Deploy: commit that folder and enable GitHub Pages for "
              "it (or drop it on any static host).")
        return 0

    items = list(args.notebooks)
    local = [Path(n) for n in items if not _is_url(n)]

    if args.embed_deck:
        if len(items) != 1 or not local:
            p.error("--embed-deck needs exactly one local notebook")
        if not local[0].exists():
            print(f"error: {local[0]} not found", file=sys.stderr)
            return 1
        embed_deck(local[0], Path(args.embed_deck))
        print(f"embedded {args.embed_deck} into {local[0]} "
              "(metadata.semantic.presentations)")
        return 0

    if args.app or not items:
        root = Path(args.root) if args.root else \
            (local[0].parent if local else Path.cwd())
        if not root.is_dir():
            p.error(f"--root {root} is not a folder")
        preload = [n if _is_url(n) else Path(n) for n in items]
        return run_app(root, preload, port=args.port,
                       open_browser=not args.no_browser)

    missing = [f for f in local if not f.exists()]
    for m in missing:
        print(f"error: {m} not found", file=sys.stderr)
    if missing:
        return 1

    deck = Path(args.deck) if args.deck else None
    single = len(items) == 1
    if not single and args.title:
        print("note: --title is ignored for multi-notebook bundles",
              file=sys.stderr)
    docs, taken = [], set()
    for n in items:
        if _is_url(n):
            doc = doc_from_url(n)
            if single and args.title:
                doc.title = args.title
            if single and deck is not None:
                pres = _as_presentations(
                    json.loads(deck.read_text(encoding="utf-8")))
                if pres:
                    doc.presentations = pres
            doc.source_name = _stem_for(
                Path(doc.source_name + ".ipynb"), taken)
        else:
            doc = load_doc(Path(n),
                           title=args.title if single else None,
                           deck_path=deck if single else None)
            doc.source_name = _stem_for(Path(n), taken)
        taken.add(doc.source_name)
        docs.append(doc)
    cfg = {}
    if not single and deck is not None:
        cfg["presentations"] = _as_presentations(
            json.loads(deck.read_text(encoding="utf-8")))
    html_out = render_page(docs, app_cfg=cfg)
    if args.output:
        out_path = Path(args.output)
    elif single and not _is_url(items[0]):
        out_path = Path(items[0]).with_suffix(".html")
    elif single:
        out_path = Path(docs[0].source_name + ".html")
    else:
        out_path = (local[0].parent if local else Path.cwd()) \
            / "semantic_view.html"
    out_path.write_text(html_out, encoding="utf-8")
    print(f"wrote {out_path}  ({len(html_out)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
