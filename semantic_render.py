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
    python semantic_render.py NOTEBOOK.ipynb [-o OUT.html] [--title "..."]
    python semantic_render.py NOTEBOOK.ipynb --deck DECK.json
    python semantic_render.py NOTEBOOK.ipynb --embed-deck DECK.json
    python semantic_render.py NOTEBOOK.ipynb --self-test

The rendered page includes a Present mode (toolbar) with a slide builder;
decks persist in the notebook's metadata.semantic.deck, in a
<notebook>.deck.json sidecar, or via --embed-deck. Slides reference cards
by stable anchors (`#| id:` first, else the nbformat cell id).
"""

from __future__ import annotations

import argparse
import ast
import base64
import html
import io
import json
import keyword
import re
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def _title_from_code(code: str) -> str:
    """Best-effort title: first comment line, else first call/assignment."""
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip() or "Code"
        if s:
            return (s[:60] + "...") if len(s) > 60 else s
    return "Code"


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

    item.title = (next((m["d"]["title"] for m in members if m["d"].get("title")), "")
                  or _title_from_code(primary["code"]))
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


def _as_presentations(obj: Any) -> list:
    """Normalize saved presentation data to [{name, slides}, ...].

    Accepts the current schema (a list, or {"presentations": [...]}) plus
    the legacy single-deck schema ({"slides": [{kind, anchor, beside}]}),
    whose card slides are converted to pane layouts.
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
            if "panes" in s:
                panes = [a if isinstance(a, str) and a else None
                         for a in (s.get("panes") or [None])][:4]
                lay = s.get("layout")
                if lay not in ("full", "halves", "quarters"):
                    lay = {1: "full", 2: "halves"}.get(len(panes), "quarters")
                slides.append({"layout": lay, "panes": panes})
            elif s.get("kind") == "card" and s.get("anchor"):   # legacy
                panes = [s["anchor"]] + [b for b in (s.get("beside") or [])
                                         if isinstance(b, str)][:3]
                lay = {1: "full", 2: "halves"}.get(len(panes), "quarters")
                slides.append({"layout": lay, "panes": panes})
        out.append({"name": str(p.get("name") or "deck"), "slides": slides})
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


def render_item(item: Item) -> str:
    badge = _BADGE.get(item.kind, item.kind)
    kclass = _kind_class(item.kind)
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
        f'<h3 class="cardtitle">{html.escape(item.title)}</h3>'
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


def render_html(doc: Document, source_name: str | None = None) -> str:
    if source_name:
        doc.source_name = source_name
    return _TEMPLATE.format(
        title=html.escape(doc.title),
        meta=html.escape(doc_meta(doc)),
        nav=render_nav(doc),
        graph_panel=render_graph_panel(doc),
        sections=render_sections(doc),
        css=_CSS,
        js=_JS,
        mathjax=_MATHJAX,
        deck_shell=_DECK_HTML,
        deck_data=deck_payload(doc),
        deck_css=_DECK_CSS,
        deck_js=_DECK_JS,
    )


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
.navitem.k-figure .dot{background:var(--cyan);}
.navitem.k-dataset .dot{background:#4d90c0;}
.navitem.k-transform .dot{background:#5b7589;}
.navitem.k-metric .dot{background:#46a892;}
.navitem.k-note .dot{background:var(--amber);border-radius:50%;}
.navitem.k-code .dot{background:#56627033;border:1px solid #ffffff22;}

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
.nodeid{font-family:var(--mono);font-size:10px;color:var(--ink-3);
  background:var(--paper-2);padding:2px 7px;border-radius:4px;flex:none;}

.cardbody{padding-left:6px;}
.figframe{background:#fff;border:1px solid var(--paper-3);border-radius:8px;
  padding:8px;overflow:auto;text-align:center;}
.figframe img{max-width:100%;height:auto;display:block;margin:0 auto;}
.figframe svg{max-width:100%;height:auto;}

pre.result,pre.stream,pre.error{font-family:var(--mono);font-size:12px;
  background:var(--paper-2);border:1px solid var(--paper-3);
  border-radius:7px;padding:11px 13px;overflow:auto;margin:0;line-height:1.45;}
pre.error{background:#fbf0ee;border-color:#f0d2cc;color:#8a3221;}
.metric .cardbody pre.result{font-size:14px;background:#46a8920d;
  border-color:#46a89233;color:#1f5f54;font-weight:500;}

.note{font-family:var(--serif);font-size:15px;line-height:1.65;
  color:var(--ink-2);}
.note .caption{font-family:var(--serif);font-style:normal;color:var(--ink-2);
  margin:0;padding:0;border:none;font-size:15px;}

.caption{font-family:var(--serif);font-size:14px;
  color:var(--ink-2);margin:13px 0 0;padding-left:6px;line-height:1.6;}

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

_JS = r"""
(function(){
  var byId = function(s,r){return (r||document).querySelector(s);};
  var all = function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s));};

  /* ---- reveal on scroll ---- */
  var cards = all('.card');
  if('IntersectionObserver' in window){
    var io = new IntersectionObserver(function(es){
      es.forEach(function(e){ if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target);} });
    },{rootMargin:'0px 0px -8% 0px',threshold:0.04});
    cards.forEach(function(c){io.observe(c);});
  } else { cards.forEach(function(c){c.classList.add('in');}); }

  /* ---- scroll-spy: active section + item + graph node ---- */
  var navSecs = {}, navItems = {}, graphNodes = {};
  all('.navsec').forEach(function(a){navSecs[a.dataset.sec]=a;});
  all('.navitem').forEach(function(a){navItems[a.dataset.item]=a;});
  all('.provnode').forEach(function(g){graphNodes[g.dataset.node]=g;});

  function setActiveSection(id){
    all('.navsec.active').forEach(function(a){a.classList.remove('active');});
    if(navSecs[id]) navSecs[id].classList.add('active');
  }
  function setActiveItem(item){
    all('.navitem.active').forEach(function(a){a.classList.remove('active');});
    if(navItems[item]) navItems[item].classList.add('active');
    var node = byId('.card[id="card-'+item+'"]');
    var nodeId = node ? node.dataset.node : '';
    all('.provnode.active').forEach(function(g){g.classList.remove('active');});
    all('.provedge.lit').forEach(function(p){p.classList.remove('lit');});
    if(nodeId && graphNodes[nodeId]){
      graphNodes[nodeId].classList.add('active');
      all('.provedge').forEach(function(p){
        if(p.dataset.to===nodeId||p.dataset.from===nodeId) p.classList.add('lit');
      });
    }
  }

  if('IntersectionObserver' in window){
    var visible = {};
    var spy = new IntersectionObserver(function(es){
      es.forEach(function(e){
        if(e.isIntersecting) visible[e.target.id]=e.intersectionRatio;
        else delete visible[e.target.id];
      });
      var bestC=null,bc=0;
      Object.keys(visible).forEach(function(k){
        if(k.indexOf('card-')===0 && visible[k]>=bc){bc=visible[k];bestC=k;}
      });
      if(bestC){
        var item=bestC.slice(5);
        setActiveItem(item);
        var card=byId('.card[id="'+bestC+'"]');
        var sec=card.closest('.section');
        if(sec) setActiveSection(sec.dataset.sec);
      }
    },{rootMargin:'-12% 0px -55% 0px',threshold:[0,0.25,0.6,1]});
    cards.forEach(function(c){spy.observe(c);});
  }

  /* ---- code toggles ---- */
  all('.codetoggle').forEach(function(btn){
    btn.addEventListener('click',function(){
      var wrap=btn.closest('.codewrap');
      var open=wrap.hasAttribute('data-open');
      if(open){wrap.removeAttribute('data-open');btn.setAttribute('aria-expanded','false');}
      else{wrap.setAttribute('data-open','');btn.setAttribute('aria-expanded','true');}
    });
  });

  /* ---- toolbar: per-type Show/Hide buttons ----
     Label follows the state (Hide X while shown, Show X while hidden);
     hidden cards collapse to slim stubs that expand in place on click. */
  var vis={figs:true,markup:true,code:true};
  var TYPES=[['figs','figures'],['markup','markup'],['code','code']];
  function renderTypeButtons(){
    TYPES.forEach(function(p){
      var b=byId('#tv-'+p[0]); if(!b) return;
      b.innerHTML='<span class="tdot"></span>'
        +(vis[p[0]]?'Hide ':'Show ')+p[1];
      b.classList.toggle('off',!vis[p[0]]);
    });
  }
  function applyFilters(){
    document.body.classList.toggle('filtered',
      !(vis.figs&&vis.markup&&vis.code));
    all('.card').forEach(function(c){
      var kind=c.dataset.kind, note=c.dataset.note==='1';
      var show=note?vis.markup
        :(kind==='figure'||kind==='diagnostic')?vis.figs:vis.code;
      c.classList.toggle('is-stub',!show);
      if(show) c.classList.remove('stub-open');
    });
    renderTypeButtons();
  }
  TYPES.forEach(function(p){
    var b=byId('#tv-'+p[0]);
    if(b) b.addEventListener('click',function(){
      vis[p[0]]=!vis[p[0]];applyFilters();
    });
  });
  renderTypeButtons();
  all('.card').forEach(function(c){
    var head=byId('.cardhead',c);
    if(head){
      head.addEventListener('click',function(){
        if(c.classList.contains('is-stub')) c.classList.toggle('stub-open');
      });
    }
  });

  /* ---- graph node click -> scroll to card ---- */
  function gotoItem(itemId){
    var card=byId('.card[id="card-'+itemId+'"]');
    if(!card) return;
    card.scrollIntoView({behavior:'smooth',block:'center'});
    card.classList.add('target-flash');
    setTimeout(function(){card.classList.remove('target-flash');},1400);
  }
  all('.provnode').forEach(function(g){
    function act(){gotoItem(g.dataset.target);}
    g.addEventListener('click',act);
    g.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();act();}});
  });
  /* dependency chips jump to the source node's card */
  all('.depchip').forEach(function(a){
    a.addEventListener('click',function(e){
      e.preventDefault();
      var dep=a.dataset.dep;
      var src=byId('.card[data-node="'+dep+'"]');
      if(src){ src.scrollIntoView({behavior:'smooth',block:'center'});
        src.classList.add('target-flash');
        setTimeout(function(){src.classList.remove('target-flash');},1400);}
    });
  });

  /* ---- rail graph collapse ---- */
  var rgBtn=byId('.rg-collapse');
  if(rgBtn){
    rgBtn.addEventListener('click',function(){
      var rg=rgBtn.closest('.railgraph');
      var c=rg.classList.toggle('collapsed');
      rgBtn.setAttribute('aria-expanded',(!c).toString());
      rgBtn.textContent=c?'+':'\u2013';
    });
  }

  /* ---- mobile drawer ---- */
  var menu=byId('#menubtn'), rail=byId('#rail'), scrim=byId('#scrim');
  function closeRail(){rail.classList.remove('open');scrim.classList.remove('show');}
  if(menu){
    menu.addEventListener('click',function(){
      rail.classList.toggle('open');scrim.classList.toggle('show');
    });
    scrim.addEventListener('click',closeRail);
    all('.navitem,.navsec').forEach(function(a){a.addEventListener('click',function(){
      if(window.innerWidth<=860) closeRail();
    });});
  }
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
            <button class="dc-mi" id="mi-dl">Download JSON</button>
            <button class="dc-mi" id="mi-discard">Discard changes</button>
          </div>
        </div>
        <span class="deck-status" id="deck-status"></span>
      </div>
      <div class="dc-block">
        <span class="dc-label">Presentation</span>
        <select id="pres-select" title="Switch presentation"></select>
        <input id="pres-name" type="text" placeholder="presentation name"
          spellcheck="false" autocomplete="off" hidden>
      </div>
      <div class="dc-block">
        <span class="dc-label">Slide layout</span>
        <div class="dc-row" id="layout-row">
          <button class="dbtn lay" data-lay="full">Full</button>
          <button class="dbtn lay" data-lay="halves">Halves</button>
          <button class="dbtn lay" data-lay="quarters">Quarters</button>
        </div>
        <div class="pane-editor" id="pane-editor"></div>
        <p class="dc-hint">Pick a pane, then click a card in the document
        to place it there.</p>
      </div>
      <div class="dc-block dc-film">
        <span class="dc-label">Slides</span>
        <div class="film-list" id="film-list"></div>
        <button class="dbtn addslide" id="film-add">+ Add slide</button>
      </div>
    </aside>
    <div class="deck-stagewrap" id="deck-stagewrap">
      <button class="deck-arrow prev" id="deck-prev" aria-label="Previous slide">&#8249;</button>
      <div class="deck-stage" id="deck-stage"></div>
      <button class="deck-arrow next" id="deck-next" aria-label="Next slide">&#8250;</button>
      <div class="deck-foot">
        <button class="dbtn" id="deck-codebtn" hidden aria-expanded="false">Show code</button>
        <span class="deck-count" id="deck-count"></span>
      </div>
      <div class="deck-drawer" id="deck-drawer" hidden></div>
    </div>
  </div>
  <div class="deck-toast" id="deck-toast" hidden></div>
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
.deck-stage{flex:1;min-height:0;display:flex;padding:26px 78px 6px;}

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
.deck.creating{width:min(352px,94vw);right:auto;
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
body.creating-docs .shell{margin-left:min(352px,94vw);}
body.creating-docs .card{cursor:copy;}
body.creating-docs .card:hover{outline:2px solid var(--cyan);
  outline-offset:2px;}

.deck-create{flex:1;overflow-y:auto;display:flex;flex-direction:column;
  min-height:0;background:#0e1926;}
.dc-block{padding:14px 14px 12px;border-bottom:1px solid #ffffff14;}
.dc-block.dc-film{flex:1;display:flex;flex-direction:column;min-height:120px;
  border-bottom:none;padding-bottom:8px;}
.dc-label{display:block;font-family:var(--mono);font-size:9.5px;
  letter-spacing:.16em;text-transform:uppercase;color:#7e93a4;
  margin-bottom:8px;}
.dc-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px;}
.dc-hint{font-size:11.5px;color:#7e93a4;line-height:1.5;margin:9px 0 0;}
#pres-select{width:100%;background:#16273a;border:1px solid #ffffff22;
  color:#dce6ee;font-family:var(--sans);font-size:12.5px;padding:7px 8px;
  border-radius:6px;}
#pres-name{width:100%;background:#16273a;border:1px solid #ffffff22;
  color:#dce6ee;font-family:var(--sans);font-size:12.5px;padding:7px 9px;
  border-radius:6px;margin-top:7px;box-sizing:border-box;}
#pres-name:focus{outline:none;border-color:var(--cyan);}
.dbtn.lay[aria-pressed="true"]{background:var(--cyan-deep);
  border-color:var(--cyan-deep);color:#fff;}

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
  margin-bottom:3px;}
.film-row.current{background:#39a9c01c;outline:1px solid #39a9c055;}
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
  var dataEl=document.getElementById('deck-data');
  var deckEl=document.getElementById('deck');
  if(!dataEl||!deckEl) return;
  var DATA;
  try{DATA=JSON.parse(dataEl.textContent);}catch(e){return;}

  var $=function(s,r){return (r||document).querySelector(s);};
  var $$=function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s));};
  function esc(t){var d=document.createElement('div');d.textContent=(t==null?'':String(t));return d.innerHTML;}

  var stage=$('#deck-stage'), drawer=$('#deck-drawer'), codeBtn=$('#deck-codebtn');
  var itemsByAnchor={};
  DATA.items.forEach(function(it){itemsByAnchor[it.anchor]=it;});
  var PANES={full:1,halves:2,quarters:4};
  var PFX='sempres:'+(DATA.stem||DATA.title)+':';

  /* ---------- presentations: saved copies + working draft ---------- */
  var saved=Array.isArray(DATA.presentations)
    ?JSON.parse(JSON.stringify(DATA.presentations)):[];
  function lsGet(k){try{return localStorage.getItem(k);}catch(e){return null;}}
  function lsSet(k,v){try{localStorage.setItem(k,v);}catch(e){}}
  function lsDel(k){try{localStorage.removeItem(k);}catch(e){}}
  function loadDraft(name){
    var raw=lsGet(PFX+name); if(!raw) return null;
    try{var d=JSON.parse(raw);
      return (d&&Array.isArray(d.slides))?d:null;}catch(e){return null;}
  }
  function savedByName(name){
    return saved.filter(function(p){return p.name===name;})[0]||null;
  }
  function emptySlide(){return {layout:'full',panes:[null]};}
  function autoSlides(withDocs){
    return DATA.items.filter(function(it){
      var fig=it.kind==='figure'||it.kind==='diagnostic';
      return fig||(withDocs&&it.kind==='note');
    }).map(function(it){return {layout:'full',panes:[it.anchor]};});
  }
  function defaultPres(){return {name:'presentation',slides:autoSlides(false)};}

  var pres=null, source='auto', mode='view', cur=0, activePane=0;
  function loadPresentation(name){
    var d=loadDraft(name);
    if(d){pres=d;source='draft';return;}
    var s=savedByName(name);
    if(s){pres=JSON.parse(JSON.stringify(s));source='saved';return;}
    pres=defaultPres();source='auto';
  }
  var last=lsGet(PFX+'last');
  if(last&&(loadDraft(last)||savedByName(last))) loadPresentation(last);
  else if(saved.length) loadPresentation(saved[0].name);
  else {pres=defaultPres();source='auto';}

  function status(){
    var el=$('#deck-status');
    el.textContent=source==='draft'?'unsaved draft'
      :(source==='saved'?'saved':'auto');
    el.className='deck-status '+source;
  }
  function markDirty(){
    source='draft';
    lsSet(PFX+(pres.name||'untitled'),JSON.stringify(pres));
    lsSet(PFX+'last',pres.name||'untitled');
    status();
  }

  /* ---------- DOM cloning from the cards already on the page ---------- */
  function cardEl(anchor){
    return $('.card[data-anchor="'+String(anchor||'').replace(/"/g,'\\"')+'"]');
  }
  function stripIds(node){
    if(node.removeAttribute) node.removeAttribute('id');
    $$('[id]',node).forEach(function(n){n.removeAttribute('id');});
    return node;
  }
  function cloneBody(anchor){
    var c=cardEl(anchor); if(!c) return null;
    var b=$('.cardbody',c); if(!b) return null;
    return stripIds(b.cloneNode(true));
  }
  function cloneCode(anchor){
    var c=cardEl(anchor); if(!c) return null;
    var inner=$('.codeinner',c); if(!inner) return null;
    return stripIds(inner.cloneNode(true));
  }
  function typeset(el){
    if(window.MathJax&&MathJax.typesetPromise){
      MathJax.typesetPromise([el]).catch(function(){});}
  }
  function chainSection(kind,title,codeEl,open){
    var sec=document.createElement('div');sec.className='chain-sec';
    var h=document.createElement('button');h.className='chain-h';
    h.setAttribute('aria-expanded',open?'true':'false');
    h.innerHTML='<span class="chain-chev">&#8250;</span>'
      +'<span class="chain-badge">'+esc(kind)+'</span>'
      +'<span class="chain-t">'+esc(title)+'</span>';
    var b=document.createElement('div');b.className='chain-b';
    if(!open) b.hidden=true;
    b.appendChild(codeEl);
    h.addEventListener('click',function(){
      var o=!b.hidden; b.hidden=o;
      h.setAttribute('aria-expanded',(!o).toString());
    });
    sec.appendChild(h);sec.appendChild(b);
    return sec;
  }
  /* full upstream story: open data -> transforms -> this card's own code.
     Upstream sections start collapsed; the card's own code starts open. */
  function buildChain(it,target){
    var any=false;
    (it.chain||[]).forEach(function(a){
      var cc=cloneCode(a); if(!cc) return;
      var up=itemsByAnchor[a];
      target.appendChild(chainSection(up?up.kind:'step',up?up.title:a,cc,false));
      any=true;
    });
    var own=cloneCode(it.anchor);
    if(own){
      if(any) target.appendChild(chainSection(it.kind,'this '+it.kind,own,true));
      else target.appendChild(own);
      any=true;
    }
    return any;
  }

  /* ---------- view mode: slide rendering ---------- */
  function paneEl(anchor){
    var p=document.createElement('div');
    var it=anchor?itemsByAnchor[anchor]:null;
    if(!it){
      p.className='spane empty';
      p.textContent=anchor?('missing: '+anchor):'empty';
      return p;
    }
    p.className='spane';
    var t=document.createElement('h4');t.className='spane-t';
    t.textContent=it.title;p.appendChild(t);
    var b=cloneBody(anchor); if(b) p.appendChild(b);
    return p;
  }
  function renderSlide(){
    var s=pres.slides[cur];
    stage.innerHTML='';
    drawer.hidden=true;drawer.innerHTML='';
    codeBtn.hidden=true;codeBtn.textContent='Show code';
    codeBtn.setAttribute('aria-expanded','false');
    if(!s){
      stage.innerHTML='<div class="slide slide-empty"><p>No slides yet.'
        +'<br>Use <b>Create</b> to build some.</p></div>';
    } else if(s.layout==='full'){
      var a=s.panes[0];
      var it=a?itemsByAnchor[a]:null;
      if(!it){
        stage.innerHTML='<div class="slide slide-empty"><p>'
          +(a?('Card not found: <code>'+esc(a)+'</code>')
             :'Empty slide — add a card in Create.')+'</p></div>';
      } else {
        var slide=document.createElement('div');
        slide.className='slide slide-card';
        var h=document.createElement('div');h.className='slide-head';
        h.innerHTML='<h3>'+esc(it.title)+'</h3>';
        slide.appendChild(h);
        var body=document.createElement('div');body.className='slide-body';
        var fig=document.createElement('div');fig.className='slide-fig';
        var b=cloneBody(a); if(b) fig.appendChild(b);
        var card=cardEl(a);
        var cap=card?$('.caption',card):null;
        if(cap) fig.appendChild(stripIds(cap.cloneNode(true)));
        body.appendChild(fig);
        slide.appendChild(body);
        stage.appendChild(slide);
        if(buildChain(it,drawer)) codeBtn.hidden=false;
        typeset(slide);
      }
    } else {
      var slide2=document.createElement('div');slide2.className='slide';
      var grid=document.createElement('div');
      grid.className='slide-grid '+s.layout;
      var n=PANES[s.layout]||1;
      for(var i=0;i<n;i++) grid.appendChild(paneEl(s.panes[i]));
      slide2.appendChild(grid);
      stage.appendChild(slide2);
      typeset(slide2);
    }
    $('#deck-count').textContent=pres.slides.length
      ?((cur+1)+' / '+pres.slides.length):'0 / 0';
    $('#deck-prev').disabled=cur<=0;
    $('#deck-next').disabled=cur>=pres.slides.length-1;
  }
  function go(n){
    cur=Math.max(0,Math.min(pres.slides.length-1,n));
    if(mode==='view') renderSlide(); else renderCreate();
  }

  /* ---------- create mode: sidebar UI ---------- */
  function firstEmpty(s){
    if(!s) return 0;
    var n=PANES[s.layout]||1;
    for(var i=0;i<n;i++) if(!s.panes[i]) return i;
    return 0;
  }
  function renderPresRow(){
    var sel=$('#pres-select');sel.innerHTML='';
    var names=saved.map(function(p){return p.name;});
    if(names.indexOf(pres.name)<0) names.unshift(pres.name);
    names.forEach(function(nm){
      var o=document.createElement('option');
      o.value=nm;o.textContent=nm||'(unnamed)';
      if(nm===pres.name) o.selected=true;
      sel.appendChild(o);
    });
    var inp=$('#pres-name');
    if(document.activeElement!==inp&&inp.value!==pres.name)
      inp.value=pres.name;
  }
  function renderLayoutRow(){
    var s=pres.slides[cur];
    $$('#layout-row .lay').forEach(function(b){
      b.setAttribute('aria-pressed',
        (!!s&&s.layout===b.dataset.lay).toString());
      b.disabled=!s;
    });
  }
  function renderPaneEditor(){
    var ed=$('#pane-editor');ed.innerHTML='';
    var s=pres.slides[cur];
    ed.className='pane-editor '+(s?s.layout:'full');
    if(!s){
      ed.innerHTML='<div class="pane empty">'
        +'<span class="pane-t">no slide</span></div>';
      return;
    }
    var n=PANES[s.layout]||1;
    for(var i=0;i<n;i++)(function(i){
      var a=s.panes[i];
      var it=a?itemsByAnchor[a]:null;
      var p=document.createElement('div');
      p.className='pane'+(a?' filled':' empty')
        +(i===activePane?' active':'');
      var src=a?paneImgSrc(a):null;
      if(src){
        var pim=document.createElement('img');
        pim.className='pane-img';pim.src=src;pim.alt='';
        p.appendChild(pim);
      }
      var t=document.createElement('span');t.className='pane-t';
      t.textContent=it?it.title:(a?('missing: '+a):'empty');
      p.appendChild(t);
      if(a){
        var x=document.createElement('button');x.className='pane-x';
        x.textContent='✕';x.title='Clear pane';
        x.addEventListener('click',function(e){e.stopPropagation();
          s.panes[i]=null;activePane=i;markDirty();renderCreate();});
        p.appendChild(x);
      }
      p.addEventListener('click',function(){activePane=i;renderCreate();});
      ed.appendChild(p);
    })(i);
  }
  function paneImgSrc(anchor){
    var card=anchor?cardEl(anchor):null;
    var img=card?$('.figframe img',card):null;
    return img?img.getAttribute('src'):null;
  }
  function paneThumb(anchor){
    var w=document.createElement('span');w.className='mini-pane';
    var it=anchor?itemsByAnchor[anchor]:null;
    if(!it){w.className+=' empty';return w;}
    var src=paneImgSrc(anchor);
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
    d.className='mini-diagram '+s.layout;
    var n=PANES[s.layout]||1;
    for(var i=0;i<n;i++) d.appendChild(paneThumb(s.panes[i]));
    return d;
  }
  function slideTitle(s){
    for(var i=0;i<s.panes.length;i++){
      var it=s.panes[i]&&itemsByAnchor[s.panes[i]];
      if(it) return it.title;
    }
    return 'empty slide';
  }
  function renderFilm(){
    var list=$('#film-list');list.innerHTML='';
    pres.slides.forEach(function(s,i){
      var row=document.createElement('div');
      row.className='film-row'+(i===cur?' current':'');
      var lbl=document.createElement('button');lbl.className='film-label';
      var num=document.createElement('span');num.className='film-n';
      num.textContent=(i+1);lbl.appendChild(num);
      lbl.appendChild(miniDiagram(s));
      var tt=document.createElement('span');tt.className='film-t';
      tt.textContent=slideTitle(s);lbl.appendChild(tt);
      lbl.addEventListener('click',function(){
        cur=i;activePane=firstEmpty(s);renderCreate();});
      row.appendChild(lbl);
      var ctr=document.createElement('span');ctr.className='film-ctr';
      [['↑',function(){moveSlide(i,-1);}],
       ['↓',function(){moveSlide(i,1);}],
       ['✕',function(){delSlide(i);}]].forEach(function(p){
        var b=document.createElement('button');b.className='film-mini';
        b.textContent=p[0];
        b.addEventListener('click',function(ev){
          ev.stopPropagation();p[1]();});
        ctr.appendChild(b);
      });
      row.appendChild(ctr);
      list.appendChild(row);
    });
  }
  function renderCreate(){
    renderPresRow();renderLayoutRow();renderPaneEditor();renderFilm();
  }
  function moveSlide(i,d){
    var j=i+d; if(j<0||j>=pres.slides.length) return;
    var t=pres.slides[i];pres.slides[i]=pres.slides[j];pres.slides[j]=t;
    if(cur===i)cur=j; else if(cur===j)cur=i;
    markDirty();renderCreate();
  }
  function delSlide(i){
    pres.slides.splice(i,1);
    if(cur>=pres.slides.length) cur=Math.max(0,pres.slides.length-1);
    activePane=firstEmpty(pres.slides[cur]);
    markDirty();renderCreate();
  }

  /* ---------- mode switching ---------- */
  function setToolbarMode(open){
    var d=$('#tb-docs'), p=$('#tb-present');
    if(d) d.setAttribute('aria-pressed',(!open).toString());
    if(p) p.setAttribute('aria-pressed',open.toString());
  }
  function setUIMode(m){
    mode=m;
    var creating=(m==='create');
    deckEl.classList.toggle('creating',creating);
    $('#deck-create').hidden=!creating;
    document.body.classList.toggle('creating-docs',
      creating&&!deckEl.hidden);
    document.body.classList.toggle('deck-open',
      !creating&&!deckEl.hidden);
    if(creating){
      activePane=firstEmpty(pres.slides[cur]);
      renderCreate();
    } else renderSlide();
  }
  function openDeck(m){
    deckEl.hidden=false;
    setToolbarMode(true);status();
    setUIMode(m||'view');
  }
  function closeDeck(){
    deckEl.hidden=true;
    document.body.classList.remove('deck-open');
    document.body.classList.remove('creating-docs');
    deckEl.classList.remove('creating');
    setToolbarMode(false);
  }
  var presentBtn=$('#tb-present');
  if(presentBtn) presentBtn.addEventListener('click',function(){
    openDeck('create');});
  var tbDocs=$('#tb-docs');
  if(tbDocs) tbDocs.addEventListener('click',function(){
    if(!deckEl.hidden) closeDeck();});
  $('#deck-docs').addEventListener('click',closeDeck);
  $('#dc-play').addEventListener('click',function(){setUIMode('view');});
  $('#deck-exit').addEventListener('click',function(){
    setUIMode('create');});
  $('#deck-prev').addEventListener('click',function(){go(cur-1);});
  $('#deck-next').addEventListener('click',function(){go(cur+1);});
  codeBtn.addEventListener('click',function(){
    var open=!drawer.hidden;
    drawer.hidden=open;
    codeBtn.textContent=open?'Show code':'Hide code';
    codeBtn.setAttribute('aria-expanded',(!open).toString());
  });
  document.addEventListener('keydown',function(e){
    if(deckEl.hidden) return;
    var tag=(e.target.tagName||'').toLowerCase();
    if(tag==='input'||tag==='select'||tag==='textarea') return;
    if(e.key==='Escape'){
      if(mode==='view') setUIMode('create'); else closeDeck();
    }
    else if(mode==='view'){
      if(e.key==='ArrowRight'||e.key==='PageDown'
         ||(e.key===' '&&tag!=='button')){e.preventDefault();go(cur+1);}
      else if(e.key==='ArrowLeft'||e.key==='PageUp'){
        e.preventDefault();go(cur-1);}
    }
  });

  /* ---------- create mode: click a card in the document to place it */
  document.addEventListener('click',function(e){
    if(deckEl.hidden||mode!=='create') return;
    var t=e.target;
    if(!t||!t.closest) return;
    if(deckEl.contains(t)) return;
    var card=t.closest('.card');
    if(!card) return;
    if(t.closest('.codetoggle,.depchip,a')) return;
    e.preventDefault();e.stopPropagation();
    if(!pres.slides.length){
      pres.slides.push(emptySlide());cur=0;activePane=0;
    }
    var s=pres.slides[cur];
    s.panes[activePane]=card.dataset.anchor;
    var n=PANES[s.layout]||1;
    for(var k=1;k<=n;k++){
      var j=(activePane+k)%n;
      if(!s.panes[j]){activePane=j;break;}
    }
    markDirty();renderCreate();
    card.classList.add('target-flash');
    setTimeout(function(){card.classList.remove('target-flash');},700);
  },true);

  /* ---------- create mode: slide + presentation operations ---------- */
  $('#film-add').addEventListener('click',function(){
    var at=pres.slides.length?cur+1:0;
    pres.slides.splice(at,0,emptySlide());
    cur=at;activePane=0;markDirty();renderCreate();
  });
  $$('#layout-row .lay').forEach(function(b){
    b.addEventListener('click',function(){
      var s=pres.slides[cur]; if(!s) return;
      var n=PANES[b.dataset.lay];
      var panes=s.panes.slice(0,n);
      while(panes.length<n) panes.push(null);
      s.layout=b.dataset.lay;s.panes=panes;
      if(activePane>=n) activePane=0;
      markDirty();renderCreate();
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
  menuAction('#mi-new',function(){
    var n2=1, name='presentation';
    while(savedByName(name)||loadDraft(name)){
      n2++;name='presentation-'+n2;}
    pres={name:name,slides:[emptySlide()]};
    cur=0;activePane=0;markDirty();renderCreate();
  });
  menuAction('#mi-rename',function(){
    var inp=$('#pres-name');
    inp.hidden=false;inp.value=pres.name;
    inp.focus();inp.select();
  });
  menuAction('#mi-auto-figs',function(){
    pres.slides=autoSlides(false);cur=0;activePane=0;
    markDirty();renderCreate();
    toast(pres.slides.length+' slides: one per figure, in order');
  });
  menuAction('#mi-auto-figdocs',function(){
    pres.slides=autoSlides(true);cur=0;activePane=0;
    markDirty();renderCreate();
    toast(pres.slides.length+' slides: figures + docs, in order');
  });
  $('#pres-select').addEventListener('change',function(){
    var v=this.value;
    if(v===pres.name) return;
    lsSet(PFX+'last',v);
    loadPresentation(v);
    cur=0;activePane=firstEmpty(pres.slides[0]);
    status();renderCreate();
  });
  $('#pres-name').addEventListener('input',function(){
    var old=pres.name;
    pres.name=this.value.trim();
    if(old&&old!==pres.name) lsDel(PFX+old);
    markDirty();
  });
  $('#pres-name').addEventListener('keydown',function(e){
    if(e.key==='Enter'||e.key==='Escape') this.blur();
  });
  $('#pres-name').addEventListener('blur',function(){
    this.hidden=true;renderPresRow();
  });

  /* ---------- persistence ---------- */
  var toastTimer;
  function toast(msg){
    var t=$('#deck-toast');t.textContent=msg;t.hidden=false;
    clearTimeout(toastTimer);
    toastTimer=setTimeout(function(){t.hidden=true;},3600);
  }
  function mergedPresentations(){
    var out=saved.filter(function(p){return p.name!==pres.name;});
    out.push(JSON.parse(JSON.stringify(pres)));
    return out;
  }
  var writeBtn=$('#mi-save');
  if(!window.showOpenFilePicker) writeBtn.hidden=true;
  writeBtn.addEventListener('click',function(){
    closeMenu();
    if(!pres.name){
      toast('Give the presentation a name first');
      var ni=$('#pres-name');ni.hidden=false;ni.focus();return;
    }
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
        nb.metadata.semantic.presentations=mergedPresentations();
        delete nb.metadata.semantic.deck;
        var w=await h.createWritable();
        await w.write(JSON.stringify(nb,null,1));
        await w.close();
        saved=mergedPresentations();
        lsDel(PFX+(pres.name||'untitled'));
        source='saved';status();renderPresRow();
        toast('Saved "'+pres.name+'" into '+f.name);
      }catch(e){
        if(!e||e.name!=='AbortError')
          toast('Save failed: '+(e&&e.message?e.message:e));
      }
    })();
  });
  menuAction('#mi-dl',function(){
    var blob=new Blob(
      [JSON.stringify({presentations:mergedPresentations()},null,2)],
      {type:'application/json'});
    var a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download=(DATA.stem||'notebook')+'.deck.json';
    a.click();
    setTimeout(function(){URL.revokeObjectURL(a.href);},2000);
    toast('Downloaded. Keep it next to the .ipynb (auto-loads) '
      +'or bake in with --embed-deck.');
  });
  menuAction('#mi-discard',function(){
    lsDel(PFX+(pres.name||'untitled'));
    loadPresentation(pres.name);
    cur=0;activePane=firstEmpty(pres.slides[0]);
    status();
    if(mode==='create') renderCreate(); else renderSlide();
  });

  status();
})();
"""

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
<style>{deck_css}</style>
{mathjax}
</head>
<body>
<div class="scrim" id="scrim"></div>
<div class="shell">
  <aside class="rail" id="rail">
    <div class="railhead">
      <p class="brand">semantic notebook</p>
      <h1 class="railtitle">{title}</h1>
      <div class="railmeta">{meta}</div>
    </div>
    {nav}
    {graph_panel}
  </aside>
  <main class="stage">
    <div class="toolbar">
      <button class="menubtn" id="menubtn" aria-label="Toggle sections"><span></span></button>
      <span class="tb-title">{title}</span>
      <div class="tb-actions">
        <button class="toggle tv" id="tv-figs"
          title="Show or hide figure cards"></button>
        <button class="toggle tv" id="tv-markup"
          title="Show or hide markdown/equation cards"></button>
        <button class="toggle tv" id="tv-code"
          title="Show or hide code, dataset and metric cards"></button>
        <span class="tb-sep"></span>
        <button class="toggle mode" id="tb-docs" aria-pressed="true"
          title="Document view">Docs</button>
        <button class="toggle mode" id="tb-present" aria-pressed="false"
          title="Build and play slide decks">Presentation mode</button>
      </div>
    </div>
    <div class="content">
      {sections}
    </div>
  </main>
</div>
{deck_shell}
<script type="application/json" id="deck-data">{deck_data}</script>
<script>{js}</script>
<script>{deck_js}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def render_notebook_file(path: Path, title: str | None = None,
                         deck_path: Path | None = None) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    doc = parse_notebook(nb, title=title)
    # deck priority: --deck file > sidecar next to the notebook > embedded
    # metadata (parse_notebook already loaded metadata.semantic.deck)
    if deck_path is None:
        sidecar = path.with_suffix(".deck.json")
        if sidecar.exists():
            deck_path = sidecar
    if deck_path is not None:
        pres = _as_presentations(
            json.loads(Path(deck_path).read_text(encoding="utf-8")))
        if pres:
            doc.presentations = pres
    return render_html(doc, source_name=path.stem)


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
            {"cell_type": "code", "id": "c-prep",
             "source": "#| title: Open dataset\nds = open_thing()",
             "outputs": []},
            {"cell_type": "code", "id": "c-fig2",
             "source": "#| display: figure\n#| id: fig2\n#| title: Second figure\nplot(ds)",
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
    assert 'id="deck-data"' in out and 'id="tb-present"' in out
    assert 'id="tv-markup"' in out and 'id="deck-docs"' in out
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
    print("self-test ok:", len(out), "bytes;",
          sum(len(s.items) for s in doc.sections), "items;",
          "presentations:", len(doc.presentations))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render a notebook into a semantic analysis environment.")
    p.add_argument("notebook", nargs="?", help="path to an executed .ipynb")
    p.add_argument("-o", "--output", help="output .html (default: alongside the notebook)")
    p.add_argument("--title", help="override the analysis title")
    p.add_argument("--deck", help="presentation deck JSON to use "
                   "(default: <notebook>.deck.json sidecar, then embedded metadata)")
    p.add_argument("--embed-deck", metavar="DECK_JSON",
                   help="write DECK_JSON into the notebook's "
                   "metadata.semantic.deck (modifies the .ipynb) and exit")
    p.add_argument("--self-test", action="store_true", help="run a built-in sanity check and exit")
    args = p.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.notebook:
        p.error("provide a notebook path (or --self-test)")

    src = Path(args.notebook)
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 1

    if args.embed_deck:
        embed_deck(src, Path(args.embed_deck))
        print(f"embedded {args.embed_deck} into {src} "
              "(metadata.semantic.deck)")
        return 0

    html_out = render_notebook_file(
        src, title=args.title,
        deck_path=Path(args.deck) if args.deck else None)
    out_path = Path(args.output) if args.output else src.with_suffix(".html")
    out_path.write_text(html_out, encoding="utf-8")
    print(f"wrote {out_path}  ({len(html_out)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
