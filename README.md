# Semantic notebook renderer

Turn an **executed** Jupyter notebook into a figure-first, nonlinear analysis
environment. Instead of rendering every cell with equal weight (the Quarto /
nbconvert model), it recovers the scientific structure underneath the notebook —

```
Dataset → Transform → Diagnostic → Figure → Interpretation
```

— and renders figures as the primary objects, with code collapsed behind them,
sections in a navigable rail, and a live **provenance graph** of how each
diagnostic derives from the data.

It is a **static** renderer: no kernel, no server, no re-execution. Run your
notebook once, normally, in Jupyter; the renderer reads the outputs already
stored in the `.ipynb`. The result is a single self-contained `.html` file you
can open in any browser or email to a collaborator.

---

## Run it

```bash
# 1. run your notebook in Jupyter so outputs are saved, then:
python semantic_render.py my_notebook.ipynb

# writes my_notebook.html next to it. Or choose the output / title:
python semantic_render.py my_notebook.ipynb -o report.html --title "Run 42"
```

Only dependency is the Python standard library. Open the example to see it:

```bash
python semantic_render.py example_climate_analysis.ipynb
open example_climate_analysis.html
```

To rebuild the example notebook from scratch (needs `nbformat`, `nbclient`,
`xarray`, `matplotlib`):

```bash
python make_example_notebook.py
jupyter execute example_climate_analysis.ipynb   # or run it in Jupyter
python semantic_render.py example_climate_analysis.ipynb
```

---

## How to format a notebook for it

You annotate cells with `#| key: value` **directive lines at the very top of a
code cell**. They are parsed and then stripped from the displayed source.
Everything is optional — with no directives at all the renderer still infers a
sensible layout from each cell's outputs.

### Directives

| Directive       | What it does                                                         |
|-----------------|----------------------------------------------------------------------|
| `#| section:`   | Start a top-level section (also doable with a Markdown `##` heading). |
| `#| subsection:`| Nested group inside the current section.                             |
| `#| title:`     | Human title for the card (otherwise inferred from the code).        |
| `#| display:`   | Card type: `figure` `dataset` `transform` `diagnostic` `metric` `text` `code` `hidden`. |
| `#| code:`      | Default code visibility: `hidden` (default) or `show`.              |
| `#| id:`        | Stable slug for this cell — makes it a node in the provenance graph. |
| `#| depends:`   | Comma-separated `id`s this cell derives from — draws the graph edges.|
| `#| caption:`   | Interpretation text / what to look for, shown under the output.     |
| `#| group:`     | Merge several cells into **one** card (alias: `tag:`).              |
| `#| order:`     | Sort this cell within its group (integer; defaults to appearance).  |
| `#| step:`      | Label this cell's chunk in the folded code.                         |
| `#| stack:`     | Fold the code of cells with these `id`s under this card; reusable.  |

### A figure cell

```python
#| display: figure
#| id: block_comp
#| depends: anom, block_freq
#| title: Composite Z500 anomaly on blocked days
#| caption: The localised positive centre is the blocking high.
comp = z_anom.sel(time=blocked).mean('time')
comp.plot(cmap='RdBu_r')
```

This renders as a figure card titled "Composite Z500 anomaly on blocked days",
with the caption beneath it, the code tucked behind a **Show code** toggle, a
`derives from anom · block_freq` provenance line, and a node in the rail graph
wired to the `anom` and `block_freq` nodes.

### Grouping several cells under one figure

A figure is usually the last step of a small pipeline — regrid, composite,
plot. Give those cells the same `#| group:` name and they collapse into a
single card: the cell that draws the figure is the face, and the prep folds
behind one **Show code** toggle as numbered steps.

```python
#| group: fig_zonal
#| order: 1
#| step: zonal mean + 30-day smoothing
zm = z_anom.mean('lon')
zm_mon = zm.rolling(time=30, center=True).mean().resample(time='1MS').mean()
```

```python
#| group: fig_zonal
#| order: 2
#| step: plot Hovmöller
#| display: figure
#| id: zonal_hov
#| depends: anom
#| title: Zonal-mean Z500 anomaly (time–latitude)
zm_mon.plot(x='time', cmap='RdBu_r')
```

Both cells become the one **zonal_hov** card. Notes on the merge:

- **Face** = the cell with `display: figure` (or, absent that, the last cell
  that produces an image / any output). Its output is shown; the others' code
  folds underneath, and any *intermediate* output (a printed shape, a repr)
  is tucked under its own step.
- **Title / caption / id** come from the group — the first member that sets
  each wins, preferring the figure cell.
- **`depends`** is the union across all members, so the prep cell's inputs and
  the plot cell's inputs both feed the one node. The group is therefore a
  **single** vertex in the provenance graph — grouping declutters the graph as
  well as the page.
- **Section / subsection** is taken from where the group's first cell sits (a
  `##` / `###` heading above it, or a `subsection:` on that cell).

`step:` is the clean way to label a chunk; `subsection:` on a grouped member
is also accepted as a chunk label, to match the obvious shorthand.

### Stacking shared cells under a figure (reuse)

Grouping is *push* — each cell tags itself into one group, so a cell can only
live under a single figure. When the same prep feeds **several** figures
(opening the data, regridding, a shared plotting helper), use `#| stack:`
instead. A figure names the upstream cells by `id`, and they fold in front of
its own code:

```python
#| id: maphelper                 # define the shared cell once
#| step: shared map helper
def plot_anom_map(da, ax, title, vmax=None):
    ...
```

```python
#| display: figure
#| id: block_comp
#| depends: anom, block_freq
#| stack: maphelper              # ← fold the helper under this figure
comp = z_anom.sel(time=blocked).mean('time')
pc = plot_anom_map(comp, ax, 'Blocked-day composite')
```

```python
#| display: figure
#| id: enso_comp
#| depends: anom, nino34
#| stack: maphelper              # ← and under this one too (same cell)
pc = plot_anom_map(tele, ax, 'Warm-phase composite')
```

`maphelper` now folds in as step 1 of **both** composite cards. Key points:

- A cell named in any `stack:` list is **consumed**: it gets no card of its
  own and no graph node — it lives only under the figures that stack it. The
  same id may be stacked under any number of figures.
- Stacked cells render **before** the card's own code, in the order listed;
  the figure's own code is the final step. Use `#| order:`-style intent by
  ordering the ids in the list.
- Stacking folds **code only**; it does *not* add provenance edges. Use
  `depends:` for lineage you want drawn in the graph.

#### group vs stack — which to use

| | `group:` (push) | `stack:` (pull) |
|---|---|---|
| Who references whom | each cell tags itself | the figure names cells by `id` |
| Cell can belong to | one card | any number of cards |
| Best for | a few adjacent cells authored as a unit | shared prep reused across figures |
| Standalone card | merged away | consumed (no card, no node) |

A useful split to remember: **`depends:` keeps a cell as its own node in the
graph; `stack:` folds its code into a figure and collapses it.** One is about
scientific lineage, the other about reproducibility of a single figure.



A Markdown cell whose first line is a heading opens structure:

```markdown
## Blocking diagnostics      ← H2 opens a section
### Regional composite       ← H3 opens a subsection
```

Any prose under a heading (or any plain Markdown cell) becomes an
**interpretation note** — rendered in a serif face to set human commentary
apart from machine output.

You can mix styles: use `##` headings for some sections and `#| section:` on a
code cell for others. The example notebook does both.

### What happens with no directives

| Cell produces…            | Inferred card |
|---------------------------|---------------|
| an image                  | `figure`      |
| an xarray HTML repr        | `dataset`     |
| only short text / stdout   | `metric`      |
| longer text                | `text`        |
| no output                  | `code` (collapsed) |

So an un-annotated notebook still renders cleanly; directives are how you take
control — naming diagnostics, writing captions, and declaring the provenance
graph with `id` / `depends`.

---

## What you get in the page

- **Left rail** — section tree plus a live analysis graph. Scrolling highlights
  the active section and its node; clicking a node jumps to that card.
- **Figure stage** — each diagnostic as a card: title, output, serif caption,
  amber `derives from …` provenance chips (click to jump to a source), and a
  collapsible code block.
- **Toolbar** — **Docs** and **Presentation mode** are the two view
  buttons, always top-right; the one you are in shows as selected.
  Beside them, three standalone buttons whose labels follow the state:
  *Hide/Show figures*, *Hide/Show markup* (the markdown/equation cells)
  and *Hide/Show code* (code, dataset and metric cards). Any combination
  works — hide code for a figures-plus-documentation reading view, leave
  only markup for just the prose. A hidden card collapses to a slim
  dashed stub that expands in place when clicked, so nothing is ever
  more than a click away.
- Responsive to mobile, keyboard-navigable, respects reduced-motion.

---

## Presentation deck

**Presentation mode** works like PowerPoint: it opens the **builder**
(below), and the **&#9654; Present** button at the top of the panel plays
the deck full-screen — arrow buttons / arrow keys to move, `Esc` or
**&#10005; Exit** to drop back to the builder, **Docs** to leave
altogether. With nothing saved yet you get *auto: figures* — one
full-screen slide per figure, in document order, each with its caption
and a **Show code** drawer underneath.

The drawer tells the figure's whole computational story, not just its
plotting call: every upstream card is a collapsible subheaded section in
execution order — *open data → regrid → statistics → this figure* — with
the upstream steps folded and the figure's own code open.
The chain is the union of your declared `depends:` edges and **automatic
variable tracing**: the renderer parses each cell's code and links a
figure to whichever cells last assigned the variables it reads, so even
un-annotated notebooks get the full lineage. (Static best-effort: it
can't see mutation without assignment, e.g. `ds.load()`; declare
`depends:` where the trace misses something.)

### Create mode

The builder docks on the left with the real document view beside it —
toolbar, filters and all. **&#9654; Present** at the top plays the deck;
the top-right **Docs** button exits back to the document. You build
slides by pointing at the document:

- **+ Add slide**, then pick a layout: **Full**, **Halves** or
  **Quarters**.
- Click a pane in the layout diagram, then **click a card in the
  document** to place it there; the next empty pane is selected
  automatically, and ✕ on a pane clears it. Figure panes show a faint
  live preview of their image.
- The filmstrip shows PowerPoint-style thumbnails of every slide with the
  actual content — scaled-down figures, text stripes for markup — click
  to select, ↑ ↓ to reorder, ✕ to delete.
- Everything else lives in the **File ▾** menu: *New presentation*,
  *Rename*, the two auto-builders (*figures* / *figures + docs*, in
  document order), *Save to notebook*, *Download JSON* and *Discard
  changes*.

Markdown cards render with bullets/bold and **LaTeX equations**
(`$...$`, `$$...$$`, typeset by MathJax — needs internet on first view),
so "figure with its equations beside it" is a *Halves* slide: the figure
in one pane, the markdown card in the other.

Edits autosave as a **draft** in the browser (`localStorage`), per
presentation — refresh and nothing is lost; the status pill shows
*auto / saved / unsaved draft*, and *Discard* reverts to the saved copy.

### Named presentations, saved in the notebook

A notebook can hold **multiple named presentations** — switch between
them (or start a new one) with the selector in Create mode. They live
under `metadata.semantic.presentations`, so they survive editing,
re-running and re-rendering. Two routes:

1. **Save to .ipynb** (Chrome / Edge) — pick the notebook file once and
   every presentation is written into its metadata. Reload the notebook
   in Jupyter afterwards if it is open there.
2. **Download** — saves `<notebook>.deck.json` next to the page. Put it
   beside the `.ipynb`: the renderer auto-loads the sidecar on every run.
   Bake it into the notebook with
   `python semantic_render.py nb.ipynb --embed-deck nb.deck.json`.

Slides reference cards by a **stable anchor** — the cell's `#| id:` if it
has one, else the notebook's built-in cell id (nbformat ≥ 4.5) — never by
position. Reordering, editing or adding cells does not break the deck;
deleting a referenced cell just skips that slide with a note. Prefer
`#| id:` anchors: they also survive copy-pasting cells between notebooks.

`--deck other.json` renders with a specific deck file (overrides the
sidecar and the embedded metadata).

---

## Design notes / extending

- The renderer is one file (`semantic_render.py`) with the directive parser,
  a tokenizer-based Python highlighter, output renderers, the semantic model,
  the graph-layout, and the inlined HTML/CSS/JS. `python semantic_render.py
  --self-test` runs a built-in sanity check.
- Because the static page is precomputed, its interactivity is limited to what
  can be baked in (navigation, provenance highlighting, code toggles). For a
  *live* kernel — recompute and persistent arrangement — use the widget below.

---

## Live in Jupyter (`semantic_widget.py`)

The static page is for sharing — one self-contained file, no Jupyter needed.
When you want to *explore*, `SemanticNotebook` renders the same figure-first
view **inside a notebook**, against the live kernel. It imports the same parser
and the same card HTML from `semantic_render`, so the directive format and the
look are identical; it just adds the two things a dead file can't do.

```python
from semantic_widget import SemanticNotebook

view = SemanticNotebook.from_ipynb("analysis.ipynb")   # an executed notebook
view
```

**1 · View-state that persists back to Python.** Hide a card (hover → ✕), switch
the layout (Column / Grid / Compact), or collapse the graph, and the choices are
synced to the kernel:

```python
view.view_state          # -> {'hidden': ['enso_spec'], 'layout': 'grid', ...}
view.export_html("clean.html")   # static page with hidden cards dropped
```

`export_html` is the bridge back to the shareable artifact: arrange it live,
then export a clean page.

**2 · Live recompute.** Attach a function to a figure `id`; its parameters
become controls, and changing them re-runs the function on the kernel and swaps
that figure in place. Every other figure stays static.

```python
@view.recompute("block_comp",
                threshold=(0.5, 2.5, 0.1),       # slider
                region=["Tasman", "Ross", "Weddell"])   # dropdown
def _(threshold, region):
    box = z_anom.sel(**REGIONS[region]).mean(["lat", "lon"])
    comp = z_anom.sel(time=box > box.std() * threshold).mean("time")
    fig, ax = plt.subplots()
    plot_anom_map(comp, ax, f"{region} composite")
    return fig          # return a matplotlib Figure (or just draw one)
```

Parameter shorthands:

| you write | control |
|-----------|---------|
| `(lo, hi)` or `(lo, hi, step)` | slider |
| `["a", "b", "c"]` | dropdown |
| `5` / `1.5` | number box |
| `True` | checkbox |
| `"text"` | text box |
| `{"type": "range", "min": …, "max": …, "value": …}` | full control |

The recompute function closes over your kernel namespace, so it can use the same
variables and helpers your analysis already defined (`z_anom`, `plot_anom_map`,
…). `make_widget_demo.py` builds `example_widget.ipynb`, a complete runnable
example with one live composite figure.

Constructors: `SemanticNotebook.from_ipynb(path)`, `.from_notebook(nb)` (an
`nbformat` object), or `SemanticNotebook(document=…)` if you already parsed one.
`height=` sets the panel height.

**Requirements / notes.** Needs `anywidget` and `ipywidgets` (`pip install
anywidget`). Runs in JupyterLab, Notebook 7, VS Code, and Colab. Fonts load from
Google Fonts, and recompute needs a live kernel — so both only show up when you
actually run it in Jupyter, not in a previewed `.ipynb`. The widget CSS is
scoped to a private root so it can't bleed into the rest of your notebook.

---

## The three tiers

The durable asset is the **directive spec + parser/model** in `semantic_render`;
every frontend consumes it.

- **Static HTML** (`render_html`) — share, publish, attach to CI. No Jupyter.
- **Widget** (`SemanticNotebook`) — explore in-kernel: recompute + persistent
  arrange/hide. This is Tier 1.
- A full JupyterLab extension (directive autocomplete, continuous two-way model
  sync, editor squiggles) would be Tier 2 — worth building only if this becomes
  the primary way several people read notebooks day to day.
# semantic-rendering
