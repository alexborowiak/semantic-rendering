"""
semantic_widget.py
==================

Tier-1 of the semantic notebook renderer: an in-notebook `anywidget` that
renders the *same* figure-first semantic view as the static renderer (it
imports the same parser and the same card HTML), but adds the two things a
dead HTML file cannot do:

  1. View-state that persists back to Python -- hide cards, switch layout,
     collapse the graph; read it back as `widget.view_state`, or
     `widget.export_html(...)` a clean static page with your arrangement baked
     in.
  2. Live recompute -- attach a function (with parameters) to a figure `id`;
     moving a slider re-runs it on the kernel and swaps the figure in place.

Usage
-----
    from semantic_widget import SemanticNotebook

    view = SemanticNotebook.from_ipynb("analysis.ipynb")   # executed notebook

    # optional: make one figure live (others stay static)
    @view.recompute("block_comp", threshold=(0.5, 2.5, 0.1),
                    region=["Tasman", "Ross", "Weddell"])
    def _(threshold, region):
        ...                      # uses your kernel's variables
        return fig               # return a matplotlib Figure

    view                          # display it

Parameter shorthands for `recompute(...)`:
    (lo, hi)        -> slider          (lo, hi, step) -> slider with step
    ["a", "b"]      -> dropdown        5 / 1.5        -> number box
    True            -> checkbox        "text"         -> text box
A dict like {"type": "range", "min": 0, "max": 1, "step": .1, "value": .5}
gives full control.
"""

from __future__ import annotations

import base64
import copy
import html
import io
import json
from pathlib import Path

import anywidget
import traitlets

from semantic_render import (
    parse_notebook, render_nav, render_sections, render_graph_panel,
    doc_meta, render_html, _CSS as _BASE_CSS,
)


# --------------------------------------------------------------------------
# CSS scoping: wrap the static stylesheet so it cannot leak into the notebook
# --------------------------------------------------------------------------

def _scope_selectors(prelude: str, root: str) -> str:
    out = []
    for sel in prelude.split(","):
        s = sel.strip()
        if not s:
            continue
        if ":root" in s:
            out.append(s.replace(":root", root))
        elif s in ("body", "html"):
            out.append(root)
        elif s.startswith(root):
            out.append(s)
        else:
            out.append(f"{root} {s}")
    return ",".join(out)


def _scope_css(css: str, root: str = ".snb-root") -> str:
    """Prefix every rule in `css` with `root` (quote/paren/brace aware)."""
    out, i, n = [], 0, len(css)
    while i < n:
        # scan to the next top-level '{' or ';'
        j, instr, paren = i, None, 0
        while j < n:
            c = css[j]
            if instr:
                if c == instr:
                    instr = None
            elif c in "\"'":
                instr = c
            elif c == "(":
                paren += 1
            elif c == ")":
                paren = max(0, paren - 1)
            elif paren == 0 and c in "{;":
                break
            j += 1
        if j >= n:
            tail = css[i:].strip()
            if tail:
                out.append(tail)
            break
        if css[j] == ";":                      # at-statement, e.g. @import ...;
            out.append(css[i:j + 1].strip())
            i = j + 1
            continue
        prelude = css[i:j].strip()             # block: find matching '}'
        depth, k, instr = 1, j + 1, None
        while k < n and depth:
            c = css[k]
            if instr:
                if c == instr:
                    instr = None
            elif c in "\"'":
                instr = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            k += 1
        inner = css[j + 1:k - 1]
        if prelude.startswith(("@media", "@supports")):
            out.append(prelude + "{" + _scope_css(inner, root) + "}")
        elif prelude.startswith("@"):          # keyframes / font-face: leave
            out.append(prelude + "{" + inner + "}")
        else:
            out.append(_scope_selectors(prelude, root) + "{" + inner + "}")
        i = k
    return "".join(out)


# widget-only styling (plain selectors; scoped by _scope_css below)
_WIDGET_EXTRA = r"""
.shell{min-height:0;}
.rail{height:auto;max-height:var(--snb-h,760px);}
.content{max-height:var(--snb-h,760px);overflow:auto;scroll-behavior:smooth;}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;
  margin:0 0 12px 6px;padding:10px 12px;background:var(--paper-2);
  border:1px solid var(--paper-3);border-radius:8px;}
.ctl{display:inline-flex;align-items:center;gap:7px;font-family:var(--mono);
  font-size:11px;color:var(--ink-2);}
.ctl .cl{letter-spacing:.05em;text-transform:uppercase;color:var(--ink-3);}
.ctl input[type=range]{accent-color:var(--cyan);width:120px;}
.ctl input[type=number],.ctl input[type=text],.ctl select{font-family:var(--mono);
  font-size:12px;border:1px solid var(--line);border-radius:5px;padding:3px 6px;
  background:#fff;color:var(--ink);}
.ctl output{font-family:var(--mono);font-size:11px;color:var(--cyan-deep);
  min-width:30px;}
.ctl-cb{text-transform:none;}
.recbtn{font-family:var(--mono);font-size:11px;border:1px solid var(--cyan);
  background:var(--cyan);color:#fff;border-radius:5px;padding:4px 11px;
  cursor:pointer;letter-spacing:.04em;}
.recbtn:hover{background:var(--cyan-deep);border-color:var(--cyan-deep);}
.recstatus{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);}
.card.recomputing .cardbody{opacity:.45;transition:opacity .2s;}
.hidebtn{margin-left:4px;flex:none;width:22px;height:22px;border-radius:5px;
  border:1px solid transparent;background:none;color:var(--ink-3);cursor:pointer;
  font-size:11px;line-height:1;opacity:0;transition:opacity .15s;}
.card:hover .hidebtn{opacity:.8;}
.hidebtn:hover{background:#fbecea;color:#b8402f;border-color:#f0d2cc;opacity:1;}
.card.is-hidden,.card.filtered-out{display:none;}
.hidden-menu{display:none;}
.hidden-menu.open{display:block;margin:0 0 12px;padding:8px 10px;
  border:1px solid var(--line);border-radius:8px;background:#fff;}
.hidrow{display:flex;justify-content:space-between;align-items:center;gap:12px;
  padding:4px 6px;font-size:12px;}
.hidrow button{font-family:var(--mono);font-size:10.5px;border:1px solid var(--line);
  background:var(--paper-2);border-radius:4px;padding:2px 9px;cursor:pointer;}
.hidempty{font-size:11px;color:var(--ink-3);padding:4px 6px;}
.content[data-layout=grid]>.section{display:grid;
  grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;
  align-items:start;}
.content[data-layout=grid] .sectionhead{grid-column:1/-1;}
.content[data-layout=compact] .codewrap,
.content[data-layout=compact] .caption,
.content[data-layout=compact] .prov{display:none;}
.content[data-layout=compact] .card{margin:8px 0;}
.tb-layout{padding-right:6px;}
"""

_WIDGET_MEDIA = r"""
@media (max-width:860px){
  .rail{position:static;transform:none;width:auto;max-height:none;height:auto;
    box-shadow:none;}
  .menubtn{display:none;}
}
"""

# strip the @import (fonts get injected via <link> from JS) then scope
_BASE_NO_IMPORT = "\n".join(
    ln for ln in _BASE_CSS.splitlines() if not ln.strip().startswith("@import"))

_WIDGET_CSS = _scope_css(_BASE_NO_IMPORT + _WIDGET_EXTRA + _WIDGET_MEDIA) + r"""
.snb-root{display:block;border:1px solid var(--line);border-radius:12px;
  overflow:hidden;}
"""


# --------------------------------------------------------------------------
# Front-end (ES module)
# --------------------------------------------------------------------------

_ESM = r"""
function esc(s){return (s==null?'':String(s)).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}

function ensureFonts(){
  if(document.getElementById('snb-fonts'))return;
  const l=document.createElement('link');
  l.id='snb-fonts'; l.rel='stylesheet';
  l.href='https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Serif:ital,wght@0,400;1,400&display=swap';
  document.head.appendChild(l);
}

function control(p){
  const lbl='<span class="cl">'+esc(p.name)+'</span>';
  if(p.type==='range')
    return '<label class="ctl">'+lbl+'<input data-p="'+esc(p.name)+'" type="range" min="'
      +p.min+'" max="'+p.max+'" step="'+(p.step||'any')+'" value="'+p.value
      +'"><output data-out="'+esc(p.name)+'">'+esc(p.value)+'</output></label>';
  if(p.type==='number')
    return '<label class="ctl">'+lbl+'<input data-p="'+esc(p.name)
      +'" type="number" step="'+(p.step||'any')+'" value="'+esc(p.value)+'"></label>';
  if(p.type==='select')
    return '<label class="ctl">'+lbl+'<select data-p="'+esc(p.name)+'">'
      +(p.options||[]).map(o=>'<option'+(String(o)===String(p.value)?' selected':'')
        +'>'+esc(o)+'</option>').join('')+'</select></label>';
  if(p.type==='checkbox')
    return '<label class="ctl ctl-cb"><input data-p="'+esc(p.name)+'" type="checkbox"'
      +(p.value?' checked':'')+'>'+lbl+'</label>';
  return '<label class="ctl">'+lbl+'<input data-p="'+esc(p.name)
    +'" type="text" value="'+esc(p.value)+'"></label>';
}

function shell(model){
  const d=model.get('data'); const t=esc(d.title), m=esc(d.meta);
  return '<div class="shell">'
    +'<aside class="rail"><div class="railhead"><p class="brand">semantic notebook</p>'
    +'<h1 class="railtitle">'+t+'</h1><div class="railmeta">'+m+'</div></div>'
    +(d.nav_html||'')+(d.graph_panel||'')+'</aside>'
    +'<main class="stage"><div class="toolbar"><span class="tb-title">'+t+'</span>'
    +'<div class="tb-actions">'
    +'<select class="toggle tb-layout" data-layout-select>'
    +'<option value="column">Column</option><option value="grid">Grid</option>'
    +'<option value="compact">Compact</option></select>'
    +'<button class="toggle" data-figs aria-pressed="false"><span class="tdot"></span>Figures only</button>'
    +'<button class="toggle" data-code aria-pressed="false"><span class="tdot"></span>All code</button>'
    +'<button class="toggle" data-hidden hidden><span class="tdot"></span>Hidden <span data-hidden-n></span></button>'
    +'</div></div><div class="hidden-menu" data-hidden-menu></div>'
    +'<div class="content" data-content>'+(d.stage_html||'')+'</div></main></div>';
}

function mount(model, el){
  ensureFonts();
  el.innerHTML='';
  const root=document.createElement('div');
  root.className='snb-root';
  root.style.setProperty('--snb-h', (model.get('height')||760)+'px');
  root.innerHTML=shell(model);
  el.appendChild(root);

  const $=(s,r)=>(r||root).querySelector(s);
  const $$=(s,r)=>Array.prototype.slice.call((r||root).querySelectorAll(s));
  const content=$('[data-content]');

  function vs(){return Object.assign({hidden:[],layout:'column',railCollapsed:false},
    model.get('view_state')||{});}
  function saveVs(p){const v=Object.assign(vs(),p);model.set('view_state',v);
    model.save_changes();}

  const state=vs();
  content.setAttribute('data-layout', state.layout||'column');
  const layoutSel=$('[data-layout-select]');
  if(layoutSel) layoutSel.value=state.layout||'column';
  if(state.railCollapsed){const rg=$('.railgraph');
    if(rg){rg.classList.add('collapsed');const b=$('.rg-collapse');
      if(b){b.textContent='+';b.setAttribute('aria-expanded','false');}}}

  function scrollTo(t){content.scrollTo({top:t.offsetTop-content.offsetTop-8,
    behavior:'smooth'});}
  function flash(c){c.classList.add('target-flash');
    setTimeout(()=>c.classList.remove('target-flash'),1300);}

  // hide buttons + restore the hidden set
  $$('.card').forEach(card=>{
    const head=$('.cardhead',card);
    if(head && !$('.hidebtn',head)){
      const b=document.createElement('button');
      b.className='hidebtn'; b.title='Hide card'; b.innerHTML='&#x2715;';
      b.addEventListener('click',()=>{card.classList.add('is-hidden');
        const h=new Set(vs().hidden); h.add(card.id.slice(5));
        saveVs({hidden:[...h]}); refreshHidden();});
      head.appendChild(b);
    }
  });
  (state.hidden||[]).forEach(id=>{const c=$('.card[id="card-'+id+'"]');
    if(c)c.classList.add('is-hidden');});

  const hiddenBtn=$('[data-hidden]'), hiddenMenu=$('[data-hidden-menu]');
  function refreshHidden(){
    const ids=vs().hidden||[];
    $('[data-hidden-n]').textContent=ids.length?('('+ids.length+')'):'';
    if(hiddenBtn) hiddenBtn.hidden=ids.length===0;
    if(hiddenMenu){
      hiddenMenu.innerHTML=ids.length? ids.map(id=>{
        const c=$('.card[id="card-'+id+'"]');
        const t=c?((($('.cardtitle',c)||{}).textContent)||id):id;
        return '<div class="hidrow"><span>'+esc(t)
          +'</span><button data-restore="'+esc(id)+'">Restore</button></div>';
      }).join(''):'<div class="hidempty">nothing hidden</div>';
      $$('[data-restore]',hiddenMenu).forEach(b=>b.addEventListener('click',()=>{
        const id=b.getAttribute('data-restore');
        const c=$('.card[id="card-'+id+'"]'); if(c)c.classList.remove('is-hidden');
        const h=new Set(vs().hidden); h.delete(id);
        saveVs({hidden:[...h]}); refreshHidden();}));
    }
  }
  if(hiddenBtn){hiddenBtn.addEventListener('click',()=>hiddenMenu.classList.toggle('open'));
    refreshHidden();}

  if(layoutSel) layoutSel.addEventListener('change',()=>{
    content.setAttribute('data-layout',layoutSel.value);
    saveVs({layout:layoutSel.value});});

  $$('.codetoggle').forEach(btn=>btn.addEventListener('click',()=>{
    const w=btn.closest('.codewrap'); const open=w.hasAttribute('data-open');
    if(open){w.removeAttribute('data-open');btn.setAttribute('aria-expanded','false');}
    else{w.setAttribute('data-open','');btn.setAttribute('aria-expanded','true');}}));

  const codeBtn=$('[data-code]');
  if(codeBtn) codeBtn.addEventListener('click',()=>{
    const on=codeBtn.getAttribute('aria-pressed')==='true';
    codeBtn.setAttribute('aria-pressed',String(!on));
    $$('.codewrap').forEach(w=>{if(!on)w.setAttribute('data-open','');
      else w.removeAttribute('data-open');
      const b=$('.codetoggle',w); if(b)b.setAttribute('aria-expanded',String(!on));});});

  const figBtn=$('[data-figs]');
  if(figBtn) figBtn.addEventListener('click',()=>{
    const on=figBtn.getAttribute('aria-pressed')==='true';
    figBtn.setAttribute('aria-pressed',String(!on));
    $$('.card').forEach(c=>{const k=c.dataset.kind;
      c.classList.toggle('filtered-out', !on && !(k==='figure'||k==='diagnostic'));});});

  const rgBtn=$('.rg-collapse');
  if(rgBtn) rgBtn.addEventListener('click',()=>{
    const rg=rgBtn.closest('.railgraph'); const c=rg.classList.toggle('collapsed');
    rgBtn.setAttribute('aria-expanded',String(!c)); rgBtn.textContent=c?'+':'\u2013';
    saveVs({railCollapsed:c});});

  $$('.navitem,.navsec').forEach(a=>a.addEventListener('click',e=>{
    const href=a.getAttribute('href');
    if(href && href[0]==='#'){e.preventDefault();
      const t=$('[id="'+href.slice(1)+'"]'); if(t)scrollTo(t);}}));

  $$('.provnode').forEach(g=>{
    const go=()=>{const c=$('.card[id="card-'+g.getAttribute('data-target')+'"]');
      if(c){scrollTo(c);flash(c);}};
    g.addEventListener('click',go);
    g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go();}});});

  $$('.depchip').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();
    const c=$('.card[data-node="'+a.getAttribute('data-dep')+'"]');
    if(c){scrollTo(c);flash(c);}}));

  const cards=$$('.card');
  if('IntersectionObserver' in window){
    const io=new IntersectionObserver(es=>es.forEach(e=>{
      if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target);}}),
      {root:content,rootMargin:'0px 0px -6% 0px',threshold:0.04});
    cards.forEach(c=>io.observe(c));
    const navItems={},navSecs={},graphNodes={};
    $$('.navitem').forEach(a=>navItems[a.dataset.item]=a);
    $$('.navsec').forEach(a=>navSecs[a.dataset.sec]=a);
    $$('.provnode').forEach(g=>graphNodes[g.dataset.node]=g);
    const vis={};
    const spy=new IntersectionObserver(es=>{
      es.forEach(e=>{if(e.isIntersecting)vis[e.target.id]=e.intersectionRatio;
        else delete vis[e.target.id];});
      let best=null,br=-1;
      Object.keys(vis).forEach(k=>{if(vis[k]>=br){br=vis[k];best=k;}});
      if(best){const item=best.slice(5);
        $$('.navitem.active').forEach(a=>a.classList.remove('active'));
        if(navItems[item])navItems[item].classList.add('active');
        const card=$('.card[id="'+best+'"]'); const sec=card&&card.closest('.section');
        $$('.navsec.active').forEach(a=>a.classList.remove('active'));
        if(sec&&navSecs[sec.dataset.sec])navSecs[sec.dataset.sec].classList.add('active');
        const nodeId=card?card.dataset.node:'';
        $$('.provnode.active').forEach(g=>g.classList.remove('active'));
        $$('.provedge.lit').forEach(p=>p.classList.remove('lit'));
        if(nodeId&&graphNodes[nodeId]){graphNodes[nodeId].classList.add('active');
          $$('.provedge').forEach(p=>{if(p.dataset.to===nodeId||p.dataset.from===nodeId)
            p.classList.add('lit');});}}
    },{root:content,rootMargin:'-10% 0px -55% 0px',threshold:[0,0.25,0.6,1]});
    cards.forEach(c=>spy.observe(c));
  } else cards.forEach(c=>c.classList.add('in'));

  // recompute controls
  const rc=(model.get('data')||{}).recompute||{};
  Object.keys(rc).forEach(id=>{
    const card=$('.card[id="card-'+id+'"]'); if(!card)return;
    const spec=rc[id].params||[];
    const strip=document.createElement('div'); strip.className='controls';
    strip.innerHTML=spec.map(control).join('')
      +'<button class="recbtn" data-run>Recompute</button>'
      +'<span class="recstatus" data-status></span>';
    const head=$('.cardhead',card); head.insertAdjacentElement('afterend',strip);
    function gather(){const out={};spec.forEach(p=>{
      const inp=$('[data-p="'+p.name+'"]',strip); if(!inp)return;
      if(p.type==='checkbox')out[p.name]=inp.checked;
      else if(p.type==='range'||p.type==='number')out[p.name]=inp.valueAsNumber;
      else out[p.name]=inp.value;}); return out;}
    let timer=null;
    function run(){card.classList.add('recomputing');
      const st=$('[data-status]',strip); if(st)st.textContent='recomputing\u2026';
      model.send({type:'recompute',id:id,params:gather()});}
    $$('input,select',strip).forEach(inp=>{
      inp.addEventListener('input',()=>{const o=$('output[data-out="'+inp.dataset.p+'"]',strip);
        if(o)o.textContent=inp.value;});
      inp.addEventListener('change',()=>{clearTimeout(timer);timer=setTimeout(run,200);});});
    const runBtn=$('[data-run]',strip); if(runBtn)runBtn.addEventListener('click',run);
  });

  model.on('msg:custom',msg=>{
    if(!msg||msg.type!=='recomputed')return;
    const card=$('.card[id="card-'+msg.id+'"]'); if(!card)return;
    const body=$('.cardbody',card); if(body)body.innerHTML=msg.output_html;
    card.classList.remove('recomputing');
    const st=$('[data-status]',card); if(st)st.textContent='';});
}

function render({model, el}){
  mount(model, el);
  const cb=()=>mount(model, el);
  model.on('change:data', cb);
  model.on('change:height', cb);
  return ()=>{model.off('change:data', cb); model.off('change:height', cb);};
}
export default { render };
"""


# --------------------------------------------------------------------------
# Parameter handling + figure capture (Python side)
# --------------------------------------------------------------------------

def _normalize_params(specs: dict) -> list[dict]:
    out: list[dict] = []
    for name, sp in specs.items():
        d: dict = {"name": name}
        if isinstance(sp, dict):
            d.update(sp)
            d.setdefault("type", "text")
            d.setdefault("value", "")
            d.setdefault("_py", {"range": "float", "number": "float",
                                 "checkbox": "bool"}.get(d["type"], "str"))
        elif isinstance(sp, bool):
            d.update(type="checkbox", value=sp, _py="bool")
        elif isinstance(sp, tuple) and len(sp) in (2, 3) \
                and all(isinstance(x, (int, float)) for x in sp):
            mn, mx = sp[0], sp[1]
            step = sp[2] if len(sp) == 3 else round((mx - mn) / 100, 6)
            is_int = all(isinstance(x, int) for x in sp)
            d.update(type="range", min=mn, max=mx, step=step, value=mn,
                     _py="int" if is_int else "float")
        elif isinstance(sp, list):
            opts = [str(x) for x in sp]
            first = sp[0] if sp else ""
            py = "int" if isinstance(first, int) else (
                "float" if isinstance(first, float) else "str")
            d.update(type="select", options=opts, value=str(first), _py=py)
        elif isinstance(sp, int):
            d.update(type="number", value=sp, step=1, _py="int")
        elif isinstance(sp, float):
            d.update(type="number", value=sp, step="any", _py="float")
        else:
            d.update(type="text", value=str(sp), _py="str")
        out.append(d)
    return out


def _coerce(spec: dict, value):
    py = spec.get("_py", "str")
    try:
        if py == "int":
            return int(round(float(value)))
        if py == "float":
            return float(value)
        if py == "bool":
            return bool(value)
    except (TypeError, ValueError):
        return spec.get("value")
    return value


def _fig_to_b64(fig, dpi: int = 108) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# The widget
# --------------------------------------------------------------------------

class SemanticNotebook(anywidget.AnyWidget):
    _esm = _ESM
    _css = _WIDGET_CSS

    data = traitlets.Dict().tag(sync=True)
    view_state = traitlets.Dict().tag(sync=True)
    height = traitlets.Int(760).tag(sync=True)

    def __init__(self, document=None, *, nb=None, title=None, height=760, **kw):
        super().__init__(**kw)
        self.height = height
        self._recompute: dict[str, tuple] = {}
        if document is None:
            if nb is None:
                raise ValueError(
                    "provide a parsed `document=`, an `nb=` notebook, "
                    "or use SemanticNotebook.from_ipynb(path)")
            nbd = nb if isinstance(nb, dict) else json.loads(json.dumps(nb))
            document = parse_notebook(nbd, title=title)
        self._doc = document
        self.on_msg(self._on_msg)
        self._build_model()

    # ---- constructors ----------------------------------------------------
    @classmethod
    def from_ipynb(cls, path, *, title=None, height=760) -> "SemanticNotebook":
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(document=parse_notebook(nb, title=title), height=height)

    @classmethod
    def from_notebook(cls, nb, *, title=None, height=760) -> "SemanticNotebook":
        return cls(nb=nb, title=title, height=height)

    # ---- model build -----------------------------------------------------
    def _build_model(self) -> None:
        rc = {}
        for iid, (_fn, specs) in self._recompute.items():
            rc[iid] = {"params": [
                {k: v for k, v in s.items() if not k.startswith("_")}
                for s in specs]}
        self.data = {
            "title": self._doc.title,
            "meta": doc_meta(self._doc),
            "nav_html": render_nav(self._doc),
            "graph_panel": render_graph_panel(self._doc),
            "stage_html": render_sections(self._doc),
            "recompute": rc,
        }

    # ---- live recompute --------------------------------------------------
    def recompute(self, item_id: str, **param_specs):
        """Decorator: attach a recompute function to a figure `id`.

        The wrapped function receives the named parameters and should return a
        matplotlib Figure (or draw the current figure)."""
        specs = _normalize_params(param_specs)

        def deco(fn):
            self._recompute[item_id] = (fn, specs)
            self._build_model()        # re-sync so the controls appear
            return fn
        return deco

    def _on_msg(self, _widget, content, _buffers):
        if not isinstance(content, dict) or content.get("type") != "recompute":
            return
        item_id = content.get("id")
        entry = self._recompute.get(item_id)
        if entry is None:
            return
        fn, specs = entry
        raw = content.get("params", {}) or {}
        kwargs = {sp["name"]: _coerce(sp, raw.get(sp["name"], sp.get("value")))
                  for sp in specs}
        output_html = self._run_recompute(fn, kwargs)
        self.send({"type": "recomputed", "id": item_id, "output_html": output_html})

    def _run_recompute(self, fn, kwargs) -> str:
        import matplotlib
        prev = matplotlib.get_backend()
        try:
            matplotlib.use("Agg", force=True)         # headless capture
            import matplotlib.pyplot as plt
            result = fn(**kwargs)
            fig = result if hasattr(result, "savefig") else plt.gcf()
            return (f'<div class="figframe"><img alt="recomputed figure" '
                    f'src="data:image/png;base64,{_fig_to_b64(fig)}"></div>')
        except Exception:
            import traceback
            return f'<pre class="error">{html.escape(traceback.format_exc())}</pre>'
        finally:
            try:
                matplotlib.use(prev, force=True)
            except Exception:
                pass

    # ---- export back to a static page ------------------------------------
    def export_html(self, path) -> str:
        """Write a self-contained static HTML page honouring the current
        view-state (hidden cards are dropped)."""
        doc = copy.deepcopy(self._doc)
        hidden = set((self.view_state or {}).get("hidden", []))
        for s in doc.sections:
            s.items = [it for it in s.items if it.item_id not in hidden]
        doc.sections = [s for s in doc.sections if s.items]
        Path(path).write_text(render_html(doc), encoding="utf-8")
        return str(path)
