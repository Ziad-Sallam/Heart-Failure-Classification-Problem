# ---------------------------------------------------------------------------
# HTML Report Generator
# ---------------------------------------------------------------------------

def save_html_report(
    best_params:    dict,
    all_results:    list[dict],
    split_results:  dict,
    tree_depth:     int,
    leaf_count:     int,
    scoring:        str,
    output_path:    str,
) -> None:
    """Render a self-contained HTML report of all tuning iterations and final metrics."""

    total     = len(all_results)
    best_iter = min(all_results, key=lambda r: r["score"] != max(x["score"] for x in all_results))
    best_score = all_results[0]["score"]   # sorted best-first

    # ---- build tuning table rows (restore original iteration order) --------
    sorted_by_iter = sorted(all_results, key=lambda r: r["iteration"])
    rows_html = ""
    for r in sorted_by_iter:
        p      = r["params"]
        s      = r["score"]
        m      = r.get("metrics", {})
        is_best = abs(s - best_score) < 1e-9
        row_cls = "best-row" if is_best else ""
        badge   = '<span class="best-badge">★ BEST</span>' if is_best else ""

        acc     = m.get("accuracy",  "—")
        f1_mac  = m.get("f1_macro",  "—")
        roc     = m.get("roc_auc",   "—")

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        rows_html += f"""
        <tr class="{row_cls}">
          <td class="iter-num">#{r['iteration']}</td>
          <td>{p.get('max_depth', '—')}</td>
          <td>{p.get('min_samples_split', '—')}</td>
          <td>{p.get('min_samples_leaf', '—')}</td>
          <td class="score-cell">{fmt(s)}</td>
          <td>{fmt(acc)}</td>
          <td>{fmt(f1_mac)}</td>
          <td>{fmt(roc)}</td>
          <td>{badge}</td>
        </tr>"""

    # ---- build final split cards ------------------------------------------
    split_cards_html = ""
    for split_name, m in split_results.items():
        cm    = m["confusion_matrix"]
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        roc   = f"{m['roc_auc']:.4f}" if "roc_auc" in m else "N/A"
        split_cards_html += f"""
        <div class="split-card">
          <h3 class="split-title">{split_name}</h3>
          <div class="metric-grid">
            <div class="metric-box">
              <span class="metric-label">Accuracy</span>
              <span class="metric-val">{m['accuracy']:.4f}</span>
            </div>
            <div class="metric-box">
              <span class="metric-label">F1 Binary</span>
              <span class="metric-val">{m['f1_binary']:.4f}</span>
            </div>
            <div class="metric-box">
              <span class="metric-label">F1 Macro</span>
              <span class="metric-val">{m['f1_macro']:.4f}</span>
            </div>
            <div class="metric-box">
              <span class="metric-label">ROC-AUC</span>
              <span class="metric-val">{roc}</span>
            </div>
          </div>
          <div class="cm-wrap">
            <p class="cm-label">Confusion Matrix</p>
            <table class="cm-table">
              <thead><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr></thead>
              <tbody>
                <tr><th>Act 0</th><td class="cm-tn">{tn}</td><td class="cm-fp">{fp}</td></tr>
                <tr><th>Act 1</th><td class="cm-fn">{fn}</td><td class="cm-tp">{tp}</td></tr>
              </tbody>
            </table>
          </div>
        </div>"""

    # ---- sparkline data (scores in iteration order) -----------------------
    spark_scores = [f"{r['score']:.4f}" for r in sorted_by_iter]
    spark_js     = ", ".join(spark_scores)

    best_params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Decision Tree — Hyperparameter Tuning Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,300;9..144,700&display=swap');

  :root {{
    --bg:        #0d0f14;
    --surface:   #151820;
    --border:    #252b38;
    --accent:    #e8c96d;
    --accent2:   #6de8b4;
    --danger:    #e87a6d;
    --text:      #d4dae8;
    --muted:     #606880;
    --best:      rgba(232,201,109,0.07);
    --best-bd:   rgba(232,201,109,0.4);
    --radius:    10px;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    padding: 0 0 80px;
  }}

  /* ── Header ── */
  header {{
    background: linear-gradient(135deg, #0d0f14 0%, #181d2a 100%);
    border-bottom: 1px solid var(--border);
    padding: 48px 60px 36px;
    position: relative;
    overflow: hidden;
  }}
  header::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(232,201,109,0.05) 0%, transparent 60%);
    pointer-events: none;
  }}
  header h1 {{
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.2rem;
    color: #fff;
    letter-spacing: -0.02em;
    line-height: 1.15;
  }}
  header h1 span {{ color: var(--accent); }}
  header p.subtitle {{
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }}

  /* ── Summary strip ── */
  .summary-strip {{
    display: flex;
    gap: 1px;
    background: var(--border);
    border-bottom: 1px solid var(--border);
  }}
  .strip-item {{
    flex: 1;
    background: var(--surface);
    padding: 20px 28px;
  }}
  .strip-item .s-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 6px;
  }}
  .strip-item .s-val {{
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
  }}
  .strip-item .s-sub {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
  }}

  /* ── Main layout ── */
  main {{ padding: 48px 60px; }}

  section {{ margin-bottom: 52px; }}
  section h2 {{
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  section h2 .tag {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 400;
    background: var(--border);
    color: var(--muted);
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}

  /* ── Sparkline canvas ── */
  .spark-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-bottom: 28px;
  }}
  .spark-wrap p {{
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}
  canvas#sparkline {{ width: 100%; height: 80px; display: block; }}

  /* ── Tuning table ── */
  .table-scroll {{ overflow-x: auto; border-radius: var(--radius); border: 1px solid var(--border); }}
  table.tune-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  .tune-table thead tr {{
    background: #1a1f2e;
    border-bottom: 2px solid var(--border);
  }}
  .tune-table thead th {{
    text-align: left;
    padding: 12px 16px;
    color: var(--muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 10px;
    white-space: nowrap;
  }}
  .tune-table tbody tr {{
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }}
  .tune-table tbody tr:hover {{ background: rgba(255,255,255,0.025); }}
  .tune-table tbody tr.best-row {{
    background: var(--best);
    border-left: 2px solid var(--accent);
  }}
  .tune-table td {{
    padding: 10px 16px;
    color: var(--text);
  }}
  .tune-table td.iter-num {{ color: var(--muted); font-size: 11px; }}
  .tune-table td.score-cell {{ color: var(--accent2); font-weight: 500; }}
  .best-badge {{
    display: inline-block;
    background: var(--accent);
    color: #0d0f14;
    font-size: 10px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.06em;
  }}

  /* ── Filter bar ── */
  .filter-bar {{
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    align-items: center;
  }}
  .filter-bar label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }}
  .filter-bar select, .filter-bar input {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    padding: 6px 10px;
    border-radius: 6px;
    outline: none;
  }}
  .filter-bar select:focus, .filter-bar input:focus {{ border-color: var(--accent); }}
  #row-count {{ font-size: 11px; color: var(--muted); margin-left: auto; }}

  /* ── Best params banner ── */
  .best-banner {{
    background: var(--best);
    border: 1px solid var(--best-bd);
    border-radius: var(--radius);
    padding: 18px 24px;
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }}
  .best-banner .bb-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent);
  }}
  .best-banner .bb-val {{
    font-size: 13px;
    color: #fff;
  }}
  .best-banner .bb-score {{
    margin-left: auto;
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
  }}

  /* ── Split cards ── */
  .split-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
  }}
  .split-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
  }}
  .split-title {{
    font-family: 'Fraunces', serif;
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 16px;
  }}
  .metric-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 20px;
  }}
  .metric-box {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .metric-label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }}
  .metric-val {{ font-family: 'Fraunces', serif; font-size: 1.25rem; font-weight: 700; color: var(--accent2); }}

  /* Confusion matrix */
  .cm-wrap {{ margin-top: 4px; }}
  .cm-label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }}
  .cm-table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
  .cm-table th, .cm-table td {{ padding: 8px 12px; text-align: center; border: 1px solid var(--border); }}
  .cm-table thead th {{ background: var(--bg); color: var(--muted); font-size: 10px; font-weight: 400; }}
  .cm-table tbody th {{ background: var(--bg); color: var(--muted); font-weight: 400; text-align: left; padding-left: 10px; }}
  .cm-tn, .cm-tp {{ color: var(--accent2); font-weight: 500; }}
  .cm-fp, .cm-fn {{ color: var(--danger); font-weight: 500; }}

  /* ── Tree info ── */
  .tree-info {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }}
  .tree-pill {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 22px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 140px;
  }}
  .tree-pill .tp-label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }}
  .tree-pill .tp-val {{ font-family: 'Fraunces', serif; font-size: 1.4rem; font-weight: 700; color: #fff; }}

  footer {{
    text-align: center;
    font-size: 11px;
    color: var(--muted);
    padding-top: 32px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
  }}
</style>
</head>
<body>

<header>
  <h1>Decision Tree <span>Tuning Report</span></h1>
  <p class="subtitle">Heart Failure Prediction &nbsp;·&nbsp; Grid Search &nbsp;·&nbsp; Scoring: {scoring}</p>
</header>

<div class="summary-strip">
  <div class="strip-item">
    <div class="s-label">Total Iterations</div>
    <div class="s-val">{total}</div>
    <div class="s-sub">exhaustive grid</div>
  </div>
  <div class="strip-item">
    <div class="s-label">Best {scoring}</div>
    <div class="s-val">{best_score:.4f}</div>
    <div class="s-sub">on validation set</div>
  </div>
  <div class="strip-item">
    <div class="s-label">Tree Depth</div>
    <div class="s-val">{tree_depth}</div>
    <div class="s-sub">final model</div>
  </div>
  <div class="strip-item">
    <div class="s-label">Leaf Count</div>
    <div class="s-val">{leaf_count}</div>
    <div class="s-sub">final model</div>
  </div>
</div>

<main>

  <!-- ── Tuning iterations ── -->
  <section>
    <h2>Hyperparameter Tuning Iterations <span class="tag">{total} combos</span></h2>

    <div class="spark-wrap">
      <p>F1 score across iterations (chronological order)</p>
      <canvas id="sparkline"></canvas>
    </div>

    <div class="best-banner">
      <div>
        <div class="bb-label">Best Configuration</div>
        <div class="bb-val">{best_params_str}</div>
      </div>
      <div class="bb-score">{best_score:.4f}</div>
    </div>

    <div class="filter-bar">
      <label>max_depth</label>
      <select id="filter-depth">
        <option value="">All</option>
        <option>3</option><option>5</option><option>7</option><option>10</option><option>None</option>
      </select>
      <label>min_samples_split</label>
      <select id="filter-split">
        <option value="">All</option>
        <option>2</option><option>5</option><option>10</option>
      </select>
      <label>min score ≥</label>
      <input id="filter-score" type="number" step="0.01" min="0" max="1" placeholder="0.00" style="width:80px"/>
      <span id="row-count"></span>
    </div>

    <div class="table-scroll">
      <table class="tune-table" id="tune-table">
        <thead>
          <tr>
            <th>#</th>
            <th>max_depth</th>
            <th>min_samples_split</th>
            <th>min_samples_leaf</th>
            <th>{scoring}</th>
            <th>accuracy</th>
            <th>f1_macro</th>
            <th>roc_auc</th>
            <th></th>
          </tr>
        </thead>
        <tbody id="tune-tbody">
{rows_html}
        </tbody>
      </table>
    </div>
  </section>

  <!-- ── Final evaluation ── -->
  <section>
    <h2>Final Model Evaluation <span class="tag">train + val → test</span></h2>
    <div class="split-grid">
{split_cards_html}
    </div>
  </section>

  <!-- ── Tree structure ── -->
  <section>
    <h2>Tree Structure</h2>
    <div class="tree-info">
      <div class="tree-pill">
        <span class="tp-label">Max Depth</span>
        <span class="tp-val">{tree_depth}</span>
      </div>
      <div class="tree-pill">
        <span class="tp-label">Leaf Nodes</span>
        <span class="tp-val">{leaf_count}</span>
      </div>
      <div class="tree-pill">
        <span class="tp-label">Best max_depth param</span>
        <span class="tp-val">{best_params.get('max_depth', '—')}</span>
      </div>
      <div class="tree-pill">
        <span class="tp-label">min_samples_split</span>
        <span class="tp-val">{best_params.get('min_samples_split', '—')}</span>
      </div>
      <div class="tree-pill">
        <span class="tp-label">min_samples_leaf</span>
        <span class="tp-val">{best_params.get('min_samples_leaf', '—')}</span>
      </div>
    </div>
  </section>

  <footer>Generated by train_decision_tree.py &nbsp;·&nbsp; Heart Failure Prediction</footer>

</main>

<script>
// ── Sparkline ──────────────────────────────────────────────────────────────
(function() {{
  const scores = [{spark_js}];
  const canvas = document.getElementById('sparkline');
  const dpr    = window.devicePixelRatio || 1;
  canvas.width  = canvas.offsetWidth  * dpr;
  canvas.height = canvas.offsetHeight * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  const pad = {{ t: 8, r: 12, b: 24, l: 40 }};
  const min = Math.min(...scores) - 0.01;
  const max = Math.max(...scores) + 0.01;
  const xScale = i => pad.l + (i / (scores.length - 1)) * (W - pad.l - pad.r);
  const yScale = v => pad.t + (1 - (v - min) / (max - min)) * (H - pad.t - pad.b);

  // grid lines
  ctx.strokeStyle = '#252b38';
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75, 1].forEach(t => {{
    const y = pad.t + (1 - t) * (H - pad.t - pad.b);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
  }});

  // fill
  ctx.beginPath();
  scores.forEach((s, i) => i === 0 ? ctx.moveTo(xScale(i), yScale(s)) : ctx.lineTo(xScale(i), yScale(s)));
  ctx.lineTo(xScale(scores.length - 1), H - pad.b);
  ctx.lineTo(xScale(0), H - pad.b);
  ctx.closePath();
  ctx.fillStyle = 'rgba(109,232,180,0.08)';
  ctx.fill();

  // line
  ctx.beginPath();
  scores.forEach((s, i) => i === 0 ? ctx.moveTo(xScale(i), yScale(s)) : ctx.lineTo(xScale(i), yScale(s)));
  ctx.strokeStyle = '#6de8b4';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // best dot
  const bestScore = Math.max(...scores);
  const bestIdx   = scores.indexOf(bestScore);
  ctx.beginPath();
  ctx.arc(xScale(bestIdx), yScale(bestScore), 4, 0, Math.PI * 2);
  ctx.fillStyle = '#e8c96d';
  ctx.fill();

  // y-axis labels
  ctx.fillStyle = '#606880';
  ctx.font = '10px DM Mono, monospace';
  ctx.textAlign = 'right';
  [min + 0.01, (min + max) / 2, max - 0.01].forEach(v => {{
    ctx.fillText(v.toFixed(3), pad.l - 6, yScale(v) + 3);
  }});
}})();

// ── Filter logic ──────────────────────────────────────────────────────────
const tbody    = document.getElementById('tune-tbody');
const allRows  = Array.from(tbody.querySelectorAll('tr'));
const countEl  = document.getElementById('row-count');

function applyFilters() {{
  const depth  = document.getElementById('filter-depth').value;
  const split  = document.getElementById('filter-split').value;
  const minSc  = parseFloat(document.getElementById('filter-score').value) || 0;
  let visible  = 0;
  allRows.forEach(row => {{
    const cells = row.querySelectorAll('td');
    const d     = cells[1].textContent.trim();
    const s     = cells[2].textContent.trim();
    const score = parseFloat(cells[4].textContent.trim());
    const show  = (depth === '' || d === depth)
               && (split === '' || s === split)
               && (score >= minSc);
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  countEl.textContent = `${{visible}} / ${{allRows.length}} rows`;
}}

['filter-depth','filter-split','filter-score'].forEach(id =>
  document.getElementById(id).addEventListener('input', applyFilters)
);
applyFilters();
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

