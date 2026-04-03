"""
report.py — reads all result JSONs from results/ and generates report.html
Usage: python report.py
"""

import json
from pathlib import Path

TASK_LABELS = {
    "summarization": "Summarization",
    "reasoning": "Math Reasoning",
    "code": "Code Gen",
    "classification": "Classification",
    "creative": "Creative Writing",
    "factual": "Factual Q&A",
    "instruction": "Instruction Following",
    "long_context": "Long Context Q&A",
}

FAMILY_COLORS = {
    "Mistral":   "#185FA5",
    "Llama":     "#3B6D11",
    "Phi":       "#533fb7",
    "Gemma":     "#BA7517",
    "Qwen":      "#993556",
    "DeepSeek":  "#993C1D",
    "TinyLlama": "#5F5E5A",
}

def load_results():
    files = [f for f in Path("results").glob("*.json") if f.name != "latest.json"]
    results = []
    for f in files:
        with open(f) as fp:
            results.append(json.load(fp))
    results.sort(key=lambda x: (x.get("size_gb", 0)))
    return results

def generate_report(results):
    if not results:
        print("No results found. Run benchmark.py first.")
        return

    task_keys = list(TASK_LABELS.keys())

    models_js = json.dumps([{
        "key": r["model_key"],
        "name": r["model_name"],
        "family": r["family"],
        "size_gb": r["size_gb"],
        "avg_score": r["avg_score"],
        "avg_tps": r["avg_tps"],
        "load_seconds": r["load_seconds"],
        "download_seconds": r.get("download_seconds"),
        "tasks": {k: r["tasks"].get(k, {}) for k in task_keys if k in r.get("tasks", {})},
    } for r in results], indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Model Benchmark Results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8f8f6; color: #1a1a1a; font-size: 14px; line-height: 1.6; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }}
  h1 {{ font-size: 22px; font-weight: 500; margin-bottom: 4px; }}
  .subtitle {{ color: #666; font-size: 13px; margin-bottom: 2rem; }}
  .grid {{ display: grid; gap: 1.25rem; }}
  .grid-2 {{ grid-template-columns: 1fr 1fr; }}
  .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
  .card {{ background: #fff; border: 0.5px solid #e0e0d8; border-radius: 12px; padding: 1.25rem; }}
  .card-title {{ font-size: 12px; font-weight: 500; color: #888; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  .metric {{ font-size: 28px; font-weight: 500; }}
  .metric-label {{ font-size: 12px; color: #888; margin-top: 2px; }}
  .filters {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1.5rem; }}
  .filter-btn {{ font-size: 12px; padding: 4px 14px; border-radius: 20px; border: 0.5px solid #ccc; background: transparent; color: #666; cursor: pointer; transition: all 0.15s; }}
  .filter-btn.active {{ background: #1a1a1a; color: #fff; border-color: #1a1a1a; }}
  .model-select {{ font-size: 13px; padding: 6px 10px; border-radius: 8px; border: 0.5px solid #ccc; background: #fff; color: #1a1a1a; width: 100%; margin-bottom: 1rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ text-align: left; padding: 8px 10px; border-bottom: 0.5px solid #e0e0d8; color: #888; font-weight: 500; white-space: nowrap; }}
  td {{ padding: 7px 10px; border-bottom: 0.5px solid #f0f0ea; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #fafaf8; }}
  .score-bar {{ display: flex; align-items: center; gap: 8px; }}
  .score-fill {{ height: 6px; border-radius: 3px; background: #185FA5; }}
  .score-val {{ font-size: 11px; color: #666; min-width: 28px; }}
  .family-badge {{ font-size: 10px; padding: 2px 8px; border-radius: 10px; background: #f0f0ea; color: #555; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; font-size: 12px; color: #666; }}
  .legend-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 4px; }}
  @media (max-width: 700px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
  <h1>AI Model Benchmark Results</h1>
  <p class="subtitle" id="subtitle">Loading...</p>

  <div class="grid grid-3" style="margin-bottom:1.5rem;" id="summary-cards"></div>

  <div class="filters" id="filters">
    <button class="filter-btn active" data-family="all">All</button>
  </div>

  <div class="grid grid-2" style="margin-bottom:1.25rem;">
    <div class="card">
      <p class="card-title">quality score vs speed</p>
      <div class="legend" id="scatter-legend"></div>
      <div style="position:relative;height:260px;"><canvas id="scatterChart"></canvas></div>
    </div>
    <div class="card">
      <p class="card-title">tokens per second</p>
      <div id="bar-wrap" style="position:relative;"><canvas id="barChart"></canvas></div>
    </div>
  </div>

  <div class="grid grid-2" style="margin-bottom:1.25rem;">
    <div class="card">
      <p class="card-title">task scores by model</p>
      <select class="model-select" id="radarSelect"></select>
      <div style="position:relative;height:240px;"><canvas id="radarChart"></canvas></div>
    </div>
    <div class="card">
      <p class="card-title">score vs size (GB)</p>
      <div style="position:relative;height:280px;"><canvas id="sizeChart"></canvas></div>
    </div>
  </div>

  <div class="card">
    <p class="card-title">full results table</p>
    <div style="overflow-x:auto;">
      <table id="resultsTable">
        <thead></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const MODELS = {models_js};
const TASK_LABELS = {json.dumps(TASK_LABELS)};
const FAMILY_COLORS = {json.dumps(FAMILY_COLORS)};
const TASK_KEYS = {json.dumps(task_keys)};

let activeFamily = "all";
let scatterChart, barChart, radarChart, sizeChart;

function getColor(family) {{
  return FAMILY_COLORS[family] || "#888";
}}

function filtered() {{
  return activeFamily === "all" ? MODELS : MODELS.filter(m => m.family === activeFamily);
}}

function buildSummaryCards() {{
  const total = MODELS.length;
  const avgScore = (MODELS.reduce((s,m) => s + m.avg_score, 0) / total).toFixed(1);
  const fastest = MODELS.reduce((a,b) => a.avg_tps > b.avg_tps ? a : b);
  const smartest = MODELS.reduce((a,b) => a.avg_score > b.avg_score ? a : b);
  document.getElementById("summary-cards").innerHTML = `
    <div class="card"><p class="metric-label">models tested</p><p class="metric">${{total}}</p></div>
    <div class="card"><p class="metric-label">fastest model</p><p class="metric" style="font-size:18px;">${{fastest.name}}</p><p class="metric-label">${{fastest.avg_tps}} tok/s avg</p></div>
    <div class="card"><p class="metric-label">highest quality</p><p class="metric" style="font-size:18px;">${{smartest.name}}</p><p class="metric-label">${{smartest.avg_score}}/10 avg score</p></div>
  `;
  const ts = new Date(MODELS[0]?.tasks ? "now" : "");
  document.getElementById("subtitle").textContent = `${{total}} models · ${{MODELS.reduce((s,m)=>s+Object.keys(m.tasks).length,0)}} total task runs`;
}}

function buildFilters() {{
  const families = [...new Set(MODELS.map(m => m.family))].sort();
  const el = document.getElementById("filters");
  families.forEach(f => {{
    const btn = document.createElement("button");
    btn.className = "filter-btn";
    btn.dataset.family = f;
    btn.textContent = f;
    btn.style.borderColor = getColor(f);
    el.appendChild(btn);
  }});
  el.addEventListener("click", e => {{
    const btn = e.target.closest(".filter-btn");
    if (!btn) return;
    activeFamily = btn.dataset.family;
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    updateScatter();
    updateBar();
    updateTable();
  }});
}}

function buildScatterLegend() {{
  const families = [...new Set(filtered().map(m => m.family))];
  document.getElementById("scatter-legend").innerHTML = families.map(f =>
    `<span><span class="legend-dot" style="background:${{getColor(f)}}"></span>${{f}}</span>`
  ).join("");
}}

function initScatter() {{
  const ctx = document.getElementById("scatterChart").getContext("2d");
  scatterChart = new Chart(ctx, {{
    type: "bubble",
    data: {{ datasets: [] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      layout: {{ padding: 16 }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: i => `${{i.raw._m.name}} — ${{i.raw.x}} tok/s, score ${{i.raw.y}}, ${{i.raw._m.size_gb}}GB` }} }}
      }},
      scales: {{
        x: {{ title: {{ display:true, text:"tok/s (speed)", font:{{size:11}}, color:"#888" }}, min:0, ticks:{{color:"#888",font:{{size:10}}}}, grid:{{color:"rgba(0,0,0,0.06)"}} }},
        y: {{ title: {{ display:true, text:"avg quality score", font:{{size:11}}, color:"#888" }}, min:0, max:11, ticks:{{color:"#888",font:{{size:10}}}}, grid:{{color:"rgba(0,0,0,0.06)"}} }}
      }}
    }}
  }});
  updateScatter();
}}

function updateScatter() {{
  const models = filtered();
  const families = [...new Set(models.map(m => m.family))];
  scatterChart.data.datasets = families.map(f => ({{
    label: f,
    data: models.filter(m => m.family === f).map(m => ({{
      x: m.avg_tps, y: m.avg_score, r: Math.max(6, m.size_gb * 3.5), _m: m
    }})),
    backgroundColor: getColor(f) + "99",
    borderColor: getColor(f),
    borderWidth: 1.5,
  }}));
  scatterChart.update();
  buildScatterLegend();
}}

function initBar() {{
  const ctx = document.getElementById("barChart").getContext("2d");
  barChart = new Chart(ctx, {{
    type: "bar",
    data: {{ labels: [], datasets: [{{ data: [], backgroundColor: [], borderColor: [], borderWidth: 1, borderRadius: 3 }}] }},
    options: {{
      indexAxis: "y",
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: i => ` ${{i.raw}} tok/s` }} }} }},
      scales: {{
        x: {{ ticks: {{ color:"#888", font:{{size:10}}, callback: v => v+" t/s" }}, grid:{{color:"rgba(0,0,0,0.06)"}} }},
        y: {{ ticks: {{ color:"#888", font:{{size:10}} }}, grid:{{display:false}} }}
      }}
    }}
  }});
  updateBar();
}}

function updateBar() {{
  const models = [...filtered()].sort((a,b) => b.avg_tps - a.avg_tps);
  const h = Math.max(280, models.length * 32 + 60);
  document.getElementById("bar-wrap").style.height = h + "px";
  barChart.data.labels = models.map(m => m.name);
  barChart.data.datasets[0].data = models.map(m => m.avg_tps);
  barChart.data.datasets[0].backgroundColor = models.map(m => getColor(m.family) + "aa");
  barChart.data.datasets[0].borderColor = models.map(m => getColor(m.family));
  barChart.update();
}}

function initRadar() {{
  const sel = document.getElementById("radarSelect");
  MODELS.forEach(m => {{
    const opt = document.createElement("option");
    opt.value = m.key; opt.textContent = m.name;
    sel.appendChild(opt);
  }});
  sel.addEventListener("change", updateRadar);
  const ctx = document.getElementById("radarChart").getContext("2d");
  radarChart = new Chart(ctx, {{
    type: "radar",
    data: {{ labels: TASK_KEYS.map(k => TASK_LABELS[k] || k), datasets: [] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{ r: {{
        min: 0, max: 10,
        ticks: {{ display: false, stepSize: 2 }},
        pointLabels: {{ color:"#888", font:{{size:10}} }},
        grid: {{ color:"rgba(0,0,0,0.08)" }},
        angleLines: {{ color:"rgba(0,0,0,0.08)" }}
      }} }}
    }}
  }});
  updateRadar();
}}

function updateRadar() {{
  const key = document.getElementById("radarSelect").value;
  const m = MODELS.find(x => x.key === key);
  if (!m) return;
  const color = getColor(m.family);
  radarChart.data.datasets = [{{
    label: m.name,
    data: TASK_KEYS.map(k => m.tasks[k]?.score ?? 0),
    backgroundColor: color + "22",
    borderColor: color,
    borderWidth: 2,
    pointBackgroundColor: color,
    pointRadius: 3,
  }}];
  radarChart.update();
}}

function initSize() {{
  const ctx = document.getElementById("sizeChart").getContext("2d");
  const sorted = [...MODELS].sort((a,b) => a.size_gb - b.size_gb);
  sizeChart = new Chart(ctx, {{
    type: "scatter",
    data: {{
      datasets: [{{
        data: sorted.map(m => ({{ x: m.size_gb, y: m.avg_score, _m: m }})),
        backgroundColor: sorted.map(m => getColor(m.family) + "bb"),
        borderColor: sorted.map(m => getColor(m.family)),
        borderWidth: 1.5,
        pointRadius: 7,
        pointHoverRadius: 9,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      layout: {{ padding: 16 }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: i => `${{i.raw._m.name}} — ${{i.raw.x}}GB, score ${{i.raw.y}}` }} }}
      }},
      scales: {{
        x: {{ title:{{display:true, text:"model size (GB)", font:{{size:11}}, color:"#888"}}, min:0, ticks:{{color:"#888",font:{{size:10}}}}, grid:{{color:"rgba(0,0,0,0.06)"}} }},
        y: {{ title:{{display:true, text:"avg quality score", font:{{size:11}}, color:"#888"}}, min:0, max:11, ticks:{{color:"#888",font:{{size:10}}}}, grid:{{color:"rgba(0,0,0,0.06)"}} }}
      }}
    }}
  }});
}}

function buildTable() {{
  const thead = document.querySelector("#resultsTable thead");
  const tbody = document.querySelector("#resultsTable tbody");
  const taskCols = TASK_KEYS.filter(k => MODELS.some(m => m.tasks[k]));
  thead.innerHTML = `<tr>
    <th>Model</th><th>Family</th><th>GB</th><th>Tok/s</th><th>Load</th><th>Avg Score</th>
    ${{taskCols.map(k=>`<th>${{TASK_LABELS[k]}}</th>`).join("")}}
  </tr>`;
  updateTable();
  document.getElementById("radarSelect").value = MODELS[0]?.key;
  updateRadar();
}}

function updateTable() {{
  const tbody = document.querySelector("#resultsTable tbody");
  const models = filtered().sort((a,b) => b.avg_score - a.avg_score);
  const taskCols = TASK_KEYS.filter(k => MODELS.some(m => m.tasks[k]));
  tbody.innerHTML = models.map(m => `
    <tr>
      <td style="font-weight:500">${{m.name}}</td>
      <td><span class="family-badge" style="background:${{getColor(m.family)}}22;color:${{getColor(m.family)}}">${{m.family}}</span></td>
      <td>${{m.size_gb}}</td>
      <td>${{m.avg_tps}}</td>
      <td>${{m.load_seconds}}s</td>
      <td>
        <div class="score-bar">
          <div class="score-fill" style="width:${{m.avg_score * 10}}%;background:${{getColor(m.family)}}"></div>
          <span class="score-val">${{m.avg_score}}</span>
        </div>
      </td>
      ${{taskCols.map(k => `<td>${{m.tasks[k]?.score ?? "—"}}</td>`).join("")}}
    </tr>
  `).join("");
}}

buildSummaryCards();
buildFilters();
initScatter();
initBar();
initRadar();
initSize();
buildTable();
</script>
</body>
</html>"""

    out_path = "results/report.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report saved to {out_path}")
    print(f"Open it in a browser: open {out_path}")

if __name__ == "__main__":
    results = load_results()
    print(f"Found {len(results)} result files")
    generate_report(results)