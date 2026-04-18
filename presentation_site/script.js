// Allosteric Site Prediction — Live Demo
// Loads manifest + per-PDB JSON, fetches structure from RCSB, renders with 3Dmol.js.

const COLORS = {
  tp: "#22c55e",
  fn: "#3b82f6",
  fp: "#ef4444",
  act: "#f59e0b",
  base: "#3b4656",
  cartoon: "#2f3a4c",
};

const METRICS = {
  "1UU7": { auroc: 0.990, auprc: 0.910 },
  "4KSQ": { auroc: 0.958, auprc: 0.808 },
  "3ME3": { auroc: 0.953, auprc: 0.748 },
  "1HQ6": { auroc: 0.255, auprc: 0.001 },
  "3W8L": { auroc: 0.342, auprc: 0.035 },
};

let viewer = null;
let currentData = null;
let currentMode = "both";
let threshold = 0.4;
let pdbCache = {};

function el(id) { return document.getElementById(id); }

async function loadManifest() {
  const r = await fetch("data/manifest.json");
  return await r.json();
}

async function loadProteinData(pdb) {
  const r = await fetch(`data/${pdb}.json`);
  return await r.json();
}

async function fetchPDB(pdb) {
  if (pdbCache[pdb]) return pdbCache[pdb];
  const r = await fetch(`https://files.rcsb.org/view/${pdb}.pdb`);
  if (!r.ok) throw new Error(`Failed to fetch PDB ${pdb}`);
  const text = await r.text();
  pdbCache[pdb] = text;
  return text;
}

function classifyResidue(res, mode) {
  // returns "tp" | "fn" | "fp" | "act" | "base" | "hidden"
  const predPos = res.p >= threshold;
  const truePos = res.t === 1;
  const active = res.a === 1;

  if (mode === "truth") {
    if (truePos) return "tp";         // real site (green)
    if (active) return "act";
    return "base";
  }
  if (mode === "pred") {
    if (predPos && truePos) return "tp";
    if (predPos && !truePos) return "fp";
    if (active) return "act";
    return "base";
  }
  if (mode === "prob") {
    return { probColor: probToColor(res.p) };
  }
  // both (default)
  if (predPos && truePos) return "tp";
  if (!predPos && truePos) return "fn";
  if (predPos && !truePos) return "fp";
  if (active) return "act";
  return "base";
}

function probToColor(p) {
  // heat: low = dark blue, mid = yellow, high = red
  const clamped = Math.max(0, Math.min(1, p));
  if (clamped < 0.5) {
    const t = clamped / 0.5;
    const r = Math.round(30 + (234 - 30) * t);
    const g = Math.round(60 + (179 - 60) * t);
    const b = Math.round(120 + (8 - 120) * t);
    return `rgb(${r},${g},${b})`;
  } else {
    const t = (clamped - 0.5) / 0.5;
    const r = Math.round(234 + (239 - 234) * t);
    const g = Math.round(179 + (68 - 179) * t);
    const b = Math.round(8 + (68 - 8) * t);
    return `rgb(${r},${g},${b})`;
  }
}

function renderStructure(data) {
  viewer.removeAllModels();
  viewer.removeAllShapes();
  viewer.removeAllSurfaces();
  viewer.removeAllLabels();

  const pdbText = pdbCache[data.pdb_id];
  viewer.addModel(pdbText, "pdb");

  // base cartoon
  viewer.setStyle({}, { cartoon: { color: COLORS.cartoon, opacity: 0.75 } });

  // group residues by class for efficient styling
  const groups = { tp: [], fn: [], fp: [], act: [], base: [] };
  const probResidues = [];

  for (const res of data.residues) {
    if (currentMode === "prob") {
      probResidues.push(res);
    } else {
      const cls = classifyResidue(res, currentMode);
      if (cls in groups) groups[cls].push(res);
    }
  }

  if (currentMode === "prob") {
    // color cartoon by probability
    const byChain = {};
    for (const r of probResidues) {
      if (!byChain[r.c]) byChain[r.c] = {};
      byChain[r.c][r.n] = probToColor(r.p);
    }
    viewer.setStyle({}, {
      cartoon: {
        colorfunc: (atom) => {
          return (byChain[atom.chain] && byChain[atom.chain][atom.resi]) || COLORS.base;
        },
      },
    });
    // highlight high-prob residues with spheres
    const highP = probResidues.filter((r) => r.p >= threshold);
    if (highP.length) {
      viewer.addStyle(
        { resi: highP.map((r) => r.n), chain: [...new Set(highP.map((r) => r.c))] },
        { stick: { radius: 0.2, colorscheme: { prop: "b", gradient: "rwb" } } },
      );
    }
  } else {
    applyGroup(groups.base, null, null);       // no override (cartoon stays)
    applyGroup(groups.act, COLORS.act, "stick");
    applyGroup(groups.fp, COLORS.fp, "sphereAndStick");
    applyGroup(groups.fn, COLORS.fn, "sphereAndStick");
    applyGroup(groups.tp, COLORS.tp, "sphereAndStick");

    // highlight all true allosteric with cartoon thickening
    const truePos = data.residues.filter((r) => r.t === 1);
    if (truePos.length) {
      const sel = buildSel(truePos);
      viewer.addStyle(sel, { cartoon: { color: currentMode === "truth" ? COLORS.tp : undefined, thickness: 1.4 } });
    }
  }

  viewer.zoomTo();
  viewer.zoom(1.1, 400);
  viewer.render();
}

function applyGroup(resList, color, kind) {
  if (!resList.length || !kind) return;
  const sel = buildSel(resList);
  const style = {};
  if (kind === "stick") {
    style.stick = { color, radius: 0.2 };
  } else if (kind === "sphereAndStick") {
    style.stick = { color, radius: 0.22 };
    style.sphere = { color, radius: 0.55, opacity: 0.95 };
  }
  viewer.addStyle(sel, style);
}

function buildSel(resList) {
  const byChain = {};
  for (const r of resList) {
    if (!byChain[r.c]) byChain[r.c] = [];
    byChain[r.c].push(r.n);
  }
  const or = Object.entries(byChain).map(([c, ns]) => ({ chain: c, resi: ns }));
  return or.length === 1 ? or[0] : { or };
}

function updateStats(data) {
  let tp = 0, fn = 0, fp = 0, pred = 0, truePos = 0;
  for (const r of data.residues) {
    const pp = r.p >= threshold;
    const tt = r.t === 1;
    if (pp) pred++;
    if (tt) truePos++;
    if (pp && tt) tp++;
    else if (!pp && tt) fn++;
    else if (pp && !tt) fp++;
  }
  el("s-res").textContent = data.n_residues.toLocaleString();
  el("s-true").textContent = truePos;
  el("s-pred").textContent = pred;
  el("s-tp").textContent = tp;
  el("s-fn").textContent = fn;
  el("s-fp").textContent = fp;
  el("s-thr").textContent = threshold.toFixed(3);

  // Performance metrics for this protein
  const m = METRICS[data.pdb_id] || {};
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : null;
  const recall = truePos > 0 ? tp / truePos : null;
  el("s-auroc").textContent = m.auroc !== undefined ? m.auroc.toFixed(3) : "—";
  el("s-auprc").textContent = m.auprc !== undefined ? m.auprc.toFixed(3) : "—";
  el("s-prec").textContent = precision !== null ? precision.toFixed(3) : "—";
  el("s-recall").textContent = recall !== null ? recall.toFixed(3) : "—";

  // Color-code AUROC / AUPRC to match the category
  const aurocEl = el("s-auroc");
  const auprcEl = el("s-auprc");
  aurocEl.classList.toggle("good", data.category === "win");
  aurocEl.classList.toggle("bad", data.category === "fail");
  auprcEl.classList.toggle("good", data.category === "win");
  auprcEl.classList.toggle("bad", data.category === "fail");
}

function updateChips(data) {
  el("chip-pdb").textContent = `PDB ${data.pdb_id}`;
  const catChip = el("chip-cat");
  catChip.textContent = data.category === "win" ? "Success case" : "Failure mode";
  catChip.classList.remove("win", "fail");
  catChip.classList.add(data.category);
  el("chip-family").textContent = data.family.toUpperCase();
}

function updateNarrative(data) {
  el("n-title").textContent = data.title;
  el("n-blurb").textContent = data.blurb;
}

async function selectProtein(pdb) {
  showLoading(true);
  document.querySelectorAll(".card").forEach((c) => {
    c.classList.toggle("active", c.dataset.pdb === pdb);
  });
  document.querySelectorAll(".tab").forEach((t) => {
    const isActive = t.dataset.pdb === pdb;
    t.classList.toggle("active", isActive);
    if (isActive && t.scrollIntoView) {
      t.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    }
  });

  try {
    const data = await loadProteinData(pdb);
    await fetchPDB(pdb);
    currentData = data;
    updateChips(data);
    updateNarrative(data);
    updateStats(data);
    renderStructure(data);
  } catch (e) {
    console.error(e);
    alert(`Error loading ${pdb}: ${e.message}`);
  } finally {
    showLoading(false);
  }
}

function showLoading(yes) {
  el("viewer-loading").classList.toggle("hidden", !yes);
}

function renderCards(manifest) {
  const list = el("protein-list");
  list.innerHTML = "";
  for (const p of manifest.proteins) {
    const m = METRICS[p.pdb_id] || {};
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.pdb = p.pdb_id;
    card.innerHTML = `
      <div class="row1">
        <span class="pdb">${p.pdb_id}</span>
        <span class="badge ${p.category}">${p.category === "win" ? "Success" : "Failure"}</span>
      </div>
      <div class="title">${p.title}</div>
      <div class="metrics">
        <span class="${p.category === 'win' ? 'mgood' : 'mbad'}">AUROC <b>${m.auroc !== undefined ? m.auroc.toFixed(3) : '—'}</b></span>
        <span class="${p.category === 'win' ? 'mgood' : 'mbad'}">AUPRC <b>${m.auprc !== undefined ? m.auprc.toFixed(3) : '—'}</b></span>
        <span>n=<b>${p.n_residues}</b></span>
      </div>
    `;
    card.addEventListener("click", () => selectProtein(p.pdb_id));
    list.appendChild(card);
  }
}

function renderMobileTabs(manifest) {
  const host = el("mobile-tabs");
  if (!host) return;
  host.innerHTML = "";
  for (const p of manifest.proteins) {
    const m = METRICS[p.pdb_id] || {};
    const tab = document.createElement("button");
    tab.className = `tab ${p.category}`;
    tab.dataset.pdb = p.pdb_id;
    tab.innerHTML = `
      <span class="tab-pdb">${p.pdb_id}</span>
      <span class="tab-auroc">${m.auroc !== undefined ? m.auroc.toFixed(2) : '—'}</span>
    `;
    tab.addEventListener("click", () => selectProtein(p.pdb_id));
    host.appendChild(tab);
  }
}

function wireModeBar() {
  document.querySelectorAll(".mode-bar button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-bar button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentMode = btn.dataset.mode;
      if (currentData) renderStructure(currentData);
    });
  });
}

function wireCalloutLinks() {
  document.querySelectorAll(".inline-link[data-jump]").forEach((btn) => {
    btn.addEventListener("click", () => {
      selectProtein(btn.dataset.jump);
      document.querySelector(".stage").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

async function main() {
  viewer = $3Dmol.createViewer("viewer", {
    backgroundColor: "rgba(0,0,0,0)",
    antialias: true,
  });
  wireModeBar();
  wireCalloutLinks();

  const manifest = await loadManifest();
  threshold = manifest.threshold || 0.4;
  renderCards(manifest);
  renderMobileTabs(manifest);

  // auto-load first protein so the stage is never empty
  if (manifest.proteins.length) {
    selectProtein(manifest.proteins[0].pdb_id);
  }
}

window.addEventListener("load", main);
