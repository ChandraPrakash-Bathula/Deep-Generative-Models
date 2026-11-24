/* From Pixels to Latent Spaces: shared navigation, data loading, plots and widgets. */
const BASE = document.body.dataset.base || "";
const A = p => `${BASE}assets/${p}`;
const CHAPTERS = [
  {f: "01-foundations.html", t: "Understanding Autoencoders", d: "What an autoencoder is, why we compress, and the reconstruction loss"},
  {f: "02-architecture.html", t: "Inside the Network", d: "Encoder, bottleneck and decoder, layer by layer, and how it trains"},
  {f: "03-latent-space.html", t: "Understanding the Latent Space", d: "What a latent dimension holds, from 2 numbers to 128"},
  {f: "04-visualizing.html", t: "Visualizing Latent Spaces", d: "PCA, t-SNE and UMAP in 2D and 3D, and how not to misread them"},
  {f: "05-experiments.html", t: "Experiments: Brains and Faces", d: "IXI brain MRI and CelebA under the same protocol, failures included"},
  {f: "06-vae.html", t: "Variational Autoencoders", d: "From points to distributions: μ, σ, sampling, the reparameterisation trick and KL"},
  {f: "07-generation.html", t: "Generation and Interpolation", d: "Real vs reconstructed vs generated, temperature, interpolation, FID"},
  {f: "08-ae-vs-vae.html", t: "Autoencoder vs VAE", d: "Side by side on every experiment, and what I take away"},
];
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const MODELS = {ae: "AE", vae01: "VAE β=0.1", vae1: "VAE β=1"};
const MCOL = {ae: "#2563eb", vae01: "#ea580c", vae1: "#059669"};

/* ---------------- sidebar + pager ---------------- */
function buildChrome() {
  const here = location.pathname.split("/").pop() || "index.html";
  const side = document.createElement("aside"); side.className = "side";
  let html = `<a class="home" href="${BASE}index.html"><div class="s">Deep Generative Models · Part 1</div><div class="t">From Pixels to Latent Spaces</div></a>`;
  html += `<div class="grp">Chapters</div>`;
  CHAPTERS.forEach((c, i) => {
    const on = c.f === here;
    html += `<a class="ch${on ? " on" : ""}" href="${BASE}chapters/${c.f}"><span class="n">${String(i + 1).padStart(2, "0")}</span><span>${c.t}</span></a>`;
    if (on) {
      const hs = $$("main h2[id]");
      if (hs.length) html += `<ul class="toc">${hs.map(h => `<li><a href="#${h.id}">${h.dataset.short || h.textContent.replace(/^\d+\.\s*/, "")}</a></li>`).join("")}</ul>`;
    }
  });
  html += `<div class="grp">Tools</div><a class="ch${here === "explorer.html" ? " on" : ""}" href="${BASE}explorer.html"><span class="n">⌘</span><span>Experiment Explorer</span></a>`;
  html += `<div class="grp">Coming next in the series</div>
    <a class="ch soon"><span class="n">03</span><span>Diffusion Models</span></a>
    <a class="ch soon"><span class="n">04</span><span>World Models</span></a>
    <a class="ch soon"><span class="n">05</span><span>JEPA</span></a>`;
  html += `<div class="foot">Chandra Prakash Bathula<br><a href="https://github.com/ChandraPrakash-Bathula/Deep-Generative-Models">source</a> · <a href="https://huggingface.co/spaces/ChandraPrakashBathula/medlatent-studio">live demo</a></div>`;
  side.innerHTML = html;
  const layout = $(".layout"); layout.prepend(side);
  const btn = document.createElement("button"); btn.className = "menu-btn"; btn.textContent = "☰ Chapters";
  btn.onclick = () => side.classList.toggle("open"); document.body.append(btn);
  side.addEventListener("click", e => { if (e.target.closest("a")) side.classList.remove("open"); });
  // highlight current section in the toc
  const tocLinks = $$(".toc a", side);
  if (tocLinks.length) {
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) tocLinks.forEach(a => a.classList.toggle("on", a.hash === "#" + e.target.id)); }), {rootMargin: "-20% 0px -70% 0px"});
    $$("main h2[id]").forEach(h => io.observe(h));
  }
  // pager
  const i = CHAPTERS.findIndex(c => c.f === here);
  const pager = $(".pager");
  if (pager && i >= 0) {
    const prev = CHAPTERS[i - 1], next = CHAPTERS[i + 1];
    pager.innerHTML = (prev ? `<a href="${prev.f}"><div class="d">← Previous</div><div class="t">${prev.t}</div></a>` : `<a href="${BASE}index.html"><div class="d">← Start</div><div class="t">Overview</div></a>`) +
      (next ? `<a class="nx" href="${next.f}"><div class="d">Next →</div><div class="t">${next.t}</div></a>` : `<a class="nx" href="${BASE}explorer.html"><div class="d">Next →</div><div class="t">Experiment Explorer</div></a>`);
  }
}

/* ---------------- libraries ---------------- */
function renderMath() {
  if (window.renderMathInElement) renderMathInElement(document.body, {delimiters: [{left: "$$", right: "$$", display: true}, {left: "\\(", right: "\\)", display: false}], throwOnError: false});
  if (window.hljs) hljs.highlightAll();
}

/* ---------------- data ---------------- */
let _data;
function data() {
  return _data || (_data = Promise.all([fetch(A("data.json")).then(r => r.json()), fetch(A("data2.json")).then(r => r.json())]).then(([D, D2]) => ({D, D2})));
}
const results = (D, ds) => D.results[ds].map(r => Object.assign({}, r, {mk: r.model === "ae" ? "ae" : (r.beta === 1 ? "vae1" : "vae01")}));

/* ---------------- controls ---------------- */
function seg(el, cb) {
  el.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; $$("button", el).forEach(x => x.classList.toggle("on", x === b)); cb && cb(b.dataset.v); });
  return () => $("button.on", el)?.dataset.v;
}
function setSeg(el, values, labels, cur) {
  el.innerHTML = values.map((v, i) => `<button data-v="${v}" class="${String(v) === String(cur) ? "on" : ""}">${labels ? labels[i] : v}</button>`).join("");
}
function stepper(root, notes) {
  const btns = $(".steps", root), svg = $("svg", root), note = $(".stepnote", root);
  btns.innerHTML = notes.map((n, i) => `<button class="${i === 0 ? "on" : ""}" data-i="${i + 1}">Step ${i + 1}</button>`).join("") + `<button data-i="all">Show all</button>`;
  const show = k => {
    $$("[data-step]", svg).forEach(g => g.classList.toggle("hide", k !== "all" && +g.dataset.step > +k));
    $$("button", btns).forEach(b => b.classList.toggle("on", b.dataset.i === String(k)));
    note.innerHTML = k === "all" ? notes.map((n, i) => `<b>${i + 1}.</b> ${n}`).join(" ") : notes[k - 1];
  };
  btns.addEventListener("click", e => { const b = e.target.closest("button"); if (b) show(b.dataset.i === "all" ? "all" : +b.dataset.i); });
  show(1);
}

/* ---------------- plotly ---------------- */
const PCFG = {displayModeBar: false, responsive: true};
function lay(extra = {}) {
  return Object.assign({paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: {l: 56, r: 16, t: 16, b: 48},
    font: {family: "Inter, sans-serif", color: "#334155", size: 12.5},
    xaxis: {gridcolor: "#eef2f7", zerolinecolor: "#cbd5e1", linecolor: "#cbd5e1"}, yaxis: {gridcolor: "#eef2f7", zerolinecolor: "#cbd5e1", linecolor: "#cbd5e1"},
    legend: {orientation: "h", y: 1.14, x: 0, font: {size: 12}}, hoverlabel: {font: {family: "JetBrains Mono"}}}, extra);
}
const logx = (t = "latent dimension") => ({type: "log", title: t, tickvals: [2, 3, 4, 8, 16, 64, 128], ticktext: ["2", "3", "4", "8", "16", "64", "128"], gridcolor: "#eef2f7", linecolor: "#cbd5e1"});

/* ---------------- Alammar-style vector cells ---------------- */
function divColor(v, lim) {
  const t = Math.max(-1, Math.min(1, v / lim));
  const neg = [29, 78, 216], pos = [194, 65, 12], mid = [248, 250, 252];
  const c = t < 0 ? mid.map((m, k) => Math.round(m + (-t) * (neg[k] - m))) : mid.map((m, k) => Math.round(m + t * (pos[k] - m)));
  return {bg: `rgb(${c})`, dark: Math.abs(t) > .55};
}
function vec(el, vals, {lim = null, digits = 1, big = false, title = null} = {}) {
  lim = lim || Math.max(...vals.map(Math.abs)) || 1;
  el.className = "vec" + (big ? " big" : "");
  el.innerHTML = vals.map((v, j) => { const c = divColor(v, lim); return `<div class="c${c.dark ? " dark" : ""}" style="background:${c.bg}" title="${title ? title + " " : ""}z${j + 1} = ${v}">${(+v).toFixed(digits)}</div>`; }).join("");
}

/* ---------------- specimen picker ---------------- */
function picker(el, D, onPick, start = 3) {
  el.className = "picker";
  el.innerHTML = `<span class="pl">pick a slice</span>` + D.picks.map((p, i) => `<button aria-pressed="${i === start}" data-i="${i}" title="height in head ${Math.round(p.slice_pos * 100)}%"><img src="${A(`pick_${i}.png`)}" alt="slice ${i + 1}"></button>`).join("");
  el.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; $$("button", el).forEach(x => x.setAttribute("aria-pressed", x === b)); onPick(+b.dataset.i); });
  onPick(start);
}

/* ---------------- viridis ---------------- */
function viridis(t) {
  const s = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  t = Math.min(1, Math.max(0, t)) * 4; const i = Math.min(3, Math.floor(t)), f = t - i;
  return `rgb(${s[i].map((v, k) => Math.round(v + f * (s[i + 1][k] - v)))})`;
}

/* ---------------- 2-D latent plane widget ---------------- */
const _sprites = {};
const sprite = key => _sprites[key] || (_sprites[key] = new Promise(res => { const im = new Image(); im.onload = () => res(im); im.src = A(`plane_${key}.jpg`); }));
function planeWidget(D, canvas, preview, readout, getState, opts = {}) {
  const ctx = canvas.getContext("2d"), pctx = preview.getContext("2d");
  let P, spr, box, hover = null;
  const toPx = (x, y) => [(x - box.x0) / (box.x1 - box.x0) * canvas.width, (box.y1 - y) / (box.y1 - box.y0) * canvas.height];
  const toZ = (px, py) => [box.x0 + px / canvas.width * (box.x1 - box.x0), box.y1 - py / canvas.height * (box.y1 - box.y0)];
  async function draw() {
    const s = getState(), key = `${s.ds}_${s.mk}`;
    P = D.planes[key]; spr = await sprite(key);
    const sx = (P.hi[0] - P.lo[0]) / (P.n - 1), sy = (P.hi[1] - P.lo[1]) / (P.n - 1);
    box = {x0: P.lo[0] - sx / 2, x1: P.hi[0] + sx / 2, y0: P.lo[1] - sy / 2, y1: P.hi[1] + sy / 2};
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (s.layer !== "dots") { ctx.globalAlpha = s.layer === "map" ? 1 : .55; ctx.drawImage(spr, 0, 0, canvas.width, canvas.height); ctx.globalAlpha = 1; }
    if (s.layer !== "map") {
      const r = canvas.width / 250;
      for (let i = 0; i < P.x.length; i++) {
        const [px, py] = toPx(P.x[i], P.y[i]);
        ctx.fillStyle = s.ds === "ixi" ? viridis((P.c[i] - .3) / .5) : (P.c[i] > .5 ? "#60a5fa" : "#f472b6");
        ctx.globalAlpha = .85; ctx.beginPath(); ctx.arc(px, py, r, 0, 7); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    if (s.ds === "ixi" && opts.pick != null && P.pick_mu) {
      const k = opts.pick(), m = P.pick_mu[k], [px, py] = toPx(m[0], m[1]);
      ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = canvas.width / 170; ctx.beginPath(); ctx.arc(px, py, canvas.width / 40, 0, 7); ctx.stroke();
    }
    if (hover) { ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = canvas.width / 450; ctx.beginPath(); ctx.moveTo(hover[0], 0); ctx.lineTo(hover[0], canvas.height); ctx.moveTo(0, hover[1]); ctx.lineTo(canvas.width, hover[1]); ctx.stroke(); }
    tile(hover || (opts.pick != null && s.ds === "ixi" && P.pick_mu ? toPx(...P.pick_mu[opts.pick()]) : [canvas.width / 2, canvas.height / 2]));
    opts.after && opts.after(s);
  }
  function tile([px, py]) {
    const [zx, zy] = toZ(px, py);
    const j = Math.max(0, Math.min(P.n - 1, Math.round((zx - P.lo[0]) / (P.hi[0] - P.lo[0]) * (P.n - 1))));
    const i = Math.max(0, Math.min(P.n - 1, Math.round((P.hi[1] - zy) / (P.hi[1] - P.lo[1]) * (P.n - 1))));
    pctx.imageSmoothingQuality = "high"; pctx.drawImage(spr, j * P.tile, i * P.tile, P.tile, P.tile, 0, 0, preview.width, preview.height);
    readout.innerHTML = `z = [ <b>${zx.toFixed(2)}</b> , <b>${zy.toFixed(2)}</b> ] → decoder → image`;
  }
  const move = e => { const r = canvas.getBoundingClientRect(), t = e.touches ? e.touches[0] : e; hover = [(t.clientX - r.left) / r.width * canvas.width, (t.clientY - r.top) / r.height * canvas.height]; draw(); if (e.touches) e.preventDefault(); };
  canvas.addEventListener("mousemove", move); canvas.addEventListener("touchmove", move, {passive: false});
  canvas.addEventListener("mouseleave", () => { hover = null; draw(); });
  return draw;
}

/* ---------------- latent scatter (2D / 3D) from explorer data ---------------- */
const CLUSTER = ["#2563eb", "#ea580c", "#dc2626", "#0d9488", "#7c3aed", "#ca8a04"];
function scatter(el, D, s, {picks = true, height} = {}) {
  const e = D.explorer[`${s.ds}_${s.mk}_${s.z}`]; if (!e) return;
  const k = +s.dim, name = s.z <= k ? `raw_${Math.min(s.z, k)}d` : `${s.method}_${k}d`;
  const emb = e[name], kk = emb[0].length, all = [...emb.keys()];
  const xyz = idx => ({x: idx.map(i => emb[i][0]), y: idx.map(i => emb[i][1]), ...(kk === 3 ? {z: idx.map(i => emb[i][2])} : {})});
  const type = kk === 3 ? "scatter3d" : "scattergl", size = kk === 3 ? 2.6 : 5, tr = [], base = {mode: "markers", hoverinfo: "skip"};
  const lab = s.colour === "cluster" ? e.cluster : e.lab[s.colour];
  if (s.colour === "cluster") for (let c = 0; c < 6; c++) { const idx = all.filter(i => lab[i] === c); tr.push({...base, type, ...xyz(idx), name: `cluster ${c + 1}`, marker: {size, color: CLUSTER[c], opacity: .8}}); }
  else if (s.ds === "celeba") for (const [v, nm, col] of [[0, "no", "#cbd5e1"], [1, "yes", "#db2777"]]) { const idx = all.filter(i => lab[i] === v); tr.push({...base, type, ...xyz(idx), name: `${s.colour.replace("_", " ")}: ${nm}`, marker: {size, color: col, opacity: v ? .9 : .55}}); }
  else tr.push({...base, type, ...xyz(all), showlegend: false, marker: {size, color: lab, colorscale: "Viridis", opacity: .85, colorbar: {title: {text: s.colour === "mse" ? "MSE×1000" : "slice height", side: "right"}, thickness: 10, len: .6}}});
  if (picks && s.ds === "ixi" && s.pick != null) {
    const rows = D.picks.map(p => p.plot_row);
    tr.push({type, mode: "markers", ...xyz(rows), name: "example slices", showlegend: false, hoverinfo: "text", hovertext: D.picks.map((p, i) => `example slice ${i + 1}${i === s.pick ? " (picked)" : ""}`),
      marker: {size: rows.map((r, i) => i === s.pick ? (kk === 3 ? 16 : 18) : (kk === 3 ? 8 : 10)), symbol: kk === 3 ? "diamond-open" : "circle", color: "rgba(0,0,0,0)",
        line: {color: rows.map((r, i) => i === s.pick ? "#0ea5e9" : "#0f172a"), width: rows.map((r, i) => i === s.pick ? 4 : 1.5)}}});
  }
  const ax = name.startsWith("raw") ? ["z1", "z2", "z3"] : [1, 2, 3].map(i => `${s.method.toUpperCase()} ${i}`);
  const L = lay({margin: {l: 10, r: 10, t: 10, b: 10}, legend: {orientation: "h", y: 1.07, x: 0}});
  if (kk === 3) L.scene = {xaxis: {title: ax[0], showticklabels: false}, yaxis: {title: ax[1], showticklabels: false}, zaxis: {title: ax[2], showticklabels: false}};
  else { L.xaxis = {title: ax[0], showticklabels: false, zeroline: false, gridcolor: "#f1f5f9"}; L.yaxis = {title: ax[1], showticklabels: false, zeroline: false, gridcolor: "#f1f5f9"}; }
  if (height) L.height = height;
  Plotly.react(el, tr, L, PCFG);
  return name;
}

document.addEventListener("DOMContentLoaded", () => { buildChrome(); renderMath(); });
