/* ARGUS console. No build step, no framework.
   Information architecture follows GitHub Actions: a list of runs, a run summary with a job graph,
   job pages with collapsible steps, and a review gate for anything that needs a person. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function setHTML(el, html) {
  if (!el || el.__html === html) return false;
  el.innerHTML = html;
  el.__html = html;
  return true;
}

const S = {
  cfg: null, rt: null, reviewer: "operator", route: { view: "rounds" },
  rounds: [], roundsTimer: null,
  run: null, // { id, snap, trace, since, timer }
  follow: false, open: {}, showLLM: false, expandAll: null,
};

const CONDITIONS = {
  hard: "Hard conditions", glare: "Day glare", night: "Night shift", vibration: "Running pumps",
  random: "Mixed conditions", clean: "Clean conditions",
};
const COND_ORDER = ["hard", "glare", "night", "vibration", "random", "clean"];

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch (_) { /* not JSON */ }
  if (!r.ok) throw new Error((body && (body.detail || body.error)) || `${r.status} ${r.statusText}`);
  return body;
}

/* ------------------------------------------------------------------ formatting */
function decimals(span) {
  const l = Math.floor(Math.log10(Math.max(span, 1e-9)));
  return Math.max(0, Math.min(4, 2 - l));
}
function fmt(v, span) { return v == null || !isFinite(v) ? "—" : Number(v).toFixed(decimals(span)); }
function pm(v) { if (v == null || !isFinite(v)) return "—"; return v === 0 ? "0" : String(Number(Number(v).toPrecision(2))); }
function dur(s) {
  if (s == null || !isFinite(s)) return "";
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}
function ago(ts) {
  const d = Date.now() / 1000 - ts;
  if (d < 45) return "just now";
  if (d < 90) return "1 minute ago";
  if (d < 3600) return `${Math.round(d / 60)} minutes ago`;
  if (d < 5400) return "1 hour ago";
  if (d < 86400) return `${Math.round(d / 3600)} hours ago`;
  return new Date(ts * 1000).toLocaleString();
}
const nice = (s) => String(s || "").replace(/_/g, " ");
function limitsText(spec) {
  const parts = [["low_alarm", "low alarm"], ["low_warn", "low warning"], ["high_warn", "high warning"], ["high_alarm", "high alarm"]]
    .filter(([k]) => spec[k] != null).map(([k, l]) => `${l} ${spec[k]}`);
  return parts.length ? `${parts.join(", ")} ${spec.units}` : "none";
}
function roundTitle(r) {
  return `${CONDITIONS[r.profile] || r.profile}, ${r.agent === "llm" ? "LLM agent" : "policy agent"}`;
}

/* ------------------------------------------------------------------ status vocabulary */
const ST = {
  normal: { icon: "check-circle-fill", cls: "c-success", text: "Normal" },
  warning: { icon: "alert-fill", cls: "c-attention", text: "Warning" },
  alarm: { icon: "stop", cls: "c-danger", text: "Alarm" },
  instrument_fault: { icon: "x-circle-fill", cls: "c-danger", text: "Instrument fault" },
  escalated: { icon: "person", cls: "c-done", text: "Needs an operator" },
  uncertain: { icon: "alert", cls: "c-attention", text: "Uncertain" },
  pending: { icon: "circle", cls: "c-muted", text: "Waiting" },
  inspecting: { spin: true, text: "Inspecting now" },
};
function spinner() {
  return `<svg class="octicon spin c-attention" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="28 12" stroke-linecap="round"/></svg>`;
}
const stBase = (st) => String(st || "pending").replace(" (operator)", "");
function stIcon(st) {
  const m = ST[stBase(st)] || ST.pending;
  return m.spin ? spinner() : icon(m.icon, m.cls);
}
function stText(st) {
  const m = ST[stBase(st)] || { text: nice(st) };
  return m.text + (String(st).includes("(operator)") ? " (confirmed by operator)" : "");
}
function gaugeState(g, snap) {
  if (snap.status === "running" && snap.current_gauge === g.spec.id && !g.outcome.finished) return "inspecting";
  return g.outcome.status;
}
function roundState(r) {
  if (r.status === "running" || r.status === "created") return { html: spinner(), text: "In progress", cls: "running" };
  if (r.status === "error") return { html: icon("x-circle-fill", "c-danger"), text: "Stopped on an error", cls: "danger" };
  if (r.pending_human > 0) return { html: icon("clock", "c-attention"), text: "Waiting for a person", cls: "attention" };
  return { html: icon("check-circle-fill", "c-success"), text: "Complete", cls: "success" };
}

/* ------------------------------------------------------------------ dial */
function pt(theta, r) { const t = (theta * Math.PI) / 180; return [60 + r * Math.sin(t), 60 - r * Math.cos(t)]; }
function arc(t0, t1, r) {
  if (t1 < t0) [t0, t1] = [t1, t0];
  const [x0, y0] = pt(t0, r), [x1, y1] = pt(t1, r);
  return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${r} ${r} 0 ${t1 - t0 > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}
function v2t(spec, v) { return spec.theta_min + ((v - spec.min_value) / (spec.max_value - spec.min_value)) * (spec.theta_max - spec.theta_min); }
function dialSVG(spec, rd, recorded) {
  const tmin = spec.theta_min, tmax = spec.theta_max;
  let s = `<svg class="dial" viewBox="0 0 120 120" role="img" aria-label="${esc(spec.id)} dial${rd && rd.value != null ? `, reading ${esc(fmt(rd.value, spec.span))} ${esc(spec.units)}` : ""}"><circle class="face" cx="60" cy="60" r="54"/>`;
  const zones = [];
  if (spec.high_alarm != null) zones.push(["alarm", spec.high_alarm, spec.max_value]);
  if (spec.low_alarm != null) zones.push(["alarm", spec.min_value, spec.low_alarm]);
  if (spec.high_warn != null) zones.push(["warn", spec.high_warn, spec.high_alarm != null ? spec.high_alarm : spec.max_value]);
  if (spec.low_warn != null) zones.push(["warn", spec.low_alarm != null ? spec.low_alarm : spec.min_value, spec.low_warn]);
  for (const [k, a, b] of zones) s += `<path class="zone-${k}" d="${arc(v2t(spec, a), v2t(spec, b), 49)}"/>`;
  const nMaj = spec.major_ticks, nMin = (nMaj - 1) * spec.minor_per_major;
  for (let i = 0; i <= nMin; i++) {
    const th = tmin + ((tmax - tmin) * i) / nMin, major = i % spec.minor_per_major === 0;
    const [x0, y0] = pt(th, major ? 40 : 44), [x1, y1] = pt(th, 47);
    s += `<line class="tick${major ? "" : " minor"}" x1="${x0.toFixed(2)}" y1="${y0.toFixed(2)}" x2="${x1.toFixed(2)}" y2="${y1.toFixed(2)}"/>`;
  }
  let th = tmin - 8;
  if (rd && rd.value != null && isFinite(rd.value)) {
    th = Math.max(tmin - 8, Math.min(tmax + 8, v2t(spec, rd.value)));
    // the shaded cone is the reported ±2σ interval (at least 6° wide so a tight interval stays visible)
    const a = v2t(spec, rd.value - 2 * rd.sigma), b = v2t(spec, rd.value + 2 * rd.sigma);
    const mid = (a + b) / 2, half = Math.max(3, (b - a) / 2);
    const t0 = Math.max(tmin - 10, mid - half), t1 = Math.min(tmax + 10, mid + half);
    const [x0, y0] = pt(t0, 48), [x1, y1] = pt(t1, 48);
    s += `<path class="cone" d="M60 60 L${x0.toFixed(2)} ${y0.toFixed(2)} A48 48 0 ${t1 - t0 > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)} Z"><title>±2σ: ${esc(fmt(rd.value - 2 * rd.sigma, spec.span))} to ${esc(fmt(rd.value + 2 * rd.sigma, spec.span))} ${esc(spec.units)}</title></path>`;
  }
  s += `<line class="needle${recorded ? "" : " prov"}" x1="60" y1="66" x2="60" y2="16" style="transform: rotate(${th.toFixed(2)}deg)"/>`;
  s += `<circle class="hub" cx="60" cy="60" r="4"/></svg>`;
  return s;
}

function thumbs(ids, cls = "") {
  if (!ids || !ids.length) return "";
  return `<div class="thumbs">${ids.map((id) => `<button type="button" data-img="${esc(id)}" aria-label="Open evidence image"><img class="${cls}" src="/api/images/${esc(id)}" alt="" loading="lazy"></button>`).join("")}</div>`;
}

/* ------------------------------------------------------------------ routing */
function parseRoute() {
  const h = location.hash.replace(/^#\/?/, "");
  const [path, query] = h.split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  const q = new URLSearchParams(query || "");
  if (parts[0] === "photo") return { view: "photo" };
  if (parts[0] === "evidence") return { view: "evidence" };
  if (parts[0] === "how") return { view: "how" };
  if (parts[0] === "rounds" && parts[1]) return { view: "run", id: parts[1], gid: parts[2] || null };
  return { view: "rounds", profile: q.get("conditions") || null };
}
function go(hash, replace = false) {
  if (replace) history.replaceState(null, "", hash);
  else history.pushState(null, "", hash);
  route(true);
}
function route(force) {
  if (force !== true && location.hash === S.lastHash) return;
  S.lastHash = location.hash;
  const r = parseRoute();
  const prev = S.route;
  S.route = r;
  $$(".underline-nav a").forEach((a) => {
    const nav = a.dataset.nav;
    const on = nav === r.view || (nav === "rounds" && r.view === "run");
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  clearTimeout(S.roundsTimer);
  if (r.view !== "run" && S.run) { clearTimeout(S.run.timer); }
  if (r.view === "rounds") renderRounds();
  else if (r.view === "run") renderRun(prev);
  else if (r.view === "photo") renderPhoto();
  else if (r.view === "evidence") renderEvidence();
  else if (r.view === "how") renderHow();
  if (prev.view !== r.view || prev.gid !== r.gid) window.scrollTo(0, 0);
}

/* ------------------------------------------------------------------ rounds list */
function runPopoverHTML(defaults = {}) {
  const llm = S.cfg && S.cfg.llm_enabled;
  const prof = defaults.profile || "hard";
  return `<div class="popover" id="run-pop" role="dialog" aria-label="Run round" hidden>
    <h2>Run a round</h2>
    <form id="run-form">
      <div class="form-group"><label for="f-cond">Plant conditions</label>
        <select class="form-control" id="f-cond" name="profile">${COND_ORDER.filter((p) => !S.cfg || !Object.keys(S.cfg.profiles || {}).length || S.cfg.profiles[p]).map((p) => `<option value="${p}" ${p === prof ? "selected" : ""}>${esc(CONDITIONS[p])}</option>`).join("")}</select>
        <p class="note" id="f-cond-note"></p></div>
      <div class="form-group"><label for="f-agent">Inspector</label>
        <select class="form-control" id="f-agent" name="agent">
          <option value="llm" ${llm ? "selected" : "disabled"}>LLM agent on Amazon Bedrock${llm ? "" : " (not enabled here)"}</option>
          <option value="policy" ${llm ? "" : "selected"}>Deterministic policy (no LLM)</option>
        </select>
        <p class="note">${llm ? "If Bedrock fails, the policy takes over and the log says so." : "The policy uses the same OpenCV tools and safety guard."}</p></div>
      <div class="form-group"><label for="f-seed">Scenario seed <span class="muted">(optional)</span></label>
        <input class="form-control" id="f-seed" name="seed" inputmode="numeric" pattern="[0-9]*" placeholder="Random"></div>
      <button type="submit" class="btn-primary" style="width:100%">Run round</button>
      <p class="form-error" id="run-err" role="alert"></p>
    </form>
  </div>`;
}
function wireRunPopover() {
  const btn = $("#run-btn"), pop = $("#run-pop");
  if (!btn || !pop) return;
  const note = () => { const p = $("#f-cond").value; $("#f-cond-note").textContent = (S.cfg && S.cfg.profiles[p]) || ""; };
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    pop.hidden = !pop.hidden;
    btn.setAttribute("aria-expanded", String(!pop.hidden));
    if (!pop.hidden) { note(); $("#f-cond").focus(); }
  });
  $("#f-cond").addEventListener("change", note);
  pop.addEventListener("click", (e) => e.stopPropagation());
  $("#run-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const body = { profile: fd.get("profile"), agent: fd.get("agent") };
    const seed = String(fd.get("seed") || "").trim();
    if (seed) body.seed = parseInt(seed, 10);
    $("#run-err").textContent = "";
    const sub = e.target.querySelector("button[type=submit]");
    sub.disabled = true; sub.textContent = "Starting";
    try {
      const r = await api("/api/rounds", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      S.follow = true;
      go(`#/rounds/${encodeURIComponent(r.id)}`);
    } catch (err) {
      $("#run-err").textContent = err.message;
      sub.disabled = false; sub.textContent = "Run round";
    }
  });
}
document.addEventListener("click", () => { const p = $("#run-pop"); if (p && !p.hidden) { p.hidden = true; const b = $("#run-btn"); if (b) b.setAttribute("aria-expanded", "false"); } });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { const p = $("#run-pop"); if (p && !p.hidden) { p.hidden = true; $("#run-btn").focus(); } } });

function renderRounds() {
  const prof = S.route.profile;
  const main = $("#main");
  main.innerHTML = `<div class="container"><div class="layout">
    <aside class="side" aria-label="Rounds navigation">
      <h2 class="side-title">Rounds</h2>
      <ul class="side-list"><li><a class="side-item" href="#/rounds" ${prof ? "" : 'aria-current="page"'}>${icon("rows", "c-muted")}<span class="side-text">All rounds</span></a></li></ul>
      <div class="side-heading">Plant conditions</div>
      <ul class="side-list">${COND_ORDER.map((p) => `<li><a class="side-item" href="#/rounds?conditions=${p}" ${prof === p ? 'aria-current="page"' : ""}>${icon("workflow", "c-muted")}<span class="side-text">${esc(CONDITIONS[p])}</span></a></li>`).join("")}</ul>
    </aside>
    <section aria-labelledby="rounds-h">
      <div class="page-head">
        <div><h1 id="rounds-h">${prof ? esc(CONDITIONS[prof]) : "All rounds"}</h1>
          <p>${prof && S.cfg ? esc(S.cfg.profiles[prof]) : "Each round sends the robot along a panel of gauges. Open a round to see what the agent saw, what it decided, and what is waiting for you."}</p></div>
        <div class="run-wrap"><button type="button" class="btn-primary" id="run-btn" aria-haspopup="dialog" aria-expanded="false">Run round ${icon("triangle-down")}</button>${runPopoverHTML({ profile: prof })}</div>
      </div>
      <div class="box" id="rounds-box"><div class="box-header"><span id="rounds-count">Loading rounds</span></div><div id="rounds-list"></div></div>
    </section>
  </div></div>`;
  wireRunPopover();
  loadRounds();
}

async function loadRounds() {
  clearTimeout(S.roundsTimer);
  if (S.route.view !== "rounds") return;
  try {
    S.rounds = await api("/api/rounds");
  } catch (e) {
    setHTML($("#rounds-list"), `<div class="blankslate">${icon("alert")}<h3>Server not reachable</h3><p>${esc(e.message)}</p></div>`);
    return;
  }
  updateNavCount();
  if (S.route.view !== "rounds") return;
  const prof = S.route.profile;
  const rows = S.rounds.filter((r) => !prof || r.profile === prof);
  const box = $("#rounds-list");
  if (!box) return;
  $("#rounds-count").textContent = `${rows.length} round${rows.length === 1 ? "" : "s"}`;
  if (!rows.length) {
    setHTML(box, `<div class="blankslate">${icon("play")}<h3>Run your first round</h3>
      <p>The robot walks a panel of five gauges under the conditions you pick. Every capture, reading and decision is logged, and anything uncertain comes back to you.</p>
      <button type="button" class="btn-primary" data-open-run>Run round</button></div>`);
  } else {
    setHTML(box, `<ul class="run-list">${rows.map(roundRow).join("")}</ul>`);
  }
  if (S.rounds.some((r) => r.status === "running" || r.status === "created")) S.roundsTimer = setTimeout(loadRounds, 2000);
  else S.roundsTimer = setTimeout(loadRounds, 15000);
}
function roundRow(r) {
  const st = roundState(r);
  const flags = [];
  if (r.pending_human) flags.push(`<span class="label label-done">${icon("person")}${r.pending_human} waiting for you</span>`);
  if (r.alarms) flags.push(`<span class="label label-danger">${r.alarms} alarm${r.alarms > 1 ? "s" : ""}</span>`);
  if (r.warnings) flags.push(`<span class="label label-attention">${r.warnings} warning${r.warnings > 1 ? "s" : ""}</span>`);
  if (r.fallbacks) flags.push(`<span class="label label-attention">LLM fallback</span>`);
  return `<li class="run-row"><a href="#/rounds/${encodeURIComponent(r.id)}" aria-label="${esc(roundTitle(r))}, ${esc(st.text)}">
    <span title="${esc(st.text)}">${st.html}</span>
    <span><span class="run-title">${esc(roundTitle(r))}</span>
      <div class="run-sub">Round #${esc(r.number)}, seed ${esc(r.seed)}, ${esc(r.done)} of ${esc(r.gauges)} gauges closed${r.agent === "llm" && r.cost_usd ? `, $${Number(r.cost_usd).toFixed(4)} of Bedrock` : ""}</div></span>
    <span class="run-flags">${flags.join("")}</span>
    <span class="run-meta"><span>${icon("clock")}${esc(ago(r.created))}</span><span>${icon("stopwatch")}${esc(dur(r.robot_time_s))} robot time</span></span>
  </a></li>`;
}
document.addEventListener("click", (e) => {
  if (e.target.closest("[data-open-run]")) { e.stopPropagation(); const b = $("#run-btn"); if (b) b.click(); }
});
function updateNavCount() {
  const n = S.rounds.reduce((a, r) => a + (r.pending_human || 0), 0);
  const c = $("#nav-count");
  if (!c) return;
  c.hidden = !S.rounds.length;
  c.textContent = n ? `${n} waiting` : String(S.rounds.length);
  c.title = n ? `${n} items need a person` : `${S.rounds.length} rounds`;
}

/* ------------------------------------------------------------------ run page */
function renderRun(prev) {
  const { id } = S.route;
  if (!S.run || S.run.id !== id) {
    if (S.run) clearTimeout(S.run.timer);
    S.run = { id, snap: null, trace: [], since: 0, timer: null };
    S.open = {};
    $("#main").innerHTML = `<div class="container"><p class="muted">Loading round</p></div>`;
    pollRun();
    return;
  }
  if (prev && prev.view === "run" && prev.id === id && $("#run-shell")) {
    drawRun(false);
    if (S.run.snap && (S.run.snap.status === "running" || S.run.snap.status === "created")) { clearTimeout(S.run.timer); S.run.timer = setTimeout(pollRun, 700); }
    return;
  }
  $("#main").innerHTML = "";
  drawRun(true);
  pollRun();
}

async function pollRun() {
  const run = S.run;
  if (!run || S.route.view !== "run" || S.route.id !== run.id) return;
  clearTimeout(run.timer);
  try {
    const snap = await api(`/api/rounds/${encodeURIComponent(run.id)}?since=${run.since}`);
    if (S.run !== run) return;
    run.trace.push(...snap.trace);
    run.since = snap.trace_len;
    run.snap = snap;
  } catch (e) {
    $("#main").innerHTML = `<div class="container"><div class="flash flash-error">${icon("alert")}<div class="flash-text"><strong>This round could not be loaded.</strong>${esc(e.message)}</div><a class="btn" href="#/rounds">Back to rounds</a></div></div>`;
    return;
  }
  const snap = run.snap;
  const running = snap.status === "running" || snap.status === "created";
  if (S.follow && running && snap.current_gauge && S.route.gid !== snap.current_gauge) {
    go(`#/rounds/${encodeURIComponent(run.id)}/${encodeURIComponent(snap.current_gauge)}`, true);
    return;
  }
  let known = S.rounds.find((x) => x.id === run.id);
  if (!known) { known = { id: run.id, created: snap.created }; S.rounds.unshift(known); }
  known.pending_human = pendingCount(snap); known.status = snap.status;
  updateNavCount();
  if (!running && S.follow) {
    S.follow = false;
    if (S.route.gid) { go(`#/rounds/${encodeURIComponent(run.id)}`, true); return; }
  }
  drawRun(false);
  if (running) run.timer = setTimeout(pollRun, 700);
}

function drawRun(full) {
  const run = S.run, snap = run && run.snap;
  const main = $("#main");
  if (!snap) return;
  if (full || !$("#run-shell")) {
    main.innerHTML = `<div class="container" id="run-shell">
      <a class="back" href="#/rounds">${icon("arrow-left")}Rounds</a>
      <div class="run-head"><div><h1 id="run-title"></h1><div class="run-head-sub" id="run-sub"></div></div><div class="run-actions" id="run-actions"></div></div>
      <div class="layout">
        <aside class="side run-side" aria-label="Round navigation" id="run-side"></aside>
        <section id="run-content" aria-live="off"></section>
      </div></div>`;
  }
  const r = { ...snap, number: snap.number, agent: snap.agent_mode, pending_human: pendingCount(snap) };
  const st = roundState(r);
  setHTML($("#run-title"), `<span title="${esc(st.text)}">${st.html}</span>${esc(roundTitle(r))} <span class="num">#${esc(snap.number)}</span>`);
  setHTML($("#run-sub"), `${esc(st.text)}. Started ${esc(ago(snap.created))}, seed ${esc(snap.seed)}.`);
  const running = snap.status === "running" || snap.status === "created";
  setHTML($("#run-actions"), `${running ? (S.follow
      ? `<button type="button" class="btn" data-act="unfollow">${icon("eye")}Following the robot</button>`
      : `<button type="button" class="btn" data-act="follow">${icon("eye")}Follow live</button>`) : ""}
    <button type="button" class="btn" data-act="rerun">${icon("sync")}Re-run round</button>`);
  drawRunSide(snap);
  if (S.route.gid) drawJob(snap, S.route.gid);
  else drawSummary(snap);
}
function pendingCount(snap) {
  return snap.reviews.filter((x) => x.status === "pending").length + snap.work_orders.filter((x) => x.status === "pending_approval").length;
}

function drawRunSide(snap) {
  const base = `#/rounds/${encodeURIComponent(snap.id)}`;
  const gid = S.route.gid;
  setHTML($("#run-side"), `
    <ul class="side-list"><li><a class="side-item" href="${base}" ${gid ? "" : 'aria-current="page"'}>${icon("home", "c-muted")}<span class="side-text">Summary</span></a></li></ul>
    <div class="side-heading">Gauges</div>
    <ul class="side-list">${snap.gauges.map((g) => {
      const gs = gaugeState(g, snap);
      return `<li><a class="side-item" href="${base}/${encodeURIComponent(g.spec.id)}" ${gid === g.spec.id ? 'aria-current="page"' : ""} title="${esc(g.spec.name)}: ${esc(stText(gs))}">${stIcon(gs)}<span class="side-text">${esc(g.spec.id)} <span class="side-sub">${esc(g.spec.name)}</span></span></a></li>`;
    }).join("")}</ul>
    <div class="side-heading">Round details</div>
    <ul class="side-list">
      <li><a class="side-item" href="#" data-act="runner">${icon("cpu", "c-muted")}<span class="side-text">Runner</span></a></li>
      <li><a class="side-item" href="/api/rounds/${encodeURIComponent(snap.id)}/trace.json" download>${icon("download", "c-muted")}<span class="side-text">Trace (JSON)</span></a></li>
    </ul>`);
}

/* ---- summary */
function drawSummary(snap) {
  const content = $("#run-content");
  const caps = snap.gauges.reduce((a, g) => a + g.outcome.captures, 0);
  const comp = snap.gauges.reduce((a, g) => a + g.outcome.compute_ms, 0);
  const closed = snap.gauges.filter((g) => g.outcome.finished).length;
  const llm = snap.agent_mode === "llm";
  const model = S.cfg && S.cfg.llm_model ? S.cfg.llm_model.replace(/^(global|us|eu|apac)\./, "").replace(/^anthropic\./, "").replace(/-\d{8}-v\d+:\d+$/, "") : "Bedrock";
  const pendWO = snap.work_orders.filter((w) => w.status === "pending_approval");
  const pendRev = snap.reviews.filter((r) => r.status === "pending");
  const items = [
    ["Status", `${roundState({ ...snap, pending_human: pendWO.length + pendRev.length }).text}`],
    ["Robot time", dur(snap.robot_time_s) || "0s"],
    ["Gauges", `${closed} of ${snap.gauges.length} closed`],
    ["Captures", String(caps)],
    ["OpenCV compute", comp >= 1000 ? `${(comp / 1000).toFixed(1)} s` : `${Math.round(comp)} ms`],
    ["Inspector", llm ? `LLM agent <small>${esc(model)}</small>` : "Deterministic policy"],
  ];
  if (llm) items.push(["Bedrock", `${snap.usage.llm_calls} calls <small>$${Number(snap.usage.cost_usd).toFixed(4)}${snap.usage.fallbacks ? `, ${snap.usage.fallbacks} fallback` : ""}</small>`]);
  items.push(["Light sensor", `${snap.ambient_light}${snap.ambient_light < 0.15 ? " <small>night</small>" : ""}`]);
  let html = `<div class="box"><div class="status-box">${items.map(([k, v]) => `<div class="status-item"><div class="k">${esc(k)}</div><div class="v">${v}</div></div>`).join("")}</div></div>`;
  if (pendWO.length) html += `<div class="flash flash-warn">${icon("clock")}<div class="flash-text"><strong>${pendWO.length} work order${pendWO.length > 1 ? "s are" : " is"} waiting for approval</strong>The agent drafted ${pendWO.length > 1 ? "them" : "it"}; only a person can dispatch work to maintenance.</div><button type="button" class="btn" data-act="review-wo">Review work orders</button></div>`;
  if (pendRev.length) html += `<div class="flash flash-done">${icon("person")}<div class="flash-text"><strong>${pendRev.length} reading${pendRev.length > 1 ? "s need" : " needs"} an operator</strong>The robot could not get evidence it trusts, so it is asking instead of guessing.</div><button type="button" class="btn" data-act="review-readings">Review readings</button></div>`;
  if (snap.status === "error") html += `<div class="flash flash-error">${icon("stop")}<div class="flash-text"><strong>The round stopped on an error</strong>${esc(snap.error || "")}</div></div>`;
  // route graph
  const base = `#/rounds/${encodeURIComponent(snap.id)}`;
  html += `<div class="box"><div class="box-header"><span>Route <span class="muted">${snap.gauges.length} gauges in walking order</span></span><span class="muted">Select a gauge to see each step</span></div>
    <div class="graph-scroll"><div class="graph">${snap.gauges.map((g) => {
      const gs = gaugeState(g, snap), rd = g.recorded || g.latest, spec = g.spec;
      return `<a class="node${gs === "inspecting" ? " active" : ""}" href="${base}/${encodeURIComponent(spec.id)}" title="${esc(spec.name)}: ${esc(stText(gs))}">
        <div class="node-head">${stIcon(gs)}<span>${esc(spec.id)}</span><span class="dur">${g.outcome.finished ? esc(dur(g.outcome.robot_time_s) || "") : ""}</span></div>
        ${dialSVG(spec, rd, !!g.recorded)}
        <div class="node-val">${rd && rd.value != null ? `${esc(fmt(rd.value, spec.span))} <small>± ${esc(pm(2 * rd.sigma))} ${esc(spec.units)}</small>` : `<small>${esc(spec.min_value)} to ${esc(spec.max_value)} ${esc(spec.units)}</small>`}</div>
        <div class="node-name">${esc(stText(gs).replace(" (confirmed by operator)", ", operator"))}</div></a>`;
    }).join("")}</div></div></div>`;
  // annotations
  const anns = annotations(snap);
  const nWarn = anns.filter((a) => a.level === "warning").length, nErr = anns.filter((a) => a.level === "error").length, nInfo = anns.length - nWarn - nErr;
  html += `<div class="box"><div class="box-header"><span>Annotations</span><span class="muted">${[nErr && `${nErr} error${nErr > 1 ? "s" : ""}`, nWarn && `${nWarn} warning${nWarn > 1 ? "s" : ""}`, nInfo && `${nInfo} notice${nInfo > 1 ? "s" : ""}`].filter(Boolean).join(", ") || "None"}</span></div>
    ${anns.length ? `<ul class="ann-list">${anns.map((a) => `<li>${a.icon}<div><div class="ann-title">${a.gid ? `<a href="${base}/${encodeURIComponent(a.gid)}">${esc(a.title)}</a>` : esc(a.title)}</div><div class="ann-msg">${esc(a.msg)}</div></div></li>`).join("")}</ul>`
      : `<div class="box-body muted">${snap.status === "done" ? "Nothing unusual: every gauge was read with evidence the agent trusts." : "Issues, escalations and safety-guard decisions appear here as the robot works."}</div>`}</div>`;
  // decisions
  const dec = decisionsHTML(snap);
  if (dec) html += `<div class="box"><div class="box-header"><span>Approvals and reviews</span><span class="muted">people decide; the agent cannot dispatch work</span></div>${dec}</div>`;
  // hand-over
  if (snap.report) html += `<div class="box handover"><div class="box-header"><span>Shift hand-over</span><span class="muted">${llm ? "written by the LLM from the round's facts" : "generated from the round's facts"}</span></div><div class="box-body">${handoverHTML(snap.report)}</div></div>`;
  html += `<div class="box"><div class="box-header"><span>Artifacts</span><span class="muted">kept for 60 days in Amazon S3 when deployed</span></div>
    <div class="artifact">${icon("file", "c-muted")}<div class="grow"><a href="/api/rounds/${encodeURIComponent(snap.id)}/trace.json" download>trace.json</a><div class="muted" style="font-size:12px">Every perception result, decision, guard check and human action in this round</div></div><a class="btn btn-sm" href="/api/rounds/${encodeURIComponent(snap.id)}/trace.json" download>${icon("download")}Download</a></div>
    <div class="artifact">${icon("cpu", "c-muted")}<div class="grow"><a href="/api/runtime">runtime.json</a><div class="muted" style="font-size:12px">Which OpenCV build ran, on which CPU and instance</div></div><button type="button" class="btn btn-sm" data-act="runner">${icon("eye")}View</button></div></div>`;
  setHTML(content, html);
}

function annotations(snap) {
  const out = [];
  const spec = (gid) => (snap.gauges.find((g) => g.spec.id === gid) || {}).spec || { span: 100, units: "" };
  for (const g of snap.gauges) {
    const s = g.spec, rd = g.recorded, st = stBase(g.outcome.status);
    if (st === "alarm" || st === "instrument_fault") out.push({ level: "error", gid: s.id, icon: icon(st === "alarm" ? "stop" : "x-circle-fill", "c-danger"), title: `${s.id} ${stText(st).toLowerCase()}${rd ? ` at ${fmt(rd.value, s.span)} ${s.units}` : ""}`, msg: `${s.name}. ${g.outcome.work_order_id ? `Work order ${g.outcome.work_order_id} drafted for approval.` : "No work order yet."}` });
    else if (st === "warning") out.push({ level: "warning", gid: s.id, icon: icon("alert-fill", "c-attention"), title: `${s.id} is in the warning band${rd ? `: ${fmt(rd.value, s.span)} ${s.units}` : ""}`, msg: `${s.name}. Limits: ${limitsText(s)}.` });
    if (rd && rd.flutter && rd.flutter > 0.006 * s.span) out.push({ level: "notice", gid: s.id, icon: icon("pulse", "c-attention"), title: `${s.id} needle flutter ±${pm(rd.flutter)} ${s.units}`, msg: "Process pulsation: consider checking the pump or fitting a snubber." });
  }
  const guardBy = {};
  for (const ev of S.run.trace) {
    if (ev.kind === "guard") { (guardBy[ev.gauge_id] = guardBy[ev.gauge_id] || []).push(ev); }
    if (ev.kind === "fallback") out.push({ level: "warning", gid: ev.gauge_id, icon: icon("alert-fill", "c-attention"), title: `${ev.gauge_id || "Round"}: handed to the deterministic policy`, msg: ev.reason });
    if (ev.kind === "round_error") out.push({ level: "error", gid: null, icon: icon("x-circle-fill", "c-danger"), title: "Round stopped on an error", msg: ev.error });
  }
  for (const [gid, evs] of Object.entries(guardBy)) out.push({ level: "notice", gid, icon: icon("shield-check", "c-accent"), title: `${gid}: the safety guard blocked ${evs.length} action${evs.length > 1 ? "s" : ""}`, msg: evs[0].error });
  const rank = { error: 0, warning: 1, notice: 2 };
  return out.sort((a, b) => rank[a.level] - rank[b.level]);
}

function decisionsHTML(snap) {
  const rows = [];
  for (const w of snap.work_orders) {
    const pending = w.status === "pending_approval";
    rows.push(`<div class="decision">${pending ? icon("clock", "c-attention") : w.status === "dispatched" ? icon("check-circle-fill", "c-success") : icon("x-circle-fill", "c-muted")}
      <div><div class="d-title"><span class="label ${w.priority <= "P2" ? "label-danger" : "label-attention"}">${esc(w.priority)}</span> ${esc(w.id)}: ${esc(w.title)}</div>
      <div class="d-msg">${pending ? "Waiting for approval." : w.status === "dispatched" ? `Approved by ${esc(w.approved_by)} and sent to maintenance (${esc(w.dispatch_ref || "")}).` : `Rejected by ${esc(w.approved_by)}.`}${w.priority_note ? ` ${esc(w.priority_note)}.` : ""}</div></div>
      ${pending ? `<button type="button" class="btn btn-sm" data-act="review-wo">Review</button>` : ""}</div>`);
  }
  for (const r of snap.reviews) {
    const pending = r.status === "pending";
    const spec = (snap.gauges.find((g) => g.spec.id === r.gauge_id) || {}).spec || { units: "" };
    rows.push(`<div class="decision">${pending ? icon("person", "c-done") : icon("check-circle-fill", r.status === "rejected" ? "c-muted" : "c-success")}
      <div><div class="d-title">${esc(r.gauge_id)}: reading ${pending ? "needs an operator" : esc(r.status)}</div>
      <div class="d-msg">${pending ? esc(r.reason) : `${esc(r.decided_by)} ${r.status === "rejected" ? "asked for someone to check on site" : `recorded ${esc(r.final_value)} ${esc(spec.units)}`}${r.note ? `: “${esc(r.note)}”` : ""}.`}</div></div>
      ${pending ? `<button type="button" class="btn btn-sm" data-act="review-readings">Review</button>` : ""}</div>`);
  }
  return rows.join("");
}

function handoverHTML(text) {
  const lines = String(text).split("\n").map((l) => l.trim()).filter(Boolean);
  const map = { ACTION: icon("stop", "c-danger"), WATCH: icon("alert-fill", "c-attention"), REVIEW: icon("person", "c-done"), NOTE: icon("pulse", "c-attention") };
  const para = [], items = [];
  for (const l of lines) {
    const m = l.replace(/^[-•*]\s*/, "").match(/^(ACTION|WATCH|REVIEW|NOTE)[:\s]+(.*)$/);
    if (m) items.push(`<li>${map[m[1]]}<span>${esc(m[2])}</span></li>`);
    else if (/^[-•*]\s/.test(l)) items.push(`<li>${icon("dot-fill", "c-muted")}<span>${esc(l.replace(/^[-•*]\s*/, ""))}</span></li>`);
    else para.push(`<p>${esc(l)}</p>`);
  }
  return para.join("") + (items.length ? `<ul>${items.join("")}</ul>` : "");
}

/* ---- job (one gauge) */
const TOOL_NAME = { capture: "capture", read_gauge: "read gauge", record_reading: "record reading", request_human_review: "escalate", create_work_order: "draft work order", finish_gauge: "close gauge" };
function who(actor) { return actor === "llm" ? "Agent" : actor === "policy" ? "Policy" : actor === "guard" ? "Guard" : actor === "mcp-agent" ? "MCP agent" : nice(actor); }
function captureTitle(a, r) {
  const parts = [];
  const n = r ? r.frames : a.burst || 1;
  parts.push(a.hdr ? "HDR bracket" : `${n} frame${n === 1 ? "" : "s"}`);
  if (a.fast_shutter) parts.push("fast shutter");
  if (a.ev_bias) parts.push(`${a.ev_bias > 0 ? "+" : ""}${a.ev_bias} EV`);
  if (a.zoom && a.zoom !== 1) parts.push(`zoom ${a.zoom}x`);
  if ((r && r.camera && r.camera.light_on) || (!r && a.light)) parts.push("spotlight on");
  return `${r ? "Captured" : "Capturing"} the ${a.view || "front"} view: ${parts.join(", ")}`;
}

function buildSteps(trace, gid, spec) {
  const steps = [];
  let thought = null, pending = null, caps = 0;
  for (const ev of trace) {
    if (ev.gauge_id !== gid) continue;
    const id = `s${ev.seq}`;
    switch (ev.kind) {
      case "gauge_start": {
        const c = ev.context;
        steps.push({ id, icon: icon("gear", "c-muted"), title: "Set up gauge", dur: "", open: false, lines: [
          { html: `Gauge      ${esc(c.gauge_id)}, ${esc(c.name)}` },
          { html: `Range      ${esc(c.range[0])} to ${esc(c.range[1])} ${esc(c.units)}` },
          { html: `Limits     ${esc(Object.entries(c.limits).map(([k, v]) => `${nice(k)} ${v}`).join(", ") || "none")}` },
          { html: `Location   ${esc(c.location || "")}, ${esc(c.criticality)} criticality` },
          { html: `History    previous rounds ${esc((c.previous_readings || []).join(", "))}` },
          { html: `Light      sensor ${esc(c.ambient_light)}${c.ambient_light < 0.15 ? " (night level)" : ""}` },
          { cls: "dim", html: `Budget     ${esc(c.budget.max_captures)} captures, ${esc(c.budget.max_tool_calls)} tool calls` },
        ] });
        break;
      }
      case "thought":
        if (pending && ev.actor === pending.actor) break;
        thought = ev;
        break;
      case "tool_call": {
        let label = "";
        if (ev.tool === "capture") { caps += 1; if (caps > 1) label = "Re-planned from what it saw"; }
        const st = { id, tool: ev.tool, args: ev.args || {}, actor: ev.actor, robot0: ev.robot_t, running: true, icon: spinner(), label, dur: "", open: true, lines: [], images: [] };
        st.title = ev.tool === "capture" ? captureTitle(st.args, null) : `Running ${TOOL_NAME[ev.tool] || ev.tool}`;
        if (thought) { st.lines.push({ cls: "why", html: `<span class="who">${esc(who(thought.actor))}</span> ${esc(thought.text).replace(/ -&gt; /g, " → ")}` }); thought = null; }
        steps.push(st);
        pending = st;
        break;
      }
      case "tool_result":
        if (pending && pending.tool === ev.tool) { finishStep(pending, ev, spec); pending = null; }
        break;
      case "guard":
      case "tool_error": {
        const st = pending || { id, tool: ev.tool, args: {}, lines: [], images: [], label: "" };
        if (!pending) steps.push(st);
        st.running = false;
        st.icon = icon(ev.kind === "guard" ? "shield-x" : "x-circle-fill", "c-danger");
        st.title = ev.kind === "guard" ? `Safety guard blocked: ${TOOL_NAME[ev.tool] || ev.tool}` : `${TOOL_NAME[ev.tool] || ev.tool} failed`;
        st.lines.push({ html: `<span class="e">Error:</span> ${esc(ev.error)}` });
        st.dur = "";
        st.open = true;
        pending = null;
        break;
      }
      case "fallback":
        steps.push({ id, icon: icon("alert-fill", "c-attention"), title: "Handed to the deterministic policy", open: true, lines: [{ html: `<span class="w">Warning:</span> ${esc(ev.reason)}` }] });
        break;
      case "forced_handoff":
        steps.push({ id, icon: icon("shield-check", "c-attention"), title: "Safety guard closed the gauge", open: true, lines: [{ html: esc(ev.reason) }] });
        break;
      case "llm_call":
        if (S.showLLM) steps.push({ id, minor: true, icon: icon("zap", "c-muted"), title: `Model call: ${ev.input_tokens} tokens in, ${ev.output_tokens} out${ev.cache_read ? `, ${ev.cache_read} cached` : ""}`, dur: `${Math.round(ev.latency_ms)} ms`, lines: [] });
        break;
      case "human_review":
        steps.push({ id, icon: icon("person-fill", "c-done"), title: ev.value != null ? `${ev.by} ${ev.decision} the reading: ${ev.value} ${spec.units}` : `${ev.by} asked for someone to check on site`, open: true, lines: ev.note ? [{ html: `<span class="hm">Comment</span> ${esc(ev.note)}` }] : [] });
        break;
      case "human_work_order":
        steps.push({ id, icon: icon("person-fill", "c-done"), title: `${ev.by} ${ev.decision === "dispatched" ? "approved and dispatched" : "rejected"} ${ev.work_order_id}`, open: true, lines: [ev.dispatch_ref ? { html: `Sent via ${esc(ev.dispatch_ref)}` } : null, ev.note ? { html: `<span class="hm">Comment</span> ${esc(ev.note)}` } : null].filter(Boolean) });
        break;
      default:
        break;
    }
  }
  if (thought) steps.push({ id: `t${thought.seq}`, icon: icon("comment", "c-muted"), title: "Agent note", open: true, lines: [{ cls: "why", html: esc(thought.text) }] });
  return steps;
}

function finishStep(st, ev, spec) {
  const r = ev.result || {}, a = st.args || {};
  st.running = false;
  st.images = ev.images || [];
  switch (st.tool) {
    case "capture": {
      st.icon = icon("check-circle-fill", "c-success");
      st.title = captureTitle(a, r);
      st.dur = dur(r.robot_time_s);
      const cam = r.camera || {};
      st.lines.push({ cls: "dim", html: `Camera     exposure ${esc((cam.exposure_ms || []).join(", "))} ms, gain ${esc(cam.gain_db)} dB${cam.light_on ? ", spotlight on" : ""}` });
      const m = (r.quality && r.quality.metrics) || {};
      st.lines.push({ cls: "dim", html: `OpenCV     ${esc(m.localization || "")} geometry, dial ${esc(m.dial_radius_px)} px, view ratio ${esc(m.view_axis_ratio)}, sharpness ${esc(m.sharpness)}, clipped ${m.clipped_frac != null ? (m.clipped_frac * 100).toFixed(1) : "—"}%` });
      const iss = (r.quality && r.quality.issues) || [];
      if (!iss.length) st.lines.push({ html: `<span class="ok">No quality issues</span>` });
      for (const i of iss) st.lines.push({ html: `<span class="w">Warning:</span> ${esc(nice(i.code))}: ${esc(i.meaning)}` });
      st.lines.push({ cls: "dim", html: `Budget     ${esc(r.captures_left)} captures left` });
      st.imgCls = "";
      st.open = iss.length > 0 || !!st.label;
      break;
    }
    case "read_gauge": {
      st.dur = ev.latency_ms != null ? `${Math.round(ev.latency_ms)} ms` : "";
      st.imgCls = "ovl";
      if (r.ok === false) {
        st.icon = icon("x-circle-fill", "c-danger");
        st.title = "No reading: the dial could not be located";
        for (const i of r.issues || []) st.lines.push({ html: `<span class="w">Warning:</span> ${esc(i.meaning || nice(i.code))}` });
        break;
      }
      const lc = r.limit_check || {};
      const ok = r.confidence >= r.auto_accept_threshold && lc.status !== "uncertain";
      st.icon = icon(ok ? "check-circle-fill" : "alert-fill", ok ? "c-success" : "c-attention");
      st.title = `Read ${fmt(r.value, spec.span)} ± ${pm(2 * r.sigma)} ${r.units}${ok ? "" : ", not trustworthy yet"}`;
      if (a.enhance && a.enhance.length) st.lines.push({ html: `Enhance    ${esc(a.enhance.join(", "))}` });
      st.lines.push({ html: `Confidence ${esc(Number(r.confidence).toFixed(2))} <span class="${ok ? "ok" : "w"}">${r.confidence >= r.auto_accept_threshold ? "meets" : "below"} the ${esc(r.auto_accept_threshold)} needed to record</span>` });
      if ((r.confidence_reduced_by || []).length) st.lines.push({ html: `<span class="w">Warning:</span> confidence reduced by ${esc(r.confidence_reduced_by.map(nice).join(", "))}` });
      st.lines.push({ cls: "dim", html: `Method     ${esc(r.method)}${r.scale_verified ? ", printed scale verified" : ", scale not verified"}` });
      st.lines.push({ html: `Limits     ${esc(stText(lc.status))}, 2σ interval ${esc((lc.interval_2sigma || []).map((x) => fmt(x, spec.span)).join(" to "))} ${esc(spec.units)}${lc.nearest_limit ? `, ${esc(nice(lc.nearest_limit.name))} ${esc(lc.nearest_limit.value)} is ${esc(lc.margin_sigma)}σ away` : ""}` });
      if (lc.explanation) st.lines.push({ html: `<span class="w">Warning:</span> ${esc(lc.explanation)}` });
      if (r.flutter_note) st.lines.push({ html: `<span class="w">Notice:</span> ${esc(r.flutter_note)} (±${esc(pm(r.needle_flutter))} ${esc(spec.units)})` });
      if (r.trend && r.trend.note) st.lines.push({ html: `<span class="w">Notice:</span> ${esc(r.trend.note)} (${esc(r.trend.change_pct_span)}% of span vs previous mean ${esc(r.trend.previous_mean)})` });
      st.open = true;
      break;
    }
    case "record_reading":
      st.icon = icon("check-circle-fill", "c-success");
      st.title = `Recorded the reading as ${stText(r.status).toLowerCase()}`;
      if (r.work_order_required) st.lines.push({ html: `<span class="w">Notice:</span> this status needs a work order before the gauge can close` });
      st.open = st.lines.length > 1;
      break;
    case "request_human_review":
      st.icon = icon("person", "c-done");
      st.title = `Escalated to an operator (${r.review_id})`;
      st.lines.push({ html: `Reason     ${esc(a.reason)}` });
      st.open = true;
      break;
    case "create_work_order":
      st.icon = icon("issue-opened", "c-danger");
      st.title = `Drafted work order ${r.work_order_id} (${r.priority}) for approval`;
      st.lines.push({ html: `Title      ${esc(a.title)}` });
      st.lines.push({ cls: "dim", html: esc(a.description) });
      if (r.priority_adjusted) st.lines.push({ html: `<span class="w">Guard:</span> ${esc(r.priority_adjusted)}` });
      st.lines.push({ cls: "dim", html: "Dispatch needs a person." });
      st.open = true;
      break;
    case "finish_gauge":
      st.icon = icon("check", "c-muted");
      st.title = "Closed the gauge";
      if (a.summary) st.lines.push({ html: esc(a.summary) });
      st.open = false;
      break;
    default:
      st.icon = icon("check", "c-muted");
      st.title = TOOL_NAME[st.tool] || st.tool;
  }
}

function stepHTML(st) {
  const hasBody = (st.lines && st.lines.length) || (st.images && st.images.length);
  const open = S.expandAll != null ? S.expandAll : (st.id in S.open ? S.open[st.id] : (st.running || st.open));
  const lines = st.lines || [];
  const imgLine = st.images && st.images.length ? `<div class="ln"><span class="ln-no">${lines.length + 1}</span><span class="ln-txt">${thumbs(st.images, st.imgCls || "")}</span></div>` : "";
  return `<details class="step${st.minor ? " minor" : ""}" data-id="${esc(st.id)}" ${open && hasBody ? "open" : ""}>
    <summary>${hasBody ? icon("chevron-right", "chev") : '<span class="chev-space"></span>'}${st.icon}<span class="step-title">${esc(st.title)}</span>${st.label ? `<span class="lbl">${esc(st.label)}</span>` : ""}<span class="step-dur">${esc(st.dur || "")}</span></summary>
    ${hasBody ? `<div class="step-body">${lines.map((l, i) => `<div class="ln ${l.cls || ""}"><span class="ln-no">${i + 1}</span><span class="ln-txt">${l.html}</span></div>`).join("")}${imgLine}</div>` : ""}
  </details>`;
}

function drawJob(snap, gid) {
  const content = $("#run-content");
  const g = snap.gauges.find((x) => x.spec.id === gid);
  if (!g) { setHTML(content, `<div class="flash flash-error">${icon("alert")}<div class="flash-text"><strong>No gauge ${esc(gid)} in this round.</strong></div></div>`); return; }
  const spec = g.spec, oc = g.outcome, gs = gaugeState(g, snap);
  const rd = g.recorded || g.latest;
  if (!$("#job-shell") || $("#job-shell").dataset.gid !== gid) {
    content.__html = null;
    content.innerHTML = `<div id="job-shell" data-gid="${esc(gid)}">
      <div class="job-head" id="job-head"></div>
      <div class="box job-summary" id="job-summary"></div>
      <div class="log"><div class="log-head"><h3>Steps<span class="log-sub" id="log-sub"></span></h3>
        <div class="log-tools"><button type="button" data-act="expand" aria-pressed="false">Expand all</button><button type="button" data-act="collapse">Collapse all</button><button type="button" data-act="llm" aria-pressed="${S.showLLM}">Show model calls</button></div></div>
        <div class="steps" id="steps"></div></div></div>`;
  }
  const pendingRev = snap.reviews.find((r) => r.gauge_id === gid && r.status === "pending");
  const pendingWO = snap.work_orders.find((w) => w.gauge_id === gid && w.status === "pending_approval");
  setHTML($("#job-head"), `<div><h2>${stIcon(gs)}${esc(spec.id)} <span class="gname">${esc(spec.name)}</span></h2>
    <p>${gs === "inspecting" ? `${spinner()} Inspecting now: ${oc.captures} capture${oc.captures === 1 ? "" : "s"} so far` : oc.finished ? `${esc(stText(gs))}. ${oc.captures} capture${oc.captures === 1 ? "" : "s"} in ${esc(dur(oc.robot_time_s))} of robot time, ${oc.compute_ms >= 1000 ? `${(oc.compute_ms / 1000).toFixed(1)} s` : `${Math.round(oc.compute_ms)} ms`} of OpenCV compute${oc.agent ? `, inspected by the ${oc.agent === "llm" ? "LLM agent" : oc.agent === "policy-fallback" ? "policy (LLM fallback)" : "policy"}` : ""}.` : "Not inspected yet."}</p></div>
`);
  const latestOvl = (g.recorded && g.recorded.overlay_id) || (g.latest && g.latest.overlay_id);
  const recordedLine = g.recorded
    ? `<span class="big">${esc(fmt(g.recorded.value, spec.span))}</span> ± ${esc(pm(2 * g.recorded.sigma))} ${esc(spec.units)} <span class="muted">(2σ)</span>`
    : g.latest ? `<span class="muted">Not recorded. Best attempt ${esc(fmt(g.latest.value, spec.span))} ± ${esc(pm(2 * g.latest.sigma))} ${esc(spec.units)}, confidence ${esc(Number(g.latest.confidence).toFixed(2))}</span>` : `<span class="muted">No reading yet</span>`;
  setHTML($("#job-summary"), `<div>${dialSVG(spec, rd, !!g.recorded)}</div>
    <div><dl class="facts">
      <dt>Reading</dt><dd>${recordedLine}</dd>
      <dt>Status</dt><dd>${stIcon(gs)} ${esc(stText(gs))}</dd>
      ${rd ? `<dt>Confidence</dt><dd>${esc(Number(rd.confidence).toFixed(2))} <span class="muted">(0.6 needed to record)</span></dd><dt>Method</dt><dd>${esc(rd.method)}</dd>` : ""}
      <dt>Range</dt><dd>${esc(spec.min_value)} to ${esc(spec.max_value)} ${esc(spec.units)}</dd>
      <dt>Limits</dt><dd>${esc(limitsText(spec))}</dd>
      <dt>Location</dt><dd>${esc(spec.location || "")}, ${esc(spec.criticality)} criticality</dd>
      ${oc.work_order_id ? `<dt>Work order</dt><dd>${esc(oc.work_order_id)}</dd>` : ""}
    </dl>
    ${pendingRev || pendingWO ? `<div class="job-actions">${pendingWO ? `<button type="button" class="btn-primary" data-act="review-wo">Review work order</button>` : ""}${pendingRev ? `<button type="button" class="btn" data-act="review-readings">${icon("person")}Review this reading</button>` : ""}</div>` : ""}</div>
    <div class="ev-col">${latestOvl ? `<figure class="ev-figure"><button type="button" data-img="${esc(latestOvl)}" aria-label="Open evidence image"><img src="/api/images/${esc(latestOvl)}" alt="Evidence: the camera view with the detected dial rim, and the rectified dial with the needle and matched scale marks"></button>
      <figcaption>${g.recorded ? "Evidence for the recorded reading" : "Latest attempt"}: left, the camera view with the detected rim; right, the rectified dial with the needle line, matched scale marks, and clipped glare in magenta.</figcaption></figure>`
      : `<p class="muted">Evidence images appear after the first reading.</p>`}</div>`);
  const steps = buildSteps(S.run.trace, gid, spec);
  const nc = steps.filter((s) => s.tool === "capture").length, nr = steps.filter((s) => s.tool === "read_gauge").length;
  $("#log-sub").textContent = steps.length ? `${nc} capture${nc === 1 ? "" : "s"}, ${nr} reading${nr === 1 ? "" : "s"}` : "";
  const box = $("#steps");
  if (!steps.length) { setHTML(box, `<div class="log-empty">Waiting for the robot to reach this gauge.</div>`); return; }
  if (box.querySelector(".log-empty")) box.innerHTML = "";
  const want = new Set(steps.map((s) => s.id));
  for (const el of $$(":scope > .step-wrap", box)) if (!want.has(el.dataset.id)) el.remove();
  let appended = null;
  steps.forEach((st, i) => {
    let w = box.querySelector(`:scope > .step-wrap[data-id="${CSS.escape(st.id)}"]`);
    if (!w) {
      w = document.createElement("div");
      w.className = "step-wrap";
      w.dataset.id = st.id;
      const before = box.children[i] || null;
      box.insertBefore(w, before);
      appended = w;
    }
    setHTML(w, stepHTML(st));
  });
  if (appended && S.follow) appended.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

document.addEventListener("toggle", (e) => {
  const d = e.target;
  if (!(d instanceof HTMLDetailsElement) || !d.classList.contains("step")) return;
  S.open[d.dataset.id] = d.open;
}, true);

/* ------------------------------------------------------------------ dialogs */
function openDialog(title, body, footer, wide) {
  const dlg = $("#dlg");
  $("#dlg-title").textContent = title;
  $("#dlg-body").innerHTML = body;
  $("#dlg-footer").innerHTML = footer || "";
  dlg.style.width = wide ? "min(820px, calc(100vw - 32px))" : "";
  hydrateIcons(dlg);
  if (!dlg.open) dlg.showModal();
  return dlg;
}
function closeDialog() { const d = $("#dlg"); if (d.open) d.close(); }
document.addEventListener("click", (e) => {
  if (e.target.closest("[data-close]")) { const d = e.target.closest("dialog"); if (d) d.close(); }
});
for (const id of ["dlg", "lightbox"]) {
  document.getElementById(id).addEventListener("click", (e) => { if (e.target.id === id) e.currentTarget.close(); });
}

function nameField() {
  return `<div class="form-group"><label for="d-name">Your name</label><input class="form-control" id="d-name" value="${esc(S.reviewer)}" autocomplete="name"><p class="note">Saved in this browser and recorded with your decision.</p></div>`;
}
function saveName() {
  const v = ($("#d-name") && $("#d-name").value.trim()) || S.reviewer;
  S.reviewer = v.slice(0, 60);
  try { localStorage.setItem("argus-reviewer", S.reviewer); } catch (_) { /* storage blocked */ }
  drawWho();
}

function reviewWorkOrders() {
  const snap = S.run.snap;
  const pend = snap.work_orders.filter((w) => w.status === "pending_approval");
  if (!pend.length) return;
  const body = `<p class="muted" style="margin-top:0">Approved work orders are sent to maintenance through Amazon SNS. Select the ones to decide on.</p>
    ${pend.map((w) => `<label class="check-row"><input type="checkbox" name="wo" value="${esc(w.id)}" checked>
      <span><span class="cr-title"><span class="label ${w.priority <= "P2" ? "label-danger" : "label-attention"}">${esc(w.priority)}</span>${esc(w.title)}</span>
      <span class="cr-msg">${esc(w.description)}</span>${w.priority_note ? `<span class="cr-msg">${icon("shield-check")} ${esc(w.priority_note)}</span>` : ""}${thumbs(w.evidence, "ovl")}</span></label>`).join("")}
    <div class="form-group"><label for="d-note">Leave a comment <span class="muted">(optional)</span></label><textarea class="form-control" id="d-note" rows="2"></textarea></div>
    ${nameField()}<p class="form-error" id="d-err" role="alert"></p>`;
  const footer = `<button type="button" class="btn btn-danger" data-wo="reject">Reject</button><button type="button" class="btn-primary" data-wo="approve">Approve and dispatch</button>`;
  openDialog("Review work orders", body, footer, true);
}
async function decideWorkOrders(action) {
  const ids = $$("#dlg-body input[name=wo]:checked").map((x) => x.value);
  if (!ids.length) { $("#d-err").textContent = "Select at least one work order."; return; }
  saveName();
  const note = $("#d-note").value.trim();
  try {
    for (const id of ids) {
      await api(`/api/rounds/${encodeURIComponent(S.run.id)}/workorders/${encodeURIComponent(id)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action, reviewer: S.reviewer, note }) });
    }
    closeDialog();
    await refreshRun();
  } catch (err) { $("#d-err").textContent = err.message; }
}

function reviewReadings() {
  const snap = S.run.snap;
  const pend = snap.reviews.filter((r) => r.status === "pending");
  if (!pend.length) return;
  const body = `<p class="muted" style="margin-top:0">For each gauge, confirm the robot's best attempt, enter the value you read, or ask for someone to check on site.</p>
    ${pend.map((r) => {
      const g = snap.gauges.find((x) => x.spec.id === r.gauge_id) || {};
      const spec = g.spec || { units: "", span: 100 };
      const best = g.latest;
      return `<div class="review-card" data-rev="${esc(r.id)}"><h3>${icon("person", "c-done")}${esc(r.gauge_id)} <span class="muted" style="font-weight:400">${esc(spec.name || "")}</span></h3>
        <p>${esc(r.reason)}</p>${thumbs(r.evidence)}
        <div class="review-actions">
          ${r.reading_id && best ? `<button type="button" class="btn" data-rev-act="confirm">Confirm ${esc(fmt(best.value, spec.span))} ${esc(spec.units)}</button>` : ""}
          <label class="sr-only" for="v-${esc(r.id)}">Your reading in ${esc(spec.units)}</label>
          <input class="form-control" id="v-${esc(r.id)}" inputmode="decimal" placeholder="Your reading (${esc(spec.units)})">
          <button type="button" class="btn" data-rev-act="correct">Save my reading</button>
          <button type="button" class="btn btn-danger" data-rev-act="reject">Send someone to check</button>
        </div></div>`;
    }).join("")}
    ${nameField()}<p class="form-error" id="d-err" role="alert"></p>`;
  openDialog("Review readings", body, "", true);
}
async function decideReading(card, action) {
  const id = card.dataset.rev;
  const payload = { action, reviewer: S.reviewer };
  if (action === "correct") {
    const inp = card.querySelector("input");
    const v = parseFloat(inp.value);
    if (!isFinite(v)) { inp.focus(); $("#d-err").textContent = "Enter the value you read on the gauge."; return; }
    payload.value = v;
  }
  saveName();
  payload.reviewer = S.reviewer;
  try {
    await api(`/api/rounds/${encodeURIComponent(S.run.id)}/reviews/${encodeURIComponent(id)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    card.remove();
    await refreshRun();
    if (!$("#dlg-body .review-card")) closeDialog();
  } catch (err) { $("#d-err").textContent = err.message; }
}
async function refreshRun() {
  if (!S.run) return;
  clearTimeout(S.run.timer);
  await pollRun();
}

async function showRunner() {
  let rt = S.rt;
  try { rt = await api("/api/runtime"); S.rt = rt; } catch (_) { /* keep cached */ }
  if (!rt) return;
  const cpu = rt.cpu || {}, ec2 = rt.ec2 || {}, b = rt.build || {};
  const rows = [
    ["OpenCV", `${esc(rt.opencv_version)}${rt.opencv_major >= 5 ? "" : " (OpenCV 5 required)"}`],
    ["Build", rt.cool ? `${icon("check-circle-fill", "c-success")} Cloud-Optimized OpenCV Library (COOL)` : "Stock OpenCV build"],
    ["Loaded from", `<span class="mono">${esc(rt.opencv_path)}</span>`],
    ["KleidiCV in build", b.kleidicv ? "Yes" : "No"],
    ["CPU", `${esc(cpu.arm_core || cpu.model_name || "")} <span class="muted">${esc(cpu.machine || "")}, ${esc(cpu.cpus)} vCPUs${cpu.sve ? ", SVE" : ""}</span>`],
    ["EC2 instance", ec2["instance-type"] ? `${esc(ec2["instance-type"])} <span class="muted">${esc(ec2.region || "")}</span>` : '<span class="muted">not on EC2</span>'],
    ["Marketplace product code", (ec2.marketplace_product_codes || []).length ? `<span class="mono">${esc(ec2.marketplace_product_codes.join(", "))}</span>` : '<span class="muted">none</span>'],
    ["Python", esc(rt.python)],
    ["Face blurring (YuNet DNN)", rt.privacy_face_detector ? "Available" : "Unavailable"],
  ];
  const body = `<p class="muted" style="margin-top:0">This is read from the running process, so it shows which OpenCV build actually does the vision work.</p>
    <table class="kv">${rows.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${v}</td></tr>`).join("")}</table>
    ${(b.highlights || []).length ? `<details style="margin-top:12px"><summary>Build information</summary><pre class="mono" style="white-space:pre-wrap">${esc(b.highlights.join("\n"))}</pre></details>` : ""}`;
  openDialog("Runner", body, `<a class="btn" href="/api/runtime">${icon("file")}Raw JSON</a>`);
}

function showWho() {
  openDialog("Your name", `${nameField()}`, `<button type="button" class="btn-primary" data-act="save-name">Save</button>`);
  setTimeout(() => { const i = $("#d-name"); if (i) { i.focus(); i.select(); } }, 30);
}
function drawWho() {
  const parts = S.reviewer.trim().split(/\s+/);
  const ini = (parts.length > 1 ? parts[0][0] + parts[1][0] : S.reviewer.slice(0, 2)).toUpperCase();
  $("#who-initials").textContent = ini || "?";
  $("#who-btn").title = `Signed as ${S.reviewer} on approvals`;
}

/* global action buttons */
document.addEventListener("click", (e) => {
  const imgBtn = e.target.closest("[data-img],[data-img-url]");
  if (imgBtn) {
    const url = imgBtn.dataset.img ? `/api/images/${imgBtn.dataset.img}` : imgBtn.dataset.imgUrl;
    $("#lightbox-img").src = url;
    $("#lightbox").showModal();
    return;
  }
  const wo = e.target.closest("[data-wo]");
  if (wo) { decideWorkOrders(wo.dataset.wo); return; }
  const ra = e.target.closest("[data-rev-act]");
  if (ra) { decideReading(ra.closest("[data-rev]"), ra.dataset.revAct); return; }
  const b = e.target.closest("[data-act]");
  if (!b) return;
  const act = b.dataset.act;
  if (act === "runner") { e.preventDefault(); showRunner(); }
  else if (act === "review-wo") reviewWorkOrders();
  else if (act === "review-readings") reviewReadings();
  else if (act === "save-name") { saveName(); closeDialog(); }
  else if (act === "follow") { S.follow = true; refreshRun(); }
  else if (act === "unfollow") { S.follow = false; drawRun(false); }
  else if (act === "rerun") rerun();
  else if (act === "expand" || act === "collapse") {
    S.open = {};
    S.expandAll = act === "expand";
    drawJob(S.run.snap, S.route.gid);
    S.expandAll = null;
    for (const d of $$("#steps details.step")) S.open[d.dataset.id] = d.open;
  } else if (act === "llm") {
    S.showLLM = !S.showLLM;
    b.setAttribute("aria-pressed", String(S.showLLM));
    $("#steps").innerHTML = "";
    drawJob(S.run.snap, S.route.gid);
  }
});
document.addEventListener("click", (e) => {
  // leaving follow mode when the user picks a gauge or the summary themselves
  const a = e.target.closest("#run-side a.side-item, .graph a.node, .ann-list a");
  if (a) S.follow = false;
});
async function rerun() {
  const snap = S.run && S.run.snap;
  if (!snap) return;
  try {
    const r = await api("/api/rounds", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ profile: snap.profile, agent: snap.agent_mode === "llm" && S.cfg.llm_enabled ? "llm" : "policy", seed: snap.seed }) });
    S.follow = true;
    go(`#/rounds/${encodeURIComponent(r.id)}`);
  } catch (err) {
    openDialog("Could not start the round", `<p>${esc(err.message)}</p>`, `<button type="button" class="btn" data-close>Close</button>`);
  }
}

/* ------------------------------------------------------------------ photo */
const GAUGE_PRESETS = [
  ["0 to 16 bar", 0, 16, "bar", 9], ["0 to 10 bar", 0, 10, "bar", 11], ["0 to 160 psi", 0, 160, "psi", 9],
  ["0 to 100 °C", 0, 100, "°C", 11], ["-1 to 3 bar", -1, 3, "bar", 9],
];
function renderPhoto() {
  $("#main").innerHTML = `<div class="container"><div class="page-head"><div><h1>Read a gauge from a photo</h1>
      <p>No tag and no robot: one photo of a round analog gauge. ARGUS finds the dial, measures the needle, and tells you whether the reading can be trusted, or how to retake the photo.</p></div></div>
    <div class="photo-layout">
      <form class="box" id="photo-form"><div class="box-header">Photo and scale</div><div class="box-body">
        <div class="dropzone" id="dropzone" tabindex="0" role="button" aria-label="Choose a photo">
          ${icon("image")}<p><strong>Drag a photo here</strong> or <span class="c-accent">choose a file</span></p><p class="note">JPEG or PNG, up to 8 MB</p>
          <div class="file-name" id="file-name"></div>
          <input type="file" id="file" name="file" accept="image/jpeg,image/png" capture="environment" hidden required>
        </div>
        <div class="form-group"><label for="preset">Common scales</label>
          <select class="form-control" id="preset"><option value="">Custom</option>${GAUGE_PRESETS.map((p, i) => `<option value="${i}" ${i === 0 ? "selected" : ""}>${esc(p[0])}</option>`).join("")}</select></div>
        <div class="grid-2">
          <div class="form-group"><label for="min">Scale minimum</label><input class="form-control" id="min" name="min_value" inputmode="decimal" value="0" required></div>
          <div class="form-group"><label for="max">Scale maximum</label><input class="form-control" id="max" name="max_value" inputmode="decimal" value="16" required></div>
          <div class="form-group"><label for="units">Units</label><input class="form-control" id="units" name="units" value="bar"></div>
          <div class="form-group"><label for="ticks">Numbered marks</label><input class="form-control" id="ticks" name="major_ticks" inputmode="numeric" value="9"><p class="note">Lets ARGUS check the scale and the photo's rotation.</p></div>
        </div>
        <details><summary class="c-accent" style="cursor:pointer">More settings</summary><div style="margin-top:12px">
          <div class="grid-2">
            <div class="form-group"><label for="tmin">Minimum mark angle</label><input class="form-control" id="tmin" name="theta_min" inputmode="decimal" value="-135"></div>
            <div class="form-group"><label for="tmax">Maximum mark angle</label><input class="form-control" id="tmax" name="theta_max" inputmode="decimal" value="135"></div>
          </div><p class="note" style="margin:-8px 0 12px">Degrees clockwise from 12 o'clock; most gauges run from -135 (7:30) to 135 (4:30).</p>
          <div class="form-group"><label for="ncol">Needle colour</label><select class="form-control" id="ncol" name="needle_color"><option value="black">Black</option><option value="red">Red</option></select></div>
        </div></details>
        <button type="submit" class="btn-primary" style="margin-top:16px">Analyze photo</button>
        <p class="note" style="margin-top:12px">${icon("shield-check")} Faces are blurred with OpenCV's face detector before anything else happens, and the photo is not stored.</p>
      </div></form>
      <section class="box" id="photo-result" aria-live="polite"><div class="box-header">Result</div><div class="blankslate">${icon("meter")}<h3>No photo yet</h3><p>Choose a photo of a gauge. The annotated dial, the reading with its uncertainty, and retake advice appear here.</p></div></section>
    </div></div>`;
  const dz = $("#dropzone"), file = $("#file");
  const showName = () => { $("#file-name").textContent = file.files[0] ? file.files[0].name : ""; };
  dz.addEventListener("click", () => file.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    if (e.dataTransfer.files.length) { const dt = new DataTransfer(); dt.items.add(e.dataTransfer.files[0]); file.files = dt.files; showName(); }
  });
  file.addEventListener("change", showName);
  $("#preset").addEventListener("change", (e) => {
    const p = GAUGE_PRESETS[e.target.value];
    if (!p) return;
    $("#min").value = p[1]; $("#max").value = p[2]; $("#units").value = p[3]; $("#ticks").value = p[4];
  });
  $("#photo-form").addEventListener("submit", analyzePhoto);
}
async function analyzePhoto(ev) {
  ev.preventDefault();
  const out = $("#photo-result");
  if (!$("#file").files.length) { out.innerHTML = `<div class="box-header">Result</div><div class="box-body"><p class="form-error">Choose a photo first.</p></div>`; return; }
  const fd = new FormData(ev.target);
  if (!fd.get("major_ticks")) fd.set("major_ticks", "0");
  out.innerHTML = `<div class="box-header">Result</div><div class="box-body"><p class="muted">${spinner()} Measuring</p></div>`;
  try {
    const r = await api("/api/analyze", { method: "POST", body: fd });
    const span = Number(fd.get("max_value")) - Number(fd.get("min_value"));
    out.innerHTML = `<div class="box-header"><span>Result</span><span class="muted">${esc(Math.round((r.timings_ms || {}).total || 0))} ms of OpenCV processing</span></div><div class="box-body">
      ${r.ok ? `<p class="result-value">${esc(fmt(r.value, span))} <small>± ${esc(pm(2 * r.sigma))} ${esc(r.units)} (2σ)</small></p>` : `<p class="result-value">No reading</p>`}
      <p style="margin:8px 0">${r.accepted ? `<span class="label label-success">${icon("check")} Trustworthy</span>` : `<span class="label label-attention">${icon("alert")} Not trustworthy yet</span>`} <span class="muted">confidence ${r.confidence != null ? esc(Number(r.confidence).toFixed(2)) : "—"}, needs 0.6</span></p>
      <p style="margin:0 0 12px">${esc(r.decision)}</p>
      ${r.guidance && r.guidance.length ? `<div class="flash flash-warn">${icon("light-bulb")}<div class="flash-text"><strong>How to retake it</strong><ul class="guidance">${r.guidance.map((g) => `<li>${esc(g)}</li>`).join("")}</ul></div></div>` : ""}
      <figure class="ev-figure"><button type="button" data-img-url="${esc(r.overlay)}" aria-label="Open annotated photo"><img src="${esc(r.overlay)}" alt="Annotated photo: detected dial rim, needle line and matched scale marks"></button>
      <figcaption>${esc(r.privacy)}${r.faces_blurred > 0 ? `; ${esc(r.faces_blurred)} face(s) blurred` : ""}. Dial found from ${esc(({ fiducial: "the gauge's tag", predicted: "the robot's pose", circle: "its round outline (no tag)" })[r.localization] || "nothing")}${r.scale_verified ? ", printed scale verified" : ""}.</figcaption></figure></div>`;
    hydrateIcons(out);
  } catch (e) {
    out.innerHTML = `<div class="box-header">Result</div><div class="box-body"><div class="flash flash-error">${icon("alert")}<div class="flash-text"><strong>The photo could not be analyzed.</strong>${esc(e.message)}</div></div></div>`;
  }
}

/* ------------------------------------------------------------------ evidence */
const STRAT = { overview: "Wide shot of the panel", fixed: "Fixed camera, accept every value", fixed_gated: "Fixed camera with uncertainty gating", agent_policy: "ARGUS policy agent", agent_llm: "ARGUS LLM agent (Bedrock)", agent_policy_same_seeds: "ARGUS policy, same rounds as LLM" };
const METRICS = [
  ["silent_failure_rate", "Wrong value recorded without warning", "pct", "low"],
  ["missed_critical", "Missed alarms or broken gauges", "int", "low"],
  ["false_alarms", "False work orders", "int", "low"],
  ["automation_rate", "Read without a person", "pct", ""],
  ["within_1pct", "Recorded readings within ±1% of span", "pct", "high"],
  ["median_err_pct_span", "Median error, % of span", "num", "low"],
  ["sigma_coverage_2sigma", "True value inside the reported ±2σ", "pct", "high"],
  ["captures_per_gauge", "Captures per gauge", "num", ""],
  ["robot_s_per_gauge", "Robot seconds per gauge", "num", ""],
  ["compute_ms_per_gauge", "OpenCV compute per gauge (ms)", "num", ""],
];
function cell(v, kind) {
  if (v == null) return "—";
  if (kind === "pct") return (v * 100).toFixed(1) + "%";
  if (kind === "int") return String(v);
  return Number(v).toFixed(Math.abs(v) < 10 ? 2 : 1);
}
async function renderEvidence() {
  $("#main").innerHTML = `<div class="container"><article class="markdown-body" id="ev"><h1>Evidence</h1><p class="muted">Loading results</p></article></div>`;
  let ev;
  try { ev = await api("/api/evidence"); } catch (e) { setHTML($("#ev"), `<h1>Evidence</h1><p>${esc(e.message)}</p>`); return; }
  let h = `<h1>Evidence</h1>
    <div class="md-alert"><p class="md-alert-title">${icon("info")}How these numbers were made</p><p style="margin:0">Accuracy and safety are measured against known true values in a simulated plant; COOL performance is measured on the deployed EC2 instances. Every number comes from a script in the repository, so it can be reproduced.</p></div>`;
  if (ev.eval && ev.eval.summary) {
    const S2 = ev.eval.summary, keys = Object.keys(S2), m = ev.eval.meta || {};
    h += `<h2>Accuracy and safety</h2><p>${esc(S2[keys[0]].gauges)} gauge inspections per approach across ${esc(m.scenes)} simulated rounds (seeds ${esc(m.seed0)} to ${esc((m.seed0 || 0) + (m.scenes || 0) - 1)}), with glare, night, vibration, oblique mounting, occlusion, haze and broken gauges mixed in.</p>
      <table><thead><tr><th>Measure</th>${keys.map((k) => `<th>${esc(STRAT[k] || k)}</th>`).join("")}</tr></thead><tbody>`;
    for (const [k, label, kind, better] of METRICS) {
      const vals = keys.map((s) => S2[s][k]);
      const valid = vals.filter((v) => v != null);
      const best = better === "low" ? Math.min(...valid) : better === "high" ? Math.max(...valid) : null;
      h += `<tr><td>${esc(label)}</td>${vals.map((v) => `<td class="${best != null && v === best ? "best" : ""}">${cell(v, kind)}</td>`).join("")}</tr>`;
    }
    h += `</tbody></table><p class="caption">Reproduce with <code>python -m eval.run_eval --scenes ${esc(m.scenes)}</code>. Ran on ${esc(m.runtime ? m.runtime.label : "")}.</p>
      <img class="figure" src="/api/evidence/strategies.png" alt="Bar charts of silent failures, automatic reads and missed alarms per approach">`;
    if (ev.eval.by_condition) {
      const C = ev.eval.by_condition, cols = keys.filter((k) => k !== "overview");
      h += `<h3>By condition</h3><p>Share read without a person, and wrong values recorded without warning.</p><table><thead><tr><th>Condition</th>${cols.map((k) => `<th>${esc(STRAT[k] || k)}</th>`).join("")}</tr></thead><tbody>`;
      for (const [c, row] of Object.entries(C)) {
        if (!Object.keys(row).length) continue;
        h += `<tr><td>${esc(c)} (${esc((row[cols[0]] || {}).n || "")})</td>${cols.map((k) => (row[k] ? `<td>${(row[k].automated * 100).toFixed(0)}%, ${row[k].silent} wrong</td>` : "<td>—</td>")).join("")}</tr>`;
      }
      h += `</tbody></table>`;
    }
  } else {
    h += `<h2>Accuracy and safety</h2><p>No evaluation results on this server yet. Run <code>python -m eval.run_eval</code>.</p>`;
  }
  const B = ev.bench;
  h += `<h2>COOL on AWS Graviton</h2>`;
  if (B && B.configs && B.configs.length) {
    h += `<p>The full ARGUS vision pipeline on identical images, timed on each build.</p><table><thead><tr><th>Configuration</th><th>OpenCV</th><th>Median per read (ms)</th><th>p95 (ms)</th><th>Reads per second, all cores</th><th>Cost per 1,000 reads</th><th>vs. stock, same instance</th></tr></thead><tbody>`;
    for (const c of B.configs) {
      h += `<tr><td>${esc(c.label)} <span class="muted">${esc(c.instance_type || "")}</span></td><td>${esc(c.opencv_version)}${c.cool ? " COOL" : ""}</td><td>${cell(c.pipeline.median_ms)}</td><td>${cell(c.pipeline.p95_ms)}</td><td>${cell(c.pipeline.throughput_rps_all)}</td><td>${c.cost_per_1k_reads_usd != null ? "$" + Number(c.cost_per_1k_reads_usd).toFixed(5) : "—"}</td><td class="${c.cool ? "best" : ""}">${c.speedup_vs_stock_same_instance ? Number(c.speedup_vs_stock_same_instance).toFixed(2) + "×" : "—"}</td></tr>`;
    }
    h += `</tbody></table><p class="caption">${esc(B.note || "")}</p><img class="figure" src="/api/evidence/bench.png" alt="Per-operation speed-ups of COOL over stock OpenCV 5">`;
  } else {
    h += `<p>Benchmark results appear here after <code>deploy/run_bench.sh</code> runs on the deployed Graviton4 instance.</p>`;
  }
  setHTML($("#ev"), h);
  for (const img of $$("#ev img.figure")) img.addEventListener("error", () => img.remove());
}

/* ------------------------------------------------------------------ how it works */
function renderHow() {
  $("#main").innerHTML = `<div class="container"><article class="markdown-body">
    <h1>How ARGUS works</h1>
    <p>Plants still have thousands of analog gauges that people walk around and read. Robots can photograph them, but a single photo fails silently under glare, pump vibration, night-time noise, oblique mounting and things hanging in front of the dial. ARGUS measures, judges its own evidence, changes how it looks, and hands off to a person when it still cannot trust what it sees.</p>
    <h2>For every gauge</h2>
    <ol>
      <li><strong>Look.</strong> The robot points its PTZ camera at the gauge and takes one to nine frames, or a three-shot HDR bracket.</li>
      <li><strong>Measure.</strong> OpenCV 5, running on AWS Graviton with the Cloud-Optimized OpenCV Library (COOL), finds the gauge's ArUco tag, rectifies the dial with a homography, re-centres on the needle's pivot, unwraps the dial to polar coordinates to find the needle, and checks the printed scale against the expected tick pattern.</li>
      <li><strong>Judge the evidence.</strong> The same pass lists what could be wrong: glare on the needle, flutter, blur, noise, an oblique view, something in front of the dial. Each issue lowers confidence, and every reading carries a ±2σ interval.</li>
      <li><strong>Decide.</strong> The agent (Claude on Amazon Bedrock, or the deterministic policy) picks the next action from what it saw: step sideways to move a reflection, bracket exposures, freeze vibration with a fast shutter and a burst, turn the light on or off, or stop and escalate.</li>
      <li><strong>Act safely.</strong> A safety guard checks every call: only measured values can be recorded, low-confidence or limit-straddling readings cannot be, alarms must get a work order, and only a person can dispatch one.</li>
    </ol>
    <h2>Architecture</h2>
    <img class="figure" src="/static/architecture.svg" alt="Architecture: CloudFront to an EC2 Graviton4 instance running COOL OpenCV 5 and the ARGUS service; Amazon Bedrock, S3, CloudWatch, SNS and Systems Manager around it." loading="lazy">
    <p class="caption">The vision workload runs on the Arm (Graviton4) instance with COOL. Approved work orders leave through Amazon SNS.</p>
    <h2>Agent loop</h2>
    <img class="figure" src="/static/agent_workflow.svg" alt="Agent workflow: perceive with OpenCV, decide, then re-plan, record or escalate; a safety guard checks every call; people confirm readings and approve work orders." loading="lazy">
    <p class="caption">Every arrow is a logged event: open any round and download its trace.</p>
  </article></div>`;
}

/* ------------------------------------------------------------------ boot */
function hydrateIcons(root = document) {
  for (const el of $$("[data-icon]", root)) { el.outerHTML = icon(el.dataset.icon); }
}
async function loadRuntime() {
  try {
    const rt = await api("/api/runtime");
    S.rt = rt;
    const cpu = rt.cpu || {}, ec2 = rt.ec2 || {};
    const where = ec2["instance-type"] ? `${ec2["instance-type"]}${cpu.arm_core ? `, ${cpu.arm_core.replace(/^.*\((.*)\)$/, "$1")}` : ""}` : (cpu.arm_core || cpu.machine || "");
    $("#runner-label").textContent = `${where}, OpenCV ${rt.opencv_version} ${rt.cool ? "COOL" : "stock build"}`;
    $("#runner-dot").classList.toggle("on", !!rt.cool);
    $("#runner-btn").title = "Which OpenCV build is running, and where";
  } catch (_) {
    $("#runner-label").textContent = "Runtime unavailable";
  }
}
async function boot() {
  hydrateIcons();
  try { S.reviewer = localStorage.getItem("argus-reviewer") || "operator"; } catch (_) { /* storage blocked */ }
  drawWho();
  $("#runner-btn").addEventListener("click", showRunner);
  $("#who-btn").addEventListener("click", showWho);
  try { S.cfg = await api("/api/config"); } catch (_) { S.cfg = { profiles: {}, llm_enabled: false }; }
  loadRuntime();
  api("/api/rounds").then((r) => { S.rounds = r; updateNavCount(); }).catch(() => {});
  window.addEventListener("popstate", () => route());
  window.addEventListener("hashchange", () => route());
  route(true);
}
boot();
