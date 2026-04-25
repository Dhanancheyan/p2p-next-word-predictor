"use strict";

// -- Utilities ----------------------------------------------------------------

const $ = (s) => document.querySelector(s);

async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// -- App state ----------------------------------------------------------------

const state = {
  page: "editor",
  modelId: "default",
  models: [],
  sessionId: null,
  suggestions: [],
  selectedIndex: 0,
  popupVisible: false,
  lastLatencyMs: 0,
  wsConnected: false,
  idleTimeoutMs: 30_000,
  idleTimer: null,
  suppressPredictOnce: false,
  modelVersions: { local: 0, global: 0 },
  backendAvailable: true,
};

// -- Editor helpers -----------------------------------------------------------

function editorEl() { return $("#editor"); }
function getEditorText() { return editorEl().innerText ?? ""; }
function getCaretPos() {
  const el = editorEl(), sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return 0;
  const range = sel.getRangeAt(0);
  if (!el.contains(range.endContainer)) return 0;
  const pre = range.cloneRange();
  pre.selectNodeContents(el); pre.setEnd(range.endContainer, range.endOffset);
  return pre.toString().length;
}
function getCaretRect() {
  const el = editorEl(), sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return null;
  const range = sel.getRangeAt(0).cloneRange();
  if (!el.contains(range.endContainer)) return null;
  range.collapse(true);
  const rects = range.getClientRects();
  if (rects && rects.length > 0) return rects[rects.length - 1];
  const rect = range.getBoundingClientRect();
  return (rect && rect.width + rect.height > 0) ? rect : el.getBoundingClientRect();
}

/**
 * Insert text at the cursor, optionally deleting deleteBack characters behind it.
 * Used to replace a partial word on autocomplete, or insert plainly on next-word.
 */
function insertTextAtCursor(text, deleteBack = 0) {
  const el = editorEl(), sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return;
  const range = sel.getRangeAt(0);
  if (!el.contains(range.endContainer)) return;

  if (deleteBack > 0) {
    const colRange = range.cloneRange();
    const node = colRange.endContainer;
    const offset = colRange.endOffset;
    if (node.nodeType === Node.TEXT_NODE) {
      const deleteStart = Math.max(0, offset - deleteBack);
      colRange.setStart(node, deleteStart);
      colRange.setEnd(node, offset);
    } else {
      for (let i = 0; i < deleteBack; i++) {
        try { colRange.setStart(colRange.startContainer, Math.max(0, colRange.startOffset - 1)); } catch {}
      }
    }
    colRange.deleteContents();
    sel.removeAllRanges(); sel.addRange(colRange);
  } else {
    range.deleteContents();
  }

  const insertRange = sel.getRangeAt(0);
  const node = document.createTextNode(text);
  insertRange.insertNode(node);
  insertRange.setStartAfter(node); insertRange.collapse(true);
  sel.removeAllRanges(); sel.addRange(insertRange); el.focus();
}

// -- Popup --------------------------------------------------------------------

function renderPopup() {
  const popup = $("#autocomplete-popup"), list = $("#suggestion-list");
  if (!state.suggestions.length) { popup.classList.remove("visible"); state.popupVisible = false; return; }
  list.innerHTML = "";
  const maxFreq = Math.max(...state.suggestions.map(s => Math.abs(s.score)), 1);
  const mode = state.suggestions[0]?.mode || "next_word";
  const modeLabel = mode === "autocomplete" ? "complete" : "next word";
  let header = list.querySelector(".popup-mode-label");
  if (!header) { header = document.createElement("div"); header.className = "popup-mode-label"; }
  header.textContent = modeLabel;
  list.appendChild(header);

  const SOURCE_BADGE = {
    "lstm":       { label: "AI",       cls: "src-lstm"  },
    "ngram":      { label: "local",    cls: "src-ngram" },
    "peer-ngram": { label: "peer",     cls: "src-peer"  },
    "cache":      { label: "cache",    cls: "src-cache" },
    "hybrid":     { label: "AI",       cls: "src-lstm"  },
  };

  state.suggestions.forEach((s, idx) => {
    const item = document.createElement("div");
    item.className = "suggestion-item" + (idx === state.selectedIndex ? " highlighted" : "");
    const bar = document.createElement("div"); bar.className = "suggestion-bar";
    bar.style.width = `${Math.round((Math.abs(s.score)/maxFreq)*100)}%`;
    const key = document.createElement("div"); key.className = "suggestion-key"; key.textContent = String(idx+1);
    // Display the word without trailing space -- space is injected on accept.
    const word = document.createElement("div"); word.className = "suggestion-word"; word.textContent = s.text.trim();
    const srcInfo = SOURCE_BADGE[s.source] || { label: s.source || "?", cls: "src-ngram" };
    const badge = document.createElement("span");
    badge.className = "suggestion-src " + srcInfo.cls;
    badge.textContent = srcInfo.label;
    item.appendChild(bar); item.appendChild(key); item.appendChild(word); item.appendChild(badge);
    item.addEventListener("mousedown", (e) => { e.preventDefault(); acceptSuggestion(idx); });
    list.appendChild(item);
  });

  const rect = getCaretRect();
  if (rect) {
    const pw = 300, ph = 240;
    let left = rect.left, top = rect.bottom + 8;
    if (left + pw > window.innerWidth - 8) left = window.innerWidth - pw - 8;
    if (top + ph > window.innerHeight - 8) top = rect.top - ph - 8;
    popup.style.left = `${left}px`; popup.style.top = `${Math.max(8, top)}px`;
  }
  popup.classList.add("visible"); state.popupVisible = true;
}

// -- Logging ------------------------------------------------------------------

let dismissLogged = false;
async function logEvent(type, payload) {
  if (!state.sessionId) return;
  try { await apiPost("/local/logs", { session_id: state.sessionId, ts: Math.floor(Date.now()/1000), type, payload: payload ?? {} }); } catch {}
}
function clearSuggestions(reason = "dismiss") {
  if (state.popupVisible && !dismissLogged && reason === "dismiss") {
    dismissLogged = true;
    logEvent("suggest_dismissed", { latency_ms: state.lastLatencyMs });
  }
  state.suggestions = []; state.selectedIndex = 0; renderPopup();
}

/**
 * Accept a suggestion at index idx.
 *
 * Inserts the accepted word plus a trailing space so the cursor sits after a
 * word boundary, then immediately fires next-word prediction so the suggestion
 * list refreshes without any further keyboard input from the user.
 */
async function acceptSuggestion(idx) {
  const s = state.suggestions[idx]; if (!s) return;
  dismissLogged = true;
  // For autocomplete, deleteBack erases the partial word being typed.
  // For next-word, partial_word is "" so deleteBack is 0 (plain insert).
  const deleteBack = (s.partial_word || "").length;
  // Trailing space positions the cursor after the word boundary and enables
  // immediate next-word prediction without a manual Space keypress.
  insertTextAtCursor(s.text.trim() + " ", deleteBack);
  clearSuggestions("accept");
  await logEvent("suggest_accepted", {
    rank: idx + 1,
    latency_ms: state.lastLatencyMs,
    suggestion_len: s.text.trim().length,
    source: s.source || "hybrid",
    mode: s.mode || "next_word",
    lstm_conf: s.lstm_conf ?? null,
  });
  try { await apiPost("/local/personalization/observe", { model_id: state.modelId, text: s.text.trim() }); } catch {}
  schedulePredict();
}

// -- Prediction ---------------------------------------------------------------

let predictTimer = null;
function schedulePredict() { if (predictTimer) clearTimeout(predictTimer); predictTimer = setTimeout(runPredict, 150); }
async function runPredict() {
  if (state.suppressPredictOnce) { state.suppressPredictOnce = false; return; }
  const text = getEditorText(), cursor = getCaretPos();
  if (!text && cursor === 0) { clearSuggestions("silent"); return; }
  dismissLogged = false;
  const wc = text.trim().split(/\s+/).filter(Boolean).length;
  $("#word-count").textContent = `${wc} word${wc !== 1 ? "s" : ""}`;
  try {
    const res = await apiPost("/local/predict", {
      model_id: state.modelId,
      context_text: text,
      cursor_pos: cursor,
      k: 5,
      max_chars: 24,
    });
    state.lastLatencyMs = res.latency_ms ?? 0;
    state.modelVersions = res.model_versions ?? state.modelVersions;
    updateHeaderVersions();
    const suggestions = (res.suggestions ?? []).filter(x => x && x.text).map(s => ({
      ...s,
      text: (s.text || "").trim(),
      source: s.source || "hybrid",
      mode: s.mode || "next_word",
      partial_word: s.partial_word || "",
      lstm_conf: s.lstm_conf ?? null,
    }));
    state.suggestions = suggestions; state.selectedIndex = 0;
    if (suggestions.length) {
      logEvent("suggest_shown", {
        count: suggestions.length,
        latency_ms: state.lastLatencyMs,
        mode: suggestions[0]?.mode,
      });
    }
    renderPopup();
    state.backendAvailable = true;
  } catch {
    state.backendAvailable = false;
    setTrainerStatus(false);
    clearSuggestions("silent");
  }
}

// -- Sessions -----------------------------------------------------------------

async function startSessionIfNeeded() {
  if (state.sessionId) return;
  try {
    const res = await apiPost("/local/session/start", { model_id: state.modelId });
    state.sessionId = res.session_id;
  } catch {}
}
async function endSession() {
  if (!state.sessionId) return;
  const finalText = getEditorText(), sid = state.sessionId;
  state.sessionId = null; clearSuggestions("silent"); setTrainStatus("queued", "amber");
  try { await apiPost("/local/session/end", { session_id: sid, final_text: finalText }); } catch { setTrainStatus("error", "red"); }
}
function resetIdleTimer() {
  if (state.idleTimer) clearTimeout(state.idleTimer);
  state.idleTimer = setTimeout(endSession, state.idleTimeoutMs);
}

// -- Header -------------------------------------------------------------------

function setTrainerStatus(ok) {
  $("#trainer-status").textContent = ok ? "online" : "offline";
  $("#ind-trainer").className = "indicator " + (ok ? "green" : "red");
  if (!state.backendAvailable && ok) state.backendAvailable = true;
}
function setGossipStatus(text, level = "amber") { $("#gossip-status").textContent = text; $("#ind-gossip").className = "indicator " + level; }
function setTrainStatus(text, level = "amber") { $("#train-status").textContent = text; $("#ind-train").className = "indicator " + level; }
function updateHeaderVersions() {
  $("#ver-local").textContent = String(state.modelVersions.local ?? 0);
  $("#ver-global").textContent = String(state.modelVersions.global ?? 0);
}
function updateHeaderModelName() {
  const m = state.models.find(x => x.model_id === state.modelId);
  $("#active-model-name").textContent = m ? m.name : state.modelId;
  $("#popup-model-tag").textContent = m ? m.name : state.modelId;
}

// -- Models -------------------------------------------------------------------

function _populateSel(sel, models, curVal) {
  if (!sel) return;
  const prev = sel.value || curVal;
  sel.innerHTML = "";
  models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.model_id;
    opt.textContent = m.name + (m.enabled ? "" : " (disabled)");
    sel.appendChild(opt);
  });
  if (models.find(m => m.model_id === prev)) sel.value = prev;
  else if (models.length) sel.value = models[0].model_id;
}
async function refreshModels() {
  try { const res = await apiGet("/local/models/list"); state.models = res.models ?? []; } catch { return; }
  const enabled = state.models.filter(m => m.enabled);
  _populateSel($("#model-select"), enabled, state.modelId);
  if (!state.models.find(m => m.model_id === state.modelId && m.enabled) && enabled.length) {
    state.modelId = enabled[0].model_id; $("#model-select").value = state.modelId;
  }
  _populateSel($("#metrics-model-select"), state.models, state.models[0]?.model_id);
  _populateSel($("#upload-target-model"), state.models, state.models[0]?.model_id);
  _populateSel($("#pers-model-select"), state.models, state.modelId);
  renderModelRegistry(); updateHeaderModelName();
}
function renderModelRegistry() {
  const list = $("#model-registry-list"); list.innerHTML = "";
  const viewingId = $("#metrics-model-select").value;
  state.models.forEach(m => {
    const item = document.createElement("div");
    item.className = "model-reg-item" + (m.model_id === viewingId ? " active" : "");
    const dot = document.createElement("div"); dot.className = "reg-enabled " + (m.enabled ? "on" : "off");
    const name = document.createElement("div"); name.className = "reg-name"; name.textContent = m.name;
    const info = document.createElement("div"); info.className = "reg-type";
    info.textContent = `${m.ngram_entries ?? 0} hybrid entries  -  ${m.engine || "hybrid"}`;
    item.append(dot, name, info);
    item.addEventListener("click", () => { $("#metrics-model-select").value = m.model_id; renderModelRegistry(); refreshMetrics(); });
    dot.addEventListener("click", async (e) => { e.stopPropagation(); await apiPost("/local/models/update", { model_id: m.model_id, enabled: !m.enabled }); await refreshModels(); });
    list.appendChild(item);
  });
}

// -- Metrics ------------------------------------------------------------------

async function refreshMetrics() {
  const modelId = $("#metrics-model-select").value || state.modelId;
  try {
    const m = await apiGet(`/local/metrics?model_id=${encodeURIComponent(modelId)}`);
    const entries = [
      ["Words typed", m.words_typed ?? 0],
      ["Suggestions accepted", m.accepted ?? 0],
      ["Suggestions dismissed", m.dismissed ?? 0],
      ["Accept rate", `${Math.round((m.accept_rate ?? 0) * 100)}%`],
      ["Avg latency", `${Math.round(m.avg_latency_ms ?? 0)} ms`],
      ["Local version", `v${m.model_versions?.local ?? 0}`],
      ["Global version", `g${m.model_versions?.global ?? 0}`],
      ["Hybrid entries", m.ngram_entries ?? 0],
      ["Cache phrase keys", m.cache_stats?.phrase_keys ?? 0],
      ["Engine", m.engine || "hybrid"],
    ];
    const kv = $("#metrics-summary"); kv.innerHTML = "";
    entries.forEach(([k, v]) => {
      const item = document.createElement("div"); item.className = "kv-item";
      item.innerHTML = `<div class="kv-key">${k}</div><div class="kv-val">${v}</div>`;
      kv.appendChild(item);
    });
    renderNgramChart($("#accuracy-chart"), m);
    await renderNgramStats(modelId);
    const sessions = await apiGet(`/local/sessions?model_id=${encodeURIComponent(modelId)}&limit=25`);
    renderSessions(sessions.sessions ?? []);
    const eb = $("#btn-export");
    if (eb) { eb.href = `/local/ngram/export?model_id=${encodeURIComponent(modelId)}`; eb.setAttribute("download", `hybrid_${modelId}.json`); }
  } catch { $("#metrics-summary").textContent = "Metrics unavailable -- is the backend running?"; }
}

function renderNgramChart(container, m) {
  container.innerHTML = "";
  const w = container.clientWidth || 500, h = 100, pad = 20;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`); svg.setAttribute("width", "100%"); svg.setAttribute("height", "100%");
  const lv = m.model_versions?.local ?? 0, gv = m.model_versions?.global ?? 0, en = Math.min(m.ngram_entries ?? 0, Math.max(lv, gv, 1));
  const maxV = Math.max(lv, gv, en, 1);
  [[`Local v${lv}`, lv, "#5c7cfa"], [`Global g${gv}`, gv, "#56d07e"], [`${m.ngram_entries ?? 0} entries`, en, "#5cfad4"]].forEach(([label, val, color], i) => {
    const bw = (w - 2*pad)/3 - 8, bh = Math.max(4, (val/maxV)*(h-40)), x = pad + i*((w-2*pad)/3), y = h - 20 - bh;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", y); rect.setAttribute("width", bw); rect.setAttribute("height", bh);
    rect.setAttribute("fill", color); rect.setAttribute("rx", "3");
    const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
    lbl.setAttribute("x", x + bw/2); lbl.setAttribute("y", h-4); lbl.setAttribute("text-anchor", "middle");
    lbl.setAttribute("fill", "#6b7491"); lbl.setAttribute("font-size", "9"); lbl.textContent = label;
    svg.appendChild(rect); svg.appendChild(lbl);
  });
  container.appendChild(svg);
}

async function renderNgramStats(modelId) {
  const el = $("#weight-history"); if (!el) return;
  el.innerHTML = "";
  const slot = state.models.find(m => m.model_id === modelId);
  [
    ["Engine", slot?.engine || "LSTM + N-gram hybrid"],
    ["Hybrid entries", `${slot?.ngram_entries ?? "--"}`],
    ["Local updates", `v${slot?.local_version ?? 0}`],
    ["Global merges", `g${slot?.global_version ?? 0}`],
    ["LSTM weight", "0.50 (conf-gated)"],
    ["Local n-gram weight", "0.20"],
    ["Peer n-gram weight", "0.30"],
    ["Confidence threshold", "0.05"],
    ["Backend", "CPU - on-device"],
  ].forEach(([k, v]) => {
    const item = document.createElement("div"); item.className = "list-item";
    item.innerHTML = `<div><strong>${k}</strong></div><div class="mono" style="font-size:11px;color:var(--text-dim)">${v}</div>`;
    el.appendChild(item);
  });
}

function renderSessions(rows) {
  const el = $("#session-list"); el.innerHTML = "";
  if (!rows.length) { el.textContent = "No sessions yet."; return; }
  rows.forEach(r => {
    const item = document.createElement("div"); item.className = "list-item";
    const end = r.end_ts ? new Date(r.end_ts * 1000).toLocaleString() : "in progress";
    item.innerHTML = `<div><strong>${String(r.session_id).slice(0, 8)}...</strong><br><span class="mono" style="font-size:10px">events ${r.num_events}  -  ${r.text_len ?? 0} chars  -  ${end}</span></div>`;
    el.appendChild(item);
  });
}

// -- Settings -----------------------------------------------------------------

function setToggle(btn, on) { btn.classList.toggle("on", !!on); btn.dataset.val = on ? "1" : "0"; }
async function refreshSettings() {
  try {
    const res = await apiGet("/local/settings");
    setToggle($("#chk-auto-train"), res.auto_train);
    setToggle($("#chk-auto-share"), res.auto_share);
    setToggle($("#chk-gossip-enabled"), res.gossip_enabled ?? true);
    setToggle($("#chk-lan-scan"), res.discovery_enable_lan ?? false);
    if (res.max_concurrent_peer_sync) $("#max-sync-input").value = res.max_concurrent_peer_sync;
    if (res.gossip_interval_s) $("#gossip-interval-input").value = res.gossip_interval_s;
    setGossipStatus(res.gossip_enabled ? "gossip on" : "gossip off", res.gossip_enabled ? "green" : "amber");
  } catch { setGossipStatus("unavailable", "red"); }
}

// -- Discovery ----------------------------------------------------------------

const discoveryState = { peers: [], selectedUrls: new Set(), lanEnabled: false };
function updateDiscMeta(ts) {
  const base = "Auto-scan every 60 s -- Ports 8001-8020";
  if (!ts) { $("#disc-meta").textContent = `Last scan: never  -  ${base}`; return; }
  $("#disc-meta").textContent = `Last scan: ${new Date(ts * 1000).toLocaleTimeString()}  -  ${base}  -  LAN: ${discoveryState.lanEnabled ? "ON" : "OFF"}`;
}
function renderDiscoveredPeers(peers) {
  const el = $("#discovered-peers-list"); el.innerHTML = "";
  if (!peers.length) { el.innerHTML = '<div class="disc-empty">No peers found -- click Refresh to scan ports 8001-8020</div>'; return; }
  peers.forEach(p => {
    const item = document.createElement("div");
    item.className = "disc-peer-item" + (discoveryState.selectedUrls.has(p.url) ? " selected" : "");
    const lc = p.latency_ms < 20 ? "green" : p.latency_ms < 100 ? "amber" : "red";
    item.innerHTML = `<span class="indicator ${lc}" style="margin-right:6px;"></span><span class="mono" style="flex:1;font-size:11px;">${p.url}</span><span class="peer-latency ${lc}">${Math.round(p.latency_ms)}ms</span><span class="peer-meta">v11  -  ${p.status}</span>`;
    item.addEventListener("click", () => {
      discoveryState.selectedUrls.has(p.url) ? discoveryState.selectedUrls.delete(p.url) : discoveryState.selectedUrls.add(p.url);
      renderDiscoveredPeers(peers);
      $("#btn-add-discovered").disabled = discoveryState.selectedUrls.size === 0;
    });
    el.appendChild(item);
  });
}
async function loadDiscoveredPeers() {
  try {
    const data = await apiGet("/local/peers/discovered");
    discoveryState.peers = data.peers || []; discoveryState.lanEnabled = data.lan_enabled ?? false;
    setToggle($("#chk-lan-scan"), discoveryState.lanEnabled);
    renderDiscoveredPeers(discoveryState.peers); updateDiscMeta(data.last_scan_ts);
  } catch {}
}
async function runPeerScan() {
  try {
    const data = await apiPost("/local/peers/scan");
    discoveryState.peers = data.peers || [];
    renderDiscoveredPeers(discoveryState.peers); updateDiscMeta(data.last_scan_ts);
  } catch {}
}
async function loadReputation() {
  try {
    const data = await apiGet("/local/peers/reputation");
    const el = $("#reputation-list"); el.innerHTML = "";
    const peers = data.peers || [];
    if (!peers.length) { el.innerHTML = '<div class="disc-empty">No reputation data yet -- connect to peers first</div>'; return; }
    peers.sort((a, b) => b.score - a.score).forEach(p => {
      const item = document.createElement("div"); item.className = "disc-peer-item";
      const sc = p.score >= 1.5 ? "green" : p.score >= 0.5 ? "amber" : "red";
      item.innerHTML = `<span class="indicator ${sc}" style="margin-right:6px;"></span><span class="mono" style="flex:1;font-size:11px;">${p.url}</span><span class="peer-latency" style="color:var(--text-dim);font-size:10px;">score ${p.score.toFixed(2)}  -  ok:${p.successes} fail:${p.failures}</span><span class="peer-meta" style="font-size:10px;">${Math.round(p.avg_latency_ms)}ms</span>`;
      el.appendChild(item);
    });
  } catch {}
}

// -- Personalisation ----------------------------------------------------------

async function refreshPersWords(modelId) {
  try {
    const data = await apiGet(`/local/personalization/words?model_id=${encodeURIComponent(modelId)}`);
    const el = $("#pers-word-list"); el.innerHTML = "";
    const words = data.words || [];
    if (!words.length) { el.innerHTML = '<div style="color:var(--text-dim);font-size:11px;padding:8px;">No custom words yet</div>'; return; }
    words.forEach(w => {
      const item = document.createElement("div"); item.className = "list-item";
      item.innerHTML = `<span class="mono">${w.word}</span><span style="color:var(--text-dim);font-size:10px;margin-left:8px;">${w.category} x ${w.weight}</span>`;
      const rmv = document.createElement("button"); rmv.className = "tbtn"; rmv.textContent = "x";
      rmv.addEventListener("click", async () => { await apiPost("/local/personalization/word/remove", { model_id: modelId, word: w.word }); await refreshPersWords(modelId); });
      item.appendChild(rmv); el.appendChild(item);
    });
  } catch {}
}

// -- Export -------------------------------------------------------------------

function initNgramExport() {
  const setStatus = (msg, type) => { const el = $("#upload-status"); if (el) { el.textContent = msg; el.className = "upload-status " + type; } };
  const exportBtn = $("#btn-export-ngram");
  if (exportBtn) {
    exportBtn.addEventListener("click", async () => {
      const modelId = $("#upload-target-model")?.value || state.modelId;
      try {
        const r = await fetch(`/local/ngram/export?model_id=${encodeURIComponent(modelId)}`);
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a"); a.href = url; a.download = `hybrid_${modelId}_${Date.now()}.json`; a.click();
        URL.revokeObjectURL(url); setStatus("Exported", "ok");
      } catch (e) { setStatus("Export failed: " + e.message, "err"); }
    });
  }
  const saveBtn = $("#btn-save-ngram");
  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      const modelId = $("#upload-target-model")?.value || state.modelId;
      try { await apiPost(`/local/ngram/save?model_id=${encodeURIComponent(modelId)}`); setStatus("Saved to disk", "ok"); }
      catch (e) { setStatus("Save failed: " + e.message, "err"); }
    });
  }
}

// -- WebSocket ----------------------------------------------------------------

function initWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/local`);
  ws.addEventListener("open", () => { state.wsConnected = true; state.backendAvailable = true; setTrainerStatus(true); });
  ws.addEventListener("close", () => { state.wsConnected = false; setTrainerStatus(false); setTimeout(initWs, 2000); });
  ws.addEventListener("message", (ev) => {
    try {
      const msg = JSON.parse(ev.data); if (!msg || typeof msg !== "object") return;
      if (msg.type === "training_started") setTrainStatus("learning", "amber");
      if (msg.type === "training_complete") {
        const w = msg.words_learned ?? 0;
        setTrainStatus(w > 0 ? `+${w}w` : "done", "green");
        state.modelVersions.local = msg.local_version ?? state.modelVersions.local;
        state.modelVersions.global = msg.global_version ?? state.modelVersions.global;
        updateHeaderVersions();
      }
      if (msg.type === "global_model_updated") {
        setGossipStatus(`merged ${msg.deltas_merged ?? 1}`, "green");
        state.modelVersions.global = (state.modelVersions.global || 0) + (msg.deltas_merged ?? 1);
        updateHeaderVersions();
      }
      if (msg.type === "share_complete") setGossipStatus("shared", "green");
      if (msg.type === "model_created") refreshModels();
      if (msg.type === "peers_discovered") { discoveryState.peers = msg.peers || []; renderDiscoveredPeers(discoveryState.peers); }
      if (msg.type === "gossip_round_complete") {
        const ok = msg.ok;
        setGossipStatus(ok ? `gossip ok ${msg.round_id?.slice(0, 6)}` : "gossip skip", ok ? "green" : "amber");
        if (ok) { state.modelVersions.global = (state.modelVersions.global || 0) + 1; updateHeaderVersions(); }
      }
    } catch {}
  });
}

// -- Keyboard -----------------------------------------------------------------

function initKeyboard() {
  const el = editorEl();
  el.addEventListener("input", async () => { await startSessionIfNeeded(); resetIdleTimer(); schedulePredict(); });
  el.addEventListener("keydown", async (e) => {
    if (!state.popupVisible) { if (e.key !== "Tab" && e.key !== "Enter") return; }
    if (e.key === "Tab") { e.preventDefault(); if (state.popupVisible) await acceptSuggestion(state.selectedIndex); return; }
    if (e.key === "ArrowUp") { e.preventDefault(); state.selectedIndex = Math.max(0, state.selectedIndex - 1); renderPopup(); return; }
    if (e.key === "ArrowDown") { e.preventDefault(); state.selectedIndex = Math.min(state.suggestions.length - 1, state.selectedIndex + 1); renderPopup(); return; }
    if (e.key === "Escape") { e.preventDefault(); clearSuggestions("dismiss"); return; }
    if (e.altKey && e.key >= "1" && e.key <= "5") { e.preventDefault(); await acceptSuggestion(parseInt(e.key) - 1); return; }
  });
  el.addEventListener("focus", async () => { await startSessionIfNeeded(); });
}

// -- Page nav -----------------------------------------------------------------

function setPage(page) {
  state.page = page;
  for (const btn of document.querySelectorAll(".nav-btn")) btn.classList.toggle("active", btn.dataset.page === page);
  $("#page-editor").classList.toggle("hidden", page !== "editor");
  $("#page-manage").classList.toggle("hidden", page !== "manage");
  $("#page-settings").classList.toggle("hidden", page !== "settings");
  $("#page-dl").classList.toggle("hidden", page !== "dl");
  if (page === "manage") { refreshModels(); refreshMetrics(); }
  if (page === "settings") { refreshSettings(); loadDiscoveredPeers(); loadReputation(); }
  if (page === "dl") { dlRefreshSettings(); dlRefreshStatus(); dlPopulateModelSelect(); }
}

// -- Init ---------------------------------------------------------------------

async function init() {
  for (const btn of document.querySelectorAll(".nav-btn")) btn.addEventListener("click", () => setPage(btn.dataset.page));

  $("#model-select").addEventListener("change", async (e) => {
    const newId = e.target.value; if (newId === state.modelId) return;
    state.modelId = newId; clearSuggestions("silent"); state.sessionId = null;
    updateHeaderModelName();
    await startSessionIfNeeded();
  });

  $("#metrics-model-select").addEventListener("change", () => { renderModelRegistry(); refreshMetrics(); });
  $("#btn-end-session").addEventListener("click", async () => await endSession());
  $("#btn-train-now").addEventListener("click", async () => {
    const mid = $("#metrics-model-select").value || state.modelId;
    setTrainStatus("learning", "amber");
    const r = await apiPost("/local/train", { model_id: mid, reason: "manual" });
    setTrainStatus(`+${r.words_learned ?? 0}w`, "green");
    await refreshMetrics();
  });
  $("#btn-share").addEventListener("click", async () => {
    const mid = $("#metrics-model-select").value || state.modelId;
    setGossipStatus("sharing...", "amber");
    const r = await apiPost("/local/share", { model_id: mid, mode: "manual" });
    const ok = r.results?.filter(x => x.ok).length ?? 0;
    setGossipStatus(ok > 0 ? `shared -> ${ok}` : "no peers", "amber");
  });
  $("#btn-gossip-round").addEventListener("click", async () => {
    const mid = $("#metrics-model-select").value || state.modelId;
    try { const r = await apiPost("/gossip/round", { model_id: mid }); setGossipStatus(r.ok ? "gossip ok" : "gossip fail", r.ok ? "green" : "amber"); }
    catch (e) { alert("Gossip round failed: " + e.message); }
  });
  $("#btn-pull-global").addEventListener("click", async () => {
    const mid = $("#metrics-model-select").value || state.modelId;
    try { await apiPost("/local/pull_global", { model_id: mid }); setGossipStatus("pulled", "green"); }
    catch (e) { alert("Pull global failed: " + e.message); }
  });

  $("#btn-create-model").addEventListener("click", async () => {
    const name = $("#new-model-name").value.trim(); if (!name) { alert("Enter a model name first."); return; }
    await apiPost("/local/models/create", { name, creation_type: "blank" });
    $("#new-model-name").value = ""; await refreshModels(); await refreshMetrics();
  });

  for (const id of ["chk-auto-train", "chk-auto-share", "chk-gossip-enabled"])
    $(`#${id}`).addEventListener("click", function() { this.classList.toggle("on"); });

  $("#btn-save-settings").addEventListener("click", async () => {
    await apiPost("/local/settings", {
      auto_train:     $("#chk-auto-train").classList.contains("on"),
      auto_share:     $("#chk-auto-share").classList.contains("on"),
      gossip_enabled: $("#chk-gossip-enabled").classList.contains("on"),
      discovery_enable_lan: $("#chk-lan-scan").classList.contains("on"),
      max_concurrent_peer_sync: parseInt($("#max-sync-input").value) || 10,
      gossip_interval_s: parseInt($("#gossip-interval-input").value) || 300,
    });
    await refreshSettings();
  });

  $("#chk-lan-scan").addEventListener("click", async function() {
    this.classList.toggle("on");
    discoveryState.lanEnabled = this.classList.contains("on");
    await fetch("/local/peers/scan/settings?enable_lan=" + discoveryState.lanEnabled, { method: "POST" });
    updateDiscMeta(null);
  });
  $("#btn-refresh-scan").addEventListener("click", runPeerScan);
  $("#btn-refresh-reputation").addEventListener("click", loadReputation);
  $("#btn-add-discovered").addEventListener("click", async () => {
    discoveryState.selectedUrls.clear(); renderDiscoveredPeers(discoveryState.peers);
    $("#btn-add-discovered").disabled = true; await runPeerScan();
  });

  $("#pers-model-select").addEventListener("change", () => refreshPersWords($("#pers-model-select").value));
  $("#btn-pers-add").addEventListener("click", async () => {
    const word = $("#pers-word-input").value.trim(); if (!word) return;
    const mid = $("#pers-model-select").value;
    await apiPost("/local/personalization/word/add", { model_id: mid, word });
    $("#pers-word-input").value = ""; await refreshPersWords(mid);
  });
  $("#pers-word-input").addEventListener("keydown", async e => { if (e.key === "Enter") { e.preventDefault(); $("#btn-pers-add").click(); } });

  initNgramExport();
  initKeyboard();
  initWs();
  initDlTab();

  await refreshModels();
  if (state.modelId) await startSessionIfNeeded();

  document.addEventListener("mousedown", (e) => {
    if (state.popupVisible && !$("#autocomplete-popup").contains(e.target) && !editorEl().contains(e.target))
      clearSuggestions("dismiss");
  });
}

// -- DL / LSTM Tuning Tab -----------------------------------------------------

const DL_DEFAULTS = {
  lstm_conf_threshold: 0.05,
  lstm_weight: 0.50,
  local_ngram_weight: 0.20,
  global_ngram_weight: 0.30,
  local_ngram_fallback_weight: 0.40,
  global_ngram_fallback_weight: 0.60,
  lstm_train_steps: 10,
};

async function dlRefreshSettings() {
  try {
    const s = await apiGet("/local/settings");
    $("#dl-conf-threshold").value    = s.lstm_conf_threshold    ?? DL_DEFAULTS.lstm_conf_threshold;
    $("#dl-train-steps").value       = s.lstm_train_steps       ?? DL_DEFAULTS.lstm_train_steps;
    $("#dl-w-lstm").value            = s.lstm_weight            ?? DL_DEFAULTS.lstm_weight;
    $("#dl-w-local").value           = s.local_ngram_weight     ?? DL_DEFAULTS.local_ngram_weight;
    $("#dl-w-global").value          = s.global_ngram_weight    ?? DL_DEFAULTS.global_ngram_weight;
    $("#dl-fb-local").value          = s.local_ngram_fallback_weight  ?? DL_DEFAULTS.local_ngram_fallback_weight;
    $("#dl-fb-global").value         = s.global_ngram_fallback_weight ?? DL_DEFAULTS.global_ngram_fallback_weight;
    dlUpdateSums();
    dlUpdateThreshLabel();
  } catch (e) { console.warn("dlRefreshSettings error", e); }
}

function dlUpdateSums() {
  const a = parseFloat($("#dl-w-lstm").value) || 0;
  const b = parseFloat($("#dl-w-local").value) || 0;
  const c = parseFloat($("#dl-w-global").value) || 0;
  $("#dl-blend-sum-confident").textContent = `Sum: ${(a+b+c).toFixed(2)} (auto-normalised on save)`;
  const x = parseFloat($("#dl-fb-local").value) || 0;
  const y = parseFloat($("#dl-fb-global").value) || 0;
  $("#dl-blend-sum-fallback").textContent  = `Sum: ${(x+y).toFixed(2)} (auto-normalised on save)`;
}

function dlUpdateThreshLabel() {
  const v = parseFloat($("#dl-conf-threshold").value) || 0.05;
  $("#dl-conf-thresh-label").textContent = v.toFixed(3);
}

async function dlRefreshStatus() {
  try {
    const mid = state.modelId || "default";
    const res = await apiGet("/local/models/list");
    const models = res.models || [];
    const m = models.find(x => x.model_id === mid) || models[0];
    if (!m) { $("#dl-engine-status").innerHTML = '<div class="disc-empty">No models loaded</div>'; return; }

    // Probe a short context to get a live LSTM confidence sample.
    let lstmConf = null;
    try {
      const pr = await apiPost("/local/predict", { model_id: m.model_id, context_text: "the", cursor_pos: 3, k: 1 });
      lstmConf = pr.suggestions?.[0]?.lstm_conf ?? null;
    } catch (_) {}
    const confThresh = parseFloat($("#dl-conf-threshold").value) || 0.05;
    const active = lstmConf !== null && lstmConf >= confThresh;

    // Update confidence bar.
    $("#dl-conf-bar").style.width = Math.min(100, lstmConf !== null ? lstmConf * 100 / 0.5 : 0) + "%";
    $("#dl-conf-val").textContent = lstmConf !== null ? lstmConf.toFixed(4) : "--";
    $("#dl-conf-bar").style.background = active ? "var(--green)" : "var(--amber)";

    let torchStatus = null;
    try { torchStatus = await apiGet("/local/model/torch_status"); } catch (_) {}
    const torchOk  = torchStatus?.torch_importable;
    const torchErr = torchStatus?.torch_import_error;
    const pyExe    = torchStatus?.python_executable || "--";

    const rows = [
      ["Model",          m.model_id],
      ["Engine",         m.engine || "--"],
      ["PyTorch importable", torchOk === true ? "yes" : torchOk === false ? "no" : "--"],
      ["LSTM available", torchStatus?.lstm_available ? "yes" : "no"],
      ["Model ready",    torchStatus?.model_ready    ? "yes" : "no"],
      ["Last LSTM conf", lstmConf !== null ? lstmConf.toFixed(4) : "--"],
      ["DL branch active", active ? "yes" : "no (n-gram fallback)"],
      ["Weight version", m.weight_version ?? "--"],
      ["Python exe",     pyExe.length > 40 ? "..." + pyExe.slice(-38) : pyExe],
      ...(torchErr ? [["Torch error", torchErr.slice(0, 80)]] : []),
    ];
    $("#dl-engine-status").innerHTML = rows.map(([k, v]) =>
      `<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid var(--border);font-size:12px;">
         <span class="muted">${k}</span>
         <span class="mono" style="color:${v==="yes"||v.startsWith("yes")?`var(--green)`:v==="no"||v.startsWith("no")?`var(--amber)`:`var(--text)`}">${v}</span>
       </div>`).join("");
  } catch (e) {
    $("#dl-engine-status").innerHTML = '<div class="disc-empty">Error loading status</div>';
  }
}

async function dlPopulateModelSelect() {
  try {
    const res = await apiGet("/local/models/list");
    const sel = $("#dl-upload-model-select");
    sel.innerHTML = "";
    for (const m of (res.models || [])) {
      const opt = document.createElement("option");
      opt.value = m.model_id;
      opt.textContent = m.name || m.model_id;
      if (m.model_id === state.modelId) opt.selected = true;
      sel.appendChild(opt);
    }
  } catch (_) {}
}

function dlSetStatus(msg, isErr = false) {
  const el = $("#dl-save-status");
  el.textContent = msg;
  el.style.color = isErr ? "var(--red)" : "var(--green)";
  setTimeout(() => { el.textContent = ""; }, 3500);
}

function dlSetUploadStatus(msg, isErr = false) {
  const el = $("#dl-upload-status");
  el.textContent = msg;
  el.style.color = isErr ? "var(--red)" : "var(--green)";
}

async function dlUploadFile(file) {
  const modelId = $("#dl-upload-model-select").value;
  if (!modelId) { dlSetUploadStatus("Select a model first.", true); return; }
  if (!file || !file.name.endsWith(".json")) { dlSetUploadStatus("Only .json files are accepted.", true); return; }
  dlSetUploadStatus("Uploading...");
  const fd = new FormData();
  fd.append("model_id", modelId);
  fd.append("file", file);
  try {
    const res = await fetch("/local/model/upload_weights", { method: "POST", body: fd });
    const data = await res.json();
    if (data.ok) {
      dlSetUploadStatus(`Loaded -- arch: ${data.arch}, train_steps: ${data.train_steps}`);
      dlRefreshStatus();
    } else {
      const reason = data.reason || data.detail || "Unknown error.";
      const isPyTorchMissing = reason.toLowerCase().includes("pytorch not installed") ||
                               reason.toLowerCase().includes("n-gram-only mode");
      if (isPyTorchMissing) {
        dlSetUploadStatus(
          "PyTorch is not installed on this node -- running in n-gram-only mode. " +
          "Install torch (pip install torch --index-url https://download.pytorch.org/whl/cpu) " +
          "then use Re-init LSTM to activate the LSTM engine before uploading.",
          true
        );
      } else {
        dlSetUploadStatus("Upload failed: " + reason, true);
      }
    }
  } catch (e) { dlSetUploadStatus("Upload error: " + e.message, true); }
}

function initDlTab() {
  // Live weight sum display.
  for (const id of ["dl-w-lstm", "dl-w-local", "dl-w-global", "dl-fb-local", "dl-fb-global"])
    $("#" + id).addEventListener("input", dlUpdateSums);
  $("#dl-conf-threshold").addEventListener("input", dlUpdateThreshLabel);

  // Save DL settings.
  $("#btn-dl-save").addEventListener("click", async () => {
    const body = {
      lstm_conf_threshold:         parseFloat($("#dl-conf-threshold").value),
      lstm_train_steps:            parseInt($("#dl-train-steps").value),
      lstm_weight:                 parseFloat($("#dl-w-lstm").value),
      local_ngram_weight:          parseFloat($("#dl-w-local").value),
      global_ngram_weight:         parseFloat($("#dl-w-global").value),
      local_ngram_fallback_weight: parseFloat($("#dl-fb-local").value),
      global_ngram_fallback_weight:parseFloat($("#dl-fb-global").value),
    };
    for (const [k, v] of Object.entries(body)) {
      if (isNaN(v) || v < 0) { dlSetStatus(`Invalid value for ${k}`, true); return; }
    }
    try {
      await apiPost("/local/settings", body);
      dlSetStatus("Saved");
      dlUpdateSums();
      dlRefreshStatus();
    } catch (e) { dlSetStatus("Save failed: " + e.message, true); }
  });

  // Reset to defaults.
  $("#btn-dl-reset").addEventListener("click", () => {
    $("#dl-conf-threshold").value = DL_DEFAULTS.lstm_conf_threshold;
    $("#dl-train-steps").value    = DL_DEFAULTS.lstm_train_steps;
    $("#dl-w-lstm").value         = DL_DEFAULTS.lstm_weight;
    $("#dl-w-local").value        = DL_DEFAULTS.local_ngram_weight;
    $("#dl-w-global").value       = DL_DEFAULTS.global_ngram_weight;
    $("#dl-fb-local").value       = DL_DEFAULTS.local_ngram_fallback_weight;
    $("#dl-fb-global").value      = DL_DEFAULTS.global_ngram_fallback_weight;
    dlUpdateSums();
    dlUpdateThreshLabel();
    dlSetStatus("Reset to defaults (not yet saved)");
  });

  $("#btn-dl-refresh-status").addEventListener("click", dlRefreshStatus);

  // Re-init LSTM -- forces a lazy torch re-import on the server.
  $("#btn-dl-reinit-lstm").addEventListener("click", async () => {
    const modelId = state.modelId || $("#dl-upload-model-select").value;
    if (!modelId) { $("#dl-reinit-status").textContent = "No model selected."; return; }
    $("#dl-reinit-status").textContent = "Re-initialising...";
    try {
      const res = await fetch(`/local/model/reinit_lstm?model_id=${encodeURIComponent(modelId)}`, { method: "POST" });
      const data = await res.json();
      if (data.ok) {
        $("#dl-reinit-status").textContent = "LSTM re-initialised successfully.";
        $("#dl-reinit-status").style.color = "var(--green)";
      } else {
        const err = data.torch_error || "unknown";
        $("#dl-reinit-status").textContent = "Still unavailable: " + err;
        $("#dl-reinit-status").style.color = "var(--amber)";
      }
      dlRefreshStatus();
    } catch (e) {
      $("#dl-reinit-status").textContent = "Error: " + e.message;
      $("#dl-reinit-status").style.color = "var(--red)";
    }
  });

  // File picker via dropzone click.
  const dropzone = $("#dl-dropzone");
  const fileInput = $("#dl-file-input");
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
  fileInput.addEventListener("change", () => { if (fileInput.files.length) dlUploadFile(fileInput.files[0]); });

  // Drag-and-drop.
  dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer?.files?.[0];
    if (file) dlUploadFile(file);
  });
}

document.addEventListener("DOMContentLoaded", init);
