import Scene from "./scene.js";

const $ = (id) => document.getElementById(id);
const API_ROOT = String(window.TERMINAL_API_BASE || "/api").replace(/\/$/, "");
const MAX_BUFFER_FRAMES = 6000;
const MAX_IMPORTED_FRAMES = 100000;
const MAX_IMPORT_BYTES = 256 * 1024 * 1024;
const FRAME_POLL_MS = 250;
const PROFILES = [
  { id: "checkpoint12", label: "Checkpoint 12" },
  { id: "p2_initial_k3", label: "P2 · initial k3" },
  { id: "p2_trained_k3", label: "P2 · trained k3" },
  { id: "p3_task_neutral", label: "P3 · task neutral" },
];
const PROFILE_SCENARIOS = {
  checkpoint12: ["warm3", "flow3", "flow6", "flow8", "flow12"],
  p2_initial_k3: ["flow6", "flow8", "flow12", "p2_dense24"],
  p2_trained_k3: ["flow6", "flow8", "flow12", "p2_dense24"],
  p3_task_neutral: ["flow6", "flow12", "dense24"],
};
const STATUS_LABELS = {
  idle: "空闲", starting: "启动中", paused: "已暂停", running: "运行中", replay: "服务器回放", completed: "已完成", stopped: "已停止", error: "错误",
};

const state = {
  scene: null,
  session: null,
  token: null,
  generation: null,
  cursor: 0,
  frames: [],
  source: "server",
  sourceLabel: "新诊断运行",
  title: "终端区实时场景",
  metadata: {},
  runConfig: null,
  status: "idle",
  serverError: "",
  processAlive: false,
  droppedFrames: 0,
  truncated: false,
  retryMs: FRAME_POLL_MS,
  sessionRetryMs: 1000,
  pollTimer: null,
  polling: false,
  sessionPolling: false,
  pollEpoch: 0,
  offline: false,
  viewMode: "live",
  liveFollow: true,
  playing: false,
  loadingReplay: false,
  showSimulationConfig: false,
  playbackSpeed: 1,
  currentIndex: 0,
  playheadTime: 0,
  selectedUid: null,
  selectedEvent: null,
  followUid: null,
  followLost: false,
  reference: null,
  layers: {
    slots: true, rings: true, targetRings: true, gates: true, landingZones: true, grid: true, labels: true, trails: true, committed: true, conflicts: true,
  },
  heightScale: 1.6,
  historyCache: { index: -1, map: new Map() },
  flightTableCache: null,
  uiDirty: true,
  lastUiAt: 0,
  lastDisplaySignature: "",
  lastTick: 0,
  fps: { last: 0, count: 0, value: 0 },
};

function finiteNumber(value, fallback = null) {
  if (value === null || value === undefined || value === "") return fallback;
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }

function safeArray(value) { return Array.isArray(value) ? value : []; }

function aircraftIdentity(uav) { return uav?.aircraft_id != null ? String(uav.aircraft_id) : `U${uav?.uid ?? "—"}`; }

function sameAircraft(first, second) {
  if (first?.aircraft_id != null || second?.aircraft_id != null) return first?.aircraft_id != null && second?.aircraft_id != null && String(first.aircraft_id) === String(second.aircraft_id);
  return first?.arrival_id != null && second?.arrival_id != null && String(first.arrival_id) === String(second.arrival_id);
}

function discreteReplay(frame, metadata = state.metadata) {
  const m = metadata?.run_metadata ? { ...metadata, ...metadata.run_metadata } : metadata;
  return frame?.console_replay_kind === "rotating_slots" || frame?.interpolation === "none" || m?.replay_kind === "rotating_slots" || m?.source === "rotating_slots" || m?.interpolation === "none";
}

function recordedReplay() { return state.source !== "server" || state.status === "replay" || discreteReplay(state.frames[0]); }

function replayScenarioLabel(metadata) {
  const scenario = metadata.scenario_profile ?? metadata.scenario;
  if (scenario == null) return "";
  return typeof scenario === "object" ? String(scenario.id ?? scenario.name ?? scenario.profile ?? JSON.stringify(scenario)).slice(0, 120) : String(scenario);
}

function updateReplayControls() {
  const replay = recordedReplay();
  $("simulationConfig").hidden = replay && !state.showSimulationConfig;
  $("showSimulationButton").hidden = !replay || state.offline;
  $("showSimulationButton").textContent = state.showSimulationConfig ? "收起仿真设置" : "新建仿真设置";
  $("viewLiveButton").disabled = state.offline || replay;
  $("slotLegend").hidden = !discreteReplay(state.frames[state.currentIndex]);
  for (const id of ["openReplayButton", "referenceReplayButton", "replayPathInput"]) $(id).disabled = state.loadingReplay || state.offline;
  const summary = $("replaySummary");
  summary.hidden = !replay;
  document.querySelector(".brand-subtitle").textContent = discreteReplay(state.frames[0]) ? "固定多环终端区 · 旋转槽位回放" : "动态终端区 · 多 UAV 降落流工作台";
  if (replay) {
    const m = { ...state.metadata, ...(state.metadata.run_metadata || {}) };
    const bounds = `${formatTime(state.frames[0]?.time)} – ${formatTime(state.frames.at(-1)?.time)}`;
    const scenario = replayScenarioLabel(m);
    summary.textContent = `${scenario ? `${scenario} · ` : ""}${m.method ?? m.policy_id ?? "记录回放"} · seed ${m.seed ?? "—"} · ${bounds}${discreteReplay(state.frames[0]) ? " · 原始采样帧" : ""}`;
    summary.title = [m.replay_path ?? m.source_path ?? state.title, m.scenario && `scenario ${typeof m.scenario === "object" ? JSON.stringify(m.scenario) : m.scenario}`, m.code_commit && `commit ${m.code_commit}`, m.config_hash && `config ${m.config_hash}`].filter(Boolean).join(" · ");
  }
}

function formatNumber(value, digits = 1) {
  const number = finiteNumber(value);
  return number === null ? "—" : number.toFixed(digits);
}

function formatTime(value) {
  const number = finiteNumber(value);
  if (number === null) return "—";
  return `${number.toFixed(number >= 100 ? 0 : 1)} s`;
}

function formatSoc(value) {
  const number = finiteNumber(value);
  return number === null ? "—" : `${(clamp(number, 0, 1) * 100).toFixed(1)}%`;
}

function finitePoint(point) {
  return Array.isArray(point) && point.length >= 3 && point.slice(0, 3).every((value) => Number.isFinite(Number(value)));
}

function toast(message, kind = "") {
  const element = document.createElement("div");
  element.className = `toast${kind ? ` ${kind}` : ""}`;
  element.textContent = String(message);
  $("toastRegion").append(element);
  window.setTimeout(() => element.remove(), kind === "error" ? 6000 : 3600);
}

function setConnection(stateName, label) {
  const element = $("sessionStatus");
  element.dataset.state = stateName;
  $("sessionStatusText").textContent = label;
  $("connectionLed").dataset.state = stateName;
}

function setSimStatus(label) { $("simStatusText").textContent = label; }

async function requestJson(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  if (state.token) headers.set("X-Terminal-Token", state.token);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(`${API_ROOT}${path}`, { ...options, headers });
  const text = await response.text();
  let payload = null;
  if (text) {
    try { payload = JSON.parse(text); } catch { throw new Error(`服务器返回非 JSON（HTTP ${response.status}）`); }
  }
  if (!response.ok) {
    const message = typeof payload?.message === "string" ? payload.message.trim() : "";
    const error = message || payload?.error || `HTTP ${response.status}`;
    const failure = new Error(String(error));
    failure.status = response.status;
    throw failure;
  }
  return payload || {};
}

function setLoading(visible, text = "正在处理…") {
  $("sceneLoading").hidden = !visible;
  $("sceneLoadingText").textContent = text;
  $("rebuildButton").disabled = visible || state.offline;
  $("stopButton").disabled = visible || state.offline;
}

function populateSelect(element, entries, fallback = []) {
  const values = safeArray(entries).map((entry) => typeof entry === "string" ? { id: entry, label: entry } : entry).filter((entry) => entry && entry.id != null);
  const previous = element.value;
  element.replaceChildren();
  for (const entry of values.length ? values : fallback) {
    const option = document.createElement("option");
    option.value = String(entry.id);
    option.textContent = String(entry.label ?? entry.id);
    element.append(option);
  }
  if ([...element.options].some((option) => option.value === previous)) element.value = previous;
}

function normalizeProfile(entry) {
  if (typeof entry === "string") return { id: entry, label: entry };
  return entry && entry.id != null ? { id: String(entry.id), label: String(entry.label ?? entry.id) } : null;
}

function profileScenarioAllowed(profileId, scenarioId) {
  const allowed = PROFILE_SCENARIOS[profileId];
  if (!allowed || !allowed.length) return true;
  return allowed.includes(String(scenarioId));
}

function applyScenarioFilter() {
  const source = safeArray(state.session?.scenarios);
  const mode = $("modeSelect").value;
  if (mode === "checkpoint" && !$("profileSelect").value && [...$("profileSelect").options].some((option) => option.value === "p3_task_neutral")) $("profileSelect").value = "p3_task_neutral";
  const profileId = $("profileSelect").value;
  let allowed;
  if (mode === "checkpoint") allowed = source.filter((scenario) => profileScenarioAllowed(profileId, scenario.id));
  else if (mode === "fixed_rule") allowed = source.filter((scenario) => ["default", "fixed_rule_warm_flow", "fixed_rule_dense30_baseline", "fixed_rule_dense30_coordinated"].includes(String(scenario.id)));
  else allowed = source.filter((scenario) => String(scenario.id) === "default");
  const entries = allowed.length ? allowed : source;
  const previous = $("scenarioSelect").value;
  populateSelect($("scenarioSelect"), entries, [{ id: "fixed_rule_warm_flow", label: "Warm flow" }]);
  if ([...$("scenarioSelect").options].some((option) => option.value === previous)) $("scenarioSelect").value = previous;
  else if (mode === "checkpoint" && [...$("scenarioSelect").options].some((option) => option.value === "flow6")) $("scenarioSelect").value = "flow6";
  else if ([...$("scenarioSelect").options].some((option) => option.value === "default")) $("scenarioSelect").value = "default";
  else if ([...$("scenarioSelect").options].some((option) => option.value === "fixed_rule_warm_flow")) $("scenarioSelect").value = "fixed_rule_warm_flow";
  updateDeterministicControl();
  updateConfigSummary();
}

function updateDeterministicControl() {
  const profile = $("profileSelect").value;
  const fixed = profile === "checkpoint12" || profile.startsWith("p2_");
  const input = $("deterministicInput");
  input.disabled = fixed || state.offline;
  if (fixed) input.checked = true;
  updateModeFields();
}

function updateModeFields() {
  const mode = $("modeSelect").value;
  const checkpoint = mode === "checkpoint";
  const needsLow = checkpoint || mode === "fixed_low_rl" || mode === "learned_high_frozen_low";
  const needsHigh = mode === "learned_high_rule_low" || mode === "learned_high_frozen_low";
  $("profileSelect").disabled = state.offline || !checkpoint;
  $("lowCheckpointInput").disabled = state.offline || !needsLow;
  $("highCheckpointInput").disabled = state.offline || !needsHigh;
  $("profileSelect").closest("label")?.toggleAttribute("hidden", !checkpoint);
  $("lowCheckpointInput").closest("label")?.toggleAttribute("hidden", !needsLow);
  $("highCheckpointInput").closest("label")?.toggleAttribute("hidden", !needsHigh);
}

function configSummaryText(config) {
  const mode = config?.mode ?? config?.controller ?? "—";
  const profile = config?.profile ?? "—";
  const seed = config?.seed ?? "—";
  const parts = [`controller ${mode}`, `profile ${profile}`, `seed ${seed}`];
  if (config?.scenario != null) parts.push(`scenario ${config.scenario}`);
  if (config?.low_checkpoint) parts.push(`low ${config.low_checkpoint}`);
  if (config?.high_checkpoint) parts.push(`high ${config.high_checkpoint}`);
  const sha = config?.low_checkpoint_sha256 || config?.checkpoint_sha256 || config?.checkpoint_hash || config?.low_checkpoint_hash;
  if (sha) parts.push(`SHA ${sha}`);
  return parts.join(" · ");
}

function configFromMetadata(metadata) {
  if (!metadata || typeof metadata !== "object") return null;
  const nested = metadata.config && typeof metadata.config === "object" ? metadata.config : metadata.run_config && typeof metadata.run_config === "object" ? metadata.run_config : {};
  const source = { ...metadata, ...nested };
  const keys = ["mode", "controller", "profile", "scenario", "seed", "low_checkpoint", "high_checkpoint", "low_checkpoint_sha256", "checkpoint_sha256", "checkpoint_hash", "low_checkpoint_hash"];
  if (!keys.some((key) => source[key] != null)) return null;
  const config = {};
  keys.forEach((key) => { if (source[key] != null) config[key] = source[key]; });
  return config;
}

function scenarioLabelFor(scenarioId) {
  const id = scenarioId == null ? "" : String(scenarioId);
  const element = $("scenarioSelect");
  const option = element && [...element.options].find((candidate) => candidate.value === id);
  return option?.textContent?.trim() || id || "终端区实时场景";
}

function updateConfigSummary() {
  const mode = $("modeSelect").value || "—";
  const profile = $("profileSelect").value || "—";
  const seed = $("seedInput").value || "—";
  const path = $("lowCheckpointInput").value.trim();
  const highPath = $("highCheckpointInput").value.trim();
  if (state.runConfig) $("runConfigSummary").textContent = `${state.source === "server" ? "服务器运行" : "记录回放"} · ${configSummaryText(state.runConfig)}`;
  else $("runConfigSummary").textContent = `${state.source === "server" ? "待运行" : "记录回放"} · controller ${mode} · profile ${profile} · seed ${seed}${path ? ` · low ${path}` : ""}${highPath ? ` · high ${highPath}` : ""}`;
}

function applySession(session) {
  state.session = session || {};
  state.token = session?.token || null;
  const profiles = safeArray(session?.profiles).map(normalizeProfile).filter(Boolean);
  populateSelect($("profileSelect"), profiles.length ? profiles : PROFILES, PROFILES);
  if ([...$("profileSelect").options].some((option) => option.value === "p3_task_neutral")) $("profileSelect").value = "p3_task_neutral";
  populateSelect($("modeSelect"), session?.modes, [
    { id: "fixed_rule", label: "Fixed rule" },
    { id: "residual_policy", label: "Residual policy" },
    { id: "checkpoint", label: "Checkpoint" },
  ]);
  if (![...$("modeSelect").options].some((option) => option.value === "checkpoint")) {
    const option = document.createElement("option"); option.value = "checkpoint"; option.textContent = "Checkpoint"; $("modeSelect").append(option);
  }
  populateSelect($("scenarioSelect"), session?.scenarios, [{ id: "fixed_rule_warm_flow", label: "Warm flow" }]);
  applyScenarioFilter();
  setConnection("online", "已连接");
}

function scenarioFromForm() {
  return {
    mode: $("modeSelect").value || "fixed_rule",
    scenario: $("scenarioSelect").value || "fixed_rule_warm_flow",
    profile: $("profileSelect").value || "p3_task_neutral",
    seed: Math.trunc(finiteNumber($("seedInput").value, 0)),
    max_simulation_seconds: Math.max(1, finiteNumber($("maxSecondsInput").value, 600)),
    max_steps: null,
    playback_speed: 1,
    low_checkpoint: $("lowCheckpointInput").value.trim() || null,
    high_checkpoint: $("highCheckpointInput").value.trim() || null,
    start_paused: true,
    deterministic: Boolean($("deterministicInput").checked),
  };
}

function setSource(source, label, title = "") {
  state.source = source;
  state.sourceLabel = label;
  if (title) state.title = title;
}

function clearHistory() { state.historyCache = { index: -1, map: new Map() }; }

function clearFrames() {
  state.frames = [];
  state.currentIndex = 0;
  state.playheadTime = 0;
  state.droppedFrames = 0;
  state.truncated = false;
  clearHistory();
  state.uiDirty = true;
}

function normalizeServerFrame(frame) {
  if (!frame || typeof frame !== "object") return null;
  const time = finiteNumber(frame.time);
  if (time === null) return null;
  return frame;
}

function applyServerConfig(config) {
  if (!config || typeof config !== "object") return;
  const setValue = (id, value) => {
    if (value == null || !$(id)) return;
    const element = $(id);
    if ([...element.options].some((option) => option.value === String(value))) element.value = String(value);
  };
  setValue("modeSelect", config.mode);
  if (config.profile == null) {
    if (![...$("profileSelect").options].some((option) => option.value === "")) $("profileSelect").prepend(new Option("—", ""));
    $("profileSelect").value = "";
  } else setValue("profileSelect", config.profile);
  applyScenarioFilter();
  setValue("scenarioSelect", config.scenario);
  if (state.source === "server") state.title = scenarioLabelFor(config.scenario);
  if (config.seed != null) $("seedInput").value = String(config.seed);
  if (config.max_simulation_seconds != null) $("maxSecondsInput").value = String(config.max_simulation_seconds);
  $("lowCheckpointInput").value = config.low_checkpoint == null ? "" : String(config.low_checkpoint);
  $("highCheckpointInput").value = config.high_checkpoint == null ? "" : String(config.high_checkpoint);
  if (typeof config.deterministic === "boolean") $("deterministicInput").checked = config.deterministic;
  updateDeterministicControl();
  updateConfigSummary();
}

function appendServerPayload(payload) {
  const incomingGeneration = payload && payload.generation != null ? payload.generation : state.generation;
  const firstGeneration = state.generation === null && incomingGeneration !== null;
  const generationChanged = state.generation !== null && incomingGeneration !== null && incomingGeneration !== state.generation;
  if (generationChanged) {
    clearFrames();
    state.cursor = finiteNumber(payload.base_cursor, 0) ?? 0;
    state.droppedFrames = 0;
    state.truncated = false;
  }
  if (incomingGeneration !== null) state.generation = incomingGeneration;
  if (payload?.status) state.status = String(payload.status);
  state.serverError = payload?.error ? String(payload.error) : "";
  state.processAlive = Boolean(payload?.process_alive);
  if (payload?.metadata && typeof payload.metadata === "object") state.metadata = payload.metadata;
  const replay = state.status === "replay" || discreteReplay(safeArray(payload?.frames)[0]);
  if (replay && (firstGeneration || generationChanged || !state.frames.length)) {
    state.viewMode = "replay"; state.liveFollow = false; state.playing = false;
    state.currentIndex = 0; state.showSimulationConfig = false;
    state.sourceLabel = "记录回放";
  }
  const shouldSyncForm = firstGeneration || generationChanged || !state.runConfig;
  const metadataConfig = configFromMetadata(payload?.metadata);
  const serverConfig = payload?.config && typeof payload.config === "object" ? payload.config : state.metadata?.config;
  if (metadataConfig) state.runConfig = { ...(state.runConfig || {}), ...metadataConfig };
  if (serverConfig) state.runConfig = { ...(state.runConfig || {}), ...serverConfig };
  if (shouldSyncForm && state.runConfig && !replay) {
    applyServerConfig(state.runConfig);
    if (state.status === "replay" || state.runConfig.mode === "replay") state.sourceLabel = "记录回放";
  }
  if (metadataConfig) updateConfigSummary();
  const incoming = safeArray(payload?.frames).map(normalizeServerFrame).filter(Boolean);
  for (const frame of incoming) {
    const previous = state.frames[state.frames.length - 1];
    if (previous && finiteNumber(frame.time, -Infinity) < finiteNumber(previous.time, -Infinity)) continue;
    state.frames.push(frame);
  }
  const nextCursor = finiteNumber(payload?.next_cursor);
  if (nextCursor !== null) state.cursor = nextCursor;
  else state.cursor += incoming.length;
  state.droppedFrames = Math.max(state.droppedFrames, finiteNumber(payload?.dropped_frames, 0) || 0);
  if (state.droppedFrames > 0) state.truncated = true;
  if (!replay && state.frames.length > MAX_BUFFER_FRAMES) {
    const trimCount = state.frames.length - MAX_BUFFER_FRAMES;
    state.frames.splice(0, trimCount);
    state.truncated = true;
    state.currentIndex = Math.max(0, state.currentIndex - trimCount);
    clearHistory();
  }
  if (state.viewMode === "live" && state.liveFollow && state.frames.length) {
    state.currentIndex = state.frames.length - 1;
    state.playheadTime = finiteNumber(state.frames[state.currentIndex].time, 0);
  } else if (!state.frames.length) {
    state.currentIndex = 0;
    state.playheadTime = 0;
  } else {
    state.currentIndex = clamp(state.currentIndex, 0, state.frames.length - 1);
    if (replay && state.currentIndex === 0 && !state.playing) state.playheadTime = finiteNumber(state.frames[0].time, 0);
  }
  if (payload?.title) state.title = String(payload.title);
  else if (replay && state.metadata?.method) state.title = String(state.metadata.method);
  state.uiDirty = true;
}

function schedulePoll(delay = FRAME_POLL_MS) {
  if (state.offline) return;
  if (state.pollTimer) window.clearTimeout(state.pollTimer);
  state.pollTimer = window.setTimeout(() => { state.pollTimer = null; pollFrames(); }, delay);
}

function scheduleSessionRefresh(delay = 1000) {
  if (state.offline || state.sessionPolling) return;
  if (state.pollTimer) { window.clearTimeout(state.pollTimer); state.pollTimer = null; }
  state.pollTimer = window.setTimeout(() => { state.pollTimer = null; reconnectSession(); }, delay);
}

async function reconnectSession() {
  if (state.offline || state.sessionPolling) return;
  state.sessionPolling = true;
  try {
    const session = await requestJson("/session");
    applySession(session);
    state.sessionRetryMs = 1000;
    state.retryMs = FRAME_POLL_MS;
    await pollFrames();
  } catch (error) {
    state.serverError = error instanceof Error ? error.message : String(error);
    setConnection("error", "等待重连");
    state.sessionRetryMs = Math.min(15000, Math.max(1000, Math.round(state.sessionRetryMs * 1.8)));
    scheduleSessionRefresh(state.sessionRetryMs);
  } finally {
    state.sessionPolling = false;
    state.uiDirty = true;
  }
}

async function pollFrames() {
  if (state.offline || state.polling) return;
  state.polling = true;
  const requestSource = state.source;
  const requestEpoch = state.pollEpoch;
  try {
    const query = new URLSearchParams({ after: String(Math.max(0, state.cursor)) });
    if (state.generation !== null) query.set("generation", String(state.generation));
    const payload = await requestJson(`/frames?${query.toString()}`);
    if (state.source === requestSource && requestSource === "server" && state.pollEpoch === requestEpoch) {
      appendServerPayload(payload);
      // A replay is finite: drain its pages immediately while keeping the first
      // sampled frame selected. Live polling retains its existing batch cadence.
      let batch = payload;
      while (state.status === "replay" && safeArray(batch.frames).length && state.frames.length < MAX_IMPORTED_FRAMES && state.pollEpoch === requestEpoch) {
        const next = new URLSearchParams({ after: String(state.cursor), generation: String(state.generation) });
        batch = await requestJson(`/frames?${next.toString()}`);
        if (state.pollEpoch !== requestEpoch || state.source !== "server") break;
        appendServerPayload(batch);
      }
      state.retryMs = FRAME_POLL_MS;
      setConnection("online", "已连接");
      schedulePoll(FRAME_POLL_MS);
    }
  } catch (error) {
    state.serverError = error instanceof Error ? error.message : String(error);
    setConnection("error", "连接异常");
    state.retryMs = Math.min(10000, Math.max(FRAME_POLL_MS, Math.round(state.retryMs * 1.7)));
    if (state.source === requestSource && requestSource === "server" && state.pollEpoch === requestEpoch) {
      if (state.sessionPolling) schedulePoll(state.retryMs);
      else scheduleSessionRefresh(Math.max(state.retryMs, state.sessionRetryMs));
    }
  } finally {
    state.polling = false;
    state.uiDirty = true;
  }
}

function setReplayFrames(payload, source = "import", label = "记录回放") {
  if (source !== "server") {
    state.pollEpoch += 1;
    if (state.pollTimer) { window.clearTimeout(state.pollTimer); state.pollTimer = null; }
  }
  clearFrames();
  state.frames = safeArray(payload.frames);
  state.currentIndex = 0;
  state.playheadTime = finiteNumber(state.frames[0]?.time, 0);
  state.viewMode = "replay";
  state.liveFollow = false;
  state.playing = false;
  state.showSimulationConfig = false;
  state.generation = null;
  state.cursor = 0;
  state.serverError = "";
  state.metadata = payload.metadata && typeof payload.metadata === "object" ? { ...payload.metadata, ...(payload.metadata.run_metadata || {}) } : {};
  state.runConfig = configFromMetadata(state.metadata);
  setSource(source, label, payload.title || (source === "offline" ? "离线记录回放" : "记录回放"));
  if (state.metadata?.source) state.sourceLabel = "记录回放";
  updateConfigSummary();
  state.uiDirty = true;
}

function interpolateFrame(first, second, alpha) {
  if (discreteReplay(first) || discreteReplay(second)) return first;
  if (!first || !second || alpha <= 0 || alpha >= 1) return first;
  const gap = finiteNumber(second.time, 0) - finiteNumber(first.time, 0);
  if (!(gap > 0 && gap <= 1.01)) return first;
  const secondByUid = new Map(safeArray(second.uavs).map((uav) => [String(uav.uid), uav]));
  const uavs = safeArray(first.uavs).map((uav) => {
    const candidate = secondByUid.get(String(uav.uid));
    const samePhase = candidate && String(candidate.phase ?? "") === String(uav.phase ?? "");
    const samePlan = candidate && String(candidate.trajectory_plan_id ?? "") === String(uav.trajectory_plan_id ?? "");
    if (!samePhase || !samePlan || !finitePoint(uav.position) || !finitePoint(candidate.position)) return uav;
    const position = uav.position.slice(0, 3).map((value, index) => Number(value) + (Number(candidate.position[index]) - Number(value)) * alpha);
    return { ...uav, position };
  });
  return { ...first, uavs };
}

function displayFrame() {
  if (!state.frames.length) return null;
  const index = clamp(state.currentIndex, 0, state.frames.length - 1);
  const first = state.frames[index];
  if (!state.playing || index >= state.frames.length - 1) return first;
  const second = state.frames[index + 1];
  const duration = finiteNumber(second.time, 0) - finiteNumber(first.time, 0);
  const alpha = duration > 0 ? clamp((state.playheadTime - Number(first.time)) / duration, 0, 1) : 0;
  return interpolateFrame(first, second, alpha);
}

function updateHistory(index) {
  if (!state.frames.length) { clearHistory(); return state.historyCache.map; }
  index = clamp(index, 0, state.frames.length - 1);
  if (index < state.historyCache.index) clearHistory();
  const map = state.historyCache.map;
  for (let i = state.historyCache.index + 1; i <= index; i += 1) {
    for (const uav of safeArray(state.frames[i]?.uavs)) {
      if (!finitePoint(uav.position)) continue;
      const key = `${uav.uid}:${uav.phase}`;
      const points = map.get(key) || [];
      points.push(uav.position.slice(0, 3).map(Number));
      if (points.length > 900) points.splice(0, points.length - 900);
      map.set(key, points);
    }
  }
  state.historyCache.index = index;
  return map;
}

function setCurrentIndex(index, { follow = false } = {}) {
  if (!state.frames.length) return;
  state.currentIndex = clamp(Math.trunc(index), 0, state.frames.length - 1);
  state.playheadTime = finiteNumber(state.frames[state.currentIndex].time, 0);
  if (follow) { state.viewMode = "live"; state.liveFollow = true; }
  state.uiDirty = true;
}

function enterReplay(index = state.currentIndex) {
  state.viewMode = "replay";
  state.liveFollow = false;
  state.playing = false;
  setCurrentIndex(index);
  renderUI(true);
}

function enterLive() {
  if (recordedReplay() && state.source === "server") { toast("当前为记录回放；使用新建仿真设置启动一次新运行", "warn"); return; }
  if (state.offline) { toast("离线记录回放没有服务器实时视图", "warn"); return; }
  if (state.source === "import") {
    clearFrames(); state.cursor = 0; state.generation = null; state.runConfig = null; state.metadata = {}; setSource("server", "新诊断运行", scenarioLabelFor($("scenarioSelect").value)); schedulePoll(20);
  }
  state.viewMode = "live";
  state.liveFollow = true;
  state.playing = false;
  if (state.frames.length) setCurrentIndex(state.frames.length - 1, { follow: true });
  state.uiDirty = true;
}

function setPlaying(value) {
  if (!state.frames.length) return;
  state.playing = Boolean(value);
  if (state.playing) {
    if (state.currentIndex >= state.frames.length - 1) setCurrentIndex(0);
    state.playheadTime = finiteNumber(state.frames[state.currentIndex].time, 0);
  }
  state.uiDirty = true;
}

async function sendControl(action, value = undefined) {
  if (state.offline || state.source !== "server") { toast("当前记录回放不连接服务器", "warn"); return null; }
  try {
    const body = { action };
    if (value !== undefined) body.value = value;
    const payload = await requestJson("/control", { method: "POST", body: JSON.stringify(body) });
    if (payload.status) state.status = String(payload.status);
    if (payload.error) state.serverError = String(payload.error);
    state.uiDirty = true;
    return payload;
  } catch (error) {
    state.serverError = error instanceof Error ? error.message : String(error);
    toast(`控制失败：${state.serverError}`, "error");
    state.uiDirty = true;
    return null;
  }
}

async function handlePlay() {
  const terminalStatus = ["completed", "stopped", "error", "replay"].includes(state.status);
  const localOnly = state.offline || state.source !== "server" || terminalStatus || state.status === "idle";
  if (localOnly) {
    if (!state.frames.length) { toast(state.offline || state.source !== "server" ? "当前记录没有可回放帧" : "当前没有可回放帧，请重建场景", "warn"); return; }
    const shouldPlay = !state.playing;
    enterReplay(state.currentIndex);
    setPlaying(shouldPlay);
    return;
  }
  if (state.viewMode === "live" && state.liveFollow && !state.offline && state.source === "server") {
    if (state.status === "running" || state.status === "starting") await sendControl("pause");
    else await sendControl("resume");
    return;
  }
  setPlaying(!state.playing);
}

async function handleStep(direction) {
  if (!state.frames.length) { toast(state.offline || state.source !== "server" ? "当前记录没有可回放帧" : "当前没有可步进帧，请重建场景", "warn"); return; }
  if (direction < 0) {
    enterReplay(state.currentIndex - 1);
    return;
  }
  const terminalStatus = ["completed", "stopped", "error", "replay"].includes(state.status);
  if (state.source !== "server" || state.offline || terminalStatus || state.status === "idle") {
    enterReplay(state.currentIndex + direction);
    return;
  }
  if (state.viewMode === "live" && state.liveFollow && !state.offline && state.source === "server") {
    await sendControl("step");
    schedulePoll(20);
    return;
  }
  enterReplay(state.currentIndex + direction);
}

async function handleSpeed(value) {
  const speed = finiteNumber(value, 1) || 1;
  state.playbackSpeed = speed;
  if (!recordedReplay() && state.viewMode === "live" && state.liveFollow && !state.offline && state.source === "server") await sendControl("speed", speed);
  state.uiDirty = true;
}

function resetLayers() {
  state.layers = { slots: true, rings: true, targetRings: true, gates: true, landingZones: true, grid: true, labels: true, trails: true, committed: true, conflicts: true };
  syncLayerInputs();
  state.uiDirty = true;
}

function syncLayerInputs() {
  const ids = { slots: "layerSlots", rings: "layerRings", targetRings: "layerTargetRings", gates: "layerGates", landingZones: "layerLandingZones", grid: "layerGrid", labels: "layerLabels", trails: "layerTrails", committed: "layerCommitted", conflicts: "layerConflicts" };
  for (const [key, id] of Object.entries(ids)) $(id).checked = Boolean(state.layers[key]);
}

function metricsForFrame(frame) {
  if (!frame) return { active: null, queue: null, completed: null, minSoc: null };
  const uavs = safeArray(frame.uavs);
  const socs = uavs.map((uav) => finiteNumber(uav.soc)).filter((value) => value !== null);
  return { active: finiteNumber(frame.active_count, uavs.length), queue: finiteNumber(frame.rotating_slots?.upstream_waiting_count, finiteNumber(frame.external_queue_size)), completed: finiteNumber(frame.completed_count), minSoc: socs.length ? Math.min(...socs) : null };
}

function referenceAtTime(time) {
  const frames = state.reference?.frames || [];
  if (!frames.length || !Number.isFinite(Number(time))) return null;
  const target = Number(time);
  const first = finiteNumber(frames[0]?.time), last = finiteNumber(frames[frames.length - 1]?.time);
  if (first === null || last === null || target < first - 1e-3 || target > last + 1e-3) return null;
  let low = 0, high = frames.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const value = finiteNumber(frames[mid]?.time, 0);
    if (Math.abs(value - target) <= 1e-3) return frames[mid];
    if (value < target) low = mid + 1; else high = mid - 1;
  }
  const before = frames[Math.max(0, high)], after = frames[Math.min(frames.length - 1, low)];
  if (before && Math.abs(Number(before.time) - target) <= 1e-3) return before;
  if (after && Math.abs(Number(after.time) - target) <= 1e-3) return after;
  return null;
}

function updateCompare(frame, reference) {
  const box = $("referenceCompare");
  $("compareEmpty").hidden = Boolean(state.reference);
  if (!state.reference) { box.hidden = true; return; }
  box.hidden = false;
  const coverage = $("referenceCoverage");
  const currentMetadata = { ...state.metadata, ...(state.metadata.run_metadata || {}) };
  const referenceMetadata = { ...state.reference.metadata, ...(state.reference.metadata?.run_metadata || {}) };
  const mismatch = ["seed", "scenario_profile"].some((key) => currentMetadata[key] != null && referenceMetadata[key] != null && String(currentMetadata[key]) !== String(referenceMetadata[key]));
  const compareWarning = mismatch ? " · 场景/种子不同，差值仅供查看" : "";
  if (!reference) {
    const target = finiteNumber(frame?.time);
    const frames = state.reference?.frames || [];
    const first = finiteNumber(frames[0]?.time);
    const last = finiteNumber(frames[frames.length - 1]?.time);
    const outside = target === null || first === null || last === null || target < first - 1e-3 || target > last + 1e-3;
    coverage.textContent = (outside ? "当前 simulation time 超出参考覆盖范围 · 无参考，不外推" : "该时刻没有对应参考快照（采样时间未对齐）· 无参考，不外推") + compareWarning;
    for (const id of ["compareActive", "compareQueue", "compareCompleted", "compareSoc", "compareActiveRef", "compareQueueRef", "compareCompletedRef", "compareSocRef", "compareActiveDelta", "compareQueueDelta", "compareCompletedDelta", "compareSocDelta"]) $(id).textContent = "—";
    return;
  }
  const current = metricsForFrame(frame), ref = metricsForFrame(reference);
  const values = [["Active", current.active, ref.active, "compareActive", "compareActiveRef", "compareActiveDelta"], ["Queue", current.queue, ref.queue, "compareQueue", "compareQueueRef", "compareQueueDelta"], ["Completed", current.completed, ref.completed, "compareCompleted", "compareCompletedRef", "compareCompletedDelta"], ["SOC", current.minSoc, ref.minSoc, "compareSoc", "compareSocRef", "compareSocDelta"]];
  values.forEach(([, currentValue, refValue, currentId, refId, deltaId]) => {
    $(currentId).textContent = currentValue === null ? "—" : currentId === "compareSoc" ? formatSoc(currentValue) : String(currentValue);
    $(refId).textContent = refValue === null ? "—" : refId === "compareSocRef" ? formatSoc(refValue) : String(refValue);
    const delta = currentValue === null || refValue === null ? null : currentValue - refValue;
    $(deltaId).textContent = delta === null ? "—" : `${delta >= 0 ? "+" : ""}${currentId === "compareSoc" ? (delta * 100).toFixed(1) + "pp" : delta.toFixed(0)}`;
  });
  coverage.textContent = `${state.reference.title} · 按绝对时间对齐 t=${formatTime(reference.time)} · 原始快照值${compareWarning}`;
}

function renderKpis(frame) {
  const metrics = metricsForFrame(frame);
  $("kpiActive").textContent = metrics.active === null ? "—" : String(metrics.active);
  $("kpiQueue").textContent = `${frame?.rotating_slots ? "上游等待" : "队列"} ${metrics.queue === null ? "—" : metrics.queue}`;
  $("kpiService").textContent = frame ? String(finiteNumber(frame.service_count, 0)) : "—";
  $("kpiCompleted").textContent = metrics.completed === null ? "—" : String(metrics.completed);
  const occupancy = safeArray(frame?.rings).reduce((total, ring) => total + (finiteNumber(ring.occupancy, 0) || 0), 0);
  const capacity = safeArray(frame?.rings).reduce((total, ring) => total + (finiteNumber(ring.capacity, 0) || 0), 0);
  $("kpiOccupancy").textContent = frame ? `${occupancy}/${capacity}` : "—";
  const conflicts = safeArray(frame?.recent_conflict_uids);
  $("kpiConflict").textContent = frame ? String(conflicts.length) : "—";
  $("kpiConflictSub").textContent = "近期冲突飞机";
  $("kpiTime").textContent = frame ? formatTime(frame.time) : "—";
  $("kpiStep").textContent = discreteReplay(frame) ? `记录帧 ${state.currentIndex + 1}` : `step ${frame?.step_count ?? "—"}`;
}

function statusDescription() {
  if (state.offline) return "离线 · 记录回放";
  const compute = STATUS_LABELS[state.status] || state.status || "未知";
  const playback = state.viewMode === "replay" ? (state.playing ? `回放 ${state.playbackSpeed}×` : "历史查看") : state.liveFollow ? "实时跟随" : "实时 · 已脱离跟随";
  return `${compute} · ${playback}`;
}

function renderTimeline(frame) {
  const last = state.frames.length - 1;
  $("timelineInput").max = String(Math.max(0, last));
  $("timelineInput").value = String(clamp(state.currentIndex, 0, Math.max(0, last)));
  $("timelineCurrent").textContent = frame ? formatTime(frame.time) : "—";
  $("timelineEnd").textContent = last >= 0 ? formatTime(state.frames[last].time) : "—";
  $("timelineState").textContent = state.offline ? "回放状态 · 离线记录" : `计算状态 · ${STATUS_LABELS[state.status] || state.status || "—"}`;
  $("timelineFrame").textContent = last >= 0 ? `帧 ${state.currentIndex + 1} / ${state.frames.length}` : "帧 — / —";
  $("timelineFollowState").textContent = discreteReplay(frame) ? `原始采样帧 · ${state.playing ? `${state.playbackSpeed}×` : "已暂停"}` : state.liveFollow && state.viewMode === "live" ? "实时跟随" : state.playing ? `回放 ${state.playbackSpeed}×` : "历史回放";
  $("playButton").textContent = state.playing || (state.viewMode === "live" && state.liveFollow && state.status === "running") ? "Ⅱ" : "▶";
  $("playButton").setAttribute("aria-label", state.playing ? "暂停" : "播放");
  $("playbackSpeed").value = String(state.playbackSpeed);
  const buffer = state.frames.length ? `缓冲 ${state.frames.length}${state.truncated ? " · 已截断" : ""}` : "缓冲 0";
  $("bufferStatusText").textContent = buffer;
  $("bufferStatus").dataset.truncated = String(state.truncated);
  $("bufferStatus").dataset.error = String(Boolean(state.serverError));
}

function updateSceneHeader(frame) {
  const activeConfig = state.runConfig || (state.source === "server" ? { mode: $("modeSelect").value || "—", profile: $("profileSelect").value || "—", scenario: $("scenarioSelect").value || "—" } : { mode: "记录回放", profile: "—", scenario: "—" });
  const sourceText = state.offline ? "记录回放 · 离线" : state.sourceLabel;
  $("sceneTitle").textContent = state.title || "终端区场景";
  $("sceneTitle").title = state.title || "终端区场景";
  const metadata = { ...state.metadata, ...(state.metadata.run_metadata || {}) };
  const scenario = replayScenarioLabel(metadata);
  $("sceneSubtitle").textContent = discreteReplay(frame) ? `${sourceText} · ${scenario ? `${scenario} · ` : ""}${metadata.method ?? frame?.rotating_slots?.policy_id ?? "rotating_slots"} · seed ${metadata.seed ?? "—"} · 原始采样帧 · t=${formatTime(frame?.time)}` : frame ? `${sourceText} · ${configSummaryText(activeConfig)} · t=${formatTime(frame.time)}` : `${sourceText} · ${configSummaryText(activeConfig)}`;
  $("sceneSubtitle").title = $("replaySummary").title;
  $("viewModePill").textContent = state.offline ? "离线回放" : state.viewMode === "live" && state.liveFollow ? "实时跟随" : state.viewMode === "live" ? "实时 · 已脱离" : state.playing ? `回放 ${state.playbackSpeed}×` : "历史回放";
  $("viewLiveButton").classList.toggle("active", state.viewMode === "live" && state.liveFollow);
  $("viewReplayButton").classList.toggle("active", state.viewMode === "replay");
  $("projectionChip").textContent = state.scene?.camera?.projection === "top" ? "ORTHO TOP" : "PERSPECTIVE";
  $("bufferChip").textContent = `BUFFER ${state.frames.length}${state.truncated ? " · TRUNC" : ""}`;
  $("heightChip").textContent = `Z 显示 ×${state.heightScale.toFixed(1)}`;
  $("fpsChip").textContent = `FPS ${state.fps.value ? state.fps.value.toFixed(0) : "—"}`;
  const pixelsPerWorld = finiteNumber(state.scene?.metrics?.scale);
  if (pixelsPerWorld && pixelsPerWorld > 0) {
    const rawDistance = 46 / pixelsPerWorld;
    const power = Math.pow(10, Math.floor(Math.log10(Math.max(rawDistance, 1e-6))));
    const normalized = rawDistance / power;
    const niceDistance = (normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1) * power;
    $("scaleBar").querySelector("i").style.width = `${Math.max(20, niceDistance * pixelsPerWorld).toFixed(1)}px`;
    $("scaleBarText").textContent = `${niceDistance >= 10 ? niceDistance.toFixed(0) : niceDistance.toFixed(1)} m`;
  } else {
    $("scaleBarText").textContent = "— m";
  }
  $("sceneEmpty").hidden = Boolean(frame);
  $("sceneEmptyTitle").textContent = state.serverError ? "服务器数据暂不可用" : state.offline ? "离线回放为空" : "等待场景数据";
  $("sceneEmptyText").textContent = state.serverError || (state.offline ? "导入一份有效的 JSON 或 HTML 记录回放。" : "服务器连接后，WorldRenderSnapshot 将显示在这里。");
}

function renderFlights(frame) {
  const search = $("flightSearch").value.trim().toLowerCase();
  const filter = $("phaseFilter");
  let phase = filter.value;
  const sort = $("flightSort").value;
  const previous = state.flightTableCache;
  // Frame objects are replaced by live arrivals, replay switches and interpolation.
  // Preserve paused rows while unrelated telemetry and FPS continue to refresh.
  if (previous && previous.frame === frame && previous.selectedUid === state.selectedUid && previous.search === search && previous.phase === phase && previous.sort === sort) return;
  const phases = [...new Set(safeArray(frame?.uavs).map((uav) => String(uav.phase ?? "")).filter(Boolean))].sort();
  if (!previous || phases.length !== previous.phases.length || phases.some((value, index) => value !== previous.phases[index])) {
    filter.replaceChildren(new Option("全部 phase", ""));
    phases.forEach((value) => filter.append(new Option(value, value)));
    filter.value = phases.includes(phase) ? phase : "";
    phase = filter.value;
  }
  const body = $("flightTable").querySelector("tbody");
  body.replaceChildren();
  const uavs = safeArray(frame?.uavs).filter((uav) => (!phase || String(uav.phase ?? "") === phase) && (!search || `${aircraftIdentity(uav)} ${uav.uid}`.toLowerCase().includes(search)));
  uavs.sort((a, b) => sort === "soc" ? (finiteNumber(a.soc, 0) - finiteNumber(b.soc, 0)) : aircraftIdentity(a).localeCompare(aircraftIdentity(b), undefined, { numeric: true }));
  for (const uav of uavs) {
    const row = document.createElement("tr");
    row.tabIndex = 0;
    row.setAttribute("aria-selected", String(state.selectedUid != null && Number(uav.uid) === Number(state.selectedUid)));
    if (state.selectedUid != null && Number(uav.uid) === Number(state.selectedUid)) row.classList.add("selected");
    row.addEventListener("click", () => selectUid(uav.uid));
    row.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault(); event.stopPropagation(); selectUid(uav.uid);
    });
    const conflict = safeArray(frame?.recent_conflict_uids).map(Number).includes(Number(uav.uid));
    const cells = [
      aircraftIdentity(uav), String(uav.phase ?? "—"), String(uav.ring_role ?? "—"), formatSoc(uav.soc), `${formatNumber(uav.speed, 1)} m/s`, uav.assigned_lz == null ? "—" : `LZ ${uav.assigned_lz}`, uav.status ?? uav.original_status ?? (conflict ? "冲突关注" : uav.committed ? "已承诺" : "等待"),
    ];
    cells.forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
    body.append(row);
  }
  $("flightTableEmpty").hidden = uavs.length > 0;
  $("flightCountBadge").textContent = String(safeArray(frame?.uavs).length);
  $("flightTableNote").textContent = `${uavs.length} / ${safeArray(frame?.uavs).length} 架`;
  state.flightTableCache = { frame, selectedUid: state.selectedUid, search, phase, sort, phases };
}

function renderEvents(frame) {
  const events = safeArray(frame?.terminal_events);
  const body = $("eventsTable").querySelector("tbody");
  body.replaceChildren();
  for (const event of events.slice().reverse()) {
    const row = document.createElement("tr");
    row.addEventListener("click", () => selectEvent(event));
    const values = [formatTime(event.time), event.aircraft_id ?? (event.uid == null ? "—" : `U${event.uid}`), String(event.outcome ?? "—"), String(event.reason ?? "—"), String(event.phase ?? "—"), formatSoc(event.soc)];
    values.forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
    body.append(row);
  }
  $("eventTableEmpty").hidden = events.length > 0;
  $("eventCountBadge").textContent = String(events.length);
  $("eventTableNote").textContent = `${events.length} 条终态事件`;
}

function selectEvent(event) {
  state.selectedEvent = event || null;
  state.followUid = null;
  if (event?.uid != null && Number.isFinite(Number(event.uid))) state.selectedUid = Number(event.uid);
  else state.selectedUid = null;
  if (finitePoint(event?.position)) state.scene.focusPoint(event.position);
  state.uiDirty = true;
  renderUI(true);
}

function renderTelemetry(frame) {
  const grid = $("telemetryGrid");
  grid.replaceChildren();
  const entries = Object.entries(frame?.telemetry || {});
  if (!entries.length) { const empty = document.createElement("div"); empty.className = "table-empty"; empty.textContent = "暂无遥测数据"; grid.append(empty); return; }
  entries.sort(([a], [b]) => a.localeCompare(b));
  entries.forEach(([key, value]) => {
    const item = document.createElement("div"); item.className = "telemetry-item";
    const keyNode = document.createElement("span"); keyNode.className = "telemetry-key"; keyNode.textContent = key;
    const valueNode = document.createElement("strong"); valueNode.className = "telemetry-value"; valueNode.textContent = typeof value === "number" && Number.isFinite(value) ? formatNumber(value, 3) : String(value);
    item.append(keyNode, valueNode); grid.append(item);
  });
}

function renderResources(frame) {
  const content = $("resourceContent"); content.replaceChildren();
  const recorded = frame?.rotating_slots;
  $("resourceEvidence").textContent = "仅显示记录中的资源与时间证据";
  if (!recorded) { const empty = document.createElement("div"); empty.className = "table-empty"; empty.textContent = "当前记录无 rotating_slots 资源诊断"; content.append(empty); return; }
  const line = (text, kind = "") => { const row = document.createElement("div"); row.className = `resource-row ${kind}`; row.textContent = text; content.append(row); };
  const evidence = recorded.separation_evidence;
  if (evidence) $("resourceEvidence").textContent = `全回合间隔证据（非当前时刻） · ${evidence.label ?? "记录值"} · [${formatNumber(evidence.lower_bound_m, 3)}, ${formatNumber(evidence.upper_bound_m, 3)}] m · ${evidence.status ?? "未记录状态"} · witness ${formatTime(evidence.witness_time)}`;
  $("resourceEvidence").title = $("resourceEvidence").textContent;
  const metadata = { ...state.metadata, ...(state.metadata.run_metadata || {}) };
  if (metadata.fixture_only || metadata.fixture_banner || recorded.fixture_banner) line(String(recorded.fixture_banner || metadata.fixture_banner || "ANALYTIC FIXTURE · 不代表物理回合结果"), "resource-soft");
  if (metadata.verification_status || metadata.human_status) line(`记录验证状态：${metadata.verification_status ?? metadata.human_status}`, "resource-note");
  if (recorded.admission_model) line(`准入模型：${recorded.admission_model}`, "resource-note");
  for (const pad of safeArray(recorded.pads).slice(0, 32)) {
    const occupied = pad.occupant_id != null;
    line(`PAD ${pad.id} · ${occupied ? `实际服务 ${pad.occupant_id} · 服务结束 ${formatTime(pad.service_end)} · 剩余 ${formatTime(pad.service_remaining)}` : `无当前服务 · ${pad.state ?? "状态未记录"}`}`, "resource-pad");
    for (const row of safeArray(pad.reservations).slice(0, 20)) {
      const soft = row.commitment === "SOFT_UNSTARTED_REMOVABLE";
      line(`${soft ? "SOFT 可移除" : row.commitment ?? "记录区间"} · ${row.owner_id ?? row.aircraft_id ?? "—"} · ${row.future_interval ? "未来区间" : "记录区间"} [${formatTime(row.interval_start ?? row.start)}, ${formatTime(row.interval_end ?? row.end)}) · 飞行开始 ${formatTime(row.maneuver_start)} · 下降 ${formatTime(row.descent_start)} · 接地 ${formatTime(row.touchdown_time)} · 区间结束 ${formatTime(row.interval_end ?? row.end)}`, soft ? "resource-soft" : "");
    }
    if (safeArray(pad.reservations).length > 20) line(`另有 ${pad.reservations.length - 20} 条 pad 日历记录，界面仅展示前 20 条`, "resource-note");
  }
  const selected = findSelected(frame);
  const future = safeArray(recorded.slots).flatMap((slot) => safeArray(slot.reservations).filter((row) => row.future_interval || (selected && String(row.owner_id ?? row.aircraft_id) === aircraftIdentity(selected))).map((row) => ({ slot: slot.id, ...row })));
  future.sort((a, b) => Number(String(b.owner_id) === aircraftIdentity(selected)) - Number(String(a.owner_id) === aircraftIdentity(selected)));
  for (const row of future.slice(0, 12)) line(`SLOT ${row.slot} · ${row.commitment ?? "记录区间"} · ${row.owner_id ?? row.aircraft_id ?? "—"} · [${formatTime(row.interval_start ?? row.start)}, ${formatTime(row.interval_end ?? row.end)}) · 飞行开始 ${formatTime(row.maneuver_start)}`, row.commitment === "SOFT_UNSTARTED_REMOVABLE" ? "resource-soft" : "");
  if (future.length > 12) line(`另有 ${future.length - 12} 条槽位日历记录，选中飞机优先`, "resource-note");
  const timing = recorded.latest_dispatch_timing;
  if (timing) {
    const fields = Object.entries(timing).filter(([, value]) => value == null || ["string", "number", "boolean"].includes(typeof value)).slice(0, 12);
    line(`最近调度记录 · ${fields.map(([key, value]) => `${key}=${value == null ? "—" : String(value).slice(0, 100)}`).join(" · ")}`, "resource-note");
  }
}

function renderRings(frame) {
  const list = $("ringOccupancyList"); list.replaceChildren();
  const rings = safeArray(frame?.rings);
  if (!rings.length) { const empty = document.createElement("div"); empty.className = "table-empty"; empty.textContent = "暂无环数据"; list.append(empty); return; }
  rings.forEach((ring) => {
    const occupancy = Math.max(0, finiteNumber(ring.occupancy, 0) || 0), capacity = Math.max(0, finiteNumber(ring.capacity, 0) || 0);
    const row = document.createElement("div"); row.className = "ring-row";
    const main = document.createElement("div"); main.className = "ring-row-main";
    const label = document.createElement("div"); label.className = "ring-row-label"; label.innerHTML = `<span>${escapeHtml(String(ring.role ?? "ring"))}</span><span>${occupancy}/${capacity}</span>`;
    const bar = document.createElement("div"); bar.className = "ring-row-bar"; const fill = document.createElement("i"); fill.style.width = `${capacity ? clamp(occupancy / capacity * 100, 0, 100) : 0}%`; bar.append(fill); main.append(label, bar);
    const count = document.createElement("span"); count.className = "ring-row-count"; count.textContent = `z ${formatNumber(ring.height, 0)}m`;
    row.append(main, count); list.append(row);
  });
}

function escapeHtml(value) { return value.replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character])); }

function findSelected(frame) { return state.selectedUid == null ? null : safeArray(frame?.uavs).find((uav) => Number(uav.uid) === Number(state.selectedUid)) || null; }

function drawHistoryChart(uid, index) {
  const canvas = $("detailHistoryCanvas");
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(20, Math.round(rect.width * dpr)), height = Math.max(20, Math.round(rect.height * dpr));
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#fafafa"; ctx.fillRect(0, 0, rect.width, rect.height);
  const rows = state.frames.slice(0, index + 1).map((frame) => {
    const uav = safeArray(frame.uavs).find((candidate) => Number(candidate.uid) === Number(uid));
    return uav && finiteNumber(frame.time) !== null && finiteNumber(uav.soc) !== null && finitePoint(uav.position) ? { time: Number(frame.time), soc: Number(uav.soc), altitude: Number(uav.position[2]) } : null;
  }).filter(Boolean);
  if (rows.length < 2) { ctx.fillStyle = "#666666"; ctx.font = "11px Inter, system-ui, sans-serif"; ctx.fillText("等待两帧历史样本", 8, rect.height / 2 + 3); return; }
  const pad = { x: 8, y: 8, right: 6, bottom: 9 }, chartW = rect.width - pad.x - pad.right, chartH = rect.height - pad.y - pad.bottom;
  ctx.strokeStyle = "#dddddd"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad.x, pad.y); ctx.lineTo(pad.x, pad.y + chartH); ctx.lineTo(pad.x + chartW, pad.y + chartH); ctx.stroke();
  const minAlt = Math.min(...rows.map((row) => row.altitude)), maxAlt = Math.max(...rows.map((row) => row.altitude));
  const minTime = rows[0].time, maxTime = rows[rows.length - 1].time;
  const point = (row, value, low, high) => [pad.x + (maxTime > minTime ? (row.time - minTime) / (maxTime - minTime) : 1) * chartW, pad.y + chartH - (high > low ? (value - low) / (high - low) : .5) * chartH];
  const drawLine = (color, getter, low, high) => { ctx.strokeStyle = color; ctx.lineWidth = 1.3; ctx.beginPath(); rows.forEach((row, i) => { const [x, y] = point(row, getter(row), low, high); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); };
  drawLine("#333333", (row) => row.soc, 0, 1); drawLine("#a66b13", (row) => row.altitude, minAlt, maxAlt);
}

function renderDetails(frame) {
  const uav = findSelected(frame);
  const selectedExists = Boolean(uav);
  $("selectedTitle").textContent = uav ? aircraftIdentity(uav) : state.selectedEvent?.aircraft_id != null ? String(state.selectedEvent.aircraft_id) : state.selectedUid == null ? "未选择 UAV" : `U${state.selectedUid} · 当前帧无数据`;
  $("selectionPulse").classList.toggle("active", selectedExists);
  const refUav = uav && state.reference ? safeArray(referenceAtTime(frame?.time)?.uavs).find((candidate) => sameAircraft(candidate, uav)) : null;
  const eventText = state.selectedEvent ? ` · 事件 ${state.selectedEvent.outcome ?? "—"} / ${state.selectedEvent.reason ?? "—"} @ ${formatTime(state.selectedEvent.time)}` : "";
  const baseState = uav ? `${uav.status ?? uav.original_status ?? (uav.committed ? "已承诺轨迹" : "等待 / 可决策")}${refUav ? ` · 参考 ${aircraftIdentity(refUav)} 已匹配` : state.reference ? ` · 参考 ${aircraftIdentity(uav)} 无匹配` : ""}${eventText}` : state.selectedUid == null ? state.selectedEvent ? `终态事件 ${state.selectedEvent.outcome ?? "—"} / ${state.selectedEvent.reason ?? "—"} @ ${formatTime(state.selectedEvent.time)}${finitePoint(state.selectedEvent.position) ? " · 已定位事件位置" : ""}` : "点击场景中的 UAV 或航班表行查看详情。" : `${state.followLost ? "已离开当前帧，跟随已退出；" : ""}选中 UAV 不在当前帧。${eventText}`;
  $("selectedState").textContent = baseState;
  $("selectedState").classList.toggle("missing", !selectedExists && state.selectedUid != null);
  $("followSelectedButton").disabled = !selectedExists || state.offline;
  $("followSelectedButton").textContent = state.followUid != null && Number(state.followUid) === Number(state.selectedUid) ? "退出跟随" : "跟随飞机";
  $("detailPhase").textContent = uav?.phase ?? "—";
  $("detailSoc").textContent = formatSoc(uav?.soc);
  $("detailSpeed").textContent = formatNumber(uav?.speed, 2);
  $("detailAltitude").textContent = finitePoint(uav?.position) ? formatNumber(uav.position[2], 2) : "—";
  $("detailPositionState").textContent = uav ? "当前快照" : "缺席";
  $("detailX").textContent = finitePoint(uav?.position) ? formatNumber(uav.position[0], 2) : "—";
  $("detailY").textContent = finitePoint(uav?.position) ? formatNumber(uav.position[1], 2) : "—";
  $("detailZ").textContent = finitePoint(uav?.position) ? formatNumber(uav.position[2], 2) : "—";
  $("detailOriginalStatus").textContent = uav?.status ?? uav?.original_status ?? "—";
  $("detailSlot").textContent = uav?.slot_index ?? "—";
  $("detailRescue").textContent = uav?.rescue_status ?? "—";
  $("detailPower").textContent = uav?.battery_power_w == null ? "—" : `${formatNumber(uav.battery_power_w, 1)} W`;
  $("detailCommitted").textContent = uav ? (uav.committed ? "COMMITTED" : "OPEN") : "—";
  $("detailCommitted").className = `status-pill${uav?.emergency_landing ? " alert" : uav?.energy_alert && uav.energy_alert !== "normal" ? " warning" : ""}`;
  $("detailPlan").textContent = uav?.trajectory_plan_id ?? "—";
  $("detailRelease").textContent = uav?.last_release_decision ?? "—";
  $("detailSafety").textContent = uav?.last_safety_decision ?? "—";
  $("detailLz").textContent = uav?.assigned_lz == null ? "—" : `LZ ${uav.assigned_lz}`;
  $("detailTrajectoryCount").textContent = `${safeArray(uav?.trajectory_points).length} 点`;
  drawHistoryChart(state.selectedUid, state.currentIndex);
  renderRings(frame);
}

function renderScene(frame) {
  const history = updateHistory(state.currentIndex);
  const reference = state.reference ? referenceAtTime(frame?.time) : null;
  state.scene.setFrame(frame, { history, selectedUid: state.selectedUid, layers: state.layers, heightScale: state.heightScale, referenceFrame: reference });
  if (state.followUid != null && frame) {
    const followed = findSelected(frame);
    if (followed) { state.scene.focusOn(state.followUid, frame); state.followLost = false; }
    else { state.followLost = true; state.followUid = null; state.uiDirty = true; }
  }
}

function renderUI(force = false) {
  const frame = displayFrame();
  const signature = `${state.currentIndex}:${state.playing}:${state.viewMode}:${state.selectedUid}:${state.frames.length}:${state.status}:${state.reference ? state.reference.frames.length : 0}`;
  const now = performance.now();
  if (!force && !state.uiDirty && signature === state.lastDisplaySignature && now - state.lastUiAt < 100) return;
  state.lastDisplaySignature = signature; state.lastUiAt = now; state.uiDirty = false;
  updateReplayControls(); renderKpis(frame); updateSceneHeader(frame); renderTimeline(frame); renderFlights(frame); renderEvents(frame); renderTelemetry(frame); renderDetails(frame); renderResources(frame); updateCompare(frame, referenceAtTime(frame?.time));
  setSimStatus(statusDescription());
  if (state.offline) setConnection("offline", "离线记录");
  else if (state.serverError) setConnection("error", "服务器错误");
  else setConnection("online", "已连接");
}

function selectUid(uid) {
  state.selectedUid = uid == null ? null : Number(uid);
  if (state.selectedUid !== null) setRailOpen("details", true);
  state.selectedEvent = null;
  state.followLost = false;
  state.uiDirty = true;
  renderUI(true);
}

function toggleFollow() {
  const uav = findSelected(displayFrame());
  if (!uav || state.offline) return;
  if (state.followUid != null && Number(state.followUid) === Number(state.selectedUid)) {
    state.followUid = null; state.followLost = false;
  } else {
    state.followUid = state.selectedUid; state.followLost = false; state.scene.focusOn(state.followUid, displayFrame());
  }
  state.uiDirty = true;
}

function clearSelection() { state.selectedUid = null; state.selectedEvent = null; state.followUid = null; state.followLost = false; state.uiDirty = true; }

function startReplayAtCurrent() { enterReplay(state.currentIndex); setPlaying(true); }

function updateFrameAndDraw(now) {
  if (!state.lastTick) state.lastTick = now;
  const elapsed = Math.min(.25, Math.max(0, (now - state.lastTick) / 1000));
  state.lastTick = now;
  if (state.playing && state.frames.length) {
    state.playheadTime += elapsed * state.playbackSpeed;
    const lastTime = finiteNumber(state.frames[state.frames.length - 1].time, state.playheadTime);
    while (state.currentIndex < state.frames.length - 1 && finiteNumber(state.frames[state.currentIndex + 1].time, Infinity) <= state.playheadTime) state.currentIndex += 1;
    if (state.playheadTime >= lastTime) { state.playheadTime = lastTime; state.currentIndex = state.frames.length - 1; state.playing = false; }
    state.uiDirty = true;
  }
  const frame = displayFrame();
  renderScene(frame);
  state.fps.count += 1;
  if (!state.fps.last) state.fps.last = now;
  if (now - state.fps.last >= 500) { state.fps.value = state.fps.count * 1000 / (now - state.fps.last); state.fps.count = 0; state.fps.last = now; state.uiDirty = true; }
  renderUI(false);
  window.requestAnimationFrame(updateFrameAndDraw);
}

function extractJsonValue(text, marker) {
  const markerIndex = text.indexOf(marker);
  if (markerIndex < 0) return null;
  const from = markerIndex + marker.length;
  let start = from;
  while (start < text.length && /\s/.test(text[start])) start += 1;
  if (text[start] === "=") { start += 1; while (start < text.length && /\s/.test(text[start])) start += 1; }
  const opening = text[start];
  if (opening !== "{" && opening !== "[") return null;
  const stack = [opening === "{" ? "}" : "]"];
  let quote = false, escaped = false;
  for (let index = start + 1; index < text.length; index += 1) {
    const character = text[index];
    if (quote) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') quote = false;
      continue;
    }
    if (character === '"') { quote = true; continue; }
    if (character === "{" || character === "[") stack.push(character === "{" ? "}" : "]");
    else if (character === "}" || character === "]") {
      if (stack[stack.length - 1] !== character) return null;
      stack.pop();
      if (!stack.length) {
        try { return JSON.parse(text.slice(start, index + 1)); } catch { return null; }
      }
    }
  }
  return null;
}

function validateReplayFrame(frame, index) {
  if (!frame || typeof frame !== "object" || Array.isArray(frame)) throw new Error(`第 ${index + 1} 帧不是对象`);
  const time = finiteNumber(frame.time);
  if (time === null) throw new Error(`第 ${index + 1} 帧 time 不是有限数值`);
  const requireArray = (value, label, optional = false) => {
    if (value == null && optional) return [];
    if (!Array.isArray(value)) throw new Error(`第 ${index + 1} 帧 ${label} 必须是数组`);
    if (value.length > 10000) throw new Error(`第 ${index + 1} 帧 ${label} 超过数组上限`);
    return value;
  };
  const requireObject = (value, label) => { if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`第 ${index + 1} 帧 ${label} 必须是对象`); return value; };
  const requireFinite = (value, label) => { if (finiteNumber(value) === null) throw new Error(`第 ${index + 1} 帧 ${label} 不是有限数值`); };
  const requirePoint = (point, label) => { if (!finitePoint(point)) throw new Error(`第 ${index + 1} 帧 ${label} 坐标无效`); };
  requireFinite(frame.step_count, "step_count"); requireFinite(frame.external_queue_size, "external_queue_size"); requireFinite(frame.service_count, "service_count"); requireFinite(frame.completed_count, "completed_count");
  const rings = requireArray(frame.rings, "rings");
  rings.forEach((ring, ringIndex) => { requireObject(ring, `rings[${ringIndex}]`); if (typeof ring.role !== "string") throw new Error(`第 ${index + 1} 帧 rings[${ringIndex}].role 无效`); ["radius", "height", "target_radius", "target_height", "capacity", "occupancy"].forEach((key) => requireFinite(ring[key], `rings[${ringIndex}].${key}`)); });
  const gates = requireArray(frame.gates, "gates");
  gates.forEach((gate, gateIndex) => { requireObject(gate, `gates[${gateIndex}]`); requireFinite(gate.gate_id, `gates[${gateIndex}].gate_id`); requirePoint(gate.spawn_position, `gates[${gateIndex}].spawn_position`); requirePoint(gate.gate_position, `gates[${gateIndex}].gate_position`); });
  const zones = requireArray(frame.landing_zones, "landing_zones");
  zones.forEach((zone, zoneIndex) => { requireObject(zone, `landing_zones[${zoneIndex}]`); requireFinite(zone.lz_id, `landing_zones[${zoneIndex}].lz_id`); if (typeof zone.status !== "string") throw new Error(`第 ${index + 1} 帧 landing_zones[${zoneIndex}].status 无效`); requirePoint(zone.position, `landing_zones[${zoneIndex}].position`); });
  const uavs = requireArray(frame.uavs, "uavs");
  uavs.forEach((uav, uavIndex) => {
    requireObject(uav, `uavs[${uavIndex}]`); requireFinite(uav.uid, `uavs[${uavIndex}].uid`); if (typeof uav.phase !== "string") throw new Error(`第 ${index + 1} 帧 uavs[${uavIndex}].phase 无效`); requirePoint(uav.position, `uavs[${uavIndex}].position`); requirePoint(uav.velocity, `uavs[${uavIndex}].velocity`); requireFinite(uav.speed, `uavs[${uavIndex}].speed`); requireFinite(uav.soc, `uavs[${uavIndex}].soc`);
    const points = requireArray(uav.trajectory_points, `uavs[${uavIndex}].trajectory_points`); points.forEach((point, pointIndex) => requirePoint(point, `uavs[${uavIndex}].trajectory_points[${pointIndex}]`));
  });
  const conflicts = requireArray(frame.recent_conflict_uids, "recent_conflict_uids"); conflicts.forEach((uid, conflictIndex) => requireFinite(uid, `recent_conflict_uids[${conflictIndex}]`));
  const events = requireArray(frame.terminal_events, "terminal_events", true); events.forEach((event, eventIndex) => { requireObject(event, `terminal_events[${eventIndex}]`); requireFinite(event.time, `terminal_events[${eventIndex}].time`); if (!(discreteReplay(frame, {}) && event.soc === null && event.soc_source === "unavailable")) requireFinite(event.soc, `terminal_events[${eventIndex}].soc`); if (event.position != null) requirePoint(event.position, `terminal_events[${eventIndex}].position`); });
  if (frame.telemetry == null || typeof frame.telemetry !== "object" || Array.isArray(frame.telemetry)) throw new Error(`第 ${index + 1} 帧 telemetry 必须是对象`);
  Object.entries(frame.telemetry).forEach(([key, value]) => requireFinite(value, `telemetry.${key}`));
}

function validateReplayPayload(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw new Error("回放根对象无效");
  if (payload.metadata != null && (typeof payload.metadata !== "object" || Array.isArray(payload.metadata))) throw new Error("回放 metadata 必须是对象");
  const frames = safeArray(payload.frames);
  if (!frames.length) throw new Error("回放 frames 为空");
  if (frames.length > MAX_IMPORTED_FRAMES) throw new Error(`回放帧数超过限制（${MAX_IMPORTED_FRAMES}）`);
  let previousTime = -Infinity;
  frames.forEach((frame, index) => { validateReplayFrame(frame, index); const time = Number(frame.time); if (time < previousTime) throw new Error(`第 ${index + 1} 帧 time 不单调`); previousTime = time; });
  return { title: String(payload.title || "Imported replay"), frames, metadata: payload.metadata || {} };
}

function parseReplayText(text, filename = "") {
  let payload = null;
  try { payload = JSON.parse(text); } catch { /* Try the explicitly supported generated HTML formats below. */ }
  if (!payload) {
    payload = extractJsonValue(text, "window.__TERMINAL_REPLAY__");
    if (!payload) {
      const dataMarker = text.match(/const\s+data\s*=/);
      payload = dataMarker ? extractJsonValue(text, dataMarker[0]) : null;
    }
  }
  if (!payload) throw new Error(`${filename || "文件"} 中未找到有效 JSON 回放（仅支持 {title,frames} 或受支持的 HTML 注入格式）`);
  return validateReplayPayload(payload);
}

async function importFile(file, reference = false) {
  if (!file) return;
  if (file.size > MAX_IMPORT_BYTES) { toast(`文件超过 ${Math.round(MAX_IMPORT_BYTES / 1024 / 1024)} MiB 限制`, "error"); return; }
  if (state.loadingReplay) return;
  state.loadingReplay = true; setLoading(true, reference ? "正在读取参考文件…" : "正在读取回放文件…"); updateReplayControls();
  try {
    await new Promise((resolve) => window.requestAnimationFrame(resolve));
    const payload = parseReplayText(await file.text(), file.name);
    if (reference) {
      state.reference = payload;
      openDrawerPanel("comparisonPanel");
      toast(`参考回放已加载：${payload.frames.length} 帧`, "");
    } else {
      setReplayFrames(payload, "import", "记录回放");
      toast(`回放已加载：${payload.frames.length} 帧`, "");
    }
    state.uiDirty = true;
  } catch (error) { toast(`导入失败：${error instanceof Error ? error.message : String(error)}`, "error"); }
  finally { state.loadingReplay = false; setLoading(false); updateReplayControls(); }
}

async function openLocalReplay(reference = false) {
  const path = $("replayPathInput").value.trim();
  const notice = $("replayNotice");
  if (!path) { notice.hidden = false; notice.className = "inline-notice error"; notice.textContent = "请填写本机回放目录或 replay.json 的绝对路径"; return; }
  if (state.offline || state.loadingReplay) return;
  state.loadingReplay = true;
  if (!reference) {
    state.pollEpoch += 1;
    if (state.pollTimer) { window.clearTimeout(state.pollTimer); state.pollTimer = null; }
  }
  notice.hidden = false; notice.className = "inline-notice"; notice.textContent = reference ? "正在读取参考回放…" : "正在读取回放…";
  setLoading(true, reference ? "正在读取参考回放…" : "正在读取本地回放…"); updateReplayControls();
  try {
    const payload = await requestJson("/replay", { method: "POST", body: JSON.stringify({ path, reference }) });
    if (reference) {
      state.reference = validateReplayPayload(payload);
      openDrawerPanel("comparisonPanel");
      notice.textContent = `参考已加载 · ${state.reference.title} · ${state.reference.frames.length} 帧`;
    } else {
      const frames = []; let cursor = 0; let metadata = payload.metadata || {}; let batch;
      do {
        const query = new URLSearchParams({ after: String(cursor), generation: String(payload.generation) });
        batch = await requestJson(`/frames?${query.toString()}`);
        if (batch.generation !== payload.generation) throw new Error("服务器回放已被另一次操作切换，请重新打开");
        frames.push(...safeArray(batch.frames)); cursor = finiteNumber(batch.next_cursor, cursor);
        metadata = batch.metadata || metadata;
        if (frames.length > MAX_IMPORTED_FRAMES) throw new Error("回放帧数超过界面限制");
      } while (safeArray(batch.frames).length);
      const replay = validateReplayPayload({ title: payload.title || metadata.method || "固定多环终端区 · 旋转槽位回放", frames, metadata });
      setReplayFrames(replay, "server", "记录回放"); clearSelection();
      state.status = "replay"; state.generation = payload.generation; state.cursor = cursor; state.processAlive = false;
      notice.textContent = "回放已打开 · 从首帧暂停 · 播放仅推进记录";
    }
    state.uiDirty = true;
  } catch (error) {
    notice.className = "inline-notice error";
    notice.textContent = `读取失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    state.loadingReplay = false; setLoading(false); updateReplayControls(); state.uiDirty = true;
    if (!reference && state.source === "server") schedulePoll(FRAME_POLL_MS);
  }
}

function downloadText(filename, text, type = "application/json") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function exportReplay() {
  try {
    let payload;
    if (!state.offline && state.source === "server") payload = await requestJson("/export");
    else payload = { title: state.title, frames: state.frames, metadata: { ...(state.metadata || {}), source: state.sourceLabel, truncated: state.truncated, dropped_frames: state.droppedFrames } };
    const validated = validateReplayPayload(payload);
    downloadText(`${(validated.title || "terminal-replay").replace(/[^\w\-\u4e00-\u9fff]+/g, "_")}.json`, JSON.stringify(validated, null, 2));
    toast("当前回放已导出", "");
  } catch (error) { toast(`导出失败：${error instanceof Error ? error.message : String(error)}`, "error"); }
}

function exportScenePng() {
  if (!state.scene) return;
  const url = state.scene.toDataURL();
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = `${(state.title || "terminal-scene").replace(/[^\w\-\u4e00-\u9fff]+/g, "_")}.png`; anchor.click();
}

async function rebuildScenario() {
  if (state.offline) { toast("离线记录回放已禁用新诊断运行", "warn"); return; }
  const config = scenarioFromForm();
  if (["checkpoint", "fixed_low_rl", "learned_high_frozen_low"].includes(config.mode) && !config.low_checkpoint) { toast("当前模式必须填写 low checkpoint 路径", "error"); $("lowCheckpointInput").focus(); return; }
  if (["learned_high_rule_low", "learned_high_frozen_low"].includes(config.mode) && !config.high_checkpoint) { toast("当前模式必须填写 high checkpoint 路径", "error"); $("highCheckpointInput").focus(); return; }
  setLoading(true, "正在重建场景…");
  state.serverError = "";
  if (state.pollTimer) { window.clearTimeout(state.pollTimer); state.pollTimer = null; }
  state.pollEpoch += 1;
  try {
    const payload = await requestJson("/start", { method: "POST", body: JSON.stringify(config) });
    clearFrames();
    state.generation = payload.generation != null ? payload.generation : null;
    state.cursor = 0;
    state.viewMode = "live"; state.liveFollow = true; state.playing = false; state.followLost = false;
    if (payload.status) state.status = String(payload.status);
    state.metadata = payload.metadata && typeof payload.metadata === "object" ? payload.metadata : {};
    state.runConfig = payload.config && typeof payload.config === "object" ? { ...payload.config } : { ...config };
    applyServerConfig(state.runConfig);
    setSource("server", "新诊断运行", scenarioLabelFor(config.scenario));
    toast("场景已重建，初始状态为暂停", "");
    schedulePoll(20);
  } catch (error) { state.serverError = error instanceof Error ? error.message : String(error); toast(`重建失败：${state.serverError}`, "error"); schedulePoll(20); }
  finally { setLoading(false); state.uiDirty = true; }
}

async function stopScenario() {
  if (state.offline || state.source !== "server") { toast("当前记录回放不连接服务器", "warn"); return; }
  const result = await sendControl("stop");
  if (result) toast("服务器仿真已请求停止", "warn");
}

function disableOfflineControls() {
  $("rebuildButton").disabled = true; $("stopButton").disabled = true; $("viewLiveButton").disabled = true; $("scenarioSelect").disabled = true; $("modeSelect").disabled = true; $("profileSelect").disabled = true; $("seedInput").disabled = true; $("maxSecondsInput").disabled = true; $("lowCheckpointInput").disabled = true; $("highCheckpointInput").disabled = true; $("deterministicInput").disabled = true;
  $("rebuildNotice").hidden = false; $("rebuildNotice").textContent = "离线记录回放：不请求后端，启动/控制已禁用。";
}

function setRailOpen(name, open) {
  const isOpen = Boolean(open);
  $(`${name}Rail`).hidden = !isOpen;
  $(`${name}ToggleButton`).setAttribute("aria-expanded", String(isOpen));
  document.querySelector(".workspace").classList.toggle(`${name}-open`, isOpen);
  state.uiDirty = true;
}

function setDrawerOpen(open) {
  const isOpen = Boolean(open);
  $("drawerContent").hidden = !isOpen;
  document.querySelector(".app-shell").classList.toggle("drawer-collapsed", !isOpen);
  $("drawerToggleButton").classList.toggle("collapsed", !isOpen);
  $("drawerToggleButton").setAttribute("aria-expanded", String(isOpen));
  $("drawerToggleButton").setAttribute("aria-label", isOpen ? "收起底部面板" : "展开底部面板");
  const comparing = document.querySelector('.drawer-tab[data-panel="comparisonPanel"]').classList.contains("active");
  $("compareToggleButton").setAttribute("aria-expanded", String(isOpen && comparing));
  document.querySelectorAll(".drawer-tab").forEach((tab) => {
    tab.setAttribute("aria-controls", tab.dataset.panel);
    tab.setAttribute("aria-expanded", String(isOpen && tab.classList.contains("active")));
  });
  state.uiDirty = true;
}

function openDrawerPanel(panelId) {
  document.querySelectorAll(".drawer-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.panel === panelId));
  document.querySelectorAll(".drawer-panel").forEach((panel) => panel.classList.toggle("active", panel.id === panelId));
  setDrawerOpen(true);
}

function wireControls() {
  $("openReplayButton").addEventListener("click", () => openLocalReplay(false));
  $("referenceReplayButton").addEventListener("click", () => openLocalReplay(true));
  $("replayPathInput").addEventListener("keydown", (event) => { if (event.key === "Enter") openLocalReplay(false); });
  $("showSimulationButton").addEventListener("click", () => { state.showSimulationConfig = !state.showSimulationConfig; updateReplayControls(); });
  $("importButton").addEventListener("click", () => $("importFile").click()); $("importFile").addEventListener("change", (event) => importFile(event.target.files?.[0], false));
  $("importReferenceButton").addEventListener("click", () => $("importReferenceFile").click()); $("importReferenceFile").addEventListener("change", (event) => importFile(event.target.files?.[0], true));
  $("exportButton").addEventListener("click", exportReplay); $("exportPngButton").addEventListener("click", exportScenePng);
  $("removeReferenceButton").addEventListener("click", () => { state.reference = null; state.uiDirty = true; toast("参考回放已移除", ""); });
  $("profileSelect").addEventListener("change", applyScenarioFilter); $("modeSelect").addEventListener("change", () => { applyScenarioFilter(); updateDeterministicControl(); updateConfigSummary(); }); $("scenarioSelect").addEventListener("change", updateConfigSummary); $("seedInput").addEventListener("input", updateConfigSummary); $("lowCheckpointInput").addEventListener("input", updateConfigSummary); $("highCheckpointInput").addEventListener("input", updateConfigSummary);
  $("deterministicInput").addEventListener("change", updateConfigSummary); $("rebuildButton").addEventListener("click", rebuildScenario); $("stopButton").addEventListener("click", stopScenario);
  $("viewLiveButton").addEventListener("click", enterLive); $("viewReplayButton").addEventListener("click", () => enterReplay(state.currentIndex)); $("playButton").addEventListener("click", handlePlay); $("stepBackButton").addEventListener("click", () => handleStep(-1)); $("stepForwardButton").addEventListener("click", () => handleStep(1)); $("playbackSpeed").addEventListener("change", (event) => handleSpeed(event.target.value));
  $("timelineInput").addEventListener("input", (event) => { enterReplay(Number(event.target.value)); }); $("followSelectedButton").addEventListener("click", toggleFollow); $("clearSelectedButton").addEventListener("click", clearSelection); $("followCameraButton").addEventListener("click", toggleFollow);
  $("perspectiveButton").addEventListener("click", () => { state.scene.setProjection("perspective"); $("perspectiveButton").classList.add("active"); $("topButton").classList.remove("active"); state.uiDirty = true; }); $("topButton").addEventListener("click", () => { state.scene.setProjection("top"); $("topButton").classList.add("active"); $("perspectiveButton").classList.remove("active"); state.uiDirty = true; }); $("resetCameraButton").addEventListener("click", () => { state.scene.resetCamera(); $("topButton").classList.remove("active"); $("perspectiveButton").classList.add("active"); state.uiDirty = true; });
  $("heightScaleInput").addEventListener("input", (event) => { state.heightScale = finiteNumber(event.target.value, 1.6) || 1.6; $("heightScaleOutput").textContent = `${state.heightScale.toFixed(1)}×`; state.scene.setHeightScale(state.heightScale); state.uiDirty = true; });
  const layerIds = { slots: "layerSlots", rings: "layerRings", targetRings: "layerTargetRings", gates: "layerGates", landingZones: "layerLandingZones", grid: "layerGrid", labels: "layerLabels", trails: "layerTrails", committed: "layerCommitted", conflicts: "layerConflicts" }; Object.entries(layerIds).forEach(([key, id]) => $(id).addEventListener("change", (event) => { state.layers[key] = event.target.checked; state.scene.setLayers(state.layers); state.uiDirty = true; })); $("resetLayersButton").addEventListener("click", resetLayers);
  $("flightSearch").addEventListener("input", () => { state.uiDirty = true; renderUI(true); }); $("phaseFilter").addEventListener("change", () => { state.uiDirty = true; renderUI(true); }); $("flightSort").addEventListener("change", () => { state.uiDirty = true; renderUI(true); });
  document.querySelectorAll(".drawer-tab").forEach((tab) => tab.addEventListener("click", () => openDrawerPanel(tab.dataset.panel)));
  $("drawerToggleButton").addEventListener("click", () => setDrawerOpen($("drawerContent").hidden));
  $("settingsToggleButton").addEventListener("click", () => setRailOpen("settings", $("settingsRail").hidden));
  $("detailsToggleButton").addEventListener("click", () => setRailOpen("details", $("detailsRail").hidden));
  $("compareToggleButton").addEventListener("click", () => {
    const current = document.querySelector('.drawer-tab[data-panel="comparisonPanel"]').classList.contains("active");
    if (current && !$("drawerContent").hidden) setDrawerOpen(false);
    else openDrawerPanel("comparisonPanel");
  });
  $("focusViewButton").addEventListener("click", () => { setRailOpen("settings", false); setRailOpen("details", false); setDrawerOpen(false); });
  setDrawerOpen(false);
  window.addEventListener("keydown", (event) => { const target = event.target; if (target && (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target.tagName) || target.isContentEditable)) return; if (event.key === " ") { event.preventDefault(); handlePlay(); } else if (event.key === "ArrowLeft") { event.preventDefault(); handleStep(-1); } else if (event.key === "ArrowRight") { event.preventDefault(); handleStep(1); } else if (event.key.toLowerCase() === "f") { event.preventDefault(); toggleFollow(); } });
}

async function initialize() {
  wireControls();
  state.scene = new Scene($("sceneCanvas"), { onSelect: selectUid });
  syncLayerInputs(); updateDeterministicControl(); updateConfigSummary();
  const offlinePayload = window.__TERMINAL_REPLAY__;
  if (offlinePayload) {
    state.offline = true; disableOfflineControls();
    try { const payload = validateReplayPayload(offlinePayload); setReplayFrames(payload, "offline", "记录回放"); } catch (error) { state.serverError = error instanceof Error ? error.message : String(error); toast(`离线回放无效：${state.serverError}`, "error"); }
    setConnection("offline", "离线记录"); setSimStatus("离线 · 记录回放");
  } else {
    setConnection("offline", "连接中");
    try { const session = await requestJson("/session"); applySession(session); await pollFrames(); } catch (error) { state.serverError = error instanceof Error ? error.message : String(error); setConnection("error", "连接失败"); toast(`服务器连接失败：${state.serverError}`, "error"); scheduleSessionRefresh(1000); }
  }
  state.uiDirty = true;
  window.requestAnimationFrame(updateFrameAndDraw);
}

initialize();
