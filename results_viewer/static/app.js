const state = {
  experiments: [],
  current: null,
  summary: null,
  stats: null,
  trainingStats: null,
  rollouts: null,
  page: 0,
  limit: 10,
};

const $ = (id) => document.getElementById(id);

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (typeof value === "number") {
    if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
    return value.toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  return String(value);
}

function fmtMetric(key, value, digits = 3) {
  if (isPercentMetric(key, value)) {
    return `${fmt(Number(value) * 100, 1)}%`;
  }
  return fmt(value, digits);
}

function isPercentMetric(key, value) {
  if (typeof value !== "number" || value < 0 || value > 1) return false;
  const name = String(key || "");
  return (
    name === "success" ||
    name === "success_rate" ||
    name === "train success" ||
    name === "test success" ||
    name === "always_success" ||
    name === "never_success" ||
    name === "sometimes_success" ||
    name.endsWith("success_mean") ||
    name.endsWith("success_rate") ||
    name.endsWith("no_error_mean") ||
    name.endsWith("no_answer_mean") ||
    name.endsWith("overflow_mean") ||
    name.endsWith("overlong_mean") ||
    name.endsWith("success_given_overlong") ||
    name.includes("/success_") ||
    name.includes("/no_error_") ||
    name.includes("/no_answer_") ||
    name.includes("/overflow_") ||
    name.includes("/overlong_") ||
    name.includes("domain_mix_actual") ||
    name.includes("domain_mix_target")
  );
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function init() {
  $("refreshButton").addEventListener("click", refreshCurrent);
  $("experimentSelect").addEventListener("change", () => selectExperiment($("experimentSelect").value));
  $("trainingStatsTopic").addEventListener("change", loadTrainingStats);
  $("trainingMetricSelect").addEventListener("change", renderTrainingStatsChart);
  $("metricSelect").addEventListener("change", renderStatsChart);
  $("rolloutTopic").addEventListener("change", () => {
    state.page = 0;
    loadRollouts();
  });
  $("successFilter").addEventListener("change", () => {
    state.page = 0;
    loadRollouts();
  });
  $("searchInput").addEventListener("input", debounce(() => {
    state.page = 0;
    loadRollouts();
  }, 250));
  $("prevPage").addEventListener("click", () => {
    state.page = Math.max(0, state.page - 1);
    loadRollouts();
  });
  $("nextPage").addEventListener("click", () => {
    state.page += 1;
    loadRollouts();
  });
  await loadExperiments();
}

async function loadExperiments() {
  const data = await api("/api/experiments");
  state.experiments = data.experiments;
  $("rootPath").textContent = data.results_root;
  $("experimentSelect").innerHTML = state.experiments
    .map((exp) => `<option value="${escapeHtml(exp.name)}">${escapeHtml(exp.name)}</option>`)
    .join("");
  renderExperimentList();
  if (state.experiments.length) {
    await selectExperiment(state.current || state.experiments[0].name);
  }
}

function renderExperimentList() {
  $("experimentList").innerHTML = state.experiments
    .map((exp) => {
      const rows = Object.values(exp.topics).reduce((sum, topic) => sum + (topic.rows || 0), 0);
      const active = exp.name === state.current ? " active" : "";
      return `<div class="experiment-item${active}" data-exp="${escapeHtml(exp.name)}">
        <strong>${escapeHtml(exp.name)}</strong>
        <span>${rows} rows · ${escapeHtml(exp.path)}</span>
      </div>`;
    })
    .join("");
  document.querySelectorAll(".experiment-item").forEach((node) => {
    node.addEventListener("click", () => selectExperiment(node.dataset.exp));
  });
}

async function selectExperiment(name) {
  state.current = name;
  $("experimentSelect").value = name;
  renderExperimentList();
  $("experimentTitle").textContent = name;
  $("detail").innerHTML = "";
  $("detailMeta").textContent = "Select a prompt group to inspect attempts and call traces.";
  await Promise.all([loadSummary(), loadTrainingStats(), loadRollouts()]);
}

async function refreshCurrent() {
  await loadExperiments();
  if (state.current) {
    await Promise.all([loadSummary(), loadTrainingStats(), loadRollouts()]);
  }
}

async function loadSummary() {
  state.summary = await api(`/api/experiments/${encodeURIComponent(state.current)}/summary`);
  $("experimentMeta").textContent = state.summary.path;
  renderSummary();
}

function renderSummary() {
  const topics = state.summary.topics;
  const cards = [
    ["actor rows", topics.actor?.rows],
    ["actor test rows", topics.actor_test?.rows],
    ["stats rows", topics.stats?.rows],
    ["stats test rows", topics.stats_test?.rows],
    ["train success", topics.actor?.success_rate, "success_rate"],
    ["test success", topics.actor_test?.success_rate, "success_rate"],
    ["train reward", topics.actor?.reward_mean],
    ["train attempts", topics.actor?.attempts],
    ["train trace steps", topics.actor?.trace_steps],
  ];
  $("summaryCards").innerHTML = cards
    .map(([label, value, key]) => `<div class="summary-card"><span>${label}</span><strong>${fmtMetric(key || label, value)}</strong></div>`)
    .join("");
}

async function loadTrainingStats() {
  const topic = $("trainingStatsTopic").value;
  state.trainingStats = await api(`/api/experiments/${encodeURIComponent(state.current)}/stats/${topic}?max_points=1000`);
  const preferred = ["success_mean", "reward_mean", "latency_mean", "output_tokens_mean", "published_samples"];
  const keys = state.trainingStats.numeric_keys;
  const current = $("trainingMetricSelect").value;
  $("trainingMetricSelect").innerHTML = keys.map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(key)}</option>`).join("");
  $("trainingMetricSelect").value = keys.includes(current) ? current : preferred.find((key) => keys.includes(key)) || keys[0] || "";
  renderTrainingStatsChart();
  renderTrainingLatestStats();
}

function updateVisibleRolloutStats() {
  const rows = state.rollouts?.rows || [];
  const preferred = ["success_rate", "reward_mean", "reward_max", "attempts", "trace_steps", "output_tokens_mean"];
  const keys = visibleRolloutMetricKeys(rows);
  const current = $("metricSelect").value;
  $("metricSelect").innerHTML = keys.map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(key)}</option>`).join("");
  $("metricSelect").value = keys.includes(current) ? current : preferred.find((key) => keys.includes(key)) || keys[0] || "";
  state.stats = {
    rows,
    latest: aggregateVisibleRollouts(rows),
    numeric_keys: keys,
    total: rows.length,
    sampled: rows.length,
  };
  renderStatsChart();
  renderLatestStats();
}

function visibleRolloutMetricKeys(rows) {
  return Array.from(
    new Set(
      rows.flatMap((row) =>
        Object.entries(row)
          .filter(([, value]) => typeof value === "number" || typeof value === "boolean")
          .map(([key]) => key)
      )
    )
  ).sort();
}

function aggregateVisibleRollouts(rows) {
  return {
    visible_rollouts: rows.length,
    success_rate: avg(rows.map((row) => row.success_rate)),
    reward_mean: avg(rows.map((row) => row.reward_mean)),
    reward_max: maxNumber(rows.map((row) => row.reward_max)),
    attempts: sumNumber(rows.map((row) => row.attempts)),
    trace_steps: sumNumber(rows.map((row) => row.trace_steps)),
    prompt_tokens_mean: avg(rows.map((row) => row.prompt_tokens_mean)),
    output_tokens_mean: avg(rows.map((row) => row.output_tokens_mean)),
  };
}

function renderTrainingLatestStats() {
  const row = state.trainingStats?.latest || state.trainingStats?.rows?.[state.trainingStats.rows.length - 1] || {};
  $("trainingLatestStats").innerHTML = renderStatsCards(row, 24);
}

function renderLatestStats() {
  const row = state.stats.latest || state.stats.rows[state.stats.rows.length - 1] || {};
  $("latestStats").innerHTML = renderStatsCards(row, 24);
}

function renderStatsCards(row, limit = 48, wrap = false) {
  const entries = Object.entries(row || {})
    .filter(([, value]) => typeof value === "number" || typeof value === "boolean" || value === null)
    .slice(0, limit);
  if (!entries.length) {
    return `<div class="empty-note">No scalar stats available for this row.</div>`;
  }
  const cards = entries
    .map(([key, value]) => `<div class="kv"><span>${escapeHtml(key)}</span><strong>${fmtMetric(key, value)}</strong></div>`)
    .join("");
  return wrap ? `<div class="kv-grid stats-card-grid">${cards}</div>` : cards;
}

function renderStatsChart() {
  const key = $("metricSelect").value;
  renderMetricChart(state.stats, key, "statsChart");
}

function renderTrainingStatsChart() {
  const key = $("trainingMetricSelect").value;
  renderMetricChart(state.trainingStats, key, "trainingStatsChart");
}

function renderMetricChart(stats, key, canvasId) {
  const canvas = $(canvasId);
  canvas.style.width = "100%";
  canvas.style.maxWidth = "100%";
  canvas.style.height = "190px";
  canvas.style.maxHeight = "190px";
  canvas.style.display = "block";
  canvas.style.boxSizing = "border-box";
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height || Number(canvas.getAttribute("height")) || 190));
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const values = (stats?.rows || []).map((row) => row[key]).filter((value) => typeof value === "number");
  if (!key || !values.length) {
    drawEmpty(ctx, width, height, "No numeric stats for this stream.");
    return;
  }
  const pad = { left: 54, right: 16, top: 18, bottom: 34 };
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  ctx.strokeStyle = "#d9dee7";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();
  ctx.fillStyle = "#657287";
  ctx.font = "12px system-ui";
  ctx.fillText(fmtMetric(key, max), 8, pad.top + 4);
  ctx.fillText(fmtMetric(key, min), 8, height - pad.bottom);
  ctx.strokeStyle = "#1f7a6d";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = pad.left + (index / Math.max(1, values.length - 1)) * (width - pad.left - pad.right);
    const y = pad.top + ((max - value) / span) * (height - pad.top - pad.bottom);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#17202f";
  const total = stats?.total || values.length;
  const sampled = stats?.sampled || values.length;
  ctx.fillText(`${key} · ${sampled}/${total} points`, pad.left, height - 10);
}

function drawEmpty(ctx, width, height, text) {
  ctx.fillStyle = "#657287";
  ctx.font = "14px system-ui";
  ctx.fillText(text, 20, height / 2);
}

async function loadRollouts() {
  const topic = $("rolloutTopic").value;
  const offset = state.page * state.limit;
  const params = new URLSearchParams({
    offset,
    limit: state.limit,
    q: $("searchInput").value,
    success: $("successFilter").value,
  });
  state.rollouts = await api(`/api/experiments/${encodeURIComponent(state.current)}/rollouts/${topic}?${params}`);
  renderRollouts();
  updateVisibleRolloutStats();
}

function renderRollouts() {
  const data = state.rollouts;
  const scope = data.filter_scope === "page" ? " · filters apply to this loaded page" : "";
  $("rolloutCount").textContent = `${data.total} prompt groups${scope}`;
  $("pageLabel").textContent = `Page ${state.page + 1}`;
  $("prevPage").disabled = state.page === 0;
  $("nextPage").disabled = (state.page + 1) * state.limit >= data.filtered;
  $("rolloutRows").innerHTML = data.rows
    .map((row) => {
      const outcome = row.success === true ? "ok" : row.success === false ? "fail" : "na";
      const label = row.success_rate !== null && row.success_rate !== undefined ? fmtMetric("success_rate", row.success_rate) : row.success === true ? "success" : row.success === false ? "failure" : "n/a";
      const tokens = `${fmt(row.prompt_tokens_mean, 0)} / ${fmt(row.output_tokens_mean, 0)}`;
      return `<tr data-row="${row.row_index}">
        <td>${row.row_index}</td>
        <td><span class="pill ${outcome}">${label}</span></td>
        <td>${fmt(row.reward_mean)} / ${fmt(row.reward_max)}</td>
        <td>${fmt(row.attempts, 0)}</td>
        <td>${fmt(row.trace_steps, 0)} (${fmt(row.steps_min, 0)}-${fmt(row.steps_max, 0)})</td>
        <td>${tokens}</td>
        <td>${escapeHtml(row.group_id || row.dataset_name || row.domain || "")}</td>
        <td>${escapeHtml(row.preview)}</td>
      </tr>`;
    })
    .join("");
  document.querySelectorAll("#rolloutRows tr").forEach((node) => {
    node.addEventListener("click", () => loadDetail(Number(node.dataset.row)));
  });
}

async function loadDetail(rowIndex) {
  const topic = $("rolloutTopic").value;
  const data = await api(`/api/experiments/${encodeURIComponent(state.current)}/rollouts/${topic}/${rowIndex}`);
  const group = data.group || { attempts: [] };
  const attempts = group.attempts || [];
  const stepCount = attempts.reduce((sum, attempt) => sum + (attempt.calls || []).length, 0);
  $("detailMeta").textContent = `${topic} row ${rowIndex} · ${attempts.length} attempt${attempts.length === 1 ? "" : "s"} · ${stepCount} trace step${stepCount === 1 ? "" : "s"}`;
  const hasPromptStats = group.prompt_stats && Object.keys(group.prompt_stats).length > 0;
  const statsTopic = topic === "actor_test" ? "stats_test" : "stats";
  const promptStats = hasPromptStats
    ? `<div class="section-title">
        <h3>Matched Stats Row</h3>
        <p>${statsTopic} row ${rowIndex}</p>
      </div>
      ${renderStatsCards(group.prompt_stats, 48, true)}`
    : "";
  const metrics = group.metrics ? `<pre class="json-box">${escapeHtml(JSON.stringify(group.metrics, null, 2))}</pre>` : "";
  $("detail").innerHTML = `
    <div class="kv-grid">
      <div class="kv"><span>dataset</span><strong>${escapeHtml(group.dataset_name || "n/a")}</strong></div>
      <div class="kv"><span>domain</span><strong>${escapeHtml(group.domain || "n/a")}</strong></div>
      <div class="kv"><span>group</span><strong>${escapeHtml(group.group_id || firstCall(attempts)?.group_id || "n/a")}</strong></div>
      <div class="kv"><span>latency</span><strong>${fmt(group.latency)}</strong></div>
    </div>
    ${promptStats}
    <h3 style="margin:16px 0 8px">Actor Row Metrics</h3>
    ${metrics}
    <h3 style="margin:16px 0 8px">Attempts</h3>
    ${attempts.map(renderAttempt).join("")}
  `;
}

function firstCall(attempts) {
  for (const attempt of attempts) {
    if (attempt.calls?.length) return attempt.calls[0];
  }
  return null;
}

function renderAttempt(attempt) {
  const calls = attempt.calls || [];
  return `<details class="call collapsible attempt" open>
    <summary class="call-head">
      <strong>Attempt ${fmt(attempt.rollout_index, 0)}</strong>
      <span>${calls.length} step${calls.length === 1 ? "" : "s"} · final reward ${fmt(attempt.reward_final)} · success ${fmt(attempt.success)}</span>
    </summary>
    <div class="call-body">
      ${calls.map(renderCall).join("")}
    </div>
  </details>`;
}

function renderCall(call, index) {
  const meta = call.metadata || {};
  const arrays = {};
  for (const key of ["logprobs", "ref_logprobs", "input_ids", "labels"]) {
    if (call[key] && typeof call[key] === "object") arrays[key] = call[key];
  }
  return `<details class="call collapsible step" ${index === 0 ? "open" : ""}>
    <summary class="call-head">
      <strong>Step ${fmt(meta.step_index ?? index, 0)}</strong>
      <span>reward ${fmt(call.reward)} · prompt ${fmt(call.prompt_tokens, 0)} · output ${fmt(call.output_tokens, 0)} · finished ${fmt(call.finished)}</span>
    </summary>
    <div class="call-body">
      <pre>${escapeHtml(call.text || "")}</pre>
      <h3 style="margin:14px 0 8px">Metadata</h3>
      <pre class="json-box">${escapeHtml(JSON.stringify(meta, null, 2))}</pre>
      <h3 style="margin:14px 0 8px">Arrays</h3>
      <pre class="json-box">${escapeHtml(JSON.stringify(arrays, null, 2))}</pre>
    </div>
  </details>`;
}

function finiteNumbers(values) {
  return values.filter((value) => typeof value === "number" && Number.isFinite(value));
}

function avg(values) {
  const nums = finiteNumbers(values);
  return nums.length ? nums.reduce((sum, value) => sum + value, 0) / nums.length : null;
}

function sumNumber(values) {
  const nums = finiteNumbers(values);
  return nums.length ? nums.reduce((sum, value) => sum + value, 0) : null;
}

function maxNumber(values) {
  const nums = finiteNumbers(values);
  return nums.length ? Math.max(...nums) : null;
}

function debounce(fn, wait) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

init().catch((err) => {
  document.body.innerHTML = `<main class="main"><section class="panel"><h2>Could not load viewer</h2><pre>${escapeHtml(err.stack || err)}</pre></section></main>`;
});
