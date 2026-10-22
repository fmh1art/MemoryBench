"use strict";
const $ = id => document.getElementById(id);
let config, current, revision = 0, elapsedBase = 0, started = 0, saveTimer, heartbeatTimer;
let saving = Promise.resolve(), busy = false, blocked = false, suppress = new Map();
const videos = Object.fromEntries(["history", "gt", "A", "B"].map(s => [s, $("video-" + s)]));
const future = ["gt", "A", "B"];
const ready = new Set(), errors = new Set();
let messageTimer;

function message(text, persistent = false) {
  $("message").textContent = text; $("message").hidden = false;
  clearTimeout(messageTimer);
  if (!persistent) messageTimer = setTimeout(() => $("message").hidden = true, 7000);
}
async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) {
    const error = new Error(typeof result.detail === "string" ? result.detail : "请求内容无效，请检查填写项");
    error.status = response.status; throw error;
  }
  return result;
}
function key(suffix) { return "memorybench-pairwise-" + (config.session_id || config.study_id) + "-" + suffix; }
function localGet(name) { try { return JSON.parse(localStorage.getItem(key(name))); } catch { return null; } }
function localSet(name, data) { try { localStorage.setItem(key(name), JSON.stringify(data)); } catch {} }
function localRemove(name) { try { localStorage.removeItem(key(name)); } catch {} }

function checked() { return document.querySelector('input[name="choice"]:checked')?.value; }
function choose(value) {
  document.querySelectorAll('input[name="choice"]').forEach(input => input.checked = input.value === value);
}
function collect() {
  return {choice: checked() ?? null, elapsed_seconds: elapsedBase + (performance.now() - started) / 1000};
}
function restore(data) {
  choose(data.choice); elapsedBase = data.elapsed_seconds || 0;
}
function lockChoices() {
  document.querySelectorAll('input[name="choice"]').forEach(input => input.disabled = blocked || busy);
}
function keepLocal() {
  if (current) localSet("draft-" + current.assignment_id, {revision, answers: collect()});
}
function queueSave() {
  clearTimeout(saveTimer);
  if (!current || busy || blocked) return saving;
  const aid = current.assignment_id;
  saving = saving.catch(() => {}).then(async () => {
    if (!current || current.assignment_id !== aid || blocked) return;
    const answers = collect();
    $("save-state").textContent = "正在保存…";
    const result = await api(`/api/task/${aid}/draft`, {revision, answers});
    revision = result.revision; keepLocal(); $("save-state").textContent = "草稿已保存";
  }).catch(error => {
    $("save-state").textContent = "服务器未保存；本机草稿已暂存";
    if (error.status === 409) { blocked = true; lockChoices(); $("submit-vote").disabled = true; }
    message(error.message, error.status === 409); throw error;
  });
  // Catch on a separate branch so autosave failures never become unhandled rejections.
  saving.catch(() => {}); return saving;
}
function changed() {
  if (!current || busy || blocked) return;
  lockChoices(); mediaState(); keepLocal(); $("save-state").textContent = "本机已暂存";
  clearTimeout(saveTimer); saveTimer = setTimeout(queueSave, 700);
}
$("vote-form").addEventListener("change", changed);
window.addEventListener("beforeunload", keepLocal);

function promptText(prompt) {
  return !prompt.recorded ? "未记录文本条件" : prompt.text === null || prompt.text === "" ? "无文本条件" : prompt.text;
}
function requiredMediaCount() { return 4 + Object.keys(current?.references || {}).length; }
function mediaState() {
  const required = requiredMediaCount();
  $("media-state").textContent = errors.size ? "存在播放故障，请报告问题并跳过" : ready.size === required ? `${required} 段视频已就绪` : `正在加载视频（${ready.size} / ${required}）…`;
  $("submit-vote").disabled = busy || blocked || ready.size !== required || errors.size > 0 || !checked();
}
function suppressEvents(slot) { suppress.set(slot, performance.now() + 400); }
function playbackMaster() { return current?.playback_master || "gt"; }
function seekAll(time) {
  if (!current) return;
  time = Math.max(0, Math.min(current.timeline_seconds, time));
  for (const slot of future) {
    const clip = current.clips[slot], video = videos[slot];
    suppressEvents(slot);
    const target = Math.min(clip.duration_seconds - 0.001, clip.start_seconds + time);
    if (Number.isFinite(target) && Math.abs(video.currentTime - target) > 0.025) video.currentTime = target;
  }
  $("timeline").value = String(time); updateClock(time);
}
function pauseAll() {
  for (const slot of future) { suppressEvents(slot); videos[slot].pause(); }
  $("play-all").textContent = "▶ 播放 GT / A / B";
}
async function playAll(time) {
  if (!current) return;
  if (time !== undefined) seekAll(time);
  const master = playbackMaster();
  const now = Math.max(0, videos[master].currentTime - current.clips[master].start_seconds);
  if (now >= current.timeline_seconds - .03) seekAll(0);
  for (const slot of future) {
    const video = videos[slot]; suppressEvents(slot);
    if (video.currentTime < current.clips[slot].duration_seconds - .02)
      video.play().catch(() => message("浏览器未能播放视频，请点击视频播放键，或报告播放问题。"));
  }
  $("play-all").textContent = "Ⅱ 暂停 GT / A / B";
}
function updateClock(time) { if (current) $("time-label").textContent = `${time.toFixed(2)} / ${current.timeline_seconds.toFixed(2)} s`; }
for (const [slot, video] of Object.entries(videos)) {
  video.addEventListener("loadeddata", () => {
    ready.add(slot); errors.delete(slot); $("error-" + slot).hidden = true; mediaState();
    if (current && current.clips[slot].start_seconds > 0) { suppressEvents(slot); video.currentTime = current.clips[slot].start_seconds; }
  });
  video.addEventListener("error", () => { if (!current || !video.getAttribute("src")) return; errors.add(slot); $("error-" + slot).hidden = false; mediaState(); });
  video.addEventListener("dblclick", () => video.requestFullscreen?.());
  if (slot === "history") continue;
  video.addEventListener("play", () => {
    if (!current || !$("sync").checked || performance.now() < (suppress.get(slot) || 0)) return;
    playAll(video.currentTime - current.clips[slot].start_seconds);
  });
  video.addEventListener("pause", () => {
    if (!current || !$("sync").checked || video.ended || performance.now() < (suppress.get(slot) || 0)) return;
    pauseAll();
  });
  video.addEventListener("seeking", () => {
    if (!current || !$("sync").checked || performance.now() < (suppress.get(slot) || 0)) return;
    seekAll(video.currentTime - current.clips[slot].start_seconds);
  });
}
for (const slot of future) videos[slot].addEventListener("ended", () => {
  if ($("sync").checked && slot === playbackMaster()) pauseAll();
});
function tick() {
  const master = playbackMaster();
  if (current && !videos[master].paused && !videos[master].seeking) {
    const time = Math.max(0, Math.min(current.timeline_seconds, videos[master].currentTime - current.clips[master].start_seconds));
    $("timeline").value = String(time); updateClock(time);
    if ($("sync").checked) for (const slot of future.filter(slot => slot !== master)) {
      const video = videos[slot], clip = current.clips[slot];
      const desired = Math.min(clip.duration_seconds - .001, time + clip.start_seconds);
      if (!video.seeking && Math.abs(video.currentTime - desired) > .12) { suppressEvents(slot); video.currentTime = desired; }
      if (desired >= clip.duration_seconds - .01 && !video.paused) { suppressEvents(slot); video.pause(); }
    }
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
$("play-all").onclick = () => videos[playbackMaster()].paused ? playAll() : pauseAll();
$("restart").onclick = () => { pauseAll(); seekAll(0); };
$("timeline").oninput = event => seekAll(Number(event.target.value));
$("jump-read").onclick = () => seekAll(current?.read_start_seconds || 0);
$("sync").onchange = () => { if ($("sync").checked && current) { const master = playbackMaster(); pauseAll(); seekAll(videos[master].currentTime - current.clips[master].start_seconds); } };
$("future-speed").onchange = event => { for (const slot of future) videos[slot].playbackRate = Number(event.target.value); };
$("history-speed").onchange = event => videos.history.playbackRate = Number(event.target.value);
document.querySelectorAll(".fullscreen").forEach(button => button.onclick = () => videos[button.dataset.slot].requestFullscreen?.());

async function next() {
  clearTimeout(saveTimer); clearInterval(heartbeatTimer); pauseAll(); videos.history.pause();
  for (const slot of ["A", "B"]) $("reference-video-" + slot).pause();
  const response = await api("/api/next"); current = response.task; blocked = false; busy = false;
  $("completed").textContent = response.progress.completed;
  $("sampling-note").textContent = response.progress.random_assignment ?
    `当前题库：${config.models} 个模型、${config.cases} 个 case、${response.progress.total_tasks} 个比较组合。随机出题，可能重复；每次提交独立保存，可随时退出后续评。` : "";
  $("task-content").hidden = !current; $("finished").hidden = !!current;
  if (!current) {
    const personalComplete = response.progress.personal_complete;
    $("refresh-tasks").textContent = personalComplete ? "检查新增任务" : "刷新任务";
    $("finished-title").textContent = personalComplete ? "你已完成本轮全部可评任务，谢谢参与。" :
      response.progress.study_complete ? "本轮评估已完成，谢谢参与。" : "任务暂时已被其他评估者领取";
    $("finished-description").textContent = personalComplete ?
      `当前题库共 ${response.progress.total_tasks} 题，你已提交 ${response.progress.completed} 次判断，报告 ${response.progress.issues} 个播放问题。记录已保存；刷新不会重复分配已完成的题目，需要管理员增加新任务后才能继续。` :
      response.progress.study_complete ? "本轮所需判断已收齐，你的记录已保存。" :
      "你的记录已保存。其他评估者的任务占用释放后，可以刷新页面继续。";
    return response;
  }
  const references = current.references || {};
  ready.clear(); errors.clear();
  $("model-references").hidden = !Object.keys(references).length;
  for (const slot of ["A", "B"]) {
    const reference = references[slot], video = $("reference-video-" + slot);
    video.onloadeddata = null; video.onloadedmetadata = null; video.onerror = null;
    video.pause(); video.removeAttribute("src"); video.load();
    $("reference-card-" + slot).hidden = !reference;
    if (reference) {
      video.src = reference.url;
      video.preload = "auto";
      video.onloadeddata = () => { ready.add("reference-" + slot); errors.delete("reference-" + slot); mediaState(); };
      video.onerror = () => { if (video.getAttribute("src")) { errors.add("reference-" + slot); mediaState(); } };
      video.onloadedmetadata = () => { video.currentTime = reference.start_seconds; };
      $("reference-context-" + slot).textContent = [reference.action_description, reference.memory_target,
        reference.success_criterion].filter(Boolean).join("；");
    }
  }
  revision = current.revision;
  $("vote-form").reset(); started = performance.now();
  const cached = localGet("draft-" + current.assignment_id);
  restore(cached && cached.revision === revision ? cached.answers : current.draft);
  lockChoices(); $("save-state").textContent = "草稿自动保存";
  const a = promptText(current.prompts.A), b = promptText(current.prompts.B);
  $("common-prompt").hidden = a !== b; $("separate-prompts").hidden = a === b;
  $("common-prompt").textContent = a; $("prompt-a").textContent = a; $("prompt-b").textContent = b;
  for (const field of ["action-description", "memory-target", "success-criterion"]) $(field).textContent = current[field.replaceAll("-", "_")] || "未提供";
  $("timeline").max = String(current.timeline_seconds); $("timeline").value = "0"; updateClock(0);
  $("read-interval").setAttribute("x", String(1000 * current.read_start_seconds / current.timeline_seconds));
  $("read-interval").setAttribute("width", String(1000 * ((current.read_end_seconds ?? current.timeline_seconds) - current.read_start_seconds) / current.timeline_seconds));
  $("read-label").textContent = `评估段 ${current.read_start_seconds.toFixed(2)}–${(current.read_end_seconds ?? current.timeline_seconds).toFixed(2)} s`;
  for (const [slot, video] of Object.entries(videos)) {
    $("error-" + slot).hidden = true; video.src = current.clips[slot].url; video.load();
    video.playbackRate = Number($(slot === "history" ? "history-speed" : "future-speed").value);
    if (slot === "A" || slot === "B") {
      const duration = current.clips[slot].duration_seconds - current.clips[slot].start_seconds;
      $("duration-" + slot).textContent = `${duration.toFixed(2)} s` + (duration < current.timeline_seconds - .12 ? " · 预测较短；末尾停在最后一帧" : " · 原始时间轴");
    }
  }
  $("issue-description").value = ""; mediaState();
  heartbeatTimer = setInterval(() => {
    if (!current || blocked) return;
    api(`/api/task/${current.assignment_id}/heartbeat`, {}).catch(error => {
      if (error.status === 409) { blocked = true; mediaState(); message(error.message, true); }
    });
  }, 60000);
  window.scrollTo({top: 0, behavior: "instant"});
  return response;
}
$("vote-form").onsubmit = async event => {
  event.preventDefault(); if (!current || busy || blocked || errors.size || ready.size !== requiredMediaCount()) return;
  if (!checked()) return message("请选择 A、B 或 Tie。");
  clearTimeout(saveTimer); busy = true; mediaState();
  const aid = current.assignment_id;
  try {
    await saving.catch(() => {});
    const result = await api(`/api/task/${aid}/vote`, {revision, answers: collect()});
    revision = result.revision; localRemove("draft-" + aid); current = null;
    message("已提交，谢谢。" ); await next();
  } catch (error) { message(error.message, true); busy = false; if (error.status === 409) blocked = true; mediaState(); }
};
$("report-issue").onclick = async () => {
  const description = $("issue-description").value.trim();
  if (!description) return message("请简要说明哪一段视频有问题。");
  if (!current || busy) return;
  busy = true; clearTimeout(saveTimer); mediaState();
  try {
    await saving.catch(() => {});
    const aid = current.assignment_id;
    await api(`/api/task/${aid}/issue`, {description}); localRemove("draft-" + aid); current = null;
    message("问题已记录，本题未计作模型失败。" ); await next();
  } catch (error) { busy = false; message(error.message, true); mediaState(); }
};
$("refresh-tasks").onclick = async () => {
  const button = $("refresh-tasks");
  button.disabled = true;
  try {
    const response = await next();
    if (!response.task) message(response.progress.personal_complete ?
      `已检查：当前仍为 ${response.progress.total_tasks} 题，暂无新增任务。已有标注已保存。` :
      "已检查：当前没有可领取任务，请稍后再试。");
  } catch (error) { message(error.message, true); }
  finally { button.disabled = false; }
};

async function enter(name) {
  $("welcome").hidden = true; $("workspace").hidden = false; $("identity").hidden = false;
  $("rater-name").textContent = name;
  const saved = localGet("session");
  if (saved?.resume_code) { $("my-code").textContent = saved.resume_code; $("copy-code").hidden = false; }
  await next();
}
$("copy-code").onclick = async () => {
  try { await navigator.clipboard.writeText($("my-code").textContent); message("续评码已复制，请妥善保存。"); }
  catch { message("请手动选中并复制续评码。"); }
};
$("login-form").onsubmit = async event => {
  event.preventDefault();
  try {
    const result = await api("/api/session", {name: $("name").value.trim(), resume_code: $("resume-code").value.trim() || null});
    localSet("session", result); await enter(result.name);
    document.querySelector(".session-details").open = true;
  } catch (error) { message(error.message, true); }
};
$("logout").onclick = async () => {
  clearTimeout(saveTimer); clearInterval(heartbeatTimer);
  try { if (current && !blocked) await queueSave(); await saving.catch(() => {}); await api("/api/logout", {}); }
  catch (error) { return message(error.message, true); }
  localRemove("session"); location.reload();
};
(async () => {
  try {
    config = await api("/api/config"); document.title = config.title;
    try { const person = await api("/api/session"); await enter(person.name); }
    catch (error) {
      if (error.status !== 401) throw error;
      const saved = localGet("session");
      if (saved?.resume_code) { const person = await api("/api/session", {resume_code: saved.resume_code}); await enter(person.name); }
      else $("welcome").hidden = false;
    }
  } catch (error) { $("welcome").hidden = false; message(error.message, true); }
})();
