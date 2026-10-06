"use strict";

const $ = (id) => document.getElementById(id);
const STATUS = {
  scored: "有效评分",
  gate_failed: "Gate 未通过",
  insufficient_evidence: "证据不足",
  not_applicable: "不适用",
  error: "评估错误",
};
const DIMENSIONS = {
  inventory: "Inventory",
  appearance: "Appearance",
  spatial: "Spatial",
  environment: "Environment",
};
const METRICS = {
  observability: "Observability · 可观察性",
  action_following: "Action Following · 动作执行",
  visual_appearance: "Visual Appearance · 视觉外观",
  geometry_shape: "Geometry / Shape · 几何结构",
  object_localization: "Object Localization · 位置",
  relative_spatial_relation: "Relative Spatial Relation · 相对关系",
  scene_structure: "Scene Structure · 场景结构",
  background_appearance: "Background Appearance · 背景外观",
  lighting_style: "Lighting / Style · 光照与风格",
  object_presence: "Object Presence · 存在率",
  object_count: "Object Count · 数量一致性",
  f1: "Inventory F1",
};
const state = {
  cases: [],
  filtered: [],
  selected: null,
  result: null,
  view: "sheets",
  frame: 1,
  controller: null,
  images: null,
  revision: 0,
  videoLoaded: false,
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
const score = (value) => (typeof value === "number" ? value.toFixed(3) : "—");
const seconds = (value) => Number(value).toFixed(3) + " s";
const media = (kind) => `/media/${encodeURIComponent(state.selected)}/${kind}`;
const badge = (status) =>
  element("span", STATUS[status] || status, `badge ${status}`);

function filterCases() {
  const query = $("search").value.trim().toLowerCase();
  state.filtered = state.cases.filter(
    (item) =>
      (!query ||
        `${item.case_id} ${item.memory_target} ${item.action_description}`
          .toLowerCase()
          .includes(query)) &&
      (!$("category").value || item.category === $("category").value) &&
      (!$("status").value || item.status === $("status").value),
  );
  const order = $("sort").value;
  state.filtered.sort((a, b) => {
    if (order !== "name") {
      if (a.memory_score === null && b.memory_score !== null) return 1;
      if (b.memory_score === null && a.memory_score !== null) return -1;
      const delta = (a.memory_score ?? 0) - (b.memory_score ?? 0);
      if (delta) return order === "low" ? delta : -delta;
    }
    return a.case_id.localeCompare(b.case_id, undefined, { numeric: true });
  });
  renderList();
  if (!state.filtered.length) {
    clearVideos();
    state.revision += 1;
    state.result = null;
    state.controller?.abort();
    state.selected = null;
    $("detail").hidden = true;
    $("message").hidden = false;
    $("message").textContent = "没有匹配的样本，请调整筛选条件。";
    $("main").setAttribute("aria-busy", "false");
  } else if (!state.filtered.some((item) => item.case_id === state.selected)) {
    selectCase(state.filtered[0].case_id);
  } else {
    updateNavigation();
  }
}

function renderList() {
  const fragment = document.createDocumentFragment();
  for (const item of state.filtered) {
    const button = element(
      "button",
      undefined,
      `case-item${item.case_id === state.selected ? " selected" : ""}`,
    );
    button.dataset.case = item.case_id;
    button.setAttribute(
      "aria-current",
      item.case_id === state.selected ? "true" : "false",
    );
    const label = element("div");
    label.append(
      element("div", item.case_id, "case-name"),
      element("div", STATUS[item.status], "case-meta"),
    );
    button.append(
      label,
      element("span", score(item.memory_score), "list-score"),
    );
    button.addEventListener("click", () => selectCase(item.case_id));
    fragment.append(button);
  }
  $("case-list").replaceChildren(fragment);
  $("count").textContent =
    `${state.filtered.length} / ${state.cases.length} 个样本`;
}

function updateNavigation() {
  const index = state.filtered.findIndex(
    (item) => item.case_id === state.selected,
  );
  $("position").textContent = `${index + 1} / ${state.filtered.length}`;
  $("prev").disabled = index <= 0;
  $("next").disabled = index < 0 || index >= state.filtered.length - 1;
}

function navigate(delta) {
  const index = state.filtered.findIndex(
    (item) => item.case_id === state.selected,
  );
  const next = state.filtered[index + delta];
  if (next) selectCase(next.case_id);
}

async function selectCase(caseId) {
  clearVideos();
  state.result = null;
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  state.selected = caseId;
  state.revision += 1;
  state.images = null;
  state.videoLoaded = false;
  $("detail").hidden = true;
  $("message").hidden = false;
  $("message").textContent = "正在加载样本…";
  $("main").setAttribute("aria-busy", "true");
  history.replaceState(null, "", `#${encodeURIComponent(caseId)}`);
  renderList();
  updateNavigation();
  try {
    const response = await fetch(`/api/case/${encodeURIComponent(caseId)}`, {
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`加载失败（HTTP ${response.status}）`);
    const result = await response.json();
    if (state.selected !== caseId || controller.signal.aborted) return;
    state.result = result;
    renderCase();
    $("message").hidden = true;
    $("detail").hidden = false;
    $("main").setAttribute("aria-busy", "false");
    setView(
      result.memory_sampling
        ? "frames"
        : state.view === "frames"
          ? "sheets"
          : state.view,
    );
  } catch (error) {
    if (error.name !== "AbortError") {
      $("message").textContent = error.message;
      $("main").setAttribute("aria-busy", "false");
    }
  }
}

function renderCase() {
  const result = state.result;
  const metadata = state.cases.find((item) => item.case_id === result.case_id);
  $("case-title").textContent = result.case_id;
  $("case-category").textContent = metadata.category;
  $("case-status").replaceChildren(badge(result.status));
  $("target").textContent = result.case.memory_target;
  $("action").textContent = result.case.action_description;
  const sampling = result.sampling;
  $("read-window").textContent = sampling
    ? `Read ${seconds(result.case.read_start_seconds)} → ${seconds(sampling.read_end_seconds)} · ${sampling.read_frame_indices.length} 个 Read 采样帧`
    : "未生成采样信息";
  $("raw-json").href = `/api/case/${encodeURIComponent(result.case_id)}`;
  const cards = [
    ["Memory", result.memory_score],
    ...Object.keys(DIMENSIONS).map((key) => [
      DIMENSIONS[key],
      result.scores?.dimensions?.[key]?.normalized,
    ]),
  ];
  $("score-cards").replaceChildren(
    ...cards.map(([name, value]) => {
      const card = element("div", undefined, "score-card");
      const key = name.toLowerCase();
      const disabled =
        name !== "Memory" &&
        result.selected_metrics &&
        !result.selected_metrics.some((metric) => metric.startsWith(key + "."));
      const number = element(
        "div",
        disabled ? "未启用" : score(value),
        "score-value",
      );
      if (!disabled) number.append(element("span", "/ 1", "score-scale"));
      const bar = element("div", undefined, "bar");
      const fill = element("div", undefined, "bar-fill");
      fill.style.width =
        typeof value === "number"
          ? `${Math.max(0, Math.min(1, value)) * 100}%`
          : "0%";
      bar.append(fill);
      card.append(element("div", name, "score-label"), number, bar);
      const validFrames =
        name === "Memory"
          ? result.scores?.scored_frames
          : result.scores?.dimensions?.[key]?.scored_frames;
      if (!disabled && validFrames !== undefined)
        card.append(element("div", `有效帧 ${validFrames} / 4`, "score-label"));
      return card;
    }),
  );
  $("score-note").textContent =
    result.status === "scored"
      ? "分数范围 0–1；展开下方指标可查看 Judge 的 0–5 原始分及证据。Inventory 分数由语义计数计算。"
      : result.status === "gate_failed"
        ? "Gate 未通过，Memory Judge 未执行；分数未计算，不能按零分理解。"
        : result.status === "insufficient_evidence"
          ? "全部采样帧均无可靠综合分；已有子指标结果仍可查看。"
          : result.status === "not_applicable"
            ? "已启用指标均不适用，综合分未计算；原因见下方 evidence。"
            : "评估调用失败，分数未计算；错误信息见下方。";
  for (const [id, kind] of [
    ["gt-sheet", "gt_sheet"],
    ["gen-sheet", "gen_sheet"],
  ]) {
    $(id).src = media(kind);
  }
  state.frame = result.memory_sampling
    ? 1
    : (sampling?.read_frame_indices?.[0] ?? 1);
  $("tab-sheets").textContent = result.protocol?.endsWith("v2")
    ? "Gate · 12 帧拼图"
    : "拼图对照";
  $("tab-frames").textContent = result.memory_sampling
    ? "Memory · 4 组单帧"
    : "逐帧放大";
  $("tab-frames").disabled =
    result.protocol?.endsWith("v2") && !result.memory_sampling;
  if (result.protocol?.endsWith("v2"))
    $("score-note").textContent +=
      " Memory 使用 20% / 40% / 60% / 80% 四组独立单帧；平均分跳过无法可靠评分的数据；有效帧数见下方。";
  renderFrameOverview();
  renderFrameSelector();
  renderJudgments();
}

function renderFrameOverview() {
  const result = state.result;
  const frames = result.memory_frames;
  $("frame-overview").hidden = !frames;
  if (!frames) return;
  const wrapper = element("div");
  wrapper.append(element("h2", "四帧评分与平均分"));
  if (result.scores?.scored_frames !== undefined) {
    wrapper.append(
      element(
        "p",
        `Memory 有效帧 ${result.scores.scored_frames} / 4；不可靠或不适用的分数跳过，零分正常参与平均。`,
        "small muted",
      ),
    );
  }
  const table = element("table");
  const head = element("tr");
  const enabled = Object.keys(DIMENSIONS).filter((dim) =>
    result.selected_metrics.some((metric) => metric.startsWith(dim + ".")),
  );
  for (const label of [
    "采样帧 / 时间",
    "Memory",
    ...enabled.map((dim) => DIMENSIONS[dim]),
    "状态",
  ])
    head.append(element("th", label));
  table.append(head);
  for (const frame of frames) {
    const row = element("tr");
    const label = element("td");
    const button = element(
      "button",
      `#${frame.frame_index} · ${20 * frame.frame_index}% · ${seconds(frame.time_seconds)}`,
      "frame-link",
    );
    button.addEventListener("click", () => {
      state.frame = frame.frame_index;
      setView("frames");
      renderFrameSelector();
      renderJudgments();
    });
    label.append(button);
    row.append(label, element("td", score(frame.memory_score), "numeric"));
    for (const dim of enabled)
      row.append(
        element(
          "td",
          score(frame.scores?.dimensions?.[dim]?.normalized),
          "numeric",
        ),
      );
    row.append(element("td", STATUS[frame.status] || frame.status));
    table.append(row);
  }
  wrapper.append(
    table,
    element(
      "p",
      "表中分数为 0–1。上方卡片是有效帧平均分；下方 evidence、Inventory 物体计数和原始 0–5 分属于当前选中的单帧。",
      "small muted",
    ),
  );
  if (result.scores?.metrics) {
    const averages = element("div", undefined, "inventory-stats");
    for (const [name, metric] of Object.entries(result.scores.metrics)) {
      const key = name.split(".")[1];
      const value = name.startsWith("inventory.")
        ? `${score(metric.normalized)} / 1`
        : `${score(metric.score)} / 5`;
      const item = element(
        "span",
        `${METRICS[key] || key} 平均（${metric.scored_frames} / 4 帧）`,
      );
      item.append(element("strong", value));
      averages.append(item);
    }
    wrapper.append(averages);
  }
  $("frame-overview").replaceChildren(wrapper);
}

function evidence(items, memoryPair = false) {
  const wrapper = element("div");
  for (const item of items ?? []) {
    const block = element("div", undefined, "evidence");
    const links = element("div");
    for (const index of item.frame_indices ?? []) {
      const sampling = memoryPair
        ? state.result.memory_sampling
        : state.result.sampling;
      const frame = sampling?.frames?.find(
        (value) => value.frame_index === index,
      );
      const button = element(
        "button",
        `#${index}${frame ? " · " + seconds(frame.time_seconds) : ""}`,
        "frame-link",
      );
      button.addEventListener("click", () => {
        if (!memoryPair && state.result.memory_sampling) {
          setView("sheets");
          $("view-sheets").scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
          return;
        }
        state.frame = index;
        setView("frames");
        renderFrameSelector();
        renderFramePair();
        $("view-frames").scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      });
      links.append(button);
    }
    block.append(links, element("p", item.observation));
    wrapper.append(block);
  }
  return wrapper;
}

function metricRow(name, metric, binary = false, memoryPair = false) {
  const row = element("div", undefined, "metric-row");
  const heading = element("div", undefined, "metric-heading");
  let value;
  if (binary)
    value = `${metric.label} / 1 · ${metric.label === 1 ? "通过" : "未通过"}`;
  else
    value =
      metric.score === null
        ? STATUS[metric.status] || metric.status
        : `${metric.score} / 5 · ${score(metric.score / 5)} / 1`;
  heading.append(
    element("span", METRICS[name] || name, "metric-title"),
    element("span", value, "metric-value"),
  );
  row.append(heading, evidence(metric.evidence, memoryPair));
  return row;
}

function section(name, subtitle, contents, open = false) {
  const details = element("details", undefined, "judge-section");
  details.open = open;
  const heading = element("summary");
  heading.append(
    element("span", name, "section-name"),
    element("span", subtitle, "small muted"),
  );
  const body = element("div", undefined, "judge-content");
  body.append(...contents);
  details.append(heading, body);
  return details;
}

function renderJudgments() {
  const result = state.result;
  const sections = [];
  if (result.gate) {
    sections.push(
      section(
        "Gate",
        "binary 0 / 1",
        Object.entries(result.gate).map(([key, value]) =>
          metricRow(key, value, true),
        ),
        result.status !== "scored",
      ),
    );
  }
  const current = result.memory_frames?.find(
    (frame) => frame.frame_index === state.frame,
  );
  const detail = current ?? result;
  const memoryPair = Boolean(current);
  $("judgments-title").textContent = current
    ? `具体指标与证据 · Memory #${state.frame} · ${seconds(current.time_seconds)}`
    : "具体指标与证据";
  const inventory = detail.judgments?.inventory;
  if (inventory) {
    const contents = [];
    const stats = detail.scores?.dimensions?.inventory;
    if (stats) {
      const values = element("div", undefined, "inventory-stats");
      for (const [key, label] of [
        ["object_presence", "Presence"],
        ["object_count", "Count accuracy"],
        ["precision", "Precision"],
        ["recall", "Recall"],
        ["f1", "F1"],
        ["missing_objects", "Missing"],
        ["spurious_objects", "Spurious"],
      ]) {
        const item = element("span", label);
        const value = stats[key];
        item.append(
          element(
            "strong",
            key.endsWith("objects") ? (value ?? "—") : score(value),
          ),
        );
        values.append(item);
      }
      contents.push(values);
    }
    if (!inventory.region_assessable)
      contents.push(element("p", "目标区域无法可靠统计 Inventory。", "reason"));
    contents.push(evidence(inventory.evidence, memoryPair));
    const table = element("table");
    const head = element("thead");
    const header = element("tr");
    for (const label of ["物体 / 语义身份", "GT", "Gen", "Missing", "Spurious"])
      header.append(element("th", label));
    head.append(header);
    const body = element("tbody");
    for (const object of inventory.objects) {
      const row = element("tr");
      const description = element("td", object.object_id);
      description.append(
        element("div", object.description, "object-description"),
        evidence(object.evidence, memoryPair),
      );
      row.append(description);
      const known = object.gt_count !== null && object.gen_count !== null;
      const counts = [
        object.gt_count,
        object.gen_count,
        known ? Math.max(object.gt_count - object.gen_count, 0) : null,
        known ? Math.max(object.gen_count - object.gt_count, 0) : null,
      ];
      for (const value of counts)
        row.append(element("td", value ?? "未知", "numeric"));
      body.append(row);
    }
    table.append(head, body);
    contents.push(table);
    if (!inventory.objects.length)
      contents.push(
        element(
          "p",
          inventory.region_assessable
            ? "该目标区域的 Inventory 标注为空；原因见上方 evidence。"
            : "无可靠物体标注。",
          "small muted",
        ),
      );
    sections.push(
      section(
        "Inventory",
        `${score(stats?.normalized)} / 1 · Python 计算`,
        contents,
        true,
      ),
    );
  }
  for (const dimension of ["appearance", "spatial", "environment"]) {
    const judgments = detail.judgments?.[dimension];
    if (judgments) {
      sections.push(
        section(
          DIMENSIONS[dimension],
          `${score(detail.scores?.dimensions?.[dimension]?.normalized)} / 1`,
          Object.entries(judgments).map(([name, metric]) =>
            metricRow(name, metric, false, memoryPair),
          ),
        ),
      );
    }
  }
  if (result.error || result.errors) {
    sections.push(
      section(
        "评估错误",
        "未评分",
        [
          element(
            "pre",
            JSON.stringify(result.errors || result.error, null, 2),
            "reason",
          ),
        ],
        true,
      ),
    );
  }
  $("judgment-sections").replaceChildren(...sections);
}

function setView(view) {
  state.view = view;
  for (const name of ["sheets", "frames", "videos"]) {
    $("view-" + name).hidden = name !== view;
    $("tab-" + name).setAttribute("aria-selected", String(name === view));
  }
  $("expand-sheet").hidden = view !== "sheets";
  if (view !== "videos") stopVideos();
  if (view === "frames") renderFramePair();
  if (view === "videos") loadVideos();
}

function renderFrameSelector() {
  const sampling = state.result.memory_sampling ?? state.result.sampling;
  const frames = sampling?.frames ?? [];
  const read = new Set(
    state.result.memory_sampling
      ? frames
          .filter((frame) => frame.is_read)
          .map((frame) => frame.frame_index)
      : (sampling?.read_frame_indices ?? []),
  );
  $("frame-selector").replaceChildren(
    ...frames.map((frame) => {
      const button = element(
        "button",
        `#${frame.frame_index} · ${seconds(frame.time_seconds)}`,
        `${read.has(frame.frame_index) ? "read " : ""}${state.frame === frame.frame_index ? "active" : ""}`,
      );
      button.title = read.has(frame.frame_index) ? "Read 阶段" : "动作上下文";
      button.addEventListener("click", () => {
        state.frame = frame.frame_index;
        renderFrameSelector();
        renderFramePair();
        renderJudgments();
      });
      return button;
    }),
  );
}

async function sheetImages() {
  if (!state.images) {
    state.images = Promise.all(
      ["gt_sheet", "gen_sheet"].map(async (kind) => {
        const image = new Image();
        image.src = media(kind);
        await image.decode();
        return image;
      }),
    );
  }
  return state.images;
}

async function renderFramePair() {
  const revision = state.revision;
  const index = state.frame;
  const sampling = state.result.memory_sampling ?? state.result.sampling;
  if (state.result.memory_sampling) {
    const frame = sampling.frames.find((value) => value.frame_index === index);
    const figures = ["gt", "gen"].map((role) => {
      const figure = element("figure");
      figure.append(
        element(
          "figcaption",
          role === "gt" ? "GT · Reference" : "Gen · Generated",
          role === "gt" ? "reference" : "generated",
        ),
      );
      const image = element("img");
      image.src = media(`${role}_frame_${index}`);
      image.alt = `${role.toUpperCase()} memory frame #${index}`;
      figure.append(
        image,
        element(
          "p",
          `#${index} · 对齐时间 ${seconds(frame.time_seconds)} · 实际帧 ${seconds(frame[role + "_actual_seconds"])}`,
          "frame-subtitle",
        ),
      );
      return figure;
    });
    $("frame-pair").replaceChildren(...figures);
    renderJudgments();
    return;
  }
  $("frame-pair").replaceChildren(element("p", "正在载入帧…", "muted"));
  if (!sampling) return;
  try {
    const images = await sheetImages();
    if (revision !== state.revision || index !== state.frame) return;
    const [width, height] = sampling.tile_size;
    const frame = sampling.frames.find((value) => value.frame_index === index);
    const figures = images.map((image, position) => {
      const figure = element("figure");
      figure.append(
        element(
          "figcaption",
          position === 0 ? "GT · Reference" : "Gen · Generated",
          position === 0 ? "reference" : "generated",
        ),
      );
      const canvas = element("canvas");
      canvas.width = width;
      canvas.height = height;
      canvas.style.aspectRatio = `${width} / ${height}`;
      // Crop the original padded content tile; keep every source pixel and its aspect ratio.
      const x = ((index - 1) % 6) * width;
      const header = image.height - 2 * (height + 32);
      const y = header + Math.floor((index - 1) / 6) * (height + 32) + 32;
      canvas
        .getContext("2d")
        .drawImage(image, x, y, width, height, 0, 0, width, height);
      const actual =
        position === 0 ? frame.gt_actual_seconds : frame.gen_actual_seconds;
      figure.append(
        canvas,
        element(
          "p",
          `#${index} · 对齐时间 ${seconds(frame.time_seconds)} · 实际帧 ${seconds(actual)}`,
          "frame-subtitle",
        ),
      );
      return figure;
    });
    $("frame-pair").replaceChildren(...figures);
  } catch (error) {
    if (revision === state.revision)
      $("frame-pair").replaceChildren(
        element("p", `无法加载拼图：${error.message}`, "reason"),
      );
  }
}

function stopVideos() {
  $("gt-video").pause();
  $("gen-video").pause();
  $("play").textContent = "同步播放";
}

function clearVideos() {
  stopVideos();
  state.videoLoaded = false;
  for (const id of ["gt-video", "gen-video"]) {
    $(id).removeAttribute("src");
    $(id).load();
  }
}

function loadVideos() {
  if (state.videoLoaded) return;
  state.videoLoaded = true;
  $("play").disabled = true;
  $("seek").disabled = true;
  $("video-error").textContent = "正在加载视频…";
  $("seek").max = state.result.sampling?.shared_end_seconds ?? 1;
  $("gt-video").src = media("gt_video");
  $("gen-video").src = media("gen_video");
}

function seekVideos(time) {
  const offsets = [
    state.result.case.gt_start_seconds,
    state.result.case.gen_start_seconds,
  ];
  for (const [index, id] of ["gt-video", "gen-video"].entries()) {
    if ($(id).readyState >= 1)
      $(id).currentTime = Number(time) + offsets[index];
  }
  $("seek").value = time;
  $("video-time").textContent = seconds(time);
}

for (const id of ["gt-video", "gen-video"]) {
  $(id).addEventListener("loadedmetadata", () => {
    if (!state.videoLoaded || !state.result) return;
    if (["gt-video", "gen-video"].every((name) => $(name).readyState >= 1)) {
      $("play").disabled = false;
      $("seek").disabled = false;
      $("video-error").textContent =
        "两个视频按同一时间轴播放；拖动滑条可同步跳转。";
      seekVideos(state.result.case.read_start_seconds);
    }
  });
  $(id).addEventListener("error", () => {
    if (!state.videoLoaded || !state.result) return;
    $("video-error").textContent = "原视频无法加载；拼图和评分仍可查看。";
  });
}
$("gt-video").addEventListener("timeupdate", () => {
  if (!state.result || !state.videoLoaded || state.view !== "videos") return;
  const time = Math.max(
    0,
    $("gt-video").currentTime - state.result.case.gt_start_seconds,
  );
  const end = state.result.sampling?.shared_end_seconds;
  if (typeof end === "number" && time >= end) {
    stopVideos();
    seekVideos(end);
    return;
  }
  $("seek").value = time;
  $("video-time").textContent = seconds(time);
  const expected = time + state.result.case.gen_start_seconds;
  if (
    !$("gt-video").paused &&
    Math.abs($("gen-video").currentTime - expected) > 0.15
  )
    $("gen-video").currentTime = expected;
});
$("gt-video").addEventListener("ended", stopVideos);
$("play").addEventListener("click", async () => {
  if (!$("gt-video").paused) {
    stopVideos();
    return;
  }
  const end = state.result.sampling?.shared_end_seconds ?? Infinity;
  if (Number($("seek").value) >= end) seekVideos(0);
  try {
    await Promise.all([$("gt-video").play(), $("gen-video").play()]);
    $("play").textContent = "暂停";
  } catch {
    stopVideos();
    $("video-error").textContent = "无法播放视频，请尝试拖动时间滑条后再播放。";
  }
});
$("seek").addEventListener("input", () => seekVideos($("seek").value));
$("speed").addEventListener("change", () => {
  for (const id of ["gt-video", "gen-video"])
    $(id).playbackRate = Number($("speed").value);
});
for (const tab of document.querySelectorAll("[data-view]"))
  tab.addEventListener("click", () => setView(tab.dataset.view));
$("expand-sheet").addEventListener("click", () => {
  $("dialog-gt").src = media("gt_sheet");
  $("dialog-gen").src = media("gen_sheet");
  $("sheet-dialog").showModal();
});
$("close-dialog").addEventListener("click", () => $("sheet-dialog").close());
$("prev").addEventListener("click", () => navigate(-1));
$("next").addEventListener("click", () => navigate(1));
for (const id of ["search", "category", "status", "sort"])
  $(id).addEventListener(id === "search" ? "input" : "change", filterCases);
document.addEventListener("keydown", (event) => {
  if (
    ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(
      document.activeElement.tagName,
    ) ||
    $("sheet-dialog").open
  )
    return;
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    navigate(event.key === "ArrowLeft" ? -1 : 1);
  }
});

window.addEventListener("hashchange", () => {
  const linked = decodeURIComponent(location.hash.slice(1));
  if (
    linked === state.selected ||
    !state.cases.some((item) => item.case_id === linked)
  )
    return;
  for (const id of ["search", "category", "status"]) $(id).value = "";
  state.selected = linked;
  filterCases();
  selectCase(linked);
});

async function init() {
  try {
    const response = await fetch("/api/catalog");
    if (!response.ok)
      throw new Error(`目录加载失败（HTTP ${response.status}）`);
    const data = await response.json();
    state.cases = data.cases;
    for (const category of [
      ...new Set(data.cases.map((item) => item.category)),
    ].sort()) {
      const option = element("option", category);
      option.value = category;
      $("category").append(option);
    }
    for (const [label, value] of [
      ["Model", data.model],
      ["Cases", data.cases.length],
      ["可评分", `${data.summary.scored_cases ?? "—"}/${data.cases.length}`],
      [
        "Gate 通过",
        typeof data.summary.gate_pass_rate === "number"
          ? `${(100 * data.summary.gate_pass_rate).toFixed(1)}%`
          : "—",
      ],
      ["Memory 均值", score(data.summary.memory_score)],
    ]) {
      const item = element("span", label);
      item.append(element("strong", value));
      $("overview").append(item);
    }
    const linked =
      decodeURIComponent(location.hash.slice(1)) ||
      (data.protocol?.endsWith("v2")
        ? state.cases.find((item) => item.status === "scored")?.case_id
        : "");
    state.selected = state.cases.some((item) => item.case_id === linked)
      ? linked
      : null;
    filterCases();
    if (linked && state.selected === linked) selectCase(linked);
  } catch (error) {
    $("message").textContent = error.message;
    $("main").setAttribute("aria-busy", "false");
  }
}
init();
