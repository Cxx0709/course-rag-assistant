"use strict";
const $ = id => document.getElementById(id);
const state = { token: "", documents: [], selected: new Set(), configured: false, busy: false, sources: [], maxUploadBytes: 50 * 1024 * 1024 };

function message(text = "", error = false) {
  $("notice").hidden = true; $("error").hidden = true;
  if (text) { const node = $(error ? "error" : "notice"); node.textContent = text; node.hidden = false; }
}
function busy(value, label = "") {
  state.busy = value;
  for (const id of ["upload-button", "demo-button", "file-input", "select-all", "question", "compare-concept"]) $(id).disabled = value;
  $("use-model").disabled = value || !state.configured;
  document.querySelectorAll(".document input, [data-question], .comparison-senses button").forEach(node => { node.disabled = value; });
  $("ask-label").textContent = label || "开始提问";
  updateScope();
}
function updateScope() {
  const count = state.selected.size;
  $("selection-count").textContent = state.documents.length ? `已选 ${count} / ${state.documents.length} 份` : "尚未添加资料";
  $("ask-button").disabled = state.busy || !count || !$("question").value.trim();
  $("quiz-create").disabled = state.busy || !count;
  $("review-load").disabled = state.busy || !count;
  $("quiz-submit").disabled = state.busy;
  $("compare-button").disabled = state.busy || count !== 2 || $("compare-concept").value.trim().length < 2;
  $("compare-help").textContent = count === 2 ? "仅对比已选的两份资料；请填写资料中的概念名称。" : `已选 ${count} 份，请恰好选中两份资料进行对比。`;
  document.querySelectorAll("#quiz-form input").forEach(node => { node.disabled = state.busy; });
  $("select-all").textContent = count === state.documents.length ? "取消全选" : "全选";
  $("question-help").textContent = count ? `仅从已选的 ${count} 份资料中回答。Ctrl / ⌘ + Enter 提问` : "先添加并勾选资料，或使用下方示例体验。";
}
function renderDocuments() {
  $("compare-result").hidden = true;
  $("document-count").textContent = state.documents.length;
  $("empty-library").hidden = Boolean(state.documents.length);
  $("select-all").hidden = !state.documents.length;
  $("getting-started").hidden = Boolean(state.documents.length);
  $("suggestions").hidden = !state.documents.some(doc => doc.name === "机器学习入门示例.md" && state.selected.has(doc.id));
  $("documents").replaceChildren();
  for (const doc of state.documents) {
    const label = document.createElement("label"); label.className = `document${state.selected.has(doc.id) ? " selected" : ""}`;
    const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.checked = state.selected.has(doc.id); checkbox.disabled = state.busy;
    checkbox.setAttribute("aria-label", `使用资料：${doc.name}`);
    checkbox.addEventListener("change", () => { checkbox.checked ? state.selected.add(doc.id) : state.selected.delete(doc.id); renderDocuments(); });
    const info = document.createElement("span"); info.className = "document-info";
    const name = document.createElement("span"); name.className = "document-name"; name.textContent = doc.name;
    const meta = document.createElement("span"); meta.className = "document-meta";
    const tag = document.createElement("span"); tag.className = "file-tag"; tag.textContent = doc.type.toUpperCase();
    meta.append(tag, document.createTextNode(`${doc.pages ? `正文至第 ${doc.pages} ${doc.type === "pptx" ? "张" : "页"} · ` : ""}${Math.max(1, Math.round(doc.size / 1024))} KB`));
    info.append(name, meta); label.append(checkbox, info); $("documents").append(label);
  }
  updateScope();
}
async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { ...options, headers: { "X-Workspace-Token": state.token, ...options.headers } });
  } catch { throw new Error("无法连接本地服务。请确认启动窗口仍在运行，再重试。"); }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(typeof data?.detail === "string" ? data.detail : `请求未完成（${response.status}），请检查输入后重试。`);
  }
  if (!data) throw new Error("服务返回了无法读取的内容，请重试。");
  return data;
}
function rememberDocument(doc) {
  if (!state.documents.some(item => item.id === doc.id)) state.documents.push(doc);
  state.selected.add(doc.id); renderDocuments();
}
async function upload(files) {
  if (state.busy || !files.length) return;
  busy(true); let succeeded = 0; const failures = [];
  for (const file of files) {
    message(`正在读取「${file.name}」…`);
    try {
      if (file.size > state.maxUploadBytes) throw new Error(`超过 ${Math.round(state.maxUploadBytes / (1024 * 1024))} MB，请按章节拆分。`);
      if (!/\.(pdf|pptx|md)$/i.test(file.name)) throw new Error("仅支持 PDF、PPTX 和 .md 文件。");
      const doc = await api(`/api/documents?filename=${encodeURIComponent(file.name)}`, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file });
      rememberDocument(doc); succeeded++;
    } catch (error) { failures.push(`${file.name}：${error.message}`); }
  }
  $("file-input").value = ""; busy(false);
  message(failures.length ? `${succeeded ? `已添加 ${succeeded} 份。` : ""}${failures.join("；")}` : `已添加 ${succeeded} 份资料。现在可以开始提问了。`, Boolean(failures.length));
  if (succeeded) $("question").focus();
}
function openSource(id) {
  const source = state.sources.find(item => item.id === id);
  if (!source) return;
  $("source-title").textContent = source.name;
  $("source-location").textContent = source.page ? `${source.type === "pptx" ? "幻灯片" : "PDF"} 第 ${source.page} ${source.type === "pptx" ? "张" : "页"}${source.section ? ` · ${source.section}` : ""}` : source.section || "正文";
  $("source-text").textContent = source.text;
  $("source-open").href = source.url;
  $("source-open").textContent = source.type === "pdf" ? "打开 PDF 对应页 ↗" : "下载原始文件 ↓";
  $("source-page-note").textContent = source.type === "pdf" ? "页码按 PDF 文件中的页面顺序计算。" : "这里显示的是文件中提取的原文。";
  $("source-dialog").showModal();
}
function renderAnswer(data) {
  state.sources = data.sources;
  $("answer-title").textContent = data.question;
  $("answer-scope").textContent = `本次查阅：${data.document_names.join("、")}`;
  $("answer-mode").textContent = data.abstained ? "证据不足" : { source: "原文回答", model: "模型辅助", fallback: "原文回退" }[data.mode];
  $("answer-content").replaceChildren();
  for (const claim of data.claims) {
    const p = document.createElement("p"); p.className = "claim"; p.append(document.createTextNode(claim.text));
    for (const id of claim.source_ids) {
      const button = document.createElement("button"); button.type = "button"; button.className = "citation"; button.textContent = id;
      button.setAttribute("aria-label", `查看引用 ${id} 的原文`); button.addEventListener("click", () => openSource(id)); p.append(button);
    }
    $("answer-content").append(p);
  }
  $("source-list").replaceChildren();
  for (const source of data.sources) {
    const button = document.createElement("button"); button.type = "button"; button.className = "source-chip";
    const number = document.createElement("span"); number.className = "source-number"; number.textContent = source.id;
    const label = document.createElement("span"); label.textContent = `${source.name} · ${source.page ? `第 ${source.page} ${source.type === "pptx" ? "张" : "页"}` : source.section || "正文"}`;
    button.append(number, label); button.addEventListener("click", () => openSource(source.id)); $("source-list").append(button);
  }
  $("answer-sources").hidden = !data.sources.length;
  renderLearningPlans($("answer-prerequisites"), data.learning_plans || []);
  if (data.mode === "fallback") message("模型暂时不可用，已改用资料原文回答。");
}
async function ask(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (state.busy) return;
  if (!question || !state.selected.size) { message("请先选择资料并输入问题。", true); return; }
  message(); busy(true, "查阅中…");
  $("source-dialog").close(); state.sources = [];
  $("answer").hidden = false; $("answer-title").textContent = question; $("answer-scope").textContent = "正在查阅所选资料，请稍候。";
  $("answer-mode").textContent = "查阅中"; $("answer-sources").hidden = true;
  $("answer-prerequisites").replaceChildren(); $("answer-prerequisites").hidden = true;
  const loading = document.createElement("div"); loading.className = "loading";
  const spinner = document.createElement("span"); spinner.className = "spinner"; spinner.setAttribute("aria-hidden", "true");
  loading.append(spinner, document.createTextNode($("use-model").checked ? "正在查找原文并整理回答，本机模型可能需要一些时间…" : "正在查找相关原文…"));
  $("answer-content").replaceChildren(loading); $("answer-content").setAttribute("aria-busy", "true");
  try {
    const data = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, document_ids: [...state.selected], use_model: $("use-model").checked }) });
    renderAnswer(data);
  } catch (error) {
    $("answer-mode").textContent = "未完成"; $("answer-content").textContent = error.message; message(error.message, true);
  } finally {
    busy(false); $("answer-content").setAttribute("aria-busy", "false");
    $("answer").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}
$("upload-button").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", event => upload([...event.target.files]));
$("select-all").addEventListener("click", () => { state.selected = state.selected.size === state.documents.length ? new Set() : new Set(state.documents.map(d => d.id)); renderDocuments(); });
$("question").addEventListener("input", updateScope);
$("question").addEventListener("keydown", event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); $("ask-form").requestSubmit(); } });
$("ask-form").addEventListener("submit", ask);
$("close-source").addEventListener("click", () => $("source-dialog").close());
document.querySelectorAll("[data-question]").forEach(button => button.addEventListener("click", () => { $("question").value = button.dataset.question; updateScope(); $("question").focus(); }));
for (const eventName of ["dragover", "drop"]) window.addEventListener(eventName, event => { if ([...event.dataTransfer.types].includes("Files")) event.preventDefault(); });
$("dropzone").addEventListener("dragover", event => { event.preventDefault(); if (!state.busy) $("dropzone").classList.add("dragging"); });
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("dragging"));
$("dropzone").addEventListener("drop", event => { event.preventDefault(); $("dropzone").classList.remove("dragging"); upload([...event.dataTransfer.files]); });
$("demo-button").addEventListener("click", async () => {
  if (state.busy) return;
  busy(true); message("正在准备示例资料…");
  try {
    const doc = await api("/api/demo", { method: "POST" }); rememberDocument(doc);
    $("question").value = "训练集和测试集的用途有什么不同？"; message("示例资料已准备好。点击「开始提问」体验带出处的回答。");
  } catch (error) { message(error.message, true); } finally { busy(false); $("question").focus(); }
});
async function init() {
  busy(true);
  try {
    const data = await api("/api/state");
    state.maxUploadBytes = data.max_upload_bytes;
    $("upload-limit").textContent = `每份最多 ${Math.round(state.maxUploadBytes / (1024 * 1024))} MB`;
    state.token = data.token; state.documents = data.documents; state.selected = new Set(data.documents.map(d => d.id)); state.configured = data.model_configured;
    $("connection").textContent = "本地学习空间";
    $("model-note").textContent = state.configured ? "更精简，可能需要稍等" : "未配置时可直接查原文";
    renderDocuments();
    if (data.unreadable_documents) message(`${data.unreadable_documents} 份已保存资料无法读取，可重新上传原文件。`, true);
  } catch (error) { $("connection").textContent = "连接未完成"; message(error.message, true); }
  finally { busy(false); }
}
let currentQuiz = null;
function renderLearningPlans(container, plans) {
  container.replaceChildren(); container.hidden = !plans.length;
  if (!plans.length) return;
  const title = document.createElement("h3"); title.textContent = "建议先复习";
  const note = document.createElement("p"); note.className = "answer-footnote";
  note.textContent = "按所选资料明确标注的先修关系排列；你可以从尚不熟悉的概念开始。";
  container.append(title, note);
  for (const plan of plans) {
    const heading = document.createElement("h4"); heading.textContent = `理解「${plan.concept}」之前`;
    container.append(heading);
    if (plan.status !== "ready") {
      const message = document.createElement("p");
      message.textContent = {not_declared: "所选资料尚未标注该概念的先修关系。", cycle: "资料中的先修关系存在循环，暂时无法排列复习顺序。", truncated: "先修关系较多，暂时无法显示完整复习顺序。"}[plan.status];
      container.append(message);
    } else {
      const list = document.createElement("ol"); list.className = "learning-steps";
      for (const step of plan.steps) {
        const item = document.createElement("li");
        const label = document.createElement("strong"); label.textContent = step.concept;
        const reason = document.createElement("p"); reason.textContent = `用于复习：${step.required_by.join("、")}`;
        item.append(label, reason);
        if (!step.sources.length) {
          const missing = document.createElement("p"); missing.textContent = "所选资料未找到该概念的独立讲解章节，可补充相应笔记。"; item.append(missing);
        }
        for (const source of step.sources) item.append(learningSource(source));
        list.append(item);
      }
      container.append(list);
    }
    if (plan.relations.length) {
      const details = document.createElement("details"); const summary = document.createElement("summary");
      summary.textContent = "查看先修关系的原文依据"; details.append(summary);
      for (const edge of plan.relations) {
        const relation = document.createElement("p"); relation.textContent = `${edge.concept} → 先复习 ${edge.prerequisite}`; details.append(relation);
        for (const source of edge.sources) details.append(learningSource(source));
      }
      container.append(details);
    }
  }
}
function learningSource(source) {
  const details = document.createElement("details"); const summary = document.createElement("summary");
  summary.textContent = `${source.document_name} · ${source.page ? `第 ${source.page} ${source.type === "pptx" ? "张" : "页"}` : source.section || "正文"}`;
  const excerpt = document.createElement("p"); excerpt.className = "learning-excerpt"; excerpt.textContent = source.text;
  const link = document.createElement("a"); link.href = source.source_url; link.target = "_blank"; link.rel = "noopener noreferrer";
  link.textContent = source.type === "pdf" ? "打开 PDF 对应页 ↗" : "下载原始资料 ↓";
  details.append(summary, excerpt, link); return details;
}
function sourceExplanation(row) {
  const p = document.createElement("p");
  p.append(document.createTextNode(row.sentence + " "));
  const link = document.createElement("a"); link.textContent = `${row.document_name} · ${row.page ? `第 ${row.page} 页` : row.section || "正文"}`;
  link.href = row.source_url; link.target = "_blank"; link.rel = "noopener noreferrer"; p.append(link);
  return p;
}
$("quiz-create").addEventListener("click", async () => {
  if (state.busy || !state.selected.size) return;
  busy(true); message();
  try {
    const quiz = await api("/api/quiz", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({document_ids: [...state.selected], count: 5})});
    currentQuiz = quiz; $("quiz-questions").replaceChildren(); $("quiz-result").replaceChildren();
    for (const q of quiz.questions) {
      const group = document.createElement("fieldset"); const legend = document.createElement("legend"); legend.textContent = q.prompt; group.append(legend);
      q.options.forEach((option, index) => {
        const label = document.createElement("label"); const radio = document.createElement("input");
        radio.type = "radio"; radio.name = `quiz-${q.id}`; radio.value = String(index); radio.required = true;
        label.append(radio, document.createTextNode(option)); group.append(label);
      }); $("quiz-questions").append(group);
    }
    $("quiz-form").hidden = false; $("quiz-submit").hidden = false;
    message(`已生成 ${quiz.questions.length} 道练习。按资料原文选择空缺概念。`);
  } catch (error) { message(error.message, true); } finally { busy(false); }
});
$("quiz-form").addEventListener("submit", async event => {
  event.preventDefault(); if (state.busy || !currentQuiz) return;
  const data = new FormData($("quiz-form")); const choices = {};
  for (const q of currentQuiz.questions) {
    if (!data.has(`quiz-${q.id}`)) { message("请完成所有题目后提交。", true); return; }
    choices[q.id] = Number(data.get(`quiz-${q.id}`));
  }
  busy(true); message();
  try {
    const result = await api(`/api/quiz/${currentQuiz.quiz_id}/grade`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({choices})});
    const score = document.createElement("h3"); score.textContent = `本次得分 ${result.score} / ${result.total}`; $("quiz-result").replaceChildren(score);
    for (const row of result.results) {
      const heading = document.createElement("p"); heading.textContent = `${row.id}. ${row.correct ? "回答正确" : "建议复习"} · 正确概念：${row.concept}`;
      $("quiz-result").append(heading, sourceExplanation(row));
    }
    const plans = document.createElement("div"); plans.className = "learning-plans";
    renderLearningPlans(plans, result.learning_plans || []); $("quiz-result").append(plans);
    $("quiz-form").hidden = true;
    message("练习已保存。点击「查看错题复习清单」回顾相关概念。");
  } catch (error) { message(error.message, true); } finally { busy(false); }
});
$("review-load").addEventListener("click", async () => {
  if (state.busy || !state.selected.size) return;
  busy(true); message();
  try {
    const result = await api("/api/review", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({document_ids: [...state.selected]})});
    const heading = document.createElement("h3"); heading.textContent = result.gaps.length ? "待复习的概念" : "当前所选资料暂无待复习的错题";
    $("review-list").replaceChildren(heading);
    for (const gap of result.gaps) {
      const text = document.createElement("p"); text.textContent = `${gap.concept} · 累计 ${gap.attempts} 次作答，${gap.mistakes} 次答错`;
      $("review-list").append(text, sourceExplanation(gap));
    }
    const plans = document.createElement("div"); plans.className = "learning-plans";
    renderLearningPlans(plans, result.learning_plans || []); $("review-list").append(plans);
  } catch (error) { message(error.message, true); } finally { busy(false); }
});
function comparisonExcerpt(excerpt) {
  const wrapper = document.createElement("div"); wrapper.className = "comparison-excerpt";
    if (excerpt.matched_term) {
      const term = document.createElement("p"); term.className = "comparison-note";
      term.textContent = `原文术语：${excerpt.matched_term}`; wrapper.append(term);
    }
  const quote = document.createElement("blockquote");
  for (const part of excerpt.parts || [{ text: excerpt.text, changed: false }]) {
    if (part.changed) { const mark = document.createElement("mark"); mark.textContent = part.text; quote.append(mark); }
    else quote.append(document.createTextNode(part.text));
  }
  const source = document.createElement("a"); source.href = excerpt.source_url; source.target = "_blank"; source.rel = "noopener noreferrer";
  source.textContent = `[${excerpt.id}] ${excerpt.document_name} · ${excerpt.page ? `第 ${excerpt.page} ${excerpt.type === "pptx" ? "张" : "页"}` : excerpt.section || "正文"}`;
  wrapper.append(quote, source);
  if (excerpt.context) {
    const details = document.createElement("details"), summary = document.createElement("summary"), context = document.createElement("p");
    summary.textContent = "查看所在原文段落"; context.className = "learning-excerpt"; context.textContent = excerpt.context;
    details.append(summary, context); wrapper.append(details);
  }
  return wrapper;
}
function renderComparison(result) {
  const container = $("compare-result"); container.replaceChildren();
  const title = document.createElement("h3"); title.textContent = `「${result.concept}」原文对照`;
  const note = document.createElement("p"); note.textContent = result.note;
  container.append(title, note);
    if (result.alignment && result.alignment.status !== "literal_name") {
      const alignment = document.createElement("p"); alignment.className = "comparison-alignment";
      alignment.textContent = result.alignment.note; container.append(alignment);
      if (result.alignment.sense_options) {
        const choices = document.createElement("div"); choices.className = "comparison-senses";
        for (const option of result.alignment.sense_options) {
          const button = document.createElement("button"); button.type = "button";
          button.textContent = option.label; button.title = option.description;
          button.setAttribute("aria-pressed", String(result.alignment.sense === option.id));
          button.addEventListener("click", () => runComparison("方差", option.id));
          choices.append(button);
        }
        container.append(choices);
      }
    }
  for (const doc of result.documents) {
    if (doc.status === "not_found") {
      const missing = document.createElement("p"); missing.className = "comparison-missing";
        missing.textContent = `${doc.label} · ${doc.name}：${result.alignment?.sense ? "未检出支持当前所选含义的原文；不代表原书没有相关内容。" : "未找到可核对的概念正文。可尝试该资料使用的完整概念名称。"}`; container.append(missing);
    }
  }
    const rows = [...result.rows].sort((a, b) => Number(b.cells.some(cell => cell.length)) - Number(a.cells.some(cell => cell.length)));
    for (const row of rows) {
    const section = document.createElement("section"); section.className = "comparison-row";
    const heading = document.createElement("h4"); heading.textContent = row.label;
    const status = document.createElement("p"); status.className = "comparison-note"; status.textContent = row.note;
    const grid = document.createElement("div"); grid.className = "comparison-grid";
    row.cells.forEach((cell, index) => {
      const column = document.createElement("div"); column.className = "comparison-column";
      const name = document.createElement("h5"); name.textContent = `${result.documents[index].label} · ${result.documents[index].name}`;
      column.append(name);
      if (!cell.length) { const empty = document.createElement("p"); empty.textContent = "未检出本项原文"; column.append(empty); }
      for (const excerpt of cell) column.append(comparisonExcerpt(excerpt));
      grid.append(column);
    });
    section.append(heading, status, grid); container.append(section);
  }
}
  $("compare-concept").addEventListener("input", () => { $("compare-result").hidden = true; updateScope(); });
  async function runComparison(concept, sense = null) {
    if (state.busy) return;
  if (state.selected.size !== 2 || concept.length < 2) { message("请输入概念，并恰好选择两份资料。", true); return; }
    $("compare-concept").value = concept;
  message(); busy(true); $("compare-button").textContent = "对比中…";
  const container = $("compare-result"); container.hidden = false; container.textContent = "正在分别查阅两份资料…"; container.setAttribute("aria-busy", "true");
  try {
      const result = await api("/api/compare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ concept, document_ids: [...state.selected], sense }) });
    renderComparison(result);
  } catch (error) { container.textContent = error.message; message(error.message, true); }
  finally { busy(false); container.setAttribute("aria-busy", "false"); $("compare-button").textContent = "开始对比"; }
  }
  $("compare-form").addEventListener("submit", event => {
    event.preventDefault(); runComparison($("compare-concept").value.trim());
  });
init();
