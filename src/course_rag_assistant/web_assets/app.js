"use strict";
const $ = id => document.getElementById(id);
const state = { token: "", documents: [], selected: new Set(), configured: false, busy: false, sources: [] };

function message(text = "", error = false) {
  $("notice").hidden = true; $("error").hidden = true;
  if (text) { const node = $(error ? "error" : "notice"); node.textContent = text; node.hidden = false; }
}
function busy(value, label = "") {
  state.busy = value;
  for (const id of ["upload-button", "demo-button", "file-input", "select-all", "question"]) $(id).disabled = value;
  $("use-model").disabled = value || !state.configured;
  document.querySelectorAll(".document input, [data-question]").forEach(node => { node.disabled = value; });
  $("ask-label").textContent = label || "开始提问";
  updateScope();
}
function updateScope() {
  const count = state.selected.size;
  $("selection-count").textContent = state.documents.length ? `已选 ${count} / ${state.documents.length} 份` : "尚未添加资料";
  $("ask-button").disabled = state.busy || !count || !$("question").value.trim();
  $("select-all").textContent = count === state.documents.length ? "取消全选" : "全选";
  $("question-help").textContent = count ? `仅从已选的 ${count} 份资料中回答。Ctrl / ⌘ + Enter 提问` : "先添加并勾选资料，或使用下方示例体验。";
}
function renderDocuments() {
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
      if (file.size > 25 * 1024 * 1024) throw new Error("超过 25 MB，请按章节拆分。");
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
    state.token = data.token; state.documents = data.documents; state.selected = new Set(data.documents.map(d => d.id)); state.configured = data.model_configured;
    $("connection").textContent = "本地学习空间";
    $("model-note").textContent = state.configured ? "更精简，可能需要稍等" : "未配置时可直接查原文";
    renderDocuments();
    if (data.unreadable_documents) message(`${data.unreadable_documents} 份已保存资料无法读取，可重新上传原文件。`, true);
  } catch (error) { $("connection").textContent = "连接未完成"; message(error.message, true); }
  finally { busy(false); }
}
init();
