const state = {
  tenant: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `${res.status} ${res.statusText}`);
  }
  return data;
}

function setStatus(text) {
  if (text) {
    $("status-bar").classList.remove("hidden");
    $("status-text").textContent = text;
  } else {
    $("status-bar").classList.add("hidden");
  }
}

function showGenPanel(show) {
  $("gen-panel").classList.toggle("hidden", !show);
}

function setBusy(busy) {
  for (const el of document.querySelectorAll("button, select, input")) {
    el.disabled = busy;
  }
}

// ── Tenant list ────────────────────────────────────────────────

async function refreshTenantList(selectId) {
  const { tenants } = await api("/api/tenants");
  const select = $("tenant-select");
  select.innerHTML = "";
  for (const id of tenants) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id;
    select.appendChild(opt);
  }
  if (tenants.length === 0) {
    $("empty-state").classList.remove("hidden");
    $("main-view").classList.add("hidden");
    return;
  }
  const target = selectId && tenants.includes(selectId) ? selectId : tenants[tenants.length - 1];
  select.value = target;
  await selectTenant(target);
}

async function selectTenant(tenantId) {
  state.tenant = tenantId;
  $("empty-state").classList.add("hidden");
  $("main-view").classList.remove("hidden");
  await Promise.all([loadBrief(), loadEmails(), loadCalendar(), loadNotes(), loadDrafts(), loadPersona()]);
}

// ── Citations ──────────────────────────────────────────────────

// Maps a citation's source type to the tab that can display it, and the
// data-ref lookup key used on that tab's cards.
const SOURCE_TO_TAB = { email: "emails", calendar: "calendar", notes: "notes" };

function renderCitations(citations) {
  if (!citations || citations.length === 0) return "";
  const chips = citations
    .map((c) => {
      const icon = c.source === "email" ? "📧" : c.source === "calendar" ? "📅" : c.source === "notes" ? "📝" : "🔗";
      const kindClass = c.matched_via === "embedding" ? "matched-embedding" : c.matched_via === "llm" ? "matched-llm" : "";
      const title = c.matched_via === "keyword" ? "literal match" : c.matched_via === "embedding" ? "semantic (embedding) match" : "LLM-inferred match";
      return `<button type="button" class="citation-chip ${kindClass}" data-source="${escapeHtml(c.source)}" data-ref="${escapeHtml(c.ref)}" title="${escapeHtml(title)}">
        <span class="chip-icon">${icon}</span>${escapeHtml(truncate(c.label, 40))}
      </button>`;
    })
    .join("");
  return `<div class="citations">${chips}</div>`;
}

function truncate(s, n) {
  if (!s) return s;
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function wireCitationClicks(container) {
  container.querySelectorAll(".citation-chip").forEach((chip) => {
    chip.addEventListener("click", (ev) => {
      ev.stopPropagation();
      jumpToCitation(chip.dataset.source, chip.dataset.ref);
    });
  });
}

function jumpToCitation(source, ref) {
  const tabName = SOURCE_TO_TAB[source];
  if (!tabName) return;
  activateTab(tabName);
  requestAnimationFrame(() => {
    // Scoped to the destination tab's card list, not a bare [data-ref]
    // lookup — citation chips carry the same data-ref as the card they
    // point to (by design, so a chip's own click handler can read it),
    // so an unscoped selector matches the chip itself first if it happens
    // to sit earlier in document order than the real card.
    const target = document.querySelector(`#tab-${tabName} .email-card[data-ref="${cssEscape(ref)}"]`);
    if (!target) return;
    target.classList.add("open");
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.remove("card-highlight");
    void target.offsetWidth; // restart animation if clicked again
    target.classList.add("card-highlight");
  });
}

function cssEscape(s) {
  return window.CSS && CSS.escape ? CSS.escape(s) : s.replace(/["\\]/g, "\\$&");
}

function activateTab(tabName) {
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t.dataset.tab === tabName);
  for (const p of document.querySelectorAll(".tab-panel")) p.classList.toggle("active", p.id === `tab-${tabName}`);
}

// ── Loaders ────────────────────────────────────────────────────

function renderBullets(container, bullets) {
  container.innerHTML = "";
  if (!bullets || bullets.length === 0) {
    container.innerHTML = '<div class="empty-inline">(empty)</div>';
    return;
  }
  for (const b of bullets) {
    const el = document.createElement("div");
    el.className = "bullet";
    el.innerHTML = `${escapeHtml(b.text)}${renderCitations(b.citations)}`;
    container.appendChild(el);
  }
  wireCitationClicks(container);
}

async function loadBrief() {
  const data = await api(`/api/tenant/${state.tenant}/brief`);
  const noBrief = $("no-brief");
  if (!data.has_brief) {
    noBrief.classList.remove("hidden");
    $("what-matters").innerHTML = "";
    $("what-missed").innerHTML = "";
    $("freshness-notice").classList.add("hidden");
    return;
  }
  noBrief.classList.add("hidden");
  renderBullets($("what-matters"), data.what_matters_today);
  renderBullets($("what-missed"), data.what_might_be_missed);
  if (data.freshness_notice) {
    $("freshness-notice").textContent = data.freshness_notice;
    $("freshness-notice").classList.remove("hidden");
  } else {
    $("freshness-notice").classList.add("hidden");
  }
}

async function loadEmails() {
  const { emails } = await api(`/api/tenant/${state.tenant}/emails`);
  const list = $("email-list");
  list.innerHTML = "";
  if (emails.length === 0) {
    list.innerHTML = '<div class="empty-inline">No emails found for this tenant.</div>';
    return;
  }
  for (const e of emails) {
    const card = document.createElement("div");
    card.className = "email-card";
    card.dataset.ref = e.filename;
    card.innerHTML = `
      <div class="email-card-head">
        <div class="email-subject">${escapeHtml(e.subject || "(no subject)")}</div>
        <div class="email-date">${escapeHtml(e.date_key || "")}</div>
      </div>
      <div class="email-from">${escapeHtml(e.from || "")} → ${escapeHtml(e.to || "")}</div>
      <div class="email-body">${escapeHtml(e.body || "")}</div>
    `;
    card.addEventListener("click", () => card.classList.toggle("open"));
    list.appendChild(card);
  }
}

async function loadCalendar() {
  const { events } = await api(`/api/tenant/${state.tenant}/calendar`);
  const list = $("calendar-list");
  list.innerHTML = "";
  if (events.length === 0) {
    list.innerHTML = '<div class="empty-inline">No calendar events found for this tenant.</div>';
    return;
  }
  for (const e of events) {
    const card = document.createElement("div");
    card.className = "email-card";
    card.dataset.ref = e.uid;
    const time = e.start ? new Date(e.start).toLocaleString(undefined, { hour: "2-digit", minute: "2-digit" }) : "";
    card.innerHTML = `
      <div class="email-card-head">
        <div class="email-subject">${escapeHtml(e.summary || "(no title)")}</div>
        <div class="email-date">${escapeHtml(e.date_key || "")} ${escapeHtml(time)}</div>
      </div>
      <div class="email-from">${escapeHtml(e.location || "")}${e.attendees && e.attendees.length ? " — " + escapeHtml(e.attendees.join(", ")) : ""}</div>
      <div class="email-body">${escapeHtml(e.description || "(no description)")}</div>
    `;
    card.addEventListener("click", () => card.classList.toggle("open"));
    list.appendChild(card);
  }
}

async function loadNotes() {
  const { notes } = await api(`/api/tenant/${state.tenant}/notes`);
  const list = $("notes-list");
  list.innerHTML = "";
  if (notes.length === 0) {
    list.innerHTML = '<div class="empty-inline">No notes found for this tenant.</div>';
    return;
  }
  for (const n of notes) {
    const card = document.createElement("div");
    card.className = "email-card";
    card.dataset.ref = n.note_id;
    card.innerHTML = `
      <div class="email-card-head">
        <div class="email-subject">${escapeHtml(n.title || n.note_id)}</div>
        <div class="email-date">${escapeHtml(n.date_key || "")}</div>
      </div>
      <div class="email-body">${escapeHtml(n.body || "")}</div>
    `;
    card.addEventListener("click", () => card.classList.toggle("open"));
    list.appendChild(card);
  }
}

async function loadDrafts() {
  const { drafts } = await api(`/api/tenant/${state.tenant}/drafts`);
  const list = $("drafts-list");
  const noDrafts = $("no-drafts");
  list.innerHTML = "";
  if (drafts.length === 0) {
    noDrafts.classList.remove("hidden");
    return;
  }
  noDrafts.classList.add("hidden");
  for (const d of drafts) {
    const card = document.createElement("div");
    card.className = "draft-card";
    card.innerHTML = `
      <div class="draft-summary">${escapeHtml(d.summary)}</div>
      ${renderCitations(d.citations)}
      ${
        d.drafted
          ? `<div class="draft-text">${escapeHtml(d.draft_text)}</div>`
          : `<div class="draft-none">Surfaced for you to handle directly — not drafted.</div>`
      }
    `;
    list.appendChild(card);
  }
  wireCitationClicks(list);
}

async function loadPersona() {
  const { persona_text } = await api(`/api/tenant/${state.tenant}/persona`);
  $("persona-text").textContent = persona_text || "";
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

// ── Actions ────────────────────────────────────────────────────

async function generateAndRun() {
  const hint = $("hint-input").value.trim();
  const provider = $("provider-select").value;
  const model = $("model-input").value.trim() || "deepseek-chat";

  showGenPanel(false);
  setBusy(true);
  try {
    setStatus("Generating a new persona + dataset… (one LLM call, ~10-30s)");
    const gen = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hint, provider, model }),
    });

    setStatus(`Running the pipeline for '${gen.tenant_id}'… (several LLM calls, can take a minute or two)`);
    await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tenant_id: gen.tenant_id, provider, model }),
    });

    setStatus("Loading results…");
    await refreshTenantList(gen.tenant_id);
  } catch (e) {
    alert(`Failed: ${e.message}`);
  } finally {
    setBusy(false);
    setStatus(null);
  }
}

async function runPipelineForCurrentTenant() {
  if (!state.tenant) return;
  const provider = $("provider-select").value;
  const model = $("model-input").value.trim() || "deepseek-chat";

  setBusy(true);
  try {
    setStatus(`Running the pipeline for '${state.tenant}'…`);
    await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tenant_id: state.tenant, provider, model }),
    });
    setStatus("Loading results…");
    await selectTenant(state.tenant);
  } catch (e) {
    alert(`Failed: ${e.message}`);
  } finally {
    setBusy(false);
    setStatus(null);
  }
}

// ── Wiring ─────────────────────────────────────────────────────

$("new-persona-btn").addEventListener("click", () => showGenPanel(true));
$("empty-generate-btn").addEventListener("click", () => showGenPanel(true));
$("cancel-gen-btn").addEventListener("click", () => showGenPanel(false));
$("generate-btn").addEventListener("click", generateAndRun);
$("run-btn").addEventListener("click", runPipelineForCurrentTenant);
$("tenant-select").addEventListener("change", (e) => selectTenant(e.target.value));

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const t of document.querySelectorAll(".tab")) t.classList.remove("active");
    for (const p of document.querySelectorAll(".tab-panel")) p.classList.remove("active");
    tab.classList.add("active");
    $(`tab-${tab.dataset.tab}`).classList.add("active");
  });
}

refreshTenantList().catch((e) => console.error(e));
