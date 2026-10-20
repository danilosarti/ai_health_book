/* =====================================================================
   AI & ML for Health Sciences — shared interactive toolkit
   Components: progress store, gate, quick checks, challenges, missions,
   progress panel, sticky counter, reset, book-wide dashboard.
   State lives only in this browser (localStorage, one key per chapter).
   Every storage access is wrapped: the page works without storage.
   ===================================================================== */
(function () {
  "use strict";

  var PREFIX = "aihs:";

  // ---------------------------------------------------------------- i18n
  var LANG = (document.documentElement.getAttribute("lang") || "en").slice(0, 2);
  var I18N = {
 "pt": {
  "Click again to confirm": "Clique de novo para confirmar",
  "Before you start": "Antes de começar",
  "Check my answers": "Verificar minhas respostas",
  "Skip for now": "Pular por enquanto",
  "Complete or skip the task above to unlock the chapter.": "Complete ou pule a tarefa acima para liberar o capítulo.",
  "All correct — the chapter is open. ": "Tudo certo — o capítulo está liberado. ",
  "{a} of {b} correct. Red cells need another look.": "{a} de {b} corretas. Revise as células em vermelho.",
  "Skipped. You can come back to this task at any time.": "Tarefa pulada. Você pode voltar a ela quando quiser.",
  "Completed earlier.": "Concluída anteriormente.",
  "Quick check": "Verificação rápida",
  "Correct.": "Correto.",
  "Not quite.": "Não exatamente.",
  "Challenge": "Desafio",
  "Answer for item {a}": "Resposta do item {a}",
  "— choose —": "— escolha —",
  "Check": "Verificar",
  "All {a} correct!": "Todas as {a} corretas!",
  "{a} of {b} correct — try again.": "{a} de {b} corretas — tente de novo.",
  "Mission": "Missão",
  "R and Python": "R e Python",
  "Starting|Under way|Halfway|Nearly there|Complete": "Início|Em andamento|Metade|Quase lá|Completo",
  "{a} / {b} checkpoints ({c}%)": "{a} / {b} etapas ({c}%)",
  "Level: ": "Nível: ",
  "Badge earned: ": "Medalha conquistada: ",
  "Chapter complete": "Capítulo completo",
  "Your progress": "Seu progresso",
  "Chapter progress": "Progresso no capítulo",
  "Reset this chapter": "Reiniciar este capítulo",
  "Progress is saved only in this browser. Nothing is sent anywhere.": "O progresso fica salvo apenas neste navegador. Nada é enviado a lugar nenhum.",
  "Chapter progress summary": "Resumo do progresso no capítulo",
  "Chapter": "Capítulo",
  "Progress": "Progresso",
  "Stars": "Estrelas",
  "Status": "Situação",
  "coming soon": "em breve",
  "not started": "não iniciado",
  "complete ✓": "completo ✓",
  "in progress": "em andamento",
  "Chapters completed: {a} · Stars collected: ★ {b}": "Capítulos completos: {a} · Estrelas: ★ {b}",
  "Reset the whole book": "Reiniciar o livro inteiro",
  "Try it": "Experimente",
  "Language": "Idioma"
 },
 "it": {
  "Click again to confirm": "Clicca di nuovo per confermare",
  "Before you start": "Prima di iniziare",
  "Check my answers": "Verifica le mie risposte",
  "Skip for now": "Salta per ora",
  "Complete or skip the task above to unlock the chapter.": "Completa o salta il compito qui sopra per sbloccare il capitolo.",
  "All correct — the chapter is open. ": "Tutto corretto: il capitolo è sbloccato. ",
  "{a} of {b} correct. Red cells need another look.": "{a} su {b} corrette. Ricontrolla le celle in rosso.",
  "Skipped. You can come back to this task at any time.": "Compito saltato. Puoi tornarci quando vuoi.",
  "Completed earlier.": "Completato in precedenza.",
  "Quick check": "Verifica rapida",
  "Correct.": "Corretto.",
  "Not quite.": "Non proprio.",
  "Challenge": "Sfida",
  "Answer for item {a}": "Risposta per l'elemento {a}",
  "— choose —": "— scegli —",
  "Check": "Verifica",
  "All {a} correct!": "Tutte le {a} corrette!",
  "{a} of {b} correct — try again.": "{a} su {b} corrette: riprova.",
  "Mission": "Missione",
  "R and Python": "R e Python",
  "Starting|Under way|Halfway|Nearly there|Complete": "Inizio|In corso|A metà|Quasi fatto|Completato",
  "{a} / {b} checkpoints ({c}%)": "{a} / {b} tappe ({c}%)",
  "Level: ": "Livello: ",
  "Badge earned: ": "Badge ottenuto: ",
  "Chapter complete": "Capitolo completato",
  "Your progress": "I tuoi progressi",
  "Chapter progress": "Progressi nel capitolo",
  "Reset this chapter": "Azzera questo capitolo",
  "Progress is saved only in this browser. Nothing is sent anywhere.": "I progressi sono salvati solo in questo browser. Nulla viene inviato altrove.",
  "Chapter progress summary": "Riepilogo dei progressi nel capitolo",
  "Chapter": "Capitolo",
  "Progress": "Progressi",
  "Stars": "Stelle",
  "Status": "Stato",
  "coming soon": "in arrivo",
  "not started": "non iniziato",
  "complete ✓": "completato ✓",
  "in progress": "in corso",
  "Chapters completed: {a} · Stars collected: ★ {b}": "Capitoli completati: {a} · Stelle raccolte: ★ {b}",
  "Reset the whole book": "Azzera l'intero libro",
  "Try it": "Prova tu",
  "Language": "Lingua"
 }
};
  function T(key, vals) {
    var s = (I18N[LANG] && I18N[LANG][key]) || key;
    if (vals) Object.keys(vals).forEach(function (k) { s = s.split("{" + k + "}").join(vals[k]); });
    return s;
  }


  // ---------------------------------------------------------------- store
  function load(ch) {
    try {
      var raw = window.localStorage.getItem(PREFIX + ch);
      return raw ? JSON.parse(raw) : {};
    } catch (e) { return {}; }
  }
  function save(ch, state) {
    try { window.localStorage.setItem(PREFIX + ch, JSON.stringify(state)); } catch (e) { /* ignore */ }
  }
  function clearChapter(ch) {
    try { window.localStorage.removeItem(PREFIX + ch); } catch (e) { /* ignore */ }
  }
  function allKeys() {
    var out = [];
    try {
      for (var i = 0; i < window.localStorage.length; i++) {
        var k = window.localStorage.key(i);
        if (k && k.indexOf(PREFIX) === 0) out.push(k);
      }
    } catch (e) { /* ignore */ }
    return out;
  }

  var root = document.querySelector("[data-aihs-chapter]");
  var CH = root ? root.getAttribute("data-aihs-chapter") : null;
  var state = CH ? load(CH) : {};
  state.done = state.done || {};
  state.stars = state.stars || {};
  var checkpoints = [];          // ids registered on this page

  function register(id) { if (checkpoints.indexOf(id) < 0) checkpoints.push(id); }
  function complete(id, star) {
    if (!CH) return;
    var changed = false;
    if (!state.done[id]) { state.done[id] = true; changed = true; }
    if (star && !state.stars[id]) { state.stars[id] = true; changed = true; }
    if (changed) { persist(); render(); }
  }
  function persist() {
    if (!CH) return;
    state.total = checkpoints.length;
    state.title = document.title;
    state.updated = new Date().toISOString();
    save(CH, state);
  }

  // ---------------------------------------------------------------- helpers
  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text != null) e.textContent = text;
    return e;
  }
  function label(container, text) {
    var l = el("div", { "class": "aihs-label" }, text);
    container.insertBefore(l, container.firstChild);
  }
  // two-step confirmation inside the page (window.confirm is blocked in some viewers)
  function confirmButton(btn, label, action) {
    var armed = false, timer = null;
    btn.addEventListener("click", function () {
      if (!armed) {
        armed = true; btn.textContent = T("Click again to confirm");
        timer = setTimeout(function () { armed = false; btn.textContent = label; }, 4000);
        return;
      }
      clearTimeout(timer); action();
    });
  }
  function norm(s) { return String(s).trim().toLowerCase().replace(/\s+/g, " "); }

  // ---------------------------------------------------------------- gate
  function initGate(g) {
    var id = g.id || "gate";
    register(id);
    label(g, T("Before you start"));
    var content = document.querySelector(".gated-content");
    var cells = g.querySelectorAll("[data-answer]");
    cells.forEach(function (c) {
      var wrap = c.closest("td") || c.parentElement;
      wrap.classList.add("gate-cell");
      c.setAttribute("autocomplete", "off");
    });
    var fb = el("div", { "class": "aihs-feedback", "role": "status", "aria-live": "polite" });
    var check = el("button", { "class": "aihs-btn primary", "type": "button" }, T("Check my answers"));
    var skip = el("button", { "class": "aihs-btn", "type": "button" }, T("Skip for now"));
    g.appendChild(check); g.appendChild(skip); g.appendChild(fb);
    var veil = null;

    function lock() {
      if (!content) return;
      content.classList.add("locked");
      content.setAttribute("inert", "");
      content.setAttribute("aria-hidden", "true");
      veil = el("p", { "class": "gate-veil" }, T("Complete or skip the task above to unlock the chapter."));
      content.parentNode.insertBefore(veil, content);
    }
    function unlock() {
      if (!content) return;
      content.classList.remove("locked");
      content.removeAttribute("inert");
      content.removeAttribute("aria-hidden");
      if (veil) { veil.remove(); veil = null; }
      // let OJS / editors recompute layout once visible
      window.dispatchEvent(new Event("resize"));
    }
    check.addEventListener("click", function () {
      var right = 0;
      cells.forEach(function (c) {
        var ans = c.getAttribute("data-answer");
        var tol = parseFloat(c.getAttribute("data-tol") || "0");
        var v = c.value;
        var ok;
        if (tol > 0 || /^-?\d+(\.\d+)?$/.test(ans)) {
          ok = v !== "" && Math.abs(parseFloat(String(v).replace(",", ".")) - parseFloat(ans)) <= (tol || 1e-9);
        } else {
          ok = ans.split("|").map(norm).indexOf(norm(v)) >= 0;
        }
        var cell = c.closest(".gate-cell");
        cell.classList.toggle("ok", ok); cell.classList.toggle("bad", !ok);
        if (ok) right++;
      });
      if (right === cells.length) {
        fb.className = "aihs-feedback ok";
        fb.textContent = T("All correct — the chapter is open. ") + (g.getAttribute("data-success") || "");
        state.gate = "passed"; g.classList.add("done");
        complete(id, true); unlock();
      } else {
        fb.className = "aihs-feedback bad";
        fb.textContent = T("{a} of {b} correct. Red cells need another look.", {a: right, b: cells.length});
      }
    });
    skip.addEventListener("click", function () {
      state.gate = state.gate || "skipped"; persist();
      fb.className = "aihs-feedback";
      fb.textContent = T("Skipped. You can come back to this task at any time.");
      unlock();
    });
    if (state.gate === "passed") { g.classList.add("done"); fb.textContent = T("Completed earlier."); }
    else if (state.gate !== "skipped") lock();
  }

  // ---------------------------------------------------------------- quick check
  function initQuickCheck(q, idx) {
    var id = q.id || ("qc-" + idx);
    register(id);
    label(q, T("Quick check"));
    var items = q.querySelectorAll("ul.task-list > li");
    var tried = false;
    items.forEach(function (li) {
      var input = li.querySelector("input[type=checkbox]");
      var correct = !!(input && input.checked);
      var whyEl = li.querySelector(".why");
      var why = whyEl ? whyEl.innerHTML : "";
      if (whyEl) whyEl.remove();
      if (input) input.remove();
      var labelEl = li.querySelector("label");
      var html = (labelEl || li).innerHTML;
      li.innerHTML = "";
      var btn = el("button", { "class": "qc-option", "type": "button", "aria-pressed": "false" });
      btn.innerHTML = html;
      var whyBox = el("div", { "class": "qc-why", "role": "status" });
      whyBox.innerHTML = (correct ? "<strong>" + T("Correct.") + "</strong> " : "<strong>" + T("Not quite.") + "</strong> ") + why;
      li.appendChild(btn); li.appendChild(whyBox);
      btn.addEventListener("click", function () {
        btn.classList.add(correct ? "is-correct" : "is-wrong");
        btn.setAttribute("aria-pressed", "true");
        whyBox.classList.add("shown");
        if (correct) { q.classList.add("done"); complete(id, !tried); }
        tried = true;
      });
    });
    if (state.done[id]) q.classList.add("done");
  }

  // ---------------------------------------------------------------- challenge
  function initChallenge(c, idx) {
    var id = c.id || ("challenge-" + idx);
    register(id);
    label(c, T("Challenge"));
    var opts = (c.getAttribute("data-options") || "").split("|");
    var lis = c.querySelectorAll("ol > li, ul > li");
    var fb = el("div", { "class": "aihs-feedback", "role": "status", "aria-live": "polite" });
    var btn = el("button", { "class": "aihs-btn primary", "type": "button" }, T("Check"));
    var firstTry = true;
    lis.forEach(function (li, k) {
      var a = li.querySelector(".ans");
      li.setAttribute("data-key", a ? a.textContent.trim() : "");
      if (a) a.remove();
      var sel = el("select", { "aria-label": T("Answer for item {a}", {a: k + 1}), "autocomplete": "off" });
      sel.appendChild(el("option", { "value": "" }, T("— choose —")));
      opts.forEach(function (o) { sel.appendChild(el("option", { "value": o }, o)); });
      li.appendChild(sel);
    });
    btn.addEventListener("click", function () {
      var right = 0;
      lis.forEach(function (li) {
        var ok = norm(li.querySelector("select").value) === norm(li.getAttribute("data-key"));
        li.classList.toggle("ok", ok); li.classList.toggle("bad", !ok);
        if (ok) right++;
      });
      if (right === lis.length) {
        fb.className = "aihs-feedback ok"; fb.textContent = T("All {a} correct!", {a: right});
        c.classList.add("done"); complete(id, firstTry);
      } else {
        fb.className = "aihs-feedback bad"; fb.textContent = T("{a} of {b} correct — try again.", {a: right, b: lis.length});
        firstTry = false;
      }
    });
    c.appendChild(btn); c.appendChild(fb);
    if (state.done[id]) c.classList.add("done");
  }

  // ---------------------------------------------------------------- mission
  function initMission(m, idx) {
    var id = m.id || ("mission-" + idx);
    register(id);
    label(m, T("Mission"));
    var lab = m.querySelector(".aihs-label"); if (lab) { var sp = el("span", { "class": "aihs-label-sub" }, " · " + T("R and Python")); lab.appendChild(sp); }
    if (state.done[id]) m.classList.add("done");
    var solutionSeen = false;
    m.addEventListener("click", function (ev) {
      var t = ev.target;
      if (t && t.closest && t.closest(".exercise-solution, [class*='solution']") && /solution/i.test(t.textContent || "")) {
        solutionSeen = true;
      }
    });
    var obs = new MutationObserver(function () {
      if (m.querySelector(".alert-success")) {
        m.classList.add("done");
        complete(id, !solutionSeen);
      }
    });
    obs.observe(m, { childList: true, subtree: true });
  }

  // ---------------------------------------------------------------- progress panel
  var panel = document.querySelector(".progress-panel");
  var sticky = null;
  function levels() {
    var l = panel && panel.getAttribute("data-levels");
    return l ? l.split("|") : T("Starting|Under way|Halfway|Nearly there|Complete").split("|");
  }
  function render() {
    if (!CH) return;
    var total = checkpoints.length || 1;
    var done = checkpoints.filter(function (id) { return state.done[id]; }).length;
    var stars = checkpoints.filter(function (id) { return state.stars[id]; }).length;
    var pct = Math.round(100 * done / total);
    var lv = levels();
    var level = lv[Math.min(lv.length - 1, Math.floor(pct / (100 / (lv.length - 1)) + 1e-9))];
    if (pct === 100) level = lv[lv.length - 1];
    if (panel) {
      panel.querySelector(".bar > span").style.width = pct + "%";
      panel.querySelector(".bar").setAttribute("aria-valuenow", pct);
      panel.querySelector(".pp-count").textContent = T("{a} / {b} checkpoints ({c}%)", {a: done, b: total, c: pct});
      panel.querySelector(".pp-level").textContent = T("Level: ") + level;
      panel.querySelector(".pp-stars").textContent = "★ " + stars;
      var b = panel.querySelector(".pp-badge");
      b.textContent = pct === 100 ? T("Badge earned: ") + (panel.getAttribute("data-badge") || T("Chapter complete")) : "";
      b.className = "pp-badge" + (pct === 100 ? " badge-earned" : "");
    }
    if (sticky) sticky.querySelector("span").textContent = pct + "% · ★ " + stars;
  }
  function initPanel() {
    if (!panel) return;
    label(panel, T("Your progress"));
    var bar = el("div", { "class": "bar", "role": "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-label": T("Chapter progress") });
    bar.appendChild(el("span"));
    var meta = el("div", { "class": "meta" });
    ["pp-count", "pp-level", "pp-stars", "pp-badge"].forEach(function (c) { meta.appendChild(el("span", { "class": c })); });
    var reset = el("button", { "class": "aihs-btn", "type": "button" });
    reset.textContent = T("Reset this chapter");
    confirmButton(reset, T("Reset this chapter"), function () { clearChapter(CH); window.location.reload(); });
    panel.appendChild(bar); panel.appendChild(meta); panel.appendChild(reset);
    var note = el("p", { "class": "small" }, T("Progress is saved only in this browser. Nothing is sent anywhere."));
    note.style.marginTop = "0.4rem"; note.style.opacity = "0.75";
    panel.appendChild(note);
    if (!panel.id) panel.id = "progress";
    sticky = el("div", { "class": "aihs-sticky", "aria-label": T("Chapter progress summary") });
    var a = el("a", { "href": "#" + panel.id });
    a.appendChild(el("span"));
    sticky.appendChild(a);
    document.body.appendChild(sticky);
  }

  // ---------------------------------------------------------------- dashboard
  function initDashboard(d) {
    var chapters = [];
    try { chapters = JSON.parse(d.getAttribute("data-chapters") || "[]"); } catch (e) { chapters = []; }
    var table = el("table", { "class": "table table-sm" });
    var thead = el("thead"); var trh = el("tr");
    ["Chapter", "Progress", "Stars", "Status"].forEach(function (h) { trh.appendChild(el("th", { "scope": "col" }, T(h))); });
    thead.appendChild(trh); table.appendChild(thead);
    var tb = el("tbody"); var totalStars = 0; var completed = 0;
    chapters.forEach(function (c) {
      var s = load(c.id); var done = s.done ? Object.keys(s.done).length : 0;
      var total = s.total || 0; var stars = s.stars ? Object.keys(s.stars).length : 0;
      var pct = total ? Math.round(100 * Math.min(done, total) / total) : 0;
      totalStars += stars; if (total && pct === 100) completed++;
      var tr = el("tr");
      var td1 = el("td"); td1.appendChild(c.ready === false ? el("span", null, c.title) : el("a", { "href": c.href }, c.title)); tr.appendChild(td1);
      var td2 = el("td"); var mini = el("span", { "class": "mini", "role": "img", "aria-label": pct + "%" });
      var fill = el("span"); fill.style.width = pct + "%"; mini.appendChild(fill); td2.appendChild(mini);
      td2.appendChild(document.createTextNode(" " + pct + "%")); tr.appendChild(td2);
      tr.appendChild(el("td", null, "★ " + stars));
      tr.appendChild(el("td", null, !total ? (c.ready === false ? T("coming soon") : T("not started")) : pct === 100 ? T("complete ✓") : T("in progress")));
      tb.appendChild(tr);
    });
    table.appendChild(tb); d.appendChild(table);
    d.appendChild(el("p", null, T("Chapters completed: {a} · Stars collected: ★ {b}", {a: completed, b: totalStars})));
    var reset = el("button", { "class": "aihs-btn", "type": "button" });
    reset.textContent = T("Reset the whole book");
    confirmButton(reset, T("Reset the whole book"), function () {
      allKeys().forEach(function (k) { try { window.localStorage.removeItem(k); } catch (e) { /* ignore */ } });
      window.location.reload();
    });
    d.appendChild(reset);
  }

  // ---------------------------------------------------------------- boot
  function boot() {
    document.querySelectorAll("input, select, textarea").forEach(function (x) { x.setAttribute("autocomplete", "off"); });
    document.querySelectorAll(".gate").forEach(initGate);
    document.querySelectorAll(".quick-check").forEach(initQuickCheck);
    document.querySelectorAll(".challenge").forEach(initChallenge);
    document.querySelectorAll(".mission").forEach(initMission);
    document.querySelectorAll(".tryit").forEach(function (t) { label(t, T("Try it")); });
    document.querySelectorAll(".checkpoint[id]").forEach(function (c) {
      register(c.id);
      if (state.done[c.id]) { var host = c.closest(".tryit"); if (host) host.classList.add("done"); }
    });
    document.querySelectorAll(".aihs-dashboard").forEach(initDashboard);
    document.querySelectorAll(".mindmap .mm-link[data-href]").forEach(function (g) {
      var go = function () {
        var h = g.getAttribute("data-href");
        var m = (window.location.pathname.split("/").pop() || "").match(/\.(pt|it)\.html$/);
        window.location.href = (m && !/\.(pt|it)\.html$/.test(h)) ? h.replace(/\.html$/, "." + m[1] + ".html") : h;
      };
      g.addEventListener("click", go);
      g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    });
    initPanel();
    try { window.localStorage.setItem("aihs:lang", LANG); } catch (e) { /* ignore */ }
    var page = (window.location.pathname.split("/").pop() || "index.html");
    function langOf(href) {
      var m = String(href || "").match(/(?:^|\/)(en|pt|it)\/(?:index\.html)?(?:[?#].*)?$/);
      return m ? m[1] : null;
    }
    document.querySelectorAll(".sidebar-tools-main-item, .quarto-navigation-tool + .dropdown-menu a").forEach(function (a) {
      var l = langOf(a.getAttribute("href")) || langOf(a.href);
      if (!l) return;
      a.setAttribute("hreflang", l);
      if (l === LANG) a.setAttribute("aria-current", "true");
      a.addEventListener("click", function (ev) {
        ev.preventDefault();
        // two layouts: one folder per language (en/ch01.html) or a single folder (ch01.pt.html)
        var foldered = /\/(en|pt|it)\/[^\/]*$/.test(window.location.pathname);
        if (foldered) {
          window.location.href = "../" + l + "/" + page + window.location.hash;
        } else {
          var base = page.replace(/\.(pt|it)\.html$/, ".html");
          window.location.href = (l === "en" ? base : base.replace(/\.html$/, "." + l + ".html")) + window.location.hash;
        }
      });
    });
    var tgl = document.querySelector(".bi-translate");
    if (tgl && tgl.parentElement) { tgl.parentElement.setAttribute("aria-label", T("Language")); tgl.parentElement.setAttribute("title", T("Language")); }
    if (CH) { persist(); render(); }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  // public hook for OJS widgets that act as checkpoints
  window.aihs = {
    register: function (id) { register(id); persist(); render(); },
    complete: function (id, star) {
      complete(id, star !== false);
      var c = document.getElementById(id); var host = c && c.closest(".tryit");
      if (host) host.classList.add("done");
    },
    isDone: function (id) { return !!state.done[id]; }
  };
})();
