(function () {
  "use strict";

  var API_PORT = window.API_PORT || "8000";
  const API_BASE = window.API_BASE || (window.location.protocol + "//" + window.location.hostname + ":" + API_PORT);

  function getPatientId() {
    let id = sessionStorage.getItem("home_ivf_patient_id");
    if (!id) {
      id = crypto.randomUUID ? crypto.randomUUID() : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/x/g, function () { return (Math.random() * 16 | 0).toString(16); });
      sessionStorage.setItem("home_ivf_patient_id", id);
    }
    return id;
  }

  function parseList(val) {
    if (val == null || val === "") return [];
    return String(val).split(",").map(function (s) { return s.trim(); }).filter(Boolean);
  }

  function parseNum(val) {
    if (val == null || val === "") return undefined;
    const n = Number(val);
    return isNaN(n) ? undefined : n;
  }

  function api(method, path, body) {
    const opts = {
      method: method,
      headers: { "Content-Type": "application/json" },
      credentials: "omit"
    };
    if (body != null) opts.body = JSON.stringify(body);
    return fetch(API_BASE + path, opts).then(function (r) {
      if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || j.message || r.statusText); }).catch(function () { throw new Error(r.statusText); });
      return r.json();
    });
  }

  function setActivePanel(id) {
    document.querySelectorAll(".panel").forEach(function (p) { p.classList.remove("active"); });
    document.querySelectorAll(".nav-link").forEach(function (a) { a.classList.remove("active"); });
    var panel = document.getElementById(id);
    var link = document.querySelector('.nav-link[href="#' + id + '"]');
    if (panel) panel.classList.add("active");
    if (link) link.classList.add("active");
  }

  function showResult(el, content, isError) {
    el.innerHTML = content;
    el.hidden = false;
    el.classList.toggle("error", !!isError);
  }

  // --- Navigation ---
  document.querySelectorAll(".nav-link").forEach(function (a) {
    a.addEventListener("click", function (e) {
      var href = a.getAttribute("href");
      if (href && href.startsWith("#")) {
        e.preventDefault();
        setActivePanel(href.slice(1));
      }
    });
  });
  window.addEventListener("hashchange", function () {
    var hash = window.location.hash.slice(1);
    if (hash) setActivePanel(hash);
  });
  if (window.location.hash) setActivePanel(window.location.hash.slice(1));
  else setActivePanel("home");

  // --- Page translation (googletrans) ---
  var translateSelect = document.getElementById("translate-select");
  var translateLoading = document.getElementById("translate-loading");
  var TRANSLATE_SELECTOR = ".header .logo, .header .tagline, .header .nav-link, " +
    ".panel-content h2, .panel-content > p, .panel-content > .muted, .card h3, .card p, .card a.btn, " +
    ".form-engagement label, .form-engagement button, .footer > p:not(.footer-professional)";
  var translationOriginals = [];
  var translationCache = {}; // dest lang -> array of translated strings (avoids lag on re-select)

  function collectTranslatable() {
    var nodes = document.querySelectorAll(TRANSLATE_SELECTOR);
    translationOriginals = [];
    nodes.forEach(function (el) {
      if (el.querySelector("input, select, textarea")) return;
      var text = (el.textContent || "").trim();
      if (text) translationOriginals.push({ el: el, text: text });
    });
  }

  function restoreOriginals() {
    translationOriginals.forEach(function (item) { item.el.textContent = item.text; });
  }

  function applyTranslations(translations) {
    if (translations.length !== translationOriginals.length) return;
    translationOriginals.forEach(function (item, i) { item.el.textContent = translations[i] || item.text; });
  }

  function setTranslateLoading(show) {
    if (translateLoading) translateLoading.classList.toggle("visible", !!show);
    if (translateSelect) translateSelect.disabled = !!show;
  }

  if (translateSelect) {
    collectTranslatable();
    translateSelect.addEventListener("change", function () {
      var dest = (translateSelect.value || "").trim();
      if (dest === "" || dest === "en") {
        restoreOriginals();
        translationCache = {};
        return;
      }
      var texts = translationOriginals.map(function (item) { return item.text; });
      if (!texts.length) return;
      if (translationCache[dest] && translationCache[dest].length === texts.length) {
        applyTranslations(translationCache[dest]);
        return;
      }
      setTranslateLoading(true);
      api("POST", "/api/v1/translate", { texts: texts, dest: dest, src: "en" })
        .then(function (r) {
          var tr = r.translations || [];
          translationCache[dest] = tr;
          applyTranslations(tr);
        })
        .catch(function (err) {
          alert("Translation failed: " + err.message);
          translateSelect.value = "en";
          restoreOriginals();
        })
        .finally(function () { setTranslateLoading(false); });
    });
  }

  // --- Chat ---
  var chatMessages = document.getElementById("chat-messages");
  var chatForm = document.getElementById("chat-form");
  var chatInput = document.getElementById("chat-input");
  var conversationId = null;

  function appendChat(role, text, suggestedActions) {
    var div = document.createElement("div");
    div.className = "chat-message " + role;
    var html = "<div class=\"role\">" + (role === "user" ? "You" : "Assistant") + "</div><div class=\"text\">" + escapeHtml(text) + "</div>";
    if (suggestedActions && suggestedActions.length) {
      html += "<div class=\"suggested-actions\">";
      suggestedActions.forEach(function (a) {
        if (a.type === "link" && a.url) {
          html += "<a href=\"" + escapeHtml(a.url) + "\" target=\"_blank\" rel=\"noopener noreferrer\" class=\"action-link\">" + escapeHtml(a.label) + "</a>";
        } else if (a.label) {
          html += "<span class=\"action-label\">" + escapeHtml(a.label) + "</span>";
        }
      });
      html += "</div>";
    }
    div.innerHTML = html;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  chatForm.addEventListener("submit", function (e) {
    e.preventDefault();
    var msg = (chatInput.value || "").trim();
    if (!msg) return;
    appendChat("user", msg);
    chatInput.value = "";
    var btn = chatForm.querySelector('button[type="submit"]');
    btn.disabled = true;

    var body = { patient_id: getPatientId(), message: msg, language: "en" };
    if (conversationId) body.conversation_id = conversationId;

    api("POST", "/api/v1/chat/message", body)
      .then(function (res) {
        if (res.conversation_id) conversationId = res.conversation_id;
        var resp = res.response;
        var text = (resp && resp.text) || (resp && resp.message) || res.message || (typeof resp === "string" ? resp : JSON.stringify(res));
        var suggestedActions = (resp && resp.suggested_actions) || [];
        appendChat("assistant", text, suggestedActions);
      })
      .catch(function (err) {
        appendChat("assistant", "Error: " + err.message);
      })
      .finally(function () { btn.disabled = false; });
  });

  // --- Fertility Readiness ---
  document.getElementById("form-fertility-readiness").addEventListener("submit", function (e) {
    e.preventDefault();
    var f = e.target;
    var data = {
      age: parseInt(f.age.value, 10),
      medical_history: parseList(f.medical_history.value),
      lifestyle_smoking: !!f.lifestyle_smoking.checked,
      lifestyle_alcohol: f.lifestyle_alcohol.value,
      lifestyle_exercise: f.lifestyle_exercise.value,
      menstrual_pattern: f.menstrual_pattern.value,
      previous_pregnancies: parseInt(f.previous_pregnancies.value, 10) || 0,
      live_births: parseInt(f.live_births.value, 10) || 0,
      miscarriages: parseInt(f.miscarriages.value, 10) || 0,
      use_ai_insight: !!f.use_ai_insight.checked
    };
    if (f.bmi.value) data.bmi = parseFloat(f.bmi.value);
    if (f.cycle_length_days.value) data.cycle_length_days = parseInt(f.cycle_length_days.value, 10);
    if (f.years_trying.value) data.years_trying = parseFloat(f.years_trying.value);

    var box = document.getElementById("result-fertility-readiness");
    box.innerHTML = "<span class=\"loading\">Calculating…</span>";
    box.hidden = false;

    api("POST", "/api/v1/engagement/fertility-readiness", data)
      .then(function (r) {
        var html = "<h4>Result</h4>Risk score: " + r.risk_score + " — " + (r.risk_level || "") + "\n" + (r.guidance_text || "");
        if (r.next_steps && r.next_steps.length) html += "\n\nNext steps:\n<ul><li>" + r.next_steps.map(escapeHtml).join("</li><li>") + "</li></ul>";
        if (r.ai_insight) html += '<div class="ai-insight">' + escapeHtml(r.ai_insight) + "</div>";
        showResult(box, html, false);
      })
      .catch(function (err) { showResult(box, "Error: " + err.message, true); });
  });

  // --- Hormonal Predictor ---
  document.getElementById("form-hormonal-predictor").addEventListener("submit", function (e) {
    e.preventDefault();
    var f = e.target;
    var data = {
      age: parseInt(f.age.value, 10),
      sex: f.sex.value,
      irregular_cycles: !!f.irregular_cycles.checked,
      symptoms_acne: !!f.symptoms_acne.checked,
      symptoms_hirsutism: !!f.symptoms_hirsutism.checked,
      symptoms_heavy_bleeding: !!f.symptoms_heavy_bleeding.checked,
      symptoms_pain: !!f.symptoms_pain.checked,
      previous_tests_amh: !!f.previous_tests_amh.checked,
      previous_tests_semen: !!f.previous_tests_semen.checked,
      use_ai_insight: !!f.use_ai_insight.checked
    };
    if (f.cycle_length_days.value) data.cycle_length_days = parseInt(f.cycle_length_days.value, 10);
    if (f.years_trying.value) data.years_trying = parseFloat(f.years_trying.value);

    var box = document.getElementById("result-hormonal-predictor");
    box.innerHTML = "<span class=\"loading\">Getting suggestions…</span>";
    box.hidden = false;

    api("POST", "/api/v1/engagement/hormonal-predictor", data)
      .then(function (r) {
        var html = "<h4>Suggestions</h4>";
        if (r.when_to_test) html += escapeHtml(r.when_to_test) + "\n";
        if (r.suggest_amh) html += "• Suggest AMH test\n";
        if (r.suggest_semen_analysis) html += "• Suggest semen analysis\n";
        if (r.suggest_specialist) html += "• Suggest specialist visit\n";
        if (r.reasoning && r.reasoning.length) html += "\nReasoning:\n<ul><li>" + r.reasoning.map(escapeHtml).join("</li><li>") + "</li></ul>";
        if (r.ai_insight) html += '<div class="ai-insight">' + escapeHtml(r.ai_insight) + "</div>";
        showResult(box, html, false);
      })
      .catch(function (err) { showResult(box, "Error: " + err.message, true); });
  });

  // --- Visual Health ---
  document.getElementById("form-visual-health").addEventListener("submit", function (e) {
    e.preventDefault();
    var f = e.target;
    var data = { use_ai_insight: !!f.use_ai_insight.checked };
    if (f.self_reported_sleep_hours.value) data.self_reported_sleep_hours = parseFloat(f.self_reported_sleep_hours.value);
    if (f.self_reported_stress_level.value) data.self_reported_stress_level = f.self_reported_stress_level.value;
    if (f.self_reported_bmi.value) data.self_reported_bmi = parseFloat(f.self_reported_bmi.value);

    var fileInput = document.getElementById("visual-health-image");
    if (fileInput.files && fileInput.files[0]) {
      var reader = new FileReader();
      reader.onload = function () {
        var b64 = reader.result;
        if (b64.indexOf("base64,") !== -1) b64 = b64.split("base64,")[1];
        data.image_base64 = b64;
        submitVisualHealth(data);
      };
      reader.readAsDataURL(fileInput.files[0]);
    } else {
      submitVisualHealth(data);
    }
  });

  function submitVisualHealth(data) {
    var box = document.getElementById("result-visual-health");
    box.innerHTML = "<span class=\"loading\">Getting recommendations…</span>";
    box.hidden = false;

    api("POST", "/api/v1/engagement/visual-health", data)
      .then(function (r) {
        var html = "<h4>Wellness</h4>";
        if (r.disclaimer) html += "<p>" + escapeHtml(r.disclaimer) + "</p>";
        if (r.recommendations && r.recommendations.length) html += "\nRecommendations:\n<ul><li>" + r.recommendations.map(escapeHtml).join("</li><li>") + "</li></ul>";
        if (r.wellness_indicators && Object.keys(r.wellness_indicators).length) html += "\nIndicators: " + escapeHtml(JSON.stringify(r.wellness_indicators));
        if (r.ai_insight) html += '<div class="ai-insight">' + escapeHtml(r.ai_insight) + "</div>";
        showResult(box, html, false);
      })
      .catch(function (err) { showResult(box, "Error: " + err.message, true); });
  }

  // --- Treatment Pathway ---
  document.getElementById("form-treatment-pathway").addEventListener("submit", function (e) {
    e.preventDefault();
    var f = e.target;
    var data = {
      age: parseInt(f.age.value, 10),
      sex: f.sex.value,
      known_diagnosis: parseList(f.known_diagnosis.value),
      previous_treatments: parseList(f.previous_treatments.value),
      preserving_fertility: !!f.preserving_fertility.checked,
      use_ai_insight: !!f.use_ai_insight.checked
    };
    if (f.years_trying.value) data.years_trying = parseFloat(f.years_trying.value);

    var box = document.getElementById("result-treatment-pathway");
    box.innerHTML = "<span class=\"loading\">Getting pathway…</span>";
    box.hidden = false;

    api("POST", "/api/v1/engagement/treatment-pathway", data)
      .then(function (r) {
        var html = "<h4>Pathway</h4>";
        if (r.primary_recommendation) html += "Primary: " + escapeHtml(r.primary_recommendation) + "\n";
        if (r.suggested_pathways && r.suggested_pathways.length) html += "\nSuggested: " + r.suggested_pathways.map(escapeHtml).join(", ") + "\n";
        if (r.reasoning && r.reasoning.length) html += "\nReasoning:\n<ul><li>" + r.reasoning.map(escapeHtml).join("</li><li>") + "</li></ul>";
        if (r.ai_insight) html += '<div class="ai-insight">' + escapeHtml(r.ai_insight) + "</div>";
        showResult(box, html, false);
      })
      .catch(function (err) { showResult(box, "Error: " + err.message, true); });
  });

  // --- Home IVF Eligibility ---
  document.getElementById("form-home-ivf-eligibility").addEventListener("submit", function (e) {
    e.preventDefault();
    var f = e.target;
    var data = {
      female_age: parseInt(f.female_age.value, 10),
      medical_contraindications: parseList(f.medical_contraindications.value),
      has_consulted_specialist: !!f.has_consulted_specialist.checked,
      ovarian_reserve_known: !!f.ovarian_reserve_known.checked,
      semen_analysis_known: !!f.semen_analysis_known.checked,
      stable_relationship_or_single_with_donor: !!f.stable_relationship_or_single_with_donor.checked,
      use_ai_insight: !!f.use_ai_insight.checked
    };
    if (f.male_age.value) data.male_age = parseInt(f.male_age.value, 10);

    var box = document.getElementById("result-home-ivf-eligibility");
    box.innerHTML = "<span class=\"loading\">Checking eligibility…</span>";
    box.hidden = false;

    api("POST", "/api/v1/engagement/home-ivf-eligibility", data)
      .then(function (r) {
        var html = "<h4>Eligibility</h4>" + (r.eligible ? "Eligible (preliminary)." : "May not be suitable for Home IVF based on inputs.");
        if (r.reasons && r.reasons.length) html += "\n\nReasons:\n<ul><li>" + r.reasons.map(escapeHtml).join("</li><li>") + "</li></ul>";
        if (r.missing_criteria && r.missing_criteria.length) html += "\nMissing criteria:\n<ul><li>" + r.missing_criteria.map(escapeHtml).join("</li><li>") + "</li></ul>";
        if (r.booking_message) html += "\n\n" + escapeHtml(r.booking_message);
        if (r.ai_insight) html += '<div class="ai-insight">' + escapeHtml(r.ai_insight) + "</div>";
        showResult(box, html, false);
      })
      .catch(function (err) { showResult(box, "Error: " + err.message, true); });
  });
})();
