(function () {
  "use strict";

  var API_PORT = window.API_PORT || "8000";
  const API_BASE = window.API_BASE || (window.location.protocol + "//" + window.location.hostname + ":" + API_PORT);

  // Age (and other min/max number) validation: clear input when value is outside range
  function setupNumberRangeValidation() {
    document.querySelectorAll(".form-engagement input[type=number][min][max]").forEach(function (input) {
      var min = parseInt(input.getAttribute("min"), 10);
      var max = parseInt(input.getAttribute("max"), 10);
      if (isNaN(min) || isNaN(max)) return;

      function validateAndClear() {
        var val = input.value.trim();
        if (val === "") return;
        var n = parseInt(val, 10);
        if (isNaN(n) || n < min || n > max) {
          input.value = "";
          input.setCustomValidity("Please enter an age between " + min + " and " + max + ".");
        } else {
          input.setCustomValidity("");
        }
      }

      input.addEventListener("blur", validateAndClear);
      input.addEventListener("input", function () {
        if (input.value.trim() !== "") input.setCustomValidity("");
      });
    });
  }
  setupNumberRangeValidation();

  var UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

  function getPatientId() {
    var id = sessionStorage.getItem("home_ivf_patient_id");
    if (!id || !UUID_REGEX.test(id)) {
      if (crypto.randomUUID) {
        id = crypto.randomUUID();
      } else {
        // UUID v4: replace x with random hex; replace y with random variant nibble (8,9,a,b)
        id = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
          var r = (Math.random() * 16) | 0;
          return c === "y" ? (r & 3 | 8).toString(16) : r.toString(16);
        });
      }
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

  // BMI from weight (kg) and height (cm): BMI = weight / (height/100)^2
  function calcBMI(weightKg, heightCm) {
    var w = parseFloat(weightKg);
    var h = parseFloat(heightCm);
    if (isNaN(w) || isNaN(h) || h <= 0) return null;
    var heightM = h / 100;
    var bmi = w / (heightM * heightM);
    return Math.round(bmi * 10) / 10;
  }

  function setupBMIDisplay(weightId, heightId, displayId) {
    var weightEl = document.getElementById(weightId);
    var heightEl = document.getElementById(heightId);
    var displayEl = document.getElementById(displayId);
    if (!weightEl || !heightEl || !displayEl) return;
    function update() {
      var bmi = calcBMI(weightEl.value, heightEl.value);
      displayEl.textContent = bmi != null ? "BMI: " + bmi + " (from weight & height)" : "BMI will be calculated from weight and height.";
    }
    weightEl.addEventListener("input", update);
    heightEl.addEventListener("input", update);
  }
  setupBMIDisplay("fertility-weight", "fertility-height", "fertility-bmi-display");
  setupBMIDisplay("visual-weight", "visual-height", "visual-bmi-display");

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

  // Allow Enter to submit; Shift+Enter for new line
  chatInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });

  chatForm.addEventListener("submit", function (e) {
    e.preventDefault();
    var msg = (chatInput.value || "").trim();
    if (!msg) return;
    appendChat("user", msg);
    chatInput.value = "";
    var btn = chatForm.querySelector('button[type="submit"]');
    btn.disabled = true;

    // Use Translate dropdown as preferred response language (read at submit so it's always current)
    var langSelect = document.getElementById("translate-select");
    var preferredLang = (langSelect && langSelect.value) ? langSelect.value : "en";
    var chatLang = (preferredLang === "hi") ? "hi" : "en";
    var body = { patient_id: getPatientId(), message: msg, language: chatLang };
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

  // --- Fertility Readiness: medical history dropdown + Other ---
  (function () {
    var selectEl = document.getElementById("medical-history-select");
    var otherWrap = document.getElementById("medical-history-other-wrap");
    var otherInput = document.getElementById("medical-history-other");
    var addBtn = document.getElementById("medical-history-add");
    var listEl = document.getElementById("medical-history-list");
    var added = [];

    function toggleOther() {
      otherWrap.hidden = selectEl.value !== "__other__";
      if (otherWrap.hidden) otherInput.value = "";
    }
    function renderList() {
      listEl.innerHTML = "";
      added.forEach(function (item, i) {
        var li = document.createElement("li");
        li.className = "medical-history-chip";
        li.innerHTML = "<span>" + escapeHtml(item) + "</span> <button type=\"button\" class=\"chip-remove\" data-index=\"" + i + "\" aria-label=\"Remove\">&times;</button>";
        listEl.appendChild(li);
      });
    }
    function addCondition() {
      var val = selectEl.value;
      if (val === "__other__") {
        val = (otherInput.value || "").trim();
        if (!val) return;
        otherInput.value = "";
      } else if (!val) return;
      added.push(val);
      renderList();
      selectEl.value = "";
      toggleOther();
    }
    addBtn.addEventListener("click", addCondition);
    selectEl.addEventListener("change", toggleOther);
    otherInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); addCondition(); }
    });
    listEl.addEventListener("click", function (e) {
      if (e.target.classList.contains("chip-remove")) {
        var i = parseInt(e.target.getAttribute("data-index"), 10);
        if (!isNaN(i)) { added.splice(i, 1); renderList(); }
      }
    });
  })();

  // --- Fertility Readiness ---
  document.getElementById("form-fertility-readiness").addEventListener("submit", function (e) {
    e.preventDefault();
    var f = e.target;
    var medicalHistoryListEl = document.getElementById("medical-history-list");
    var medicalHistory = [];
    if (medicalHistoryListEl) {
      medicalHistoryListEl.querySelectorAll(".medical-history-chip span").forEach(function (span) {
        var t = (span.textContent || "").trim();
        if (t) medicalHistory.push(t);
      });
    }
    var data = {
      age: parseInt(f.age.value, 10),
      medical_history: medicalHistory,
      lifestyle_smoking: !!f.lifestyle_smoking.checked,
      lifestyle_alcohol: f.lifestyle_alcohol.value,
      lifestyle_exercise: f.lifestyle_exercise.value,
      menstrual_pattern: f.menstrual_pattern.value,
      previous_pregnancies: parseInt(f.previous_pregnancies.value, 10) || 0,
      live_births: parseInt(f.live_births.value, 10) || 0,
      miscarriages: parseInt(f.miscarriages.value, 10) || 0,
      use_ai_insight: !!f.use_ai_insight.checked
    };
    var bmi = calcBMI(f.weight_kg && f.weight_kg.value, f.height_cm && f.height_cm.value);
    if (bmi != null) data.bmi = bmi;
    if (f.cycle_length_days.value) data.cycle_length_days = parseInt(f.cycle_length_days.value, 10);
    if (f.years_trying.value) data.years_trying = parseFloat(f.years_trying.value);

    var box = document.getElementById("result-fertility-readiness");
    box.innerHTML = "<span class=\"loading\">Calculating…</span>";
    box.hidden = false;

    api("POST", "/api/v1/engagement/fertility-readiness", data)
      .then(function (r) {
        var html = "<h4>Result</h4>Risk score: " + r.risk_score + " — " + (r.risk_level || "") + "\n" + (r.guidance_text || "");
        if (r.next_steps && r.next_steps.length) html += "\n\nNext steps:\n<ul><li>" + r.next_steps.map(escapeHtml).join("</li><li>") + "</li></ul>";
        if (r.medical_history_recognized && r.medical_history_recognized.length) html += "\n\n<strong>Medical conditions used in score:</strong> " + r.medical_history_recognized.map(escapeHtml).join(", ") + ".";
        if (r.medical_history_unrecognized && r.medical_history_unrecognized.length) html += "\n\n<strong>Not recognized (not used in score):</strong> " + r.medical_history_unrecognized.map(escapeHtml).join(", ") + ". Check spelling or choose from the suggested options next time.";
        if (r.ai_insight) html += '<div class="ai-insight"><h4>AI Insight</h4><p>' + escapeHtml(r.ai_insight) + "</p></div>";
        showResult(box, html, false);
      })
      .catch(function (err) { showResult(box, "Error: " + err.message, true); });
  });

  // --- Hormonal Predictor: show female block for Female only, male block for Male only ---
  (function () {
    var sexSelect = document.getElementById("hormonal-sex");
    var femaleBlock = document.getElementById("hormonal-female-fields");
    var maleBlock = document.getElementById("hormonal-male-fields");
    if (!sexSelect || !femaleBlock || !maleBlock) return;
    function setBlock(block, visible) {
      block.hidden = !visible;
      block.querySelectorAll("input").forEach(function (input) {
        input.disabled = !visible;
      });
      block.querySelectorAll("label").forEach(function (label) {
        if (label.querySelector("input")) label.style.pointerEvents = visible ? "" : "none";
      });
    }
    function updateHormonalFields() {
      var sex = sexSelect.value;
      setBlock(femaleBlock, sex === "female");
      setBlock(maleBlock, sex === "male");
    }
    sexSelect.addEventListener("change", updateHormonalFields);
    updateHormonalFields();
  })();

  // --- Hormonal Predictor ---
  document.getElementById("form-hormonal-predictor").addEventListener("submit", function (e) {
    e.preventDefault();
    var f = e.target;
    var sex = f.sex.value;
    var showFemale = sex === "female";
    var showMale = sex === "male";
    var data = {
      age: parseInt(f.age.value, 10),
      sex: sex,
      irregular_cycles: showFemale ? !!f.irregular_cycles.checked : false,
      symptoms_acne: showFemale ? !!f.symptoms_acne.checked : false,
      symptoms_hirsutism: showFemale ? !!f.symptoms_hirsutism.checked : false,
      symptoms_heavy_bleeding: showFemale ? !!f.symptoms_heavy_bleeding.checked : false,
      symptoms_pain: showFemale ? !!f.symptoms_pain.checked : false,
      previous_tests_amh: showFemale ? !!f.previous_tests_amh.checked : false,
      previous_tests_semen: showMale ? !!f.previous_tests_semen.checked : false,
      use_ai_insight: !!f.use_ai_insight.checked
    };
    if (showFemale && f.cycle_length_days.value) data.cycle_length_days = parseInt(f.cycle_length_days.value, 10);
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
        if (r.ai_insight) html += '<div class="ai-insight"><h4>AI Insight</h4><p>' + escapeHtml(r.ai_insight) + "</p></div>";
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
    var bmi = calcBMI(f.weight_kg && f.weight_kg.value, f.height_cm && f.height_cm.value);
    if (bmi != null) data.self_reported_bmi = bmi;

    var fileInput = document.getElementById("visual-health-image");
    var MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024; // 5 MB
    if (fileInput.files && fileInput.files[0]) {
      var file = fileInput.files[0];
      if (file.size > MAX_IMAGE_SIZE_BYTES) {
        var box = document.getElementById("result-visual-health");
        var msg = "File too large. Maximum upload size is 5 MB. Your file is " + (file.size / (1024 * 1024)).toFixed(1) + " MB.";
        showResult(box, msg, true);
        return;
      }
      var reader = new FileReader();
      reader.onload = function () {
        var b64 = reader.result;
        if (b64.indexOf("base64,") !== -1) b64 = b64.split("base64,")[1];
        data.image_base64 = b64;
        submitVisualHealth(data);
      };
      reader.readAsDataURL(file);
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
        var html = "<h4>Your wellness overview</h4>";
        if (r.summary) html += "<p class=\"result-summary\">" + escapeHtml(r.summary) + "</p>";
        if (r.wellness_summary) html += "<p class=\"result-what-you-shared\"><strong>What you shared:</strong> " + escapeHtml(r.wellness_summary) + "</p>";
        if (r.recommendations && r.recommendations.length) html += "<p class=\"result-section-title\"><strong>Suggestions for you</strong></p><ul><li>" + r.recommendations.map(escapeHtml).join("</li><li>") + "</li></ul>";
        if (r.image_analysis || r.ai_insight) {
          html += "<div class=\"ai-insight\"><h4>AI Insight</h4>";
          if (r.image_analysis) {
            var cleanAnalysis = r.image_analysis.replace(/\n+/g, " ").trim();
            html += "<p class=\"image-analysis-text\"><strong>What we noticed from your photo</strong><br>" + escapeHtml(cleanAnalysis) + "</p>";
          }
          if (r.ai_insight) html += "<p>" + escapeHtml(r.ai_insight) + "</p>";
          html += "</div>";
        }
        if (r.disclaimer) html += "<p class=\"result-disclaimer\">" + escapeHtml(r.disclaimer) + "</p>";
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
        if (r.ai_insight) html += '<div class="ai-insight"><h4>AI Insight</h4><p>' + escapeHtml(r.ai_insight) + "</p></div>";
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
        if (r.ai_insight) html += '<div class="ai-insight"><h4>AI Insight</h4><p>' + escapeHtml(r.ai_insight) + "</p></div>";
        showResult(box, html, false);
      })
      .catch(function (err) { showResult(box, "Error: " + err.message, true); });
  });
})();
