"use strict";

/*
 * The web view is intentionally a renderer, not a second graph builder.
 * New servers return one nested Experiment view model. The legacy adapter
 * below exists only so an older v1 endpoint still renders a useful inspection
 * screen while the workspace is being migrated. Mutations remain isolated to
 * the explicit authoring commit endpoints near the end of this file.
 */

const state = {
  document: null,
  selectedVariant: null,
  view: "overview",
  runDetail: { address: null, metric: null, response: null, loading: false, error: null, request: 0 },
  comparison: {
    selectedRuns: [],
    metric: null,
    availableMetrics: [],
    response: null,
    loading: false,
    error: null,
    request: 0,
  },
  authoring: {
    document: null,
    loading: false,
    error: null,
    notice: null,
    request: 0,
  },
};

const elements = {
  experimentContext: document.getElementById("experiment-context"),
  observedAt: document.getElementById("observed-at"),
  refresh: document.getElementById("refresh"),
  overviewView: document.getElementById("overview-view"),
  compareView: document.getElementById("compare-view"),
  changesView: document.getElementById("changes-view"),
  error: document.getElementById("error"),
  experimentName: document.getElementById("experiment-name"),
  experimentMeta: document.getElementById("experiment-meta"),
  currentCount: document.getElementById("current-count"),
  currentVariants: document.getElementById("current-variants"),
  historyToggle: document.getElementById("history-toggle"),
  historyCount: document.getElementById("history-count"),
  historicalVariants: document.getElementById("historical-variants"),
  detail: document.getElementById("detail"),
};

function node(tagName, className, text) {
  const value = document.createElement(tagName);
  if (className) value.className = className;
  if (text !== undefined && text !== null) value.textContent = String(text);
  return value;
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function firstPresent(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function stringValue(value, fallback = "") {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function arrayValue(value) {
  return Array.isArray(value) ? value : [];
}

function formatValue(value) {
  if (value === undefined || value === null || value === "") return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch (_error) {
      return "[unavailable]";
    }
  }
  return String(value);
}

function humanizeKey(value) {
  return String(value)
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function flattenSummary(value, prefix = "", output = [], depth = 0) {
  if (output.length >= 12 || depth > 2) return output;
  if (!isObject(value)) {
    if (prefix) output.push([prefix, value]);
    return output;
  }
  for (const [key, child] of Object.entries(value)) {
    if (output.length >= 12) break;
    if (
      key === "schema_version" ||
      /(^|_)(hash|id|identity|revision)($|_)/i.test(key)
    ) {
      continue;
    }
    const path = prefix ? `${prefix}.${key}` : key;
    if (isObject(child)) {
      flattenSummary(child, path, output, depth + 1);
    } else if (Array.isArray(child)) {
      output.push([path, child]);
    } else {
      output.push([path, child]);
    }
  }
  return output;
}

function summarySource(value) {
  if (!isObject(value)) return {};
  return firstPresent(value.summary, value.values, value.document, value) || {};
}

function summaryRows(value, { exclude = [] } = {}) {
  const excluded = new Set(exclude);
  return flattenSummary(summarySource(value)).filter(([key]) => {
    const leaf = key.split(".").at(-1);
    return (
      !excluded.has(leaf) &&
      !/(^|_)(hash|id|identity|revision)($|_)/i.test(leaf)
    );
  });
}

function identityEntries(...sources) {
  const result = [];
  const seen = new Set();
  const identityKey =
    /(^|_)(hash|id|identity|revision|attempt|run_spec|comparison|source|profile|case)($|_)/i;

  function add(key, value) {
    if (!identityKey.test(key) || value === undefined || value === null) return;
    const normalized = typeof value === "object" ? formatValue(value) : String(value);
    const marker = `${key}\u0000${normalized}`;
    if (seen.has(marker)) return;
    seen.add(marker);
    result.push([key, normalized]);
  }

  function visit(value, prefix = "") {
    if (!isObject(value)) return;
    for (const [key, child] of Object.entries(value)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (key === "identity" || key === "identities" || key === "ids") {
        visit(child, prefix);
      } else if (isObject(child)) {
        visit(child, path);
      } else {
        add(path, child);
      }
    }
  }

  for (const source of sources) visit(source);
  return result;
}

function identityDetails(label, ...sources) {
  const entries = identityEntries(...sources);
  if (entries.length === 0) return null;
  const details = node("details", "identity-details");
  details.append(node("summary", "identity-summary", `${label} details`));
  const list = node("dl", "identity-list");
  for (const [key, value] of entries) {
    list.append(node("dt", "identity-key", humanizeKey(key)));
    list.append(node("dd", "identity-value", value));
  }
  details.append(list);
  return details;
}

function normalizeCommitState(value) {
  if (isObject(value)) {
    if (value.locked === true || value.is_locked === true) return "locked";
    if (value.dirty === true || value.uncommitted === true) return "uncommitted";
  }
  const source = isObject(value)
    ? firstPresent(value.state, value.status, value.code)
    : value;
  const raw = stringValue(source, "unknown").toLowerCase().replace(/[ -]+/g, "_");
  if (raw.includes("lock")) return "locked";
  if (raw.includes("dirty") || raw.includes("uncommit") || raw.includes("draft")) {
    return "uncommitted";
  }
  if (raw === "clean" || raw === "committed" || raw === "current" || raw === "ready") {
    return "clean";
  }
  return "unknown";
}

function stateLabel(value, kind = "State") {
  const normalized = normalizeCommitState(value);
  const labels = {
    clean: kind === "Code" ? "Code clean" : "Committed",
    uncommitted: kind === "Code" ? "Code uncommitted" : "Uncommitted",
    locked: kind === "Code" ? "Code locked" : `${kind} locked`,
    unknown: `${kind} state unavailable`,
  };
  return labels[normalized];
}

function statusPill(label, status, className = "") {
  const normalized = stringValue(status, "unknown").toLowerCase().replace(/[ _]+/g, "-");
  return node("span", `status-pill status-${normalized} ${className}`.trim(), label);
}

function statusLabel(status) {
  const value = stringValue(status, "unknown").toLowerCase();
  const labels = {
    done: "Done",
    complete: "Done",
    completed: "Done",
    failed: "Failed",
    interrupted: "Interrupted",
    abandoned: "Abandoned",
    running: "Recorded active",
    allocated: "Recorded active",
    pending: "Pending",
    unknown: "Status unavailable",
  };
  return labels[value] || humanizeKey(value);
}

function metricText(value) {
  if (!isObject(value)) return null;
  const primary = value.primary;
  if (isObject(primary) && typeof primary.name === "string") {
    return `${primary.name}: ${formatValue(primary.value)}`;
  }
  const summary = firstPresent(value.metric_summary, value.metrics, value.summary);
  if (!isObject(summary)) return null;
  const rows = flattenSummary(summary);
  if (rows.length === 0) return null;
  return `${humanizeKey(rows[0][0])}: ${formatValue(rows[0][1])}`;
}

function normalizeArtifactList(value) {
  if (Array.isArray(value)) return value;
  if (!isObject(value)) return [];
  return Object.entries(value).map(([name, item]) =>
    isObject(item) ? { name, ...item } : { name, value: item },
  );
}

function normalizeModel(value) {
  if (!value) return null;
  if (typeof value === "string") return { name: "Model", identity: { model_id: value } };
  if (!isObject(value)) return null;
  return {
    ...value,
    name: stringValue(firstPresent(value.display_name, value.name, value.label), "Model"),
    evaluations: normalizeArtifactList(
      firstPresent(
        value.evaluations,
        value.evals,
        value.evaluation,
        value.children?.evaluations,
        value.children?.eval,
        value.lineage?.evaluations,
        value.lineage?.eval,
      ),
    ),
    exports: normalizeArtifactList(
      firstPresent(
        value.exports,
        value.export_runs,
        value.export,
        value.children?.exports,
        value.children?.export,
        value.lineage?.exports,
        value.lineage?.export,
      ),
    ),
  };
}

function normalizeRun(value, index = 0) {
  const run = isObject(value) ? value : {};
  const model = normalizeModel(
    firstPresent(
      run.model,
      run.produced_model,
      run.output?.model,
      run.output_model,
      run.result?.model,
      run.lineage?.model,
      run.model_ref,
    ),
  );
  const directEvaluations = normalizeArtifactList(
    firstPresent(
      run.evaluations,
      run.evals,
      run.evaluation,
      run.children?.evaluations,
      run.children?.eval,
      run.downstream?.evaluations,
      run.downstream?.eval,
      run.lineage?.evaluations,
      run.lineage?.eval,
    ),
  );
  const directExports = normalizeArtifactList(
    firstPresent(
      run.exports,
      run.export_runs,
      run.export,
      run.children?.exports,
      run.children?.export,
      run.downstream?.exports,
      run.downstream?.export,
      run.lineage?.exports,
      run.lineage?.export,
    ),
  );
  return {
    ...run,
    name: stringValue(
      firstPresent(run.display_name, run.name, run.run_name, run.run_id),
      `Run ${index + 1}`,
    ),
    action: stringValue(firstPresent(run.action, run.kind, run.type), "Run"),
    status: stringValue(firstPresent(run.status, run.lifecycle?.status), "unknown"),
    model,
    evaluations: directEvaluations.length ? directEvaluations : model?.evaluations || [],
    exports: directExports.length ? directExports : model?.exports || [],
  };
}

function legacyRuns(historyVariant) {
  const runs = [];
  for (const group of arrayValue(historyVariant?.training_groups)) {
    for (const seed of arrayValue(group.seeds)) {
      for (const run of arrayValue(seed.runs)) {
        runs.push({
          ...run,
          comparison_group: group.name,
          seed: seed.seed,
          model: seed.model,
        });
      }
      if (seed.model && !seed.runs.length) {
        runs.push({
          action: "Train",
          comparison_group: group.name,
          seed: seed.seed,
          model: seed.model,
          status: "done",
        });
      }
    }
  }
  return runs.map(normalizeRun);
}

function legacyExperiment(value) {
  const catalog = value.authored_catalog || {};
  const authoredExperiment = catalog.experiment || {};
  const historyExperiment = arrayValue(value.generated_history?.experiments).find(
    (item) => item?.name === authoredExperiment.name,
  );
  const authored = arrayValue(catalog.variants);
  const history = arrayValue(historyExperiment?.variants);
  const byName = new Map();
  for (const variant of authored) byName.set(variant.name, { ...variant });
  for (const variant of history) {
    const current = byName.get(variant.name) || {};
    byName.set(variant.name, {
      ...current,
      ...variant,
      history_only: !authored.some((item) => item.name === variant.name),
      runs: legacyRuns(variant),
    });
  }
  return {
    ...authoredExperiment,
    name: stringValue(authoredExperiment.name, "Experiment"),
    variants: [...byName.values()],
  };
}

function nestedExperiment(value) {
  if (!isObject(value)) return null;
  const candidates = [
    value.experiment,
    value.view?.experiment,
    value.data?.experiment,
    value.view_model?.experiment,
    value.view_model,
    value.view,
    value.data,
    value,
  ];
  return (
    candidates.find(
      (candidate) => isObject(candidate) && Array.isArray(candidate.variants),
    ) || null
  );
}

function normalizeVariant(value, index = 0) {
  const variant = isObject(value) ? value : {};
  const code = isObject(variant.code) ? variant.code : {};
  const options = firstPresent(
    variant.options,
    variant.option_set,
    variant.options_snapshot,
    variant.option_summary,
    {},
  );
  const rawRuns = firstPresent(
    variant.runs,
    variant.execution?.runs,
    variant.history?.runs,
    variant.run_history,
    variant.execution_records,
    [],
  );
  const normalizedCode = isObject(code)
    ? code
    : isObject(variant.code_summary)
      ? { summary: variant.code_summary }
      : {};
  return {
    ...variant,
    name: stringValue(
      firstPresent(variant.display_name, variant.name, variant.label),
      `Variant ${index + 1}`,
    ),
    code: normalizedCode,
    codeState: normalizeCommitState(
      firstPresent(
        variant.code_state,
        variant.code?.state,
        variant.status?.code,
        variant.state?.code,
        variant.state,
      ),
    ),
    options: isObject(options) ? options : {},
    runs: arrayValue(rawRuns).map(normalizeRun),
  };
}

function normalizeDocument(value) {
  const nested = nestedExperiment(value);
  const experiment = nested || legacyExperiment(value);
  const variants = arrayValue(experiment.variants).map(normalizeVariant);
  const historical = variants.filter(
    (variant) => variant.history_only || variant.historical_only,
  );
  const current = variants.filter(
    (variant) => !variant.history_only && !variant.historical_only,
  );
  return {
    observed_at: stringValue(value?.observed_at, ""),
    experiment: {
      ...experiment,
      name: stringValue(experiment.name, "Experiment"),
      type: stringValue(experiment.type, "—"),
      question: stringValue(
        firstPresent(
          experiment.question,
          experiment.research_question,
          experiment.research?.question,
        ),
        "",
      ),
      template: firstPresent(experiment.template, {}),
      commitState: normalizeCommitState(
        firstPresent(
          experiment.commit_state,
          experiment.commit?.state,
          experiment.commit?.status,
          experiment.revision_state,
          experiment.state,
        ),
      ),
      variants,
    },
    currentVariants: current,
    historicalVariants: historical,
  };
}

function validateOverview(value) {
  if (!value || typeof value !== "object" || typeof value.observed_at !== "string") {
    throw new Error("overview response has an invalid shape");
  }
  if (
    !nestedExperiment(value) &&
    (!value.authored_catalog || !value.generated_history)
  ) {
    throw new Error("overview response has no Experiment view model");
  }
  return normalizeDocument(value);
}

function formatObserved(value) {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? `Observed ${value}`
    : `Observed ${parsed.toLocaleString()}`;
}

function variantButton(variant, selected, onSelect) {
  const button = node("button", "variant-button");
  button.type = "button";
  button.setAttribute("aria-current", selected ? "page" : "false");
  button.setAttribute("aria-label", `View Variant ${variant.name}`);
  button.addEventListener("click", onSelect);
  const row = node("span", "variant-button-row");
  row.append(node("span", "variant-button-name", variant.name));
  row.append(
    statusPill(stateLabel(variant.codeState, "Code"), variant.codeState, "compact"),
  );
  button.append(row);
  if (variant.runs.length > 0) {
    button.append(
      node(
        "span",
        "variant-button-meta",
        `${variant.runs.length} ${variant.runs.length === 1 ? "Run" : "Runs"}`,
      ),
    );
  }
  return button;
}

function selectVariant(name) {
  state.selectedVariant = name;
  if (state.view === "run") selectView("overview");
  render();
}

function renderTree(data) {
  elements.currentCount.textContent = String(data.currentVariants.length);
  elements.historyCount.textContent = String(data.historicalVariants.length);
  elements.currentVariants.replaceChildren(
    ...data.currentVariants.map((variant) =>
      variantButton(
        variant,
        state.selectedVariant === variant.name,
        () => selectVariant(variant.name),
      ),
    ),
  );
  elements.historicalVariants.replaceChildren(
    ...data.historicalVariants.map((variant) =>
      variantButton(
        variant,
        state.selectedVariant === variant.name,
        () => selectVariant(variant.name),
      ),
    ),
  );
  if (data.historicalVariants.length === 0) {
    elements.historicalVariants.hidden = true;
    elements.historyToggle.disabled = true;
    elements.historyToggle.setAttribute("aria-expanded", "false");
  } else {
    elements.historyToggle.disabled = false;
  }
}

function renderFacts(rows, className = "facts") {
  const list = node("dl", className);
  for (const [key, value] of rows) {
    list.append(node("dt", "fact-key", humanizeKey(key)));
    list.append(node("dd", "fact-value", formatValue(value)));
  }
  return list;
}

function renderExperimentSummary(experiment, variantCount) {
  const section = node("section", "experiment-panel");
  const header = node("div", "panel-header");
  const title = node("div", "panel-title");
  title.append(
    node("span", "eyebrow", "Experiment"),
    node("h2", "section-title", experiment.name),
  );
  title.append(
    statusPill(
      stateLabel(experiment.commitState, "Experiment"),
      experiment.commitState,
    ),
  );
  header.append(title);
  const identity = identityDetails("Experiment identity", experiment);
  if (identity) header.append(identity);
  section.append(header);

  const question = node("div", "question-block");
  question.append(node("div", "eyebrow", "Research question"));
  question.append(
    node("p", "question", experiment.question || "No research question recorded."),
  );
  section.append(question);

  const template = isObject(experiment.template)
    ? firstPresent(experiment.template.name, experiment.template.id)
    : experiment.template;
  section.append(
    renderFacts(
      [
        ["Type", experiment.type],
        ["Template", stringValue(template, "—")],
        ["Variants", variantCount],
      ],
      "facts experiment-facts",
    ),
  );
  return section;
}

function renderCodeCard(variant) {
  const section = node("section", "input-card code-card");
  const header = node("div", "card-header");
  header.append(node("h3", "card-title", "Code"));
  header.append(statusPill(stateLabel(variant.codeState, "Code"), variant.codeState));
  section.append(header);
  const reason = firstPresent(
    variant.code?.lock_reason,
    variant.lock_reason,
    variant.status?.lock_reason,
  );
  if (reason) section.append(node("p", "card-note", reason));
  const rows = summaryRows(variant.code, { exclude: ["state"] });
  if (rows.length) section.append(renderFacts(rows, "facts compact-facts"));
  else section.append(node("p", "card-note", "Code summary is not available."));
  const identity = identityDetails("Code identity", variant.code, variant);
  if (identity) section.append(identity);
  return section;
}

function renderOptionsCard(variant) {
  const section = node("section", "input-card options-card");
  const header = node("div", "card-header");
  header.append(node("h3", "card-title", "Options"));
  header.append(node("span", "card-context", "Run snapshot source"));
  section.append(header);
  const rows = summaryRows(variant.options, {
    exclude: ["tracker", "seed", "device"],
  });
  if (rows.length) section.append(renderFacts(rows, "facts compact-facts"));
  else section.append(node("p", "card-note", "No current options recorded."));
  const identity = identityDetails(
    "Options identity",
    variant.options,
    variant.option_set,
  );
  if (identity) section.append(identity);
  return section;
}

function renderDownstreamItem(item, action) {
  const value = isObject(item) ? item : {};
  const article = node(
    "article",
    `downstream-item downstream-${action.toLowerCase()}`,
  );
  const header = node("div", "downstream-header");
  header.append(
    node(
      "h5",
      "downstream-name",
      stringValue(firstPresent(value.display_name, value.name, value.label), humanizeKey(action)),
    ),
  );
  header.append(statusPill(statusLabel(value.status), value.status || "unknown"));
  article.append(header);
  const inspect = runDetailButton(value);
  if (inspect) article.append(inspect);
  const metric = metricText(value);
  if (metric) article.append(node("p", "secondary", metric));
  const identity = identityDetails(`${action} identity`, value);
  if (identity) article.append(identity);
  return article;
}

function renderModel(model, run) {
  if (!model) return null;
  const section = node("section", "model-card");
  const header = node("div", "lineage-header");
  header.append(
    node("span", "lineage-kind", "Model"),
    node("h4", "lineage-name", model.name),
  );
  const modelStatus = stringValue(model.status, "done");
  header.append(statusPill(statusLabel(modelStatus), modelStatus));
  section.append(header);
  const facts = [];
  if (model.producer_run || run.name) {
    facts.push(["Produced by", stringValue(model.producer_run, run.name)]);
  }
  if (model.seed !== undefined || run.seed !== undefined) {
    facts.push(["Seed", firstPresent(model.seed, run.seed)]);
  }
  if (facts.length) section.append(renderFacts(facts, "facts lineage-facts"));
  const downstream = node("div", "downstream");
  const evaluations = arrayValue(model.evaluations);
  const exports = arrayValue(model.exports);
  if (evaluations.length) {
    const group = node("section", "downstream-group");
    group.append(node("h5", "downstream-title", "Evaluations"));
    group.append(...evaluations.map((item) => renderDownstreamItem(item, "Eval")));
    downstream.append(group);
  }
  if (exports.length) {
    const group = node("section", "downstream-group");
    group.append(node("h5", "downstream-title", "Exports"));
    group.append(...exports.map((item) => renderDownstreamItem(item, "Export")));
    downstream.append(group);
  }
  if (downstream.childElementCount) section.append(downstream);
  const identity = identityDetails("Model identity", model);
  if (identity) section.append(identity);
  return section;
}

function renderRunSnapshots(run) {
  const hasCode = isObject(run.code) || isObject(run.code_snapshot);
  const hasOptions = isObject(run.options) || isObject(run.options_snapshot);
  if (!hasCode && !hasOptions) return null;
  const section = node("section", "run-snapshots");
  section.append(node("h5", "downstream-title", "Captured inputs"));
  const tags = node("div", "snapshot-tags");
  if (hasCode) tags.append(node("span", "snapshot-tag", "Code snapshot"));
  if (hasOptions) tags.append(node("span", "snapshot-tag", "Options snapshot"));
  section.append(tags);
  const identity = identityDetails(
    "Run input identity",
    run.code,
    run.code_snapshot,
    run.options,
    run.options_snapshot,
  );
  if (identity) section.append(identity);
  return section;
}

function renderRun(run) {
  const article = node("article", "run-card");
  const header = node("div", "run-header");
  const title = node("div", "run-title");
  title.append(
    node("span", "lineage-kind", run.action),
    node("h4", "lineage-name", run.name),
  );
  header.append(title, statusPill(statusLabel(run.status), run.status));
  article.append(header);
  const inspect = runDetailButton(run);
  if (inspect) article.append(inspect);
  const facts = [];
  if (run.seed !== undefined) facts.push(["Seed", run.seed]);
  if (run.comparison_group) facts.push(["Comparison", run.comparison_group]);
  if (run.created_at) facts.push(["Created", run.created_at]);
  const metric = metricText(run);
  if (metric) facts.push(["Metric", metric]);
  if (facts.length) article.append(renderFacts(facts, "facts run-facts"));
  const snapshots = renderRunSnapshots(run);
  if (snapshots) article.append(snapshots);
  const model = renderModel(run.model, run);
  if (model) article.append(model);
  const directEvaluations = arrayValue(run.evaluations).filter(
    (item) => !run.model?.evaluations?.includes(item),
  );
  const directExports = arrayValue(run.exports).filter(
    (item) => !run.model?.exports?.includes(item),
  );
  if (directEvaluations.length || directExports.length) {
    const direct = node("div", "downstream direct-downstream");
    if (directEvaluations.length) {
      direct.append(...directEvaluations.map((item) => renderDownstreamItem(item, "Eval")));
    }
    if (directExports.length) {
      direct.append(...directExports.map((item) => renderDownstreamItem(item, "Export")));
    }
    article.append(direct);
  }
  const identity = identityDetails("Run identity", run);
  if (identity) article.append(identity);
  return article;
}

function runDetailButton(run) {
  const id = firstPresent(run.run_id, run.name);
  const variant = state.selectedVariant;
  if (typeof id !== "string" || !/^run-\d{3,}$/.test(id) || !variant) return null;
  const button = node("button", "refresh run-detail-button", "View Run");
  button.type = "button";
  button.setAttribute("aria-label", `View ${variant}/${id}`);
  button.addEventListener("click", () => {
    state.runDetail.address = `${variant}/${id}`;
    state.runDetail.metric = null;
    state.view = "run";
    loadRunDetail("heading");
  });
  return button;
}

function validateRunDetail(value, address) {
  const valid = (condition, message) => {
    if (!condition) throw new Error(`Run detail response ${message}`);
  };
  valid(comparisonExactFields(value, ["observed_at", "experiment", "run", "snapshots", "metrics", "log"]), "has invalid fields");
  valid(typeof value.observed_at === "string" && value.experiment === state.document?.experiment.name, "belongs to another Experiment");
  const run = value.run;
  valid(comparisonExactFields(run, ["variant", "run_id", "action", "status", "reason", "created_at", "updated_at", "seed", "device"]), "has invalid Run fields");
  valid(`${run.variant}/${run.run_id}` === address, "changed selected Run identity");
  valid(["train", "eval", "export"].includes(run.action), "has invalid action");
  valid(["allocated", "running", "done", "failed", "interrupted", "abandoned"].includes(run.status), "has invalid status");
  valid((run.reason === null || typeof run.reason === "string") && typeof run.device === "string" && typeof run.created_at === "string" && typeof run.updated_at === "string", "has invalid metadata");
  valid(run.seed === null || (Number.isSafeInteger(run.seed) && run.seed >= 0), "has invalid seed");
  const snapshots = value.snapshots;
  valid(comparisonExactFields(snapshots, ["source", "code", "options"]) && ["content_addressed", "legacy_snapshot"].includes(snapshots.source), "has invalid snapshot source");
  valid(comparisonExactFields(snapshots.code, ["schema_version", "template", "components"]) && snapshots.code.schema_version === 2 && isObject(snapshots.code.components) && comparisonExactFields(snapshots.code.template, ["name", "version"]), "has invalid Code snapshot");
  valid(comparisonExactFields(snapshots.options, ["schema_version", "dataset", "metrics", "train", "eval", "infer"]) && snapshots.options.schema_version === 2 && ["dataset", "metrics", "train", "eval", "infer"].every((key) => isObject(snapshots.options[key])), "has invalid Options snapshot");
  const log = value.log;
  valid(comparisonExactFields(log, ["availability", "text", "bytes_total", "lines_returned", "truncated", "decoding_replaced"]), "has invalid log fields");
  valid(["available", "not_recorded"].includes(log.availability) && typeof log.text === "string" && Number.isSafeInteger(log.bytes_total) && log.bytes_total >= 0 && Number.isSafeInteger(log.lines_returned) && log.lines_returned >= 0 && log.lines_returned <= 200 && typeof log.truncated === "boolean" && typeof log.decoding_replaced === "boolean", "has invalid log data");
  valid(log.availability !== "not_recorded" || (log.text === "" && log.bytes_total === 0 && log.lines_returned === 0 && !log.truncated && !log.decoding_replaced), "has inconsistent missing log data");
  let chart = null;
  if (run.action === "train") {
    valid(comparisonExactFields(value.metrics, ["selected_metric", "available_metrics", "series"]), "has invalid metrics");
    valid(value.metrics.series?.status === run.status && value.metrics.series?.seed === run.seed, "has inconsistent metric metadata");
    chart = normalizeComparisonResponse({ observed_at: value.observed_at, experiment: value.experiment, selected_metric: value.metrics.selected_metric, available_metrics: value.metrics.available_metrics, runs: [value.metrics.series], input_differences: [], evaluation_groups: [] }, [{ variant: run.variant, id: run.run_id, key: address, address }]);
  } else {
    valid(value.metrics === null, "has a training curve for a non-Train Run");
  }
  return { ...value, chart };
}

function capturedInputPanel(label, value) {
  const panel = node("details", "captured-input");
  panel.append(node("summary", "identity-summary", label));
  const content = node("pre", "run-text", JSON.stringify(value, null, 2));
  content.tabIndex = 0;
  content.setAttribute("aria-label", label);
  panel.append(content);
  return panel;
}

function renderRunDetail() {
  const section = node("section", "run-detail-panel");
  const back = node("button", "refresh", "Back to Variant");
  back.type = "button";
  back.addEventListener("click", () => selectView("overview"));
  section.append(back);
  const heading = node("h2", "section-title run-detail-title", state.runDetail.address);
  heading.tabIndex = -1;
  section.append(heading);
  if (state.runDetail.loading) {
    section.append(node("div", "empty-card", "Loading recorded Run…"));
    return section;
  }
  if (state.runDetail.error) {
    const error = node("div", "error", state.runDetail.error);
    error.setAttribute("role", "alert");
    section.append(error);
    return section;
  }
  const data = state.runDetail.response;
  if (!data) return section;
  const run = data.run;
  section.append(statusPill(statusLabel(run.status), run.status));
  section.append(renderFacts([["Action", run.action], ["Seed", run.seed], ["Device", run.device], ["Created", run.created_at], ["Last recorded update", run.updated_at], ["Observed", data.observed_at]]));
  if (["allocated", "running"].includes(run.status)) {
    section.append(node("p", "run-notice", "Recorded active. Process liveness is unknown; this is not a live-process check."));
  }
  if (run.reason !== null) {
    const reason = node("section", "run-reason");
    reason.append(node("h3", "section-title", "Recorded stop reason"), node("pre", "run-text", run.reason));
    section.append(reason);
  }
  section.append(node("h3", "section-title", "Captured inputs"));
  section.append(node("p", "secondary", "These are the inputs captured for this Run, not the current editable Code or next-Run Options."));
  if (data.snapshots.source === "legacy_snapshot") {
    section.append(node("p", "secondary", "Legacy captured snapshot; a full content-addressed OptionSet is unavailable."));
  }
  section.append(capturedInputPanel("Code used by this Run", data.snapshots.code));
  section.append(capturedInputPanel("Options used by this Run", data.snapshots.options));
  if (data.chart) {
    const control = node("div", "run-metric-control");
    const label = node("label", "secondary", "Training metric");
    label.htmlFor = "run-metric";
    const select = node("select", "metric-select");
    select.id = "run-metric";
    select.disabled = data.chart.metric_names.length === 0;
    for (const name of data.chart.metric_names) {
      const option = node("option", "", name);
      option.value = name;
      option.selected = name === data.chart.metric;
      select.append(option);
    }
    select.addEventListener("change", () => { state.runDetail.metric = select.value; loadRunDetail("metric"); });
    control.append(label, select);
    section.append(control);
    const series = data.metrics.series;
    section.append(node("p", "secondary", `Recorded step → ${data.chart.metric || "metric"}. ${series.points_returned} of ${series.points_total} points shown; no smoothing or interpolation.`));
    if (series.points.length === 1) {
      section.append(node("div", "empty-card", `One recorded point — step ${series.points[0].step}: ${series.points[0].value}. Not enough points for a learning curve.`));
    } else {
      section.append(renderMetricChart(data.chart));
    }
  } else {
    section.append(node("p", "secondary", "Training curves apply to Train Runs only."));
  }
  const logs = node("section", "worker-log");
  logs.append(node("h3", "section-title", "Recent worker log"));
  const log = data.log;
  if (log.availability === "not_recorded") {
    logs.append(node("div", "empty-card", "No worker log was recorded."));
  } else {
    logs.append(node("p", "secondary", `${log.lines_returned} lines shown · ${log.bytes_total} bytes recorded${log.truncated ? " · Truncated to the latest 200 lines / 64 KiB" : ""}`));
    if (log.decoding_replaced) logs.append(node("p", "run-notice", "Some log bytes could not be decoded as UTF-8 and are shown with replacement characters."));
    const pre = node("pre", "run-text", log.text || "The recorded log is empty.");
    pre.tabIndex = 0;
    pre.setAttribute("aria-label", "Recent worker log");
    logs.append(pre);
  }
  section.append(logs);
  return section;
}

async function loadRunDetail(focusTarget = null) {
  if (state.view !== "run" || !state.document || !state.runDetail.address) return;
  const requestId = ++state.runDetail.request;
  const address = state.runDetail.address;
  state.runDetail.loading = true;
  state.runDetail.response = null;
  state.runDetail.error = null;
  render();
  const params = new URLSearchParams({ run: address });
  if (state.runDetail.metric) params.set("metric", state.runDetail.metric);
  try {
    const response = await fetch(`/api/v1/run?${params.toString()}`, { headers: { Accept: "application/json" }, cache: "no-store" });
    const body = await response.json();
    if (!response.ok) throw new Error(body?.error?.message || `request failed (${response.status})`);
    if (requestId !== state.runDetail.request || state.view !== "run") return;
    const value = validateRunDetail(body, address);
    state.runDetail.response = value;
    state.runDetail.metric = value.metrics?.selected_metric || null;
  } catch (error) {
    if (requestId !== state.runDetail.request || state.view !== "run") return;
    state.runDetail.error = `Could not read Run: ${error.message}. No partial detail was displayed.`;
  }
  state.runDetail.loading = false;
  render();
  if (focusTarget === "heading") elements.detail.querySelector(".run-detail-title")?.focus();
  if (focusTarget === "metric") elements.detail.querySelector("#run-metric")?.focus();
}

function renderRuns(variant) {
  const section = node("section", "runs-section");
  const header = node("div", "section-header");
  header.append(node("h3", "section-title", "Runs"));
  header.append(
    node(
      "span",
      "section-count",
      `${variant.runs.length} ${variant.runs.length === 1 ? "Run" : "Runs"}`,
    ),
  );
  section.append(header);
  if (variant.runs.length === 0) {
    section.append(node("div", "empty-card", "No execution records for this Variant."));
    return section;
  }
  const list = node("div", "run-list");
  list.append(...variant.runs.map(renderRun));
  section.append(list);
  return section;
}

function renderVariantDetail(variant) {
  const section = node("section", "variant-panel");
  const header = node("div", "variant-detail-header");
  const title = node("div", "variant-detail-title");
  title.append(
    node("span", "eyebrow", "Variant"),
    node("h2", "variant-title", variant.name),
  );
  title.append(statusPill(stateLabel(variant.codeState, "Code"), variant.codeState));
  header.append(title);
  const identity = identityDetails("Variant identity", variant);
  if (identity) header.append(identity);
  section.append(header);
  const inputs = node("div", "input-grid");
  inputs.append(renderCodeCard(variant), renderOptionsCard(variant));
  section.append(inputs, renderRuns(variant));
  if (variant.history_only || variant.historical_only) {
    section.append(
      node(
        "p",
        "note",
        "This Variant exists in preserved history but is not in the current authored catalog.",
      ),
    );
  }
  return section;
}

function authoringExactFields(value, fields) {
  return (
    isObject(value) &&
    Object.keys(value).length === fields.length &&
    fields.every((field) => Object.hasOwn(value, field))
  );
}

function validateAuthoringItem(value) {
  if (
    !authoringExactFields(value, [
      "name",
      "state",
      "committable",
      "reason",
      "validation",
      "changes_total",
      "changes_truncated",
      "changes",
    ]) ||
    typeof value.name !== "string" ||
    !["clean", "uncommitted", "locked", "unavailable"].includes(value.state) ||
    typeof value.committable !== "boolean" ||
    !(value.reason === null || typeof value.reason === "string") ||
    !Number.isInteger(value.changes_total) ||
    value.changes_total < 0 ||
    typeof value.changes_truncated !== "boolean" ||
    !Array.isArray(value.changes)
  ) {
    throw new Error("authoring response has an invalid commit item");
  }
  if (
    !authoringExactFields(value.validation, ["status", "errors"]) ||
    value.validation.status !== "valid" ||
    !Array.isArray(value.validation.errors) ||
    value.validation.errors.some((error) => typeof error !== "string")
  ) {
    throw new Error("authoring response has an invalid validation result");
  }
  for (const change of value.changes) {
    if (
      !authoringExactFields(change, ["path", "change", "before", "after"]) ||
      typeof change.path !== "string" ||
      !["added", "removed", "modified"].includes(change.change)
    ) {
      throw new Error("authoring response has an invalid change summary");
    }
    for (const side of [change.before, change.after]) {
      if (
        !authoringExactFields(side, ["present", "value"]) ||
        typeof side.present !== "boolean"
      ) {
        throw new Error("authoring response has an invalid change value");
      }
    }
  }
  return value;
}

function validateAuthoring(value) {
  if (
    !authoringExactFields(value, ["observed_at", "available", "experiment", "variants"]) ||
    typeof value.observed_at !== "string" ||
    typeof value.available !== "boolean" ||
    !Array.isArray(value.variants)
  ) {
    throw new Error("authoring response has an invalid shape");
  }
  validateAuthoringItem(value.experiment);
  for (const variant of value.variants) {
    if (
      !authoringExactFields(variant, ["name", "code", "options", "active_records"]) ||
      typeof variant.name !== "string" ||
      !authoringExactFields(variant.options, ["mode", "message", "summary"]) ||
      variant.options.mode !== "next_run" ||
      typeof variant.options.message !== "string" ||
      !isObject(variant.options.summary) ||
      !authoringExactFields(variant.active_records, ["runs", "models"]) ||
      !Array.isArray(variant.active_records.runs) ||
      !Array.isArray(variant.active_records.models) ||
      [...variant.active_records.runs, ...variant.active_records.models].some(
        (name) => typeof name !== "string",
      )
    ) {
      throw new Error("authoring response has an invalid Variant item");
    }
    validateAuthoringItem(variant.code);
  }
  return value;
}

function authoringValue(cell) {
  return cell.present ? formatValue(cell.value) : "Not present";
}

function renderAuthoringChanges(item) {
  if (item.changes.length === 0) {
    return node("p", "authoring-empty", "No committed fields differ from the current draft.");
  }
  const wrapper = node("div", "authoring-change-scroll");
  const table = node("table", "authoring-change-table");
  const head = node("thead");
  const row = node("tr");
  row.append(node("th", "", "Field"), node("th", "", "Committed"), node("th", "", "Draft"));
  head.append(row);
  const body = node("tbody");
  for (const change of item.changes) {
    const changeRow = node("tr");
    const path = node("th", "authoring-change-path", change.path);
    path.scope = "row";
    path.append(node("span", `change-kind change-${change.change}`, humanizeKey(change.change)));
    changeRow.append(
      path,
      node("td", "authoring-change-value", authoringValue(change.before)),
      node("td", "authoring-change-value", authoringValue(change.after)),
    );
    body.append(changeRow);
  }
  table.append(head, body);
  wrapper.append(table);
  if (item.changes_truncated) {
    wrapper.append(
      node(
        "p",
        "authoring-truncated",
        `Showing ${item.changes.length} of ${item.changes_total} changes.`,
      ),
    );
  }
  return wrapper;
}

function authoringCommitButton(kind, name, item) {
  const button = node(
    "button",
    "commit-button",
    state.authoring.loading ? "Working…" : "Commit changes",
  );
  button.type = "button";
  button.disabled = !item.committable || state.authoring.loading;
  button.setAttribute("aria-label", `Commit ${kind} ${name}`);
  button.addEventListener("click", () => commitAuthoring(kind, name));
  return button;
}

function renderAuthoringItem(kind, item) {
  const section = node("section", "authoring-card");
  const header = node("div", "authoring-card-header");
  const title = node("div", "authoring-card-title");
  title.append(node("span", "eyebrow", kind), node("h3", "card-title", item.name));
  title.append(statusPill(stateLabel(item.state, kind === "Variant Code" ? "Code" : "Experiment"), item.state));
  header.append(title, authoringCommitButton(kind, item.name, item));
  section.append(header);
  if (item.reason) section.append(node("p", "authoring-reason", item.reason));
  section.append(renderAuthoringChanges(item));
  return section;
}

function renderAuthoringVariant(variant) {
  const section = node("section", "authoring-variant");
  section.append(renderAuthoringItem("Variant Code", variant.code));
  const options = node("section", "authoring-options");
  const header = node("div", "authoring-options-header");
  header.append(node("h4", "card-title", "Options"), statusPill("Next Run", "clean"));
  options.append(header, node("p", "card-note", variant.options.message));
  const rows = summaryRows(variant.options.summary, {
    exclude: ["tracker", "seed", "device"],
  });
  if (rows.length) options.append(renderFacts(rows, "facts compact-facts"));
  const records = [
    ...variant.active_records.runs.map((name) => `Run ${name}`),
    ...variant.active_records.models.map((name) => `Model ${name}`),
  ];
  if (records.length) {
    const active = node("div", "active-records");
    active.append(node("h4", "active-records-title", "Active records"));
    const list = node("ul", "active-records-list");
    list.append(...records.map((name) => node("li", "", name)));
    active.append(list);
    options.append(active);
  }
  section.append(options);
  return section;
}

function renderAuthoring() {
  const panel = node("section", "authoring-panel");
  const header = node("div", "authoring-page-header");
  header.append(
    node("span", "eyebrow", "Authoring"),
    node("h2", "section-title", "Changes to commit"),
    node(
      "p",
      "note",
      "Research JSON stays user-authored. HKDL commits Code and Experiment revisions and captures Options when a Run starts.",
    ),
  );
  panel.append(header);
  if (state.authoring.notice) {
    panel.append(node("div", "authoring-notice", state.authoring.notice));
  }
  if (state.authoring.error) {
    panel.append(node("div", "authoring-error", state.authoring.error));
  }
  if (state.authoring.loading && !state.authoring.document) {
    panel.append(node("div", "loading", "Checking authored changes…"));
    return panel;
  }
  const documentValue = state.authoring.document;
  if (!documentValue) {
    panel.append(node("div", "empty-card", "Authoring state is not available."));
    return panel;
  }
  if (!documentValue.available) {
    panel.append(
      node("div", "empty-card", "Commit management requires an active v2 workspace."),
    );
  }
  panel.append(renderAuthoringItem("Experiment", documentValue.experiment));
  const variants = node("div", "authoring-variants");
  variants.append(...documentValue.variants.map(renderAuthoringVariant));
  if (!documentValue.variants.length) {
    variants.append(node("div", "empty-card", "No authored Variants are available."));
  }
  panel.append(variants);
  return panel;
}

async function loadAuthoring() {
  if (state.view !== "changes" || !state.document) return;
  const requestId = state.authoring.request + 1;
  state.authoring.request = requestId;
  state.authoring.loading = true;
  state.authoring.error = null;
  render();
  try {
    const response = await fetch("/api/v1/authoring", {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    let body = {};
    try {
      body = await response.json();
    } catch (_error) {
      body = {};
    }
    if (!response.ok) {
      throw new Error(body?.error?.message || `authoring request failed (${response.status})`);
    }
    if (requestId !== state.authoring.request) return;
    state.authoring.document = validateAuthoring(body);
    state.authoring.error = null;
  } catch (error) {
    if (requestId !== state.authoring.request) return;
    state.authoring.document = null;
    state.authoring.error = `Could not validate authored changes: ${error.message}`;
  } finally {
    if (requestId === state.authoring.request) {
      state.authoring.loading = false;
      if (state.view === "changes") render();
    }
  }
}

async function commitAuthoring(kind, name) {
  const path =
    kind === "Experiment"
      ? "/api/v1/authoring/experiment/commit"
      : "/api/v1/authoring/variant/commit";
  const payload = kind === "Experiment" ? {} : { variant: name };
  state.authoring.loading = true;
  state.authoring.error = null;
  state.authoring.notice = null;
  render();
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        "X-HKDL-Action": "commit",
      },
      body: JSON.stringify(payload),
      cache: "no-store",
    });
    let body = {};
    try {
      body = await response.json();
    } catch (_error) {
      body = {};
    }
    if (!response.ok) {
      throw new Error(body?.error?.message || `commit request failed (${response.status})`);
    }
    if (
      !authoringExactFields(body, ["kind", "name", "changed", "authoring"]) ||
      !["experiment", "variant"].includes(body.kind) ||
      typeof body.name !== "string" ||
      typeof body.changed !== "boolean"
    ) {
      throw new Error("commit response has an invalid shape");
    }
    state.authoring.document = validateAuthoring(body.authoring);
    state.authoring.notice = body.changed
      ? `${humanizeKey(body.kind)} ${body.name} was committed.`
      : `${humanizeKey(body.kind)} ${body.name} was already clean.`;
    state.authoring.loading = false;
    await loadOverview();
  } catch (error) {
    state.authoring.loading = false;
    state.authoring.error = `Could not commit changes: ${error.message}`;
    if (state.view === "changes") render();
  }
}

/*
 * Comparison is deliberately a view-model adapter.  The server owns graph
 * traversal and metric validation; the browser only turns the returned
 * records into a small, readable comparison.  Keeping this boundary here
 * also lets the read-only page display a useful unavailable state while an
 * older server is still serving only /overview.
 */
const COMPARISON_SVG_NS = "http://www.w3.org/2000/svg";
const COMPARISON_COLORS = [
  "var(--hkdl-accent)",
  "var(--hkdl-warn)",
  "var(--hkdl-good)",
  "var(--hkdl-bad)",
];

function comparisonAction(value) {
  return stringValue(value, "").toLowerCase().replace(/[ _-]+/g, "");
}

function isTrainRun(run) {
  return comparisonAction(firstPresent(run?.action, run?.kind, run?.type)) === "train";
}

function runIdValue(run, index = 0) {
  const address = firstPresent(run?.address, run?.run_address, run?.path);
  if (typeof address === "string" && address.trim()) {
    const segments = address.split("/").filter(Boolean);
    if (segments.length) return segments.at(-1);
  }
  return stringValue(
    firstPresent(run?.run_id, run?.id, run?.run_name, run?.name),
    `run-${index + 1}`,
  );
}

function runAddressValue(variantName, run, index = 0) {
  const address = firstPresent(run?.address, run?.run_address, run?.path);
  if (typeof address === "string" && address.trim()) {
    const segments = address.split("/").filter(Boolean);
    if (segments.length >= 2) return segments.slice(-2).join("/");
  }
  return `${variantName}/${runIdValue(run, index)}`;
}

function comparisonRunRecords(documentValue) {
  const records = [];
  const seen = new Set();
  for (const variant of arrayValue(documentValue?.experiment?.variants)) {
    for (const [index, run] of arrayValue(variant.runs).entries()) {
      if (!isTrainRun(run)) continue;
      const name = stringValue(variant.name, "Variant");
      const id = runIdValue(run, index);
      const address = runAddressValue(name, run, index);
      const key = `${name}\u0000${id}`;
      if (seen.has(key)) continue;
      seen.add(key);
      records.push({ key, address, variant: name, id, run });
    }
  }
  return records;
}

function comparisonMetricNames(value, output = new Set()) {
  if (Array.isArray(value)) {
    for (const item of value) {
      if (typeof item === "string" && item.trim()) output.add(item);
      else if (isObject(item)) {
        const name = firstPresent(item.name, item.metric, item.key);
        if (typeof name === "string" && name.trim()) output.add(name);
      }
    }
    return output;
  }
  if (!isObject(value)) return output;
  for (const key of Object.keys(value)) {
    if (key === "summary" || key === "values" || key === "events") continue;
    if (key.trim()) output.add(key);
  }
  return output;
}

function availableComparisonMetrics(records, response = null) {
  const names = new Set();
  for (const record of records) {
    const run = record.run || {};
    comparisonMetricNames(run.metric_names, names);
    comparisonMetricNames(run.available_metrics, names);
    comparisonMetricNames(run.metric_summary, names);
    comparisonMetricNames(run.metrics, names);
    comparisonMetricNames(run.history, names);
  }
  if (response) {
    comparisonMetricNames(response.metric_names, names);
    comparisonMetricNames(response.available_metrics, names);
    comparisonMetricNames(response.metrics, names);
  }
  return [...names].sort((left, right) => left.localeCompare(right));
}

function comparisonPoint(value, index = 0) {
  if (Array.isArray(value)) {
    const step = Number(value[0]);
    const pointValue = Number(value[1]);
    if (Number.isFinite(step) && Number.isFinite(pointValue)) {
      return { step, value: pointValue };
    }
    return null;
  }
  if (!isObject(value)) return null;
  const step = Number(firstPresent(value.step, value.x, value.epoch, value.index, index));
  const pointValue = Number(
    firstPresent(value.value, value.y, value.metric_value, value.scalar),
  );
  if (!Number.isFinite(step) || !Number.isFinite(pointValue)) return null;
  return { step, value: pointValue };
}

function comparisonPoints(value) {
  if (!Array.isArray(value)) return [];
  return value
    .map((item, index) => comparisonPoint(item, index))
    .filter(Boolean)
    .sort((left, right) => left.step - right.step);
}

function comparisonSeries(value, metric) {
  if (Array.isArray(value)) return comparisonPoints(value);
  if (!isObject(value)) return [];
  const candidates = [
    value.series,
    value.metric_series,
    value.history,
    value.events,
    value.points,
    value.metrics?.[metric],
    value.values?.[metric],
    value.metrics?.history?.[metric],
  ];
  for (const candidate of candidates) {
    const points = comparisonPoints(candidate);
    if (points.length) return points;
  }
  if (isObject(value.metrics)) {
    for (const candidate of Object.values(value.metrics)) {
      const points = comparisonPoints(candidate);
      if (points.length) return points;
    }
  }
  return [];
}

function comparisonRoot(value) {
  if (!isObject(value)) return {};
  const candidates = [
    value.comparison,
    value.view?.comparison,
    value.data?.comparison,
    value.view_model?.comparison,
    value.data,
    value.view,
    value,
  ];
  return candidates.find(isObject) || {};
}

function comparisonCollection(value, keyName = "name") {
  if (Array.isArray(value)) return value;
  if (!isObject(value)) return [];
  return Object.entries(value).map(([key, item]) => {
    if (isObject(item)) return { [keyName]: key, ...item };
    return { [keyName]: key, value: item };
  });
}

function comparisonAddressParts(value) {
  if (typeof value !== "string") return {};
  const parts = value.split("/").filter(Boolean);
  if (parts.length < 2) return {};
  return { variant: parts.at(-2), id: parts.at(-1) };
}

function comparisonSnapshot(run, kind) {
  const snapshot = isObject(run?.snapshot) ? run.snapshot : {};
  const key = kind === "code" ? "code" : "options";
  return firstPresent(
    run?.[`${key}_snapshot`],
    run?.[key],
    snapshot[key],
    run?.captured?.[key],
    run?.inputs?.[key],
    null,
  );
}

function normalizeComparisonRun(value, index, metric, fallback = null) {
  const raw = isObject(value) ? value : {};
  const fallbackRun = fallback?.run || {};
  const addressParts = comparisonAddressParts(
    firstPresent(raw.address, raw.run_address, raw.path, raw.run),
  );
  const variant = stringValue(
    firstPresent(
      raw.variant_name,
      raw.variant,
      raw.owner?.variant,
      addressParts.variant,
      fallback?.variant,
    ),
    "Variant",
  );
  const id = stringValue(
    firstPresent(raw.run_id, raw.id, raw.run_name, raw.name, raw.run, addressParts.id, fallback?.id),
    `run-${index + 1}`,
  );
  const run = {
    ...fallbackRun,
    ...raw,
  };
  const snapshots = {
    code: comparisonSnapshot(raw, "code") || comparisonSnapshot(fallbackRun, "code"),
    options:
      comparisonSnapshot(raw, "options") || comparisonSnapshot(fallbackRun, "options"),
  };
  const evaluations = comparisonCollection(
    firstPresent(
      raw.evaluations,
      raw.evals,
      raw.evaluation,
      raw.downstream?.evaluations,
      raw.children?.evaluations,
      fallbackRun.evaluations,
    ),
    "case",
  );
  const status = stringValue(firstPresent(raw.status, fallbackRun.status), "unknown");
  const availability = stringValue(
    firstPresent(raw.availability, raw.metric_availability, "available"),
    "available",
  );
  const partial =
    raw.partial === true || raw.incomplete === true ||
    ["running", "allocated", "pending"].includes(status.toLowerCase());
  return {
    ...run,
    key: fallback?.key || `${variant}\u0000${id}`,
    address: fallback?.address || `${variant}/${id}`,
    variant,
    id,
    name: stringValue(firstPresent(raw.display_name, raw.name, raw.run_id, fallback?.run?.name), id),
    status,
    availability,
    partial,
    series: comparisonSeries(
      firstPresent(raw.series, raw.metric_series, raw.history, raw.events, raw.points, raw.metrics, raw.values, raw),
      metric,
    ),
    snapshots,
    evaluations,
    last_value: firstPresent(raw.last?.value, raw.last_value, raw.lastValue, raw.summary?.last_value),
    last_step: firstPresent(raw.last?.step, raw.last_step, raw.lastStep),
    points_total: firstPresent(raw.points_total, raw.total_points),
    points_returned: firstPresent(raw.points_returned, raw.returned_points),
  };
}

function comparisonContract(condition, message) {
  if (!condition) throw new Error(`comparison response ${message}`);
}

function comparisonExactFields(value, fields) {
  return (
    isObject(value) &&
    Object.keys(value).sort().join("\u0000") === [...fields].sort().join("\u0000")
  );
}

function normalizeComparisonResponse(value, selectedRecords = []) {
  comparisonContract(
    comparisonExactFields(value, [
      "observed_at",
      "experiment",
      "selected_metric",
      "available_metrics",
      "runs",
      "input_differences",
      "evaluation_groups",
    ]),
    "has invalid fields",
  );
  comparisonContract(typeof value.observed_at === "string", "has invalid observation time");
  comparisonContract(
    value.experiment === state.document?.experiment?.name,
    "belongs to another Experiment",
  );
  comparisonContract(
    Array.isArray(value.available_metrics) &&
      value.available_metrics.every((name) => typeof name === "string" && name),
    "has invalid metric names",
  );
  comparisonContract(
    new Set(value.available_metrics).size === value.available_metrics.length,
    "has duplicate metric names",
  );
  comparisonContract(
    value.selected_metric === null ||
      (typeof value.selected_metric === "string" &&
        value.available_metrics.includes(value.selected_metric)),
    "has an invalid selected metric",
  );
  comparisonContract(
    Array.isArray(value.runs) && value.runs.length === selectedRecords.length,
    "does not contain every selected Run",
  );
  const availabilityValues = new Set([
    "available",
    "tracking_disabled",
    "not_recorded",
    "metric_not_recorded",
  ]);
  const runs = value.runs.map((run, index) => {
    const expected = selectedRecords[index];
    comparisonContract(
      comparisonExactFields(run, [
        "variant",
        "run_id",
        "status",
        "seed",
        "comparison_group",
        "availability",
        "partial",
        "last",
        "points_total",
        "points_returned",
        "points",
      ]),
      "contains an invalid Run",
    );
    comparisonContract(
      run.variant === expected.variant && run.run_id === expected.id,
      "changed selected Run order or identity",
    );
    comparisonContract(typeof run.status === "string" && run.status, "has invalid Run status");
    comparisonContract(
      Number.isSafeInteger(run.seed) && run.seed >= 0,
      "has invalid Run seed",
    );
    comparisonContract(
      typeof run.comparison_group === "string" && run.comparison_group,
      "has invalid comparison group",
    );
    comparisonContract(
      availabilityValues.has(run.availability),
      "has invalid metric availability",
    );
    comparisonContract(typeof run.partial === "boolean", "has invalid partial state");
    comparisonContract(Array.isArray(run.points), "has invalid metric points");
    let previousStep = -1;
    const points = run.points.map((point) => {
      comparisonContract(
        comparisonExactFields(point, ["step", "value"]) &&
          Number.isSafeInteger(point.step) &&
          point.step >= 0 &&
          point.step > previousStep &&
          typeof point.value === "number" &&
          Number.isFinite(point.value),
        "has an invalid metric point",
      );
      previousStep = point.step;
      return { step: point.step, value: point.value };
    });
    comparisonContract(
      Number.isSafeInteger(run.points_total) &&
        Number.isSafeInteger(run.points_returned) &&
        run.points_total >= run.points_returned &&
        run.points_returned === points.length,
      "has invalid point counts",
    );
    comparisonContract(
      run.last === null ||
        (comparisonExactFields(run.last, ["step", "value"]) &&
          Number.isSafeInteger(run.last.step) &&
          run.last.step >= 0 &&
          typeof run.last.value === "number" &&
          Number.isFinite(run.last.value)),
      "has an invalid last metric",
    );
    comparisonContract(
      (points.length === 0 && run.last === null) ||
        (points.length > 0 &&
          run.last?.step === points.at(-1).step &&
          run.last?.value === points.at(-1).value),
      "last metric disagrees with its series",
    );
    return {
      ...run,
      key: expected.key,
      address: expected.address,
      id: run.run_id,
      name: run.run_id,
      series: points,
      last_value: run.last?.value,
      last_step: run.last?.step,
    };
  });
  comparisonContract(Array.isArray(value.input_differences), "has invalid input differences");
  const inputDifferences = value.input_differences.map((row) => {
    comparisonContract(
      comparisonExactFields(row, ["path", "values"]) &&
        typeof row.path === "string" &&
        row.path &&
        Array.isArray(row.values) &&
        row.values.length === runs.length,
      "has an invalid input difference",
    );
    for (const cell of row.values) {
      comparisonContract(
        comparisonExactFields(cell, ["present", "value"]) &&
          typeof cell.present === "boolean",
        "has an invalid input value",
      );
    }
    return row;
  });
  comparisonContract(Array.isArray(value.evaluation_groups), "has invalid Eval groups");
  for (const group of value.evaluation_groups) {
    comparisonContract(
      comparisonExactFields(group, ["case", "condition", "results"]) &&
        typeof group.case === "string" &&
        group.case &&
        (group.condition === null ||
          (Number.isSafeInteger(group.condition) && group.condition > 0)) &&
        Array.isArray(group.results),
      "has an invalid Eval group",
    );
    for (const result of group.results) {
      comparisonContract(
        comparisonExactFields(result, [
          "variant",
          "train_run_id",
          "eval_run_id",
          "status",
          "values",
        ]) &&
          typeof result.variant === "string" &&
          typeof result.train_run_id === "string" &&
          typeof result.eval_run_id === "string" &&
          typeof result.status === "string" &&
          isObject(result.values),
        "has an invalid Eval result",
      );
    }
  }
  return {
    metric: value.selected_metric,
    metric_names: value.available_metrics,
    partial: runs.some((run) => run.partial),
    unavailable: false,
    message: "",
    inputDifferences,
    runs,
    evalGroups: value.evaluation_groups,
  };
}

function comparisonCleanKey(value) {
  return !/(^|[._-])(hash|id|identity|revision|attempt|run_spec|comparison|source|profile|case)($|[._-])/i.test(
    value,
  );
}

function comparisonValueText(value) {
  if (value === undefined || value === null || value === "") return "Not recorded";
  if (isObject(value) || Array.isArray(value)) {
    try {
      return JSON.stringify(value);
    } catch (_error) {
      return "Unavailable";
    }
  }
  return formatValue(value);
}

function renderInputDifferenceTable(rows, runs) {
  const table = node("table", "snapshot-difference-table");
  const caption = node("caption", "visually-hidden", "Captured input differences");
  table.append(caption);
  const head = node("thead");
  const headerRow = node("tr");
  const pathHeader = node("th", "", "Field");
  pathHeader.scope = "col";
  headerRow.append(pathHeader);
  for (const run of runs) {
    const cell = node("th", "", comparisonRunLabel(run));
    cell.scope = "col";
    headerRow.append(cell);
  }
  head.append(headerRow);
  table.append(head);
  const body = node("tbody");
  for (const rawRow of rows) {
    const row = isObject(rawRow) ? rawRow : {};
    const tableRow = node("tr");
    const path = node("th", "snapshot-difference-path", stringValue(row.path, "Field"));
    path.scope = "row";
    tableRow.append(path);
    const values = arrayValue(row.values);
    for (let index = 0; index < runs.length; index += 1) {
      const entry = isObject(values[index]) ? values[index] : {};
      tableRow.append(
        node(
          "td",
          "",
          entry.present === false ? "Not recorded" : comparisonValueText(entry.value),
        ),
      );
    }
    body.append(tableRow);
  }
  table.append(body);
  return table;
}

function svgElement(tagName, attributes = {}) {
  const element = document.createElementNS(COMPARISON_SVG_NS, tagName);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  return element;
}

function comparisonMetricLastValue(run, metric) {
  if (!metric) return undefined;
  if (isObject(run?.last)) return firstPresent(run.last.value, run.last.last_value);
  const summary = firstPresent(
    run?.metric_summary,
    run?.summary?.metrics,
    run?.metrics?.summary,
  );
  if (isObject(summary) && isObject(summary[metric])) {
    return firstPresent(summary[metric].last_value, summary[metric].lastValue);
  }
  const series = comparisonSeries(
    firstPresent(run?.series, run?.metric_series, run?.history, run?.metrics, run?.values, run),
    metric,
  );
  return series.length ? series.at(-1).value : firstPresent(run?.last_value, run?.lastValue);
}

function comparisonStatus(run) {
  const availability = String(run.availability || "").toLowerCase();
  if (availability === "unavailable" || availability === "tracking_disabled") {
    return availability === "tracking_disabled" ? "Tracking disabled" : "Unavailable";
  }
  if (availability === "not_recorded" || availability === "metric_not_recorded") {
    return "Metric not recorded";
  }
  if (run.partial) return "Partial";
  return statusLabel(run.status);
}

function comparisonRunLabel(run) {
  return `${run.variant} · ${run.name}`;
}

function comparisonRecordByKey(documentValue, key) {
  return comparisonRunRecords(documentValue).find((record) => record.key === key) || null;
}

function selectedComparisonRecords() {
  if (!state.document) return [];
  return state.comparison.selectedRuns
    .map((key) => comparisonRecordByKey(state.document, key))
    .filter(Boolean);
}

function comparisonSelectionNote(count) {
  if (count < 2) return "Select at least 2 Train Runs to compare.";
  if (count > 4) return "Select no more than 4 Train Runs.";
  return `${count} Train Runs selected. Select up to 4.`;
}

function toggleComparisonRun(record, checked) {
  const selected = new Set(state.comparison.selectedRuns);
  if (checked && selected.size < 4) selected.add(record.key);
  if (!checked) selected.delete(record.key);
  state.comparison.selectedRuns = [...selected];
  state.comparison.response = null;
  state.comparison.error = null;
  render();
  loadComparison();
}

function renderComparisonRunPicker(documentValue) {
  const section = node("section", "comparison-picker");
  const header = node("div", "section-header");
  header.append(node("h3", "section-title", "Train Runs"));
  const allRecords = comparisonRunRecords(documentValue);
  header.append(node("span", "section-count", `${allRecords.length} available`));
  section.append(header);
  const selected = new Set(state.comparison.selectedRuns);
  const note = node("p", "comparison-selection-note", comparisonSelectionNote(selected.size));
  note.setAttribute("role", "status");
  note.setAttribute("aria-live", "polite");
  section.append(note);

  if (allRecords.length === 0) {
    section.append(node("div", "empty-card", "No Train Runs are available for comparison."));
    return section;
  }

  const groups = new Map();
  for (const record of allRecords) {
    if (!groups.has(record.variant)) groups.set(record.variant, []);
    groups.get(record.variant).push(record);
  }
  const groupList = node("div", "comparison-groups");
  for (const [variant, records] of groups) {
    const group = node("section", "comparison-group");
    group.append(node("h4", "comparison-group-title", variant));
    const options = node("div", "comparison-run-options");
    for (const record of records) {
      const label = node("label", "comparison-run-option");
      const checkbox = node("input");
      checkbox.type = "checkbox";
      checkbox.checked = selected.has(record.key);
      checkbox.disabled = !checkbox.checked && selected.size >= 4;
      checkbox.setAttribute("aria-label", `Compare ${record.variant} · ${record.id}`);
      checkbox.addEventListener("change", () => toggleComparisonRun(record, checkbox.checked));
      const copy = node("span", "comparison-run-copy");
      const title = node("span", "comparison-run-name", record.id);
      const meta = node(
        "span",
        "comparison-run-meta",
        `${statusLabel(record.run.status)} · ${record.run.seed === undefined ? "Seed not recorded" : `Seed ${record.run.seed}`} · ${comparisonValueText(comparisonMetricLastValue(record.run, state.comparison.metric))}`,
      );
      copy.append(title, meta);
      label.append(checkbox, copy);
      options.append(label);
    }
    group.append(options);
    groupList.append(group);
  }
  section.append(groupList);
  return section;
}

function comparisonControl(response, records) {
  const section = node("section", "comparison-controls");
  const label = node("label", "comparison-metric-label", "Metric");
  label.htmlFor = "comparison-metric";
  const select = node("select", "comparison-metric");
  select.id = "comparison-metric";
  select.name = "metric";
  const names = availableComparisonMetrics(records, response);
  for (const name of state.comparison.availableMetrics) {
    if (!names.includes(name)) names.push(name);
  }
  names.sort((left, right) => left.localeCompare(right));
  if (names.length === 0) {
    const option = node("option", "", "No recorded metrics");
    option.value = "";
    select.append(option);
    select.disabled = true;
  } else {
    for (const name of names) {
      const option = node("option", "", name);
      option.value = name;
      option.selected = name === state.comparison.metric;
      select.append(option);
    }
  }
  select.addEventListener("change", () => {
    state.comparison.metric = select.value || null;
    state.comparison.response = null;
    state.comparison.error = null;
    render();
    loadComparison();
  });
  label.append(select);
  section.append(label);
  section.append(
    node(
      "p",
      "comparison-control-note",
      "Raw step values are shown as recorded. A missing metric remains Not recorded.",
    ),
  );
  return section;
}

function renderComparisonRunSummary(response) {
  const section = node("section", "comparison-summary");
  section.append(node("h3", "section-title", "Run summary"));
  const table = node("table", "comparison-summary-table");
  const caption = node("caption", "visually-hidden", "Selected Train Run summary");
  table.append(caption);
  const head = node("thead");
  const headerRow = node("tr");
  for (const label of ["Run", "Status", "Seed", "Points", "Last value"]) {
    const cell = node("th", "", label);
    cell.scope = "col";
    headerRow.append(cell);
  }
  head.append(headerRow);
  table.append(head);
  const body = node("tbody");
  for (const run of arrayValue(response?.runs)) {
    const row = node("tr");
    const runCell = node("th", "comparison-summary-run", comparisonRunLabel(run));
    runCell.scope = "row";
    row.append(runCell);
    row.append(node("td", "", comparisonStatus(run)));
    row.append(node("td", "", run.seed === undefined ? "Not recorded" : run.seed));
    const points = firstPresent(run.points_returned, run.series.length ? run.series.length : null);
    row.append(node("td", "", points === null || run.availability === "unavailable" ? "Not recorded" : points));
    row.append(node("td", "", comparisonValueText(firstPresent(run.last_value, comparisonMetricLastValue(run, response.metric)))));
    body.append(row);
  }
  table.append(body);
  const tableScroll = node("div", "comparison-table-scroll");
  tableScroll.append(table);
  section.append(tableScroll);
  return section;
}

function renderMetricChart(response) {
  const section = node("section", "metric-chart-panel");
  const header = node("div", "section-header");
  const metricLabel = response.metric || "Training metric";
  const hasPoints = arrayValue(response.runs).some((run) => run.series.length > 0);
  header.append(node("h3", "section-title", metricLabel));
  const stateText = response.unavailable
    ? "Unavailable"
    : !hasPoints
      ? "Not recorded"
      : response.partial
        ? "Partial data"
        : "Recorded data";
  header.append(statusPill(stateText, response.unavailable ? "failed" : response.partial ? "partial" : "done"));
  section.append(header);
  if (response.message) section.append(node("p", "comparison-result-note", response.message));
  if (response.unavailable) {
    section.append(node("div", "empty-card", "Metric comparison is unavailable for these Runs."));
    return section;
  }

  const runs = arrayValue(response.runs);
  const plottedRuns = runs.filter((run) => run.series.length > 0);
  if (plottedRuns.length === 0) {
    const disabled = runs.filter(
      (run) => run.availability === "tracking_disabled",
    ).length;
    const message =
      disabled === runs.length
        ? "Local metric tracking was disabled for the selected Runs."
        : disabled > 0
          ? "Some selected Runs disabled local tracking; the metric was not recorded for the others."
          : "This metric was not recorded for the selected Runs.";
    section.append(node("div", "empty-card", message));
    return section;
  }
  const allPoints = plottedRuns.flatMap((run) => run.series);
  const xValues = allPoints.map((point) => point.step);
  const yValues = allPoints.map((point) => point.value);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const dataMinY = Math.min(...yValues);
  const dataMaxY = Math.max(...yValues);
  const constantPadding = dataMinY === dataMaxY ? Math.max(Math.abs(dataMaxY) * 0.05, 0.5) : 0;
  const minY = dataMinY - constantPadding;
  const maxY = dataMaxY + constantPadding;
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || Math.max(Math.abs(maxY), 1);
  const width = 760;
  const height = 340;
  const pad = { top: 24, right: 24, bottom: 44, left: 56 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const xScale = (value) => pad.left + ((value - minX) / xSpan) * plotWidth;
  const yScale = (value) => pad.top + (1 - (value - minY) / ySpan) * plotHeight;
  const titleId = "comparison-chart-title";
  const descriptionId = "comparison-chart-description";
  const svg = svgElement("svg", {
    class: "metric-chart",
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-labelledby": `${titleId} ${descriptionId}`,
  });
  const title = svgElement("title", { id: titleId });
  title.textContent = `${metricLabel} by step`;
  const description = svgElement("desc", { id: descriptionId });
  description.textContent = `${plottedRuns.length} Train Run series over raw recorded steps.`;
  svg.append(title, description);
  for (let tick = 0; tick <= 4; tick += 1) {
    const fraction = tick / 4;
    const y = pad.top + fraction * plotHeight;
    const value = maxY - fraction * ySpan;
    const line = svgElement("line", {
      class: "chart-grid-line",
      x1: pad.left,
      x2: width - pad.right,
      y1: y,
      y2: y,
    });
    svg.append(line);
    const label = svgElement("text", { class: "chart-axis-label", x: pad.left - 8, y: y + 4, "text-anchor": "end" });
    label.textContent = Number(value.toPrecision(5)).toString();
    svg.append(label);
  }
  for (let tick = 0; tick <= 4; tick += 1) {
    const fraction = tick / 4;
    const x = pad.left + fraction * plotWidth;
    const value = minX + fraction * xSpan;
    const line = svgElement("line", {
      class: "chart-grid-line chart-grid-vertical",
      x1: x,
      x2: x,
      y1: pad.top,
      y2: height - pad.bottom,
    });
    svg.append(line);
    const label = svgElement("text", { class: "chart-axis-label", x, y: height - pad.bottom + 22, "text-anchor": "middle" });
    label.textContent = Number(value.toPrecision(5)).toString();
    svg.append(label);
  }
  const axisX = svgElement("line", { class: "chart-axis", x1: pad.left, x2: width - pad.right, y1: height - pad.bottom, y2: height - pad.bottom });
  const axisY = svgElement("line", { class: "chart-axis", x1: pad.left, x2: pad.left, y1: pad.top, y2: height - pad.bottom });
  svg.append(axisX, axisY);
  const xLabel = svgElement("text", { class: "chart-axis-name", x: pad.left + plotWidth / 2, y: height - 7, "text-anchor": "middle" });
  xLabel.textContent = "step";
  const yLabel = svgElement("text", { class: "chart-axis-name", x: 14, y: pad.top + plotHeight / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${pad.top + plotHeight / 2})` });
  yLabel.textContent = metricLabel;
  svg.append(xLabel, yLabel);
  for (const [index, run] of runs.entries()) {
    if (!run.series.length) continue;
    const path = svgElement("path", {
      class: "metric-series",
      d: run.series.map((point, pointIndex) => `${pointIndex === 0 ? "M" : "L"} ${xScale(point.step).toFixed(2)} ${yScale(point.value).toFixed(2)}`).join(" "),
      stroke: COMPARISON_COLORS[index % COMPARISON_COLORS.length],
    });
    path.setAttribute("aria-label", comparisonRunLabel(run));
    svg.append(path);
  }
  const chartScroll = node("div", "metric-chart-scroll");
  chartScroll.append(svg);
  section.append(chartScroll);
  const legend = node("ul", "metric-legend");
  for (const [index, run] of runs.entries()) {
    const item = node("li", "metric-legend-item");
    const swatch = node("span", "metric-legend-swatch");
    swatch.style.backgroundColor = COMPARISON_COLORS[index % COMPARISON_COLORS.length];
    swatch.setAttribute("aria-hidden", "true");
    item.append(swatch, node("span", "metric-legend-label", comparisonRunLabel(run)));
    item.append(node("span", "metric-legend-value", run.series.length ? comparisonValueText(run.series.at(-1).value) : "Not recorded"));
    legend.append(item);
  }
  section.append(legend);
  return section;
}

function renderSnapshotDifference(response) {
  const section = node("section", "snapshot-differences");
  section.append(node("h3", "section-title", "Captured input differences"));
  const runs = arrayValue(response?.runs);
  const explicitRows = arrayValue(response?.inputDifferences);
  if (explicitRows.length === 0) {
    section.append(
      node(
        "div",
        "empty-card",
        "No captured Code or Options differences were recorded.",
      ),
    );
    return section;
  }
  for (const kind of ["code", "options"]) {
    const group = node("section", "snapshot-difference-group");
    group.append(node("h4", "snapshot-difference-title", humanizeKey(kind)));
    const rows = explicitRows.filter((row) => {
      const path = stringValue(row?.path, "").toLowerCase();
      return path === kind || path.startsWith(`${kind}.`);
    });
    if (rows.length) group.append(renderInputDifferenceTable(rows, runs));
    else group.append(node("div", "empty-card", `No captured ${kind} differences.`));
    section.append(group);
  }
  return section;
}

function evaluationMetrics(value) {
  if (!isObject(value)) return [];
  const values = firstPresent(value.values, value.metrics, value.metric_values, value.summary);
  if (isObject(values)) {
    return Object.entries(values)
      .filter(([key]) => comparisonCleanKey(key))
      .map(([key, entry]) => [key, isObject(entry) ? firstPresent(entry.value, entry.last_value) : entry]);
  }
  if (isObject(value.primary)) {
    return [[stringValue(value.primary.name, "primary"), value.primary.value]];
  }
  return [];
}

function derivedEvaluationGroups(runs) {
  const groups = new Map();
  for (const run of runs) {
    for (const evaluation of arrayValue(run.evaluations)) {
      const name = stringValue(firstPresent(evaluation.case, evaluation.case_name, evaluation.name, evaluation.label), "Evaluation");
      if (!groups.has(name)) groups.set(name, { case: name, results: [] });
      groups.get(name).results.push({ ...evaluation, run: comparisonRunLabel(run) });
    }
  }
  return [...groups.values()];
}

function renderEvaluationGroups(response) {
  const section = node("section", "evaluation-groups");
  section.append(node("h3", "section-title", "Connected Eval groups"));
  const groups = arrayValue(response?.evalGroups).length
    ? response.evalGroups
    : derivedEvaluationGroups(arrayValue(response?.runs));
  if (groups.length === 0) {
    section.append(node("div", "empty-card", "No connected Eval groups are recorded for these Runs."));
    return section;
  }
  for (const rawGroup of groups) {
    const group = isObject(rawGroup) ? rawGroup : {};
    const title = stringValue(
      firstPresent(group.case, group.case_name, group.name, group.label),
      "Evaluation",
    );
    const condition = Number.isInteger(group.condition)
      ? ` · condition ${group.condition}`
      : "";
    const article = node("article", "evaluation-group");
    article.append(node("h4", "evaluation-group-title", `${title}${condition}`));
    const results = comparisonCollection(firstPresent(group.results, group.runs, group.evaluations, group.values), "run");
    if (results.length === 0) {
      article.append(node("p", "comparison-result-note", "Evaluation result is unavailable."));
      section.append(article);
      continue;
    }
    const list = node("div", "evaluation-result-list");
    for (const rawResult of results) {
      const result = isObject(rawResult) ? rawResult : {};
      const resultRow = node("article", "evaluation-result");
      const trainRun = stringValue(result.train_run_id, "Train Run");
      const evalRun = stringValue(result.eval_run_id, "Eval Run");
      const resultLabel = result.variant
        ? `${result.variant} · ${trainRun} / ${evalRun}`
        : stringValue(firstPresent(result.run, result.run_name), `${trainRun} / ${evalRun}`);
      resultRow.append(node("strong", "evaluation-run", resultLabel));
      resultRow.append(statusPill(statusLabel(result.status), result.status || "unknown"));
      const metrics = evaluationMetrics(result);
      if (metrics.length === 0) {
        resultRow.append(node("span", "evaluation-metric", "Not recorded"));
      } else {
        const metricList = node("span", "evaluation-metric");
        metricList.append(...metrics.map(([name, value]) => node("span", "evaluation-metric-item", `${humanizeKey(name)}: ${comparisonValueText(value)}`)));
        resultRow.append(metricList);
      }
      list.append(resultRow);
    }
    article.append(list);
    section.append(article);
  }
  return section;
}

function renderComparison(data) {
  const section = node("section", "comparison-panel");
  const titleHeader = node("div", "panel-header");
  const title = node("div", "panel-title");
  title.append(node("span", "eyebrow", "Experiment"), node("h2", "section-title", `${data.experiment.name} comparison`));
  titleHeader.append(title, statusPill("Read only", "clean"));
  section.append(titleHeader);
  section.append(node("p", "comparison-intro", "Select Train Runs across Variants to compare recorded metrics and their connected evaluation results."));
  section.append(renderComparisonRunPicker(data));
  const records = selectedComparisonRecords();
  section.append(comparisonControl(state.comparison.response, records));
  if (state.comparison.loading) {
    section.append(node("div", "empty-card comparison-loading", "Loading comparison data…"));
  }
  if (state.comparison.error) {
    const unavailable = node("div", "empty-card comparison-error", state.comparison.error);
    unavailable.setAttribute("role", "alert");
    section.append(unavailable);
  }
  if (records.length < 2) {
    section.append(node("div", "empty-card", comparisonSelectionNote(records.length)));
    return section;
  }
  if (!state.comparison.response) {
    section.append(node("div", "empty-card", "Choose a metric to load the selected Run comparison."));
    return section;
  }
  section.append(renderMetricChart(state.comparison.response));
  section.append(renderComparisonRunSummary(state.comparison.response));
  section.append(renderSnapshotDifference(state.comparison.response));
  section.append(renderEvaluationGroups(state.comparison.response));
  return section;
}

function ensureComparisonSelection() {
  if (!state.document) return [];
  const records = comparisonRunRecords(state.document);
  const available = new Set(records.map((record) => record.key));
  const selected = state.comparison.selectedRuns.filter((key) => available.has(key));
  if (selected.length === 0) {
    selected.push(...records.slice(0, 2).map((record) => record.key));
  }
  state.comparison.selectedRuns = selected.slice(0, 4);
  const selectedRecords = selectedComparisonRecords();
  const availableMetrics = availableComparisonMetrics(selectedRecords);
  if (!state.comparison.metric || !availableMetrics.includes(state.comparison.metric)) {
    state.comparison.metric = availableMetrics[0] || null;
  }
  return selectedRecords;
}

function selectView(view) {
  if (!state.document) return;
  state.runDetail.request += 1;
  state.runDetail.response = null;
  state.view = ["compare", "changes"].includes(view) ? view : "overview";
  if (state.view === "compare") ensureComparisonSelection();
  render();
  if (state.view === "compare") loadComparison();
  if (state.view === "changes") loadAuthoring();
}

async function loadComparison() {
  if (state.view !== "compare" || !state.document) return;
  const records = selectedComparisonRecords();
  if (records.length < 2 || records.length > 4) {
    state.comparison.loading = false;
    state.comparison.response = null;
    return;
  }
  const requestId = state.comparison.request + 1;
  state.comparison.request = requestId;
  state.comparison.loading = true;
  state.comparison.error = null;
  render();
  const params = new URLSearchParams();
  for (const record of records) params.append("run", record.address);
  const knownMetrics = new Set([
    ...state.comparison.availableMetrics,
    ...availableComparisonMetrics(records),
  ]);
  if (knownMetrics.has(state.comparison.metric)) {
    params.set("metric", state.comparison.metric);
  }
  try {
    const response = await fetch(`/api/v1/comparison?${params.toString()}`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    let body = {};
    try {
      body = await response.json();
    } catch (_error) {
      body = {};
    }
    if (!response.ok) {
      throw new Error(body?.error?.message || `comparison request failed (${response.status})`);
    }
    if (requestId !== state.comparison.request) return;
    state.comparison.response = normalizeComparisonResponse(body, records, state.comparison.metric);
    state.comparison.availableMetrics = arrayValue(state.comparison.response.metric_names).filter(
      (name) => typeof name === "string",
    );
    state.comparison.metric = state.comparison.response.metric;
    state.comparison.loading = false;
    state.comparison.error = null;
  } catch (error) {
    if (requestId !== state.comparison.request) return;
    state.comparison.loading = false;
    state.comparison.response = null;
    state.comparison.error = `Could not load comparison data: ${error.message}`;
  }
  if (state.view === "compare") render();
}

function renderDetail(data) {
  if (state.view === "run") {
    elements.detail.replaceChildren(renderRunDetail());
    return;
  }
  if (state.view === "compare") {
    elements.detail.replaceChildren(renderComparison(data));
    return;
  }
  if (state.view === "changes") {
    elements.detail.replaceChildren(renderAuthoring());
    return;
  }
  elements.detail.replaceChildren(
    renderExperimentSummary(data.experiment, data.experiment.variants.length),
  );
  if (!state.selectedVariant) {
    elements.detail.append(node("div", "empty-card", "No Variant identities are available."));
    return;
  }
  const variant = data.experiment.variants.find(
    (item) => item.name === state.selectedVariant,
  );
  if (!variant) {
    elements.detail.append(
      node("div", "empty-card", "The selected Variant is no longer available."),
    );
    return;
  }
  elements.detail.append(renderVariantDetail(variant));
}

function render() {
  if (!state.document) return;
  const documentValue = state.document;
  const data = documentValue;
  const experiment = data.experiment;
  const identities = new Set(experiment.variants.map((variant) => variant.name));
  if (!state.selectedVariant || !identities.has(state.selectedVariant)) {
    state.selectedVariant = experiment.variants[0]?.name || null;
  }
  elements.experimentContext.textContent = `${experiment.name} · ${experiment.type}`;
  elements.experimentName.textContent = experiment.name;
  elements.experimentMeta.textContent = `${experiment.variants.length} ${experiment.variants.length === 1 ? "Variant" : "Variants"} · ${stateLabel(experiment.commitState, "Experiment")}`;
  elements.observedAt.textContent = formatObserved(documentValue.observed_at);
  elements.overviewView.setAttribute("aria-pressed", String(state.view === "overview"));
  elements.compareView.setAttribute("aria-pressed", String(state.view === "compare"));
  elements.changesView.setAttribute("aria-pressed", String(state.view === "changes"));
  renderTree(data);
  renderDetail(data);
}

function clearRenderedState() {
  elements.experimentContext.textContent = "Loading…";
  elements.observedAt.textContent = "";
  elements.experimentName.textContent = "Experiment";
  elements.experimentMeta.textContent = "";
  elements.currentCount.textContent = "0";
  elements.historyCount.textContent = "0";
  elements.currentVariants.replaceChildren();
  elements.historicalVariants.replaceChildren();
  elements.detail.replaceChildren(node("div", "loading", "Loading Experiment state…"));
  elements.overviewView.setAttribute("aria-pressed", "false");
  elements.compareView.setAttribute("aria-pressed", "false");
  elements.changesView.setAttribute("aria-pressed", "false");
}

function setNavigationAvailable(available) {
  for (const button of [elements.overviewView, elements.compareView, elements.changesView]) {
    button.disabled = !available;
  }
}

async function loadOverview() {
  elements.refresh.disabled = true;
  setNavigationAvailable(false);
  elements.error.hidden = true;
  state.authoring.request += 1;
  state.authoring.document = null;
  state.authoring.loading = false;
  state.authoring.error = null;
  state.comparison.request += 1;
  state.comparison.response = null;
  state.comparison.availableMetrics = [];
  state.comparison.loading = false;
  state.comparison.error = null;
  state.comparison.metric = null;
  state.runDetail.request += 1;
  state.runDetail.response = null;
  state.document = null;
  clearRenderedState();
  try {
    const response = await fetch("/api/v1/overview", {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body?.error?.message || `request failed (${response.status})`);
    }
    state.document = validateOverview(body);
    setNavigationAvailable(true);
    ensureComparisonSelection();
    render();
    if (state.view === "compare") loadComparison();
    if (state.view === "changes") loadAuthoring();
    if (state.view === "run") loadRunDetail();
  } catch (error) {
    state.document = null;
    setNavigationAvailable(false);
    state.selectedVariant = null;
    state.comparison.response = null;
    state.comparison.loading = false;
    elements.error.textContent = `Could not read Experiment state: ${error.message}`;
    elements.error.hidden = false;
    elements.detail.replaceChildren(node("div", "loading", "No partial data was displayed."));
  } finally {
    elements.refresh.disabled = false;
  }
}

elements.historyToggle.addEventListener("click", () => {
  const expanded = elements.historyToggle.getAttribute("aria-expanded") === "true";
  elements.historyToggle.setAttribute("aria-expanded", String(!expanded));
  elements.historicalVariants.hidden = expanded;
});
elements.overviewView.addEventListener("click", () => selectView("overview"));
elements.compareView.addEventListener("click", () => selectView("compare"));
elements.changesView.addEventListener("click", () => selectView("changes"));
elements.refresh.addEventListener("click", loadOverview);

loadOverview();
