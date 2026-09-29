/* Shared raw-record chart presentation. Callers own data fetching and configuration. */
(() => {
  "use strict";
  // Keep Y padding independent of nice ticks; both use the same value mapping.
  function chartAxis(values, integer = false, tickCount = 4) {
    let min = Math.min(...values), max = Math.max(...values);
    if (min === max) {
      const padding = integer ? 1 : Math.abs(min) * 0.1 || 1;
      min -= padding; max += padding;
      if (integer) min = Math.max(0, min);
    } else if (!integer) {
      const padding = (max - min) * 0.05;
      min -= padding; max += padding;
    }
    const rough = (max - min) / tickCount;
    const power = 10 ** Math.floor(Math.log10(rough));
    const factor = [1, 2, 2.5, 5, 10].find(value => value * power >= rough);
    const interval = integer ? Math.max(1, Math.ceil(factor * power)) : factor * power;
    const lower = integer ? Math.floor(min / interval) : Math.ceil(min / interval);
    const upper = integer ? Math.ceil(max / interval) : Math.floor(max / interval);
    const ticks = Array.from({ length: upper - lower + 1 }, (_, index) =>
      Number(((lower + index) * interval).toPrecision(12)));
    return { min: integer ? ticks[0] : min, max: integer ? ticks.at(-1) : max, ticks };
  }

  function barAxis(values) {
    const axis = chartAxis([0, ...values]);
    if (values.every(value => value >= 0)) axis.min = 0;
    else if (values.every(value => value <= 0)) axis.max = 0;
    axis.ticks = axis.ticks.filter(value => value >= axis.min && value <= axis.max);
    return axis;
  }

  function tickLabel(value) {
    return value !== 0 && (Math.abs(value) < 0.001 || Math.abs(value) >= 100000)
      ? value.toExponential(2).replace(/\.?0+e/, "e") : String(value);
  }

  function logUnavailable(model, type) {
    if (type === "bar") return "Log10 is unavailable for bars because bars use a zero baseline.";
    if (model.some(run => run.values.some(value => value <= 0))) return "Log10 requires positive values. The displayed data includes zero or negative values.";
    return "";
  }

  function logAxis(values) {
    const logs = values.map(Math.log10);
    let min = Math.min(...logs), max = Math.max(...logs);
    const padding = min === max ? Math.log10(1.1) : (max - min) * 0.05;
    min -= padding; max += padding;
    let ticks = [];
    for (let exponent = Math.floor(min); exponent <= Math.ceil(max); exponent += 1) {
      for (const factor of [1, 2, 5]) {
        const value = factor * 10 ** exponent;
        if (Math.log10(value) >= min && Math.log10(value) <= max) ticks.push(value);
      }
    }
    if (ticks.length > 8) {
      const stride = Math.max(1, Math.ceil((max - min) / 6));
      ticks = [];
      for (let exponent = Math.ceil(min); exponent <= max; exponent += stride) ticks.push(10 ** exponent);
    } else if (ticks.length < 2) {
      const magnitude = Math.max(...values);
      const factor = magnitude > 1e100 || magnitude < 1e-100 ? magnitude : 1;
      ticks = chartAxis(values.map(value => value / factor)).ticks.map(value => value * factor)
        .filter(value => Number.isFinite(value) && value > 0 && Math.log10(value) >= min && Math.log10(value) <= max);
      if (!ticks.length) ticks = [...new Set(values)];
    }
    return { min, max, ticks: [...new Set(ticks.filter(value => Number.isFinite(value) && value > 0).map(value => Number(value.toPrecision(12))))] };
  }

  // Reserve a fixed series slot at every recorded step, including absent samples.
  function barLayout(model) {
    const steps = [...new Set(model.flatMap(run => run.steps))].sort((a, b) => a - b);
    const gap = steps.length > 1 ? Math.min(...steps.slice(1).map((step, index) => step - steps[index])) : 1;
    const min = steps[0] - gap / 2, max = steps.at(-1) + gap / 2;
    return { min, max, gap, minPlotWidth: (max - min) / gap * (model.length * 8 + 8) };
  }

  function barUnavailable(model) {
    if (!model.some(run => run.values.length)) return "";
    return barLayout(model).minPlotWidth > 32768
      ? "Bars are unavailable for this step spacing at a readable width. Use Line or Scatter; recorded steps are unchanged."
      : "";
  }

  function recordedSteps(model) {
    return [...new Set(model.flatMap(run => run.steps))].sort((a, b) => a - b);
  }

  function nearestStepIndex(steps, target) {
    return steps.reduce((nearest, step, index) =>
      Math.abs(step - target) < Math.abs(steps[nearest] - target) ? index : nearest, 0);
  }

  function inspectionAt(model, step) {
    return model.map(run => {
      const index = run.steps.indexOf(step);
      return { id: run.id, value: index < 0 ? null : run.values[index],
        missing: !run.values.length && run.missingLabel ? run.missingLabel : run.sampling === "partial" || run.sampling === "downsampled" ? "Not returned at this step" : "No record at this step" };
    });
  }
  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function svgNode(tag, attributes = {}, text) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
    if (text !== undefined) element.textContent = text;
    return element;
  }
  // Recompute plot geometry in CSS pixels so SVG text retains its semantic size.
  const chartDrawers = new WeakMap();
  // Node imports only pure helpers; browser rendering requires ResizeObserver.
  const chartResize = typeof document === "undefined" ? null : new ResizeObserver(entries => {
    for (const { target, contentRect } of entries) {
      const draw = chartDrawers.get(target);
      if (draw && contentRect.width > 0) draw(contentRect.width);
    }
  });
  function createChart(model, metric, type, scale, options = {}) {
    const height = options.height || 340;
    const idPrefix = options.idPrefix || "compare";
    const selection = options.selection || { highlightedRun: null };
    const labelFor = run => run.label || `${run.variant} · ${run.id}`;
    const highlightId = run => run.highlightId || run.id;
    const section = node("section", "metric-chart-panel chart-preview");
    section.append(node("h3", "section-title", metric));
    const values = model.flatMap(run => run.values);
    // Normalize linear arithmetic before subtracting extremes; labels/readouts retain raw values.
    const magnitude = Math.max(...values.map(Math.abs));
    const valueFactor = magnitude && (magnitude > 1e100 || magnitude < 1e-100) ? magnitude : 1;
    const normalizedValues = values.map(value => value / valueFactor);
    const yAxis = scale === "log" ? logAxis(values) : type === "bar" ? barAxis(normalizedValues) : chartAxis(normalizedValues);
    const rawTick = value => scale === "log" ? value : value * valueFactor;
    yAxis.ticks = yAxis.ticks.filter(value => Number.isFinite(rawTick(value)));
    const steps = recordedSteps(model);
    const maxStep = Math.max(...steps, 1);
    const bars = type === "bar" ? barLayout(model) : null;
    if (!model.some(run => run.values.length && highlightId(run) === selection.highlightedRun)) selection.highlightedRun = null;
    const runCount = new Set(model.map(highlightId)).size;
    const svg = svgNode("svg", { class: "metric-chart", height, role: "img", "aria-label": `${metric}, ${type} chart, ${scale === "log" ? "Log10" : "Linear"} Y scale, by raw step for ${runCount} Train Run${runCount === 1 ? "" : "s"}` });
    const scroll = node("div", "metric-chart-scroll");
    const densityNote = node("p", "authoring-reason", "Scroll horizontally to see all bars. Each bar is one recorded value; no values are combined or averaged.");
    densityNote.hidden = true;
    const hint = node("p", "chart-inspection-hint", "Move or tap to inspect · ← / → recorded steps · Home / End · Esc clears");
    hint.id = `${idPrefix}-inspection-help`;
    scroll.tabIndex = 0;
    scroll.setAttribute("role", "region");
    scroll.setAttribute("aria-label", `${metric} interactive chart`);
    scroll.setAttribute("aria-describedby", hint.id);
    scroll.append(svg);
    section.append(hint, densityNote, scroll);

    const legend = node("ul", "metric-legend");
    const legendButtons = new Map();
    for (const run of model) {
      const item = node("li", "metric-legend-item");
      const button = node("button", "chart-legend-button");
      button.type = "button";
      button.disabled = !run.values.length;
      button.append(node("span", "visually-hidden", "Highlight "));
      const swatch = node("span", "metric-legend-swatch");
      swatch.style.backgroundColor = run.dashed ? "transparent" : run.color;
      swatch.style.color = run.color;
      if (run.dashed) swatch.classList.add("chart-overlay-swatch");
      swatch.setAttribute("aria-hidden", "true");
      const copy = node("span", "chart-legend-copy");
      copy.append(node("span", "metric-legend-label", labelFor(run)),
        node("span", "chart-series-context", run.values.length ? `Last returned ${run.values.at(-1)} · step ${run.steps.at(-1)}` : (run.missingLabel || "Not recorded")),
        node("span", "chart-series-context", `${run.values.length} / ${run.total} returned${run.context ? ` · ${run.context}` : run.sampling === "partial" ? " · partial" : run.sampling === "downsampled" ? " · downsampled" : ""}`));
      button.append(swatch, copy);
      button.addEventListener("click", () => {
        selection.highlightedRun = selection.highlightedRun === highlightId(run) ? null : highlightId(run);
        updateHighlight();
      });
      legendButtons.set(run.id, button);
      item.append(button); legend.append(item);
    }
    section.append(legend);
    const inspector = node("div", "chart-inspector");
    const navigation = node("div", "chart-inspection-actions");
    const previous = node("button", "chart-step-button", "Previous step");
    const next = node("button", "chart-step-button", "Next step");
    const clear = node("button", "chart-step-button", "Clear");
    for (const button of [previous, next, clear]) button.type = "button";
    navigation.append(previous, next, clear);
    const readout = node("div", "chart-readout");
    readout.setAttribute("role", "group");
    readout.setAttribute("aria-labelledby", `${idPrefix}-inspected-step`);
    const announcement = node("p", "visually-hidden");
    announcement.setAttribute("role", "status");
    announcement.setAttribute("aria-live", "polite");
    announcement.setAttribute("aria-atomic", "true");
    const stepTitle = node("p", "chart-readout-title");
    stepTitle.id = `${idPrefix}-inspected-step`;
    const rows = node("dl", "chart-readout-values");
    const cells = new Map();
    for (const run of model) {
      const row = node("div");
      const value = node("dd");
      row.append(node("dt", null, labelFor(run)), value);
      rows.append(row); cells.set(run.id, value);
    }
    readout.append(stepTitle, rows);
    inspector.append(navigation, readout, announcement); section.append(inspector);
    if (options.contextualInspection) {
      section.classList.add("contextual-inspection");
      previous.textContent = "←";
      previous.setAttribute("aria-label", "Previous recorded step");
      previous.title = "Previous recorded step";
      next.textContent = "→";
      next.setAttribute("aria-label", "Next recorded step");
      next.title = "Next recorded step";
      clear.textContent = "Clear selection";
      clear.classList.add("inspection-clear");
      const start = node("button", "chart-step-button inspection-start", "Inspect records");
      start.type = "button";
      start.addEventListener("click", () => { next.click(); clear.focus(); });
      clear.addEventListener("click", () => start.focus());
      navigation.prepend(stepTitle, start);
    }

    let activeIndex = null, geometry = null, overlay = null;

    function updateHighlight() {
      for (const element of svg.querySelectorAll("[data-run-id]")) {
        element.style.opacity = !selection.highlightedRun || element.dataset.runId === selection.highlightedRun ? "1" : "0.25";
      }
      for (const run of model) legendButtons.get(run.id).setAttribute("aria-pressed", String(highlightId(run) === selection.highlightedRun));
    }
    function inspect(index, reveal = false, announce = false) {
      activeIndex = index;
      previous.disabled = index === null || index === 0;
      next.disabled = index === steps.length - 1;
      clear.disabled = index === null;
      overlay?.replaceChildren();
      stepTitle.textContent = index === null ? "Select a recorded step" : `Step ${steps[index]} · returned records`;
      const records = index === null ? null : inspectionAt(model, steps[index]);
      model.forEach((run, runIndex) => {
        const record = records?.[runIndex];
        cells.get(run.id).textContent = record ? record.value === null ? record.missing : String(record.value) : "—";
      });
      if (announce) announcement.textContent = `${stepTitle.textContent}. ${model.map(run => `${labelFor(run)}: ${cells.get(run.id).textContent}`).join(". ")}`;
      if (!geometry || index === null) return;
      const { x, y, pointX, left, right, top, bottom } = geometry;
      const step = steps[index];
      overlay.append(svgNode("line", { class: "chart-crosshair", x1: x(step), x2: x(step), y1: top, y2: bottom }));
      records.forEach((record, runIndex) => {
        if (record.value === null) return;
        const run = model[runIndex];
        overlay.append(svgNode("circle", { class: "chart-inspected-point", cx: pointX(step, runIndex), cy: y(record.value), r: 5, fill: "var(--hkdl-panel)", stroke: run.color, "stroke-width": 2, "data-run-id": highlightId(run) }));
      });
      updateHighlight();
      if (reveal) {
        const position = x(step);
        if (position < scroll.scrollLeft + left || position > scroll.scrollLeft + scroll.clientWidth - 24) {
          scroll.scrollLeft = Math.max(0, Math.min(right, position - scroll.clientWidth / 2));
        }
      }
    }
    function move(delta) {
      inspect(activeIndex === null ? 0 : Math.max(0, Math.min(steps.length - 1, activeIndex + delta)), true, true);
    }
    previous.addEventListener("click", () => move(-1));
    next.addEventListener("click", () => move(1));
    clear.addEventListener("click", () => inspect(null, false, true));
    scroll.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End", "Escape"].includes(event.key)) return;
      event.preventDefault();
      if (event.key === "Escape") inspect(null, false, true);
      else if (event.key === "Home") inspect(0, true, true);
      else if (event.key === "End") inspect(steps.length - 1, true, true);
      else move(event.key === "ArrowLeft" ? -1 : 1);
    });
    function inspectPointer(event, announce = false) {
      if (!geometry) return;
      const rect = svg.getBoundingClientRect();
      const pixel = event.clientX - rect.left;
      const { left, right, top, bottom } = geometry;
      if (pixel < left || pixel > right || event.clientY - rect.top < top || event.clientY - rect.top > bottom) return;
      const minStep = bars ? bars.min : 0, endStep = bars ? bars.max : maxStep;
      const step = minStep + (pixel - left) / (right - left) * (endStep - minStep);
      const index = nearestStepIndex(steps, step);
      if (index !== activeIndex || announce) inspect(index, false, announce);
    }
    svg.addEventListener("pointermove", event => { if (event.pointerType !== "touch") inspectPointer(event); });
    // A click is also emitted for a tap, but not for a touch scroll gesture.
    svg.addEventListener("click", event => inspectPointer(event, true));
    inspect(null);

    chartDrawers.set(scroll, availableWidth => {
      const left = Math.max(56, ...yAxis.ticks.map(value => tickLabel(rawTick(value)).length * 7 + 16));
      const width = bars ? Math.max(availableWidth, left + bars.minPlotWidth + 24) : availableWidth;
      const right = width - 24, top = 24, bottom = height - 44;
      const x = step => left + (bars ? (step - bars.min) / (bars.max - bars.min) : step / maxStep) * (right - left);
      const y = value => top + (1 - ((scale === "log" ? Math.log10(value) : value / valueFactor) - yAxis.min) / (yAxis.max - yAxis.min)) * (bottom - top);
      const slot = bars ? Math.min(28, bars.gap / (bars.max - bars.min) * (right - left) * 0.8 / model.length) : 0;
      const pointX = (step, runIndex) => x(step) + (bars ? (runIndex + 0.5 - model.length / 2) * slot : 0);
      geometry = { left, right, top, bottom, x, y, pointX };
      densityNote.hidden = width <= availableWidth;
      svg.style.width = `${width}px`;
      svg.style.height = `${height}px`;
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.replaceChildren();
      for (const value of yAxis.ticks) {
        svg.append(svgNode("line", { class: "chart-grid-line", x1: left, x2: right, y1: y(rawTick(value)), y2: y(rawTick(value)) }));
        svg.append(svgNode("text", { class: "chart-axis-label", x: left - 8, y: y(rawTick(value)) + 4, "text-anchor": "end" }, tickLabel(rawTick(value))));
      }
      const xAxis = chartAxis([0, maxStep], true, bars ? Math.max(4, Math.floor((right - left) / 100)) : 4);
      for (const step of xAxis.ticks.filter(step => step <= (bars ? Math.max(...steps) : maxStep))) {
        svg.append(svgNode("text", { class: "chart-axis-label", x: x(step), y: height - 22, "text-anchor": "middle" }, String(step)));
      }
      svg.append(svgNode("line", { class: "chart-axis", x1: left, x2: right, y1: bottom, y2: bottom }));
      svg.append(svgNode("line", { class: "chart-axis", x1: left, x2: left, y1: top, y2: bottom }));
      svg.append(svgNode("text", { class: "chart-axis-name", x: (left + right) / 2, y: height - 4, "text-anchor": "middle" }, "step"));
      if (bars) svg.append(svgNode("line", { class: "chart-axis", x1: left, x2: right, y1: y(0), y2: y(0) }));
      model.forEach((run, runIndex) => {
        const series = svgNode("g", { "data-run-id": highlightId(run) });
        svg.append(series);
        if (type === "line") {
          const path = run.values.map((value, index) => `${index ? "L" : "M"} ${x(run.steps[index]).toFixed(2)} ${y(value).toFixed(2)}`).join(" ");
          series.append(svgNode("path", { class: "metric-series", d: path, stroke: run.color, "stroke-dasharray": run.dashed ? "6 4" : "none" }));
        }
        run.values.forEach((value, index) => {
          const step = run.steps[index];
          let mark;
          if (bars) {
            mark = svgNode("rect", { class: "metric-bar", x: pointX(step, runIndex) - slot / 2 + 1, y: Math.min(y(value), y(0)), width: slot - 2, height: Math.abs(y(value) - y(0)), fill: run.dashed ? "var(--hkdl-panel)" : run.color, stroke: run.color, "stroke-width": run.dashed ? 1.5 : 0 });
            if (value === 0) series.append(svgNode("line", { class: "metric-zero", x1: pointX(step, runIndex) - slot / 2 + 1, x2: pointX(step, runIndex) + slot / 2 - 1, y1: y(0), y2: y(0), stroke: run.color, "stroke-width": 2 }));
          } else if (type === "scatter" || index === run.values.length - 1) {
            mark = svgNode("circle", { class: "metric-endpoint", cx: x(step), cy: y(value), r: 4, fill: run.dashed ? "var(--hkdl-panel)" : run.color, stroke: run.dashed ? run.color : "var(--hkdl-panel)", "stroke-width": 1.5 });
          }
          if (mark) series.append(mark);
        });
      });
      overlay = svgNode("g", { class: "chart-inspection-overlay", "aria-hidden": "true" });
      svg.append(overlay);
      updateHighlight();
      inspect(activeIndex);
    });
    chartResize.observe(scroll);
    return section;
  }

  function disposeChart(section) {
    const scroll = section.querySelector(".metric-chart-scroll");
    if (scroll) { chartResize.unobserve(scroll); chartDrawers.delete(scroll); }
  }
  const api = { chartAxis, barAxis, tickLabel, logUnavailable, logAxis, barLayout, recordedSteps, nearestStepIndex, inspectionAt, barUnavailable, createChart, disposeChart };
  if (typeof module !== "undefined") module.exports = api;
  else globalThis.MetricChart = api;
})();
