// --- Helper to read UARL parameters ---
function getQueryParam(name) {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get(name);
}

// --- Helper to inject Umami ---
function loadUmami() {
  if (window.__umamiLoaded) return; // prevent duplicate loads
  window.__umamiLoaded = true;

  const s = document.createElement("script");
  s.defer = true;
  s.src = "https://cloud.umami.is/script.js";
  s.setAttribute("data-website-id", "81b626e1-16f6-4963-b3b8-4f9d34cd7fb9");
  document.head.appendChild(s);
}

// --- Lazy activation when tracking=false ---
function enableUmamiOnInteraction() {
  const interactionEvents = ["click", "scroll", "keydown", "mousemove", "touchstart"];

  const triggerLoad = () => {
    loadUmami();
    interactionEvents.forEach(evt => window.removeEventListener(evt, triggerLoad));
  };

  interactionEvents.forEach(evt => {
    window.addEventListener(evt, triggerLoad, { once: true, passive: true });
  });
}

// --- On DOM ready ---
window.addEventListener("DOMContentLoaded", () => {
  // Handle dark mode
  const darkParam = getQueryParam("dark");
  if (darkParam === "1" || darkParam === "true" || darkParam === "yes") {
    document.body.classList.add("dark-mode");
  }

  // Handle tracking
  const trackingParam = getQueryParam("tracking");
  const trackingDisabled = (trackingParam === "0" || trackingParam === "false" || trackingParam === "no");

  if (trackingDisabled) {
    // Only load Umami after user interaction
    enableUmamiOnInteraction();
  } else {
    // Load Umami immediately
    loadUmami();
  }
});

// --- Dark mode toggle button ---
document.getElementById("toggleDarkMode").addEventListener("click", () => {
    document.body.classList.toggle("dark-mode");
    updateAxesAndChart();
});

let model;
const sliders = [];

let parameterDefinitions = [];
const sliderSvgUrls = [];

const fixedYAxisSettings = new Map();

let outputRanges = new Map();
let xValuesByObservable = {};

const pkOutputsByName = new Map();
const selectedRedshift = { value: 0 };

const DEFAULT_MODEL_BUNDLE_URL = "models/default_model.tflite";

let modelBundle = null;

async function loadModelBundleFromUrl(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(
            `Could not load model archive: ${response.status} ${response.statusText}`
        );
    }

    // The .tflite file selected by the iOS app is a ZIP archive.
    const archiveBuffer = await response.arrayBuffer();
    const archive = await JSZip.loadAsync(archiveBuffer);

    // Extract the actual TensorFlow Lite model from the archive.
    const modelFile = archive.file("model.tflite");

    if (!modelFile) {
        throw new Error(
            "Invalid model archive: model.tflite was not found at the archive root."
        );
    }

    const modelBuffer = await modelFile.async("arraybuffer");

    // Load the actual TFLite model into the browser runtime.
    const tfliteModel = await tflite.loadTFLiteModel(modelBuffer);

    return {
        model: tfliteModel,
        archive: archive
    };
}


async function readArchiveText(archive, filename) {
    const file = archive.file(filename);

    if (!file) {
        throw new Error(
            `Model archive is missing required file: ${filename}`
        );
    }

    return (await file.async("string"))
        .replace(/^\uFEFF/, "");
}


async function initializeFixedYAxisSettings() {
    fixedYAxisSettings.clear();

    // Use the model's best-fit values, not the current slider positions.
    const bestFitValues = parameterDefinitions.map(
        parameter => parameter.bestFit
    );

    const inputTensor = tf.tensor([bestFitValues]);
    let outputTensor = null;

    try {
        outputTensor = model.predict(inputTensor);
        const outputArray = await outputTensor.array();
        const allOutputs = outputArray[0];

        for (const [observable, options] of observableOptions) {
            for (const option of options) {
                const key = `${observable}_${option}`;

                // CMB limits are specified explicitly in updateAxes().
                if (observable === "Cl") {
                    continue;
                }

                // Match the iOS fixed log limits of 10^-2 to 10^6.
                // The web plots log10(y), so its coordinate bounds are -2 to 6.
                if (observable === "Pk") {
                    fixedYAxisSettings.set(key, {
                        mode: "log",
                        bounds: [-2, 6]
                    });

                    continue;
                }

                const range = findOutputRange(observable, option);

                if (!range) {
                    continue;
                }

                const values = allOutputs
                    .slice(range.start, range.end)
                    .map(Number)
                    .filter(Number.isFinite);

                // Match the iOS fallback values if a valid output is absent.
                const low = values.length
                    ? Math.min(...values)
                    : 1;

                const high = values.length
                    ? Math.max(...values)
                    : 2;

                const validPositiveRange =
                    low > 0 &&
                    high > low;

                const logYSpan = validPositiveRange
                    ? Math.log10(high) - Math.log10(low)
                    : 0;

                const useLogY =
                    validPositiveRange &&
                    logYSpan >= 1;

                if (useLogY) {
                    const padding = Math.max(
                        logYSpan * 0.05,
                        0.05
                    );

                    fixedYAxisSettings.set(key, {
                        mode: "log",
                        bounds: [
                            Math.log10(low) - padding,
                            Math.log10(high) + padding
                        ]
                    });
                } else {
                    const safeLow = Number.isFinite(low) ? low : 0;
                    const safeHigh = Number.isFinite(high) ? high : 1;

                    const padding = Math.max(
                        Math.max(
                            Math.abs(safeLow),
                            Math.abs(safeHigh)
                        ) * 0.05,
                        1e-10
                    );

                    const span = Math.max(
                        safeHigh - safeLow,
                        padding
                    );

                    fixedYAxisSettings.set(key, {
                        mode: "linear",
                        bounds: [
                            safeLow - 0.1 * span,
                            safeHigh + 0.2 * span
                        ]
                    });
                }
            }
        }
    } finally {
        inputTensor.dispose();

        if (outputTensor) {
            outputTensor.dispose();
        }
    }
}


async function readParameterDefinitions(archive) {
    const [inputText, bestFitText] = await Promise.all([
        readArchiveText(archive, "input_names.txt"),
        readArchiveText(archive, "best_fit.txt")
    ]);

    // Parse best-fit values by parameter name instead of assuming
    // that their order matches the input metadata.
    const bestFitByName = new Map();

    for (const line of bestFitText.split(/\r?\n/)) {
        if (!line.trim()) continue;

        const parts = line.split(",").map(part => part.trim());
        if (parts.length < 2) continue;

        const value = Number(parts[1]);

        if (parts[0] && Number.isFinite(value)) {
            bestFitByName.set(parts[0], value);
        }
    }

    const definitions = [];

    for (const line of inputText.split(/\r?\n/)) {
        if (!line.trim()) continue;

        // Format: name, xmin, xmax, step
        const parts = line.split(",").map(part => part.trim());

        if (parts.length < 4) {
            throw new Error(
                `Invalid input_names.txt entry: ${line}`
            );
        }

        const [name, minText, maxText, stepText] = parts;
        const min = Number(minText);
        const max = Number(maxText);
        const step = Number(stepText);
        const bestFit = bestFitByName.get(name);

        if (
            !name ||
            !Number.isFinite(min) ||
            !Number.isFinite(max) ||
            !Number.isFinite(step) ||
            min >= max ||
            step <= 0
        ) {
            throw new Error(
                `Invalid slider configuration for parameter "${name}".`
            );
        }

        if (!Number.isFinite(bestFit)) {
            throw new Error(
                `No best-fit value found for parameter "${name}" in best_fit.txt.`
            );
        }

        if (bestFit < min || bestFit > max) {
            console.warn(
                `Best-fit value for "${name}" is outside its slider range.`,
                { bestFit, min, max }
            );
        }

        definitions.push({
            name,
            min,
            max,
            step,
            bestFit: Math.min(max, Math.max(min, bestFit))
        });
    }

    if (definitions.length === 0) {
        throw new Error("No parameter definitions were found.");
    }

    return definitions;
}


function decimalPlacesForStep(step) {
    const text = Number(step).toString().toLowerCase();
    const [mantissa, exponentText] = text.split("e");
    const decimalPlaces = (mantissa.split(".")[1] || "").length;
    const exponent = exponentText ? Number(exponentText) : 0;

    return Math.max(0, decimalPlaces - exponent);
}


function formatParameterValue(value, step) {
    return Number(value).toFixed(decimalPlacesForStep(step));
}


function updateSliderFill(slider) {
    const min = Number(slider.min);
    const max = Number(slider.max);
    const value = Number(slider.value);

    const percentage = max > min
        ? 100 * (value - min) / (max - min)
        : 0;

    slider.style.backgroundSize =
        `${Math.max(0, Math.min(100, percentage))}% 100%`;
}


async function createParameterSliders(definitions, archive) {
    const container = document.getElementById("sliders");

    if (!container) {
        throw new Error('Could not find the "#sliders" container.');
    }

    // Clean up old labels if a different model is loaded later.
    sliderSvgUrls.forEach(url => URL.revokeObjectURL(url));
    sliderSvgUrls.length = 0;
    sliders.length = 0;

    container.replaceChildren();
    parameterDefinitions = definitions;

    for (const [index, parameter] of definitions.entries()) {
        const number = index + 1;

        const wrapper = document.createElement("div");
        wrapper.className = "slider-container";

        const label = document.createElement("label");
        label.id = `slider${number}-label`;
        label.htmlFor = `slider${number}`;
        label.className = "slider-label";

        // Use the SVG label bundled with the model, when available.
        const svgFile = archive.file(`${parameter.name}.svg`);

        if (svgFile) {
            const svgText = await svgFile.async("string");
            const blob = new Blob([svgText], {
                type: "image/svg+xml"
            });
            const svgUrl = URL.createObjectURL(blob);

            sliderSvgUrls.push(svgUrl);

            const image = document.createElement("img");
            image.src = svgUrl;
            image.alt = parameter.name;
            image.className = "svgcolor";
            image.draggable = false;

            label.appendChild(image);
        } else {
            // A missing optional SVG should not prevent the slider working.
            label.textContent = parameter.name;
        }

        const valueLabel = document.createElement("span");
        valueLabel.id = `slider${number}-value`;
        valueLabel.className = "slider-value";

        const slider = document.createElement("input");
        slider.type = "range";
        slider.id = `slider${number}`;
        slider.className = "slider";
        slider.min = parameter.min;
        slider.max = parameter.max;
        slider.step = parameter.step;
        slider.value = parameter.bestFit;

        valueLabel.textContent = formatParameterValue(
            slider.value,
            parameter.step
        );

        slider.addEventListener("input", () => {
            updateSliderFill(slider);

            valueLabel.textContent = formatParameterValue(
                slider.value,
                parameter.step
            );

            updateDataAndChart();
        });

        updateSliderFill(slider);

        wrapper.append(label, valueLabel, slider);
        container.appendChild(wrapper);
        sliders.push(slider);
    }
}

function createRedshiftSlider() {
    const container = document.getElementById("sliders");

    if (!container) {
        throw new Error('Could not find the "#sliders" container.');
    }

    // Avoid creating duplicates if initialization runs again.
    if (document.getElementById("redshiftSliderContainer")) {
        return;
    }

    const wrapper = document.createElement("div");
    wrapper.id = "redshiftSliderContainer";
    wrapper.className = "slider-container redshift-slider-container";
    wrapper.hidden = true;

    const label = document.createElement("label");
    label.className = "slider-label";
    label.htmlFor = "redshiftSlider";

    const image = document.createElement("img");
    image.src = "./svg_images/z_slider.svg";
    image.alt = "Redshift z";
    image.className = "svgcolor";
    image.draggable = false;

    label.appendChild(image);

    const valueLabel = document.createElement("span");
    valueLabel.id = "redshiftSliderValue";
    valueLabel.className = "slider-value";
    valueLabel.textContent = "0.0";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.id = "redshiftSlider";
    slider.className = "slider";
    slider.min = "0";
    slider.max = "0";
    slider.step = "0.1";
    slider.value = "0";

    slider.addEventListener("input", () => {
        selectedRedshift.value = Number(slider.value);
        valueLabel.textContent = selectedRedshift.value.toFixed(1);

        updateSliderFill(slider);
        updateDataAndChart();
    });

    wrapper.append(label, valueLabel, slider);

    // Place redshift below the cosmological parameter sliders.
    // Crucially, do not add it to the global `sliders` array.
    container.append(wrapper);

    updateSliderFill(slider);
}

function updateRedshiftSliderForSelection() {
    const wrapper = document.getElementById(
        "redshiftSliderContainer"
    );
    const slider = document.getElementById("redshiftSlider");
    const valueLabel = document.getElementById(
        "redshiftSliderValue"
    );

    if (!wrapper || !slider || !valueLabel) {
        return;
    }

    if (selectedObservable.value !== "Pk") {
        wrapper.hidden = true;
        return;
    }

    const family = pkOutputsByName.get(
        String(selectedOption.value).toLowerCase()
    );

    if (!family || family.redshifts.length === 0) {
        wrapper.hidden = true;
        return;
    }

    const redshifts = family.redshifts;
    const minZ = redshifts[0];
    const maxZ = redshifts[redshifts.length - 1];

    slider.min = String(minZ);
    slider.max = String(maxZ);
    slider.step = "0.1";

    // Preserve the previously selected redshift if this family
    // supports it; otherwise clamp it to the family range.
    let z = Number(selectedRedshift.value);

    if (!Number.isFinite(z)) {
        z = minZ;
    }

    z = Math.max(minZ, Math.min(maxZ, z));

    // Snap to the 0.1 redshift resolution, relative to the minimum.
    z = minZ + Math.round((z - minZ) / 0.1) * 0.1;
    z = Math.max(minZ, Math.min(maxZ, z));

    selectedRedshift.value = Number(z.toFixed(10));
    slider.value = String(selectedRedshift.value);
    valueLabel.textContent = selectedRedshift.value.toFixed(1);

    updateSliderFill(slider);
    wrapper.hidden = false;
}

async function readOutputRanges(archive) {
    const text = await readArchiveText(archive, "output_indices.txt");
    const ranges = new Map();

    for (const line of text.split(/\r?\n/)) {
        if (!line.trim()) continue;

        const parts = line.split(",").map(part => part.trim());

        if (parts.length < 3) continue;

        const [name, startText, endText] = parts;
        const start = Number(startText);
        const end = Number(endText);

        if (
            name &&
            Number.isInteger(start) &&
            Number.isInteger(end) &&
            start >= 0 &&
            end > start
        ) {
            ranges.set(name, { start, end });
        }
    }

    if (ranges.size === 0) {
        throw new Error("No valid output ranges found in output_indices.txt.");
    }

    return ranges;
}


async function readXValues(archive) {
    const text = await readArchiveText(archive, "x_values.txt");
    const result = {};
    let currentKey = null;
    let currentValues = [];

    for (const rawLine of text.split(/\r?\n/)) {
        const line = rawLine.trim();

        if (!line) continue;

        const value = Number(line);

        if (Number.isFinite(value)) {
            currentValues.push(value);
        } else {
            if (currentKey !== null) {
                result[currentKey] = currentValues;
            }

            currentKey = line.replace(/:/g, "").trim();
            currentValues = [];
        }
    }

    if (currentKey !== null) {
        result[currentKey] = currentValues;
    }

    return result;
}

function getCurrentHForPkUnits() {
    const index = parameterDefinitions.findIndex(parameter => {
        const name = String(parameter.name)
            .toLowerCase()
            .replace(/[^a-z0-9]/g, "");

        return (
            name === "h" ||
            name === "h0" ||
            name === "hubble" ||
            name === "hubbleconstant"
        );
    });

    if (index !== -1) {
        const parameter = parameterDefinitions[index];
        const name = String(parameter.name)
            .toLowerCase()
            .replace(/[^a-z0-9]/g, "");

        const rawValue = Number(
            sliders[index]?.value ?? parameter.bestFit
        );

        if (Number.isFinite(rawValue) && rawValue > 0) {
            // A parameter called h is already dimensionless.
            // H0 is normally supplied in km/s/Mpc.
            const h = name === "h"
                ? rawValue
                : rawValue / 100;

            if (Number.isFinite(h) && h > 0) {
                return h;
            }
        }
    }

    // 100*theta_s alone does not determine H0 uniquely.
    // Use the declared fiducial fallback if there is no H0/h input.
    if (!warnedAboutPkHFallback) {
        console.warn(
            `No H0/h input slider found. Using fixed h = ` +
            `${FIXED_H_FOR_PK_UNITS} for Pk unit conversion.`
        );

        warnedAboutPkHFallback = true;
    }

    return FIXED_H_FOR_PK_UNITS;
}


function findOutputRange(observable, option) {
    const requestedName = `${observable}_${option}`.toLowerCase();

    for (const [name, range] of outputRanges) {
        if (name.toLowerCase() === requestedName) {
            return range;
        }
    }

    return null;
}


function loadData(name) {
    const request = new XMLHttpRequest();
    request.open('GET', name, false); // false makes the request synchronous
    request.send(null);

    if (request.status === 200) {
        return JSON.parse(request.responseText);
    } else {
        console.error('Failed to load data.json');
        return null;
    }
}

const tt_data = loadData('tt_data.json');
const te_data = loadData('te_data.json');
const ee_data = loadData('ee_data.json');
const pp_data = loadData('pp_data.json');
const pkCompilationData = loadData("pk_data.json");

const PK_DATA_COLORS = [
    "#32CD32", // Lime green
    "#FAC811", // Orange
    "#FC656C", // Red
    "#05F1FA", // Cyan
    "#F520EE", // Magenta
    "#9B71C7",//#A800E0"  // Purple
];

const CMB_OBSERVED_COLOR = "limeGreen";
const FIXED_H_FOR_PK_UNITS = 0.677;
let warnedAboutPkHFallback = false;

// Shared chart responsiveness and observational-data styling.
const COMPACT_CHART_MAX_WIDTH = 500;
const COMPACT_CHART_MAX_HEIGHT = 400;
const OBSERVATION_MARKER_RADIUS = 4;
const COMPACT_OBSERVATION_MARKER_RADIUS = 2.5;
const OBSERVATION_ERRORBAR_LINE_WIDTH = 2;
const OBSERVATION_ERRORBAR_OPACITY = 0.6;

function isCompactChart() {
    const chartElement = document.getElementById("outputChart");
    return Boolean(chartElement && (
        chartElement.offsetWidth <= COMPACT_CHART_MAX_WIDTH ||
        chartElement.offsetHeight < COMPACT_CHART_MAX_HEIGHT
    ));
}

function getObservationMarkerRadius() {
    return isCompactChart()
        ? COMPACT_OBSERVATION_MARKER_RADIUS
        : OBSERVATION_MARKER_RADIUS;
}

var data = null;
var checkboxChecked = false;



function resetSlidersToBestfit() {
    sliders.forEach((slider, index) => {
        const parameter = parameterDefinitions[index];

        slider.value = parameter.bestFit;
        updateSliderFill(slider);

        const valueLabel = document.getElementById(
            `slider${index + 1}-value`
        );

        if (valueLabel) {
            valueLabel.textContent = formatParameterValue(
                slider.value,
                parameter.step
            );
        }
    });
}


function resetGraphToBestfit() {
    resetSlidersToBestfit();
    updateDataAndChart();
}


const selectedOption = { value: "TT" };

const selectedObservable = { value: "Cl" };

const observableOptions = new Map();
const observableOrder = [];

const observableDisplayNames = {
    Cl: "CMB power spectra",
    Pk: "Matter power spectra",
    bg: "Background",
    th: "Thermodynamics"
};

const xValueKeyByObservable = {
    Cl: "Cl",
    Pk: "Pk",
    bg: "Background",
    th: "Thermodynamics"
};



function buildObservableOptions() {
    observableOptions.clear();
    observableOrder.length = 0;
    pkOutputsByName.clear();

    for (const outputName of outputRanges.keys()) {
        // Derived outputs are not selectable.
        if (/^derived/i.test(outputName)) continue;

        const range = outputRanges.get(outputName);

	// Output metadata may prefix the model's Pk output name with "Pk_".
	// For example, Pk_pk_cb_20.6 becomes pk_cb_20.6 before parsing.
	const pkOutputName = outputName.replace(/^Pk_/i, "");

	const pkMatch = pkOutputName.match(
	    /^(pk(?:_.*)?)_(-?\d+(?:\.\d+)?)$/i
	);

        if (pkMatch) {
            const pkName = pkMatch[1];
            const redshift = Number(pkMatch[2]);
            const xdata = xValuesByObservable.Pk;

            // Only expose spectra with a matching usable k-grid.
            if (
                !range ||
                !Number.isFinite(redshift) ||
                !Array.isArray(xdata) ||
                xdata.length < 3 ||
                xdata.length !== range.end - range.start
            ) {
                continue;
            }

            const familyKey = pkName.toLowerCase();

            if (!pkOutputsByName.has(familyKey)) {
                pkOutputsByName.set(familyKey, {
                    name: pkName,
                    rangesByRedshift: new Map()
                });
            }

            const family = pkOutputsByName.get(familyKey);
            family.rangesByRedshift.set(redshift, range);

            // Add Pk to the observable dropdown just once.
            if (!observableOptions.has("Pk")) {
                observableOptions.set("Pk", []);
                observableOrder.push("Pk");
            }

            const pkOptions = observableOptions.get("Pk");

            if (
                !pkOptions.some(
                    name => name.toLowerCase() === familyKey
                )
            ) {
                pkOptions.push(pkName);
            }

            // Do not process this output as a regular observable.
            continue;
        }

        const separator = outputName.indexOf("_");
        if (separator <= 0) continue;

        const rawObservable = outputName.slice(0, separator);
        const rawOption = outputName.slice(separator + 1);

        if (!rawOption) continue;

        // Pk outputs with malformed suffixes should not become
        // ordinary Pk options.
        if (/^pk$/i.test(rawObservable)) continue;

        const observableAliases = {
            cl: "Cl",
            bg: "bg",
            th: "th",
            extra: "extra"
        };

        const observable =
            observableAliases[rawObservable.toLowerCase()] ||
            rawObservable;

        const xKey =
            xValueKeyByObservable[observable] || observable;
        const xdata = xValuesByObservable[xKey];

        if (
            !range ||
            !Array.isArray(xdata) ||
            xdata.length < 3 ||
            xdata.length !== range.end - range.start
        ) {
            continue;
        }

        const option = observable === "Cl"
            ? rawOption.toUpperCase()
            : rawOption;

        if (!observableOptions.has(observable)) {
            observableOptions.set(observable, []);
            observableOrder.push(observable);
        }

        const options = observableOptions.get(observable);

        if (!options.includes(option)) {
            options.push(option);
        }
    }

    // Keep the observable order stable regardless of archive order.
    const preferredOrder = ["Cl", "Pk", "bg", "th", "extra"];
    const rank = new Map(
        preferredOrder.map((name, index) => [name, index])
    );

    observableOrder.sort(
        (a, b) =>
            (rank.get(a) ?? Infinity) -
            (rank.get(b) ?? Infinity)
    );

    // Keep each family's redshifts numerically ordered.
    for (const family of pkOutputsByName.values()) {
        family.redshifts = [
            ...family.rangesByRedshift.keys()
        ].sort((a, b) => a - b);
    }
}



function formatOptionLabel(observable, option) {
    if (observable === "Cl" && option === "PP") {
        return "φφ";
    }

    if (observable === "Pk") {
        const parts = String(option).split("_");

        // Remove the base "pk" token.
        if (parts[0]?.toLowerCase() === "pk") {
            parts.shift();
        }

        const isLinear = parts.some(
            part => part.toLowerCase() === "lin"
        );

        // "lin" determines the spectrum type, so don't repeat
        // it inside the parentheses.
        const modifiers = parts.filter(
            part => part.toLowerCase() !== "lin"
        );

        const label = isLinear ? "Linear" : "Non-linear";

        return modifiers.length
            ? `${label} (${modifiers.join(", ")})`
            : label;
    }

    return option;
}


function populateOutputSelector() {
    const select = document.getElementById("dropdown");
    const options = observableOptions.get(
        selectedObservable.value
    ) || [];

    select.replaceChildren();

    for (const option of options) {
        const element = document.createElement("option");
        element.value = option;
        element.textContent = formatOptionLabel(
            selectedObservable.value,
            option
        );
        select.appendChild(element);
    }

    if (!options.includes(selectedOption.value)) {
        selectedOption.value = options[0] || "";
    }

    select.value = selectedOption.value;
}


function populateObservableSelector() {
    const select = document.getElementById("observableDropdown");
    select.replaceChildren();

    for (const observable of observableOrder) {
        const element = document.createElement("option");
        element.value = observable;
        element.textContent =
            observableDisplayNames[observable] || observable;

        select.appendChild(element);
    }

    // Keep the current observable if possible; otherwise choose
    // the first available one in model output order.
    if (!observableOptions.has(selectedObservable.value)) {
        selectedObservable.value = observableOrder[0] || "";
    }

    select.value = selectedObservable.value;
    populateOutputSelector();
}


function handleSelection(event) {
    selectedOption.value = event.target.value;
    updateRedshiftSliderForSelection();
    updateObservedDataSelection();
    updateDataAndChart();
}


function handleObservableSelection(event) {
    selectedObservable.value = event.target.value;

    // Select the first output for the new observable.
    const options = observableOptions.get(
        selectedObservable.value
    ) || [];

    selectedOption.value = options[0] || "";

    populateOutputSelector();
    updateRedshiftSliderForSelection();
    updateObservedDataSelection();
    updateDataAndChart();
}


function updateObservedDataSelection() {
    const toggle = document.getElementById("toggle");
    const observable = selectedObservable.value;

    const supportsData =
        observable === "Cl" ||
        observable === "Pk";

    if (!supportsData) {
        data = null;
        checkboxChecked = false;

        if (toggle) {
            toggle.checked = false;
            toggle.disabled = true;
        }

        return;
    }

    if (toggle) {
        toggle.disabled = false;
    }

    if (!checkboxChecked) {
        data = null;
        return;
    }

    // Pk compilation data are handled separately from the CMB arrays.
    if (observable === "Pk") {
        data = null;
        return;
    }

    // Preserve the existing CMB observational datasets.
    switch (selectedOption.value) {
        case "TT":
            data = tt_data;
            break;
        case "TE":
            data = te_data;
            break;
        case "EE":
            data = ee_data;
            break;
        case "PP":
            data = pp_data;
            break;
        default:
            data = null;
    }
}

function showData(checkboxElem) {
    if (!["Cl", "Pk"].includes(selectedObservable.value)) {
        data = null;
        checkboxChecked = false;
        checkboxElem.checked = false;
        return;
    }

    checkboxChecked = checkboxElem.checked;

    updateObservedDataSelection();
    updateDataAndChart();
}


function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[character]);
}


function axisTitleHTML(
    filename,
    fallbackText,
    width,
    height,
    labelFilter,
    rotation = 0
) {
    const compact = isCompactChart();

    const fallbackFontSize = compact ? "18px" : "26px";
    const transformStyle = rotation === 0
        ? ""
        : `transform:rotate(${rotation}deg);transform-origin:center;`;

    const fallbackHTML = `
        <span
            class="axis-title-fallback"
            style="
                display:${filename ? "none" : "inline-block"};
                font-style:italic;
                font-size:${fallbackFontSize};
                white-space:nowrap;
            "
        >${escapeHTML(fallbackText)}</span>
    `;

    const imageHTML = filename ? `
        <img
            class="axis-title-image"
            src="./svg_images/${escapeHTML(filename)}"
            alt=""
            style="
                display:inline-block;
                width:${typeof width === "number" ? `${width}px` : width};
                height:${typeof height === "number" ? `${height}px` : height};
                object-fit:contain;
                filter:${labelFilter};
            "
        />
    ` : "";

    return `
        <span style="
            display:inline-block;
            white-space:nowrap;
            ${transformStyle}
        ">
            ${imageHTML}
            ${fallbackHTML}
        </span>
    `;
}

function formatAdaptiveTickLabel(value, isLog, html = false) {
    value = Number(value);

    if (!Number.isFinite(value)) {
        return "";
    }

    if (isLog) {
        const exponent = Number(value.toFixed(1)).toString();
        return html
            ? `10<sup>${exponent}</sup>`
            : `10^${exponent}`;
    }

    if (value === 0) {
        return "0";
    }

    const absValue = Math.abs(value);

    if (absValue < 1e-3 || absValue >= 1e4) {
        const [mantissa, exponent] = value.toExponential(2).split("e");
        const exp = Number(exponent);

        return html
            ? `${mantissa}×10<sup>${exp}</sup>`
            : `${mantissa}×10^${exp}`;
    }

    return Number(value.toPrecision(4)).toString();
}


function generateAdaptiveTickPositions(
    bounds,
    isLog,
    axisPixels,
    orientation
) {
    const [min, max] = bounds;
    const span = max - min;

    if (
        !Number.isFinite(min) ||
        !Number.isFinite(max) ||
        span <= 0
    ) {
        return [min, max].filter(Number.isFinite);
    }

    // Estimate the longest label from the axis range.
    const samples = [
        min,
        min + span * 0.25,
        min + span * 0.5,
        min + span * 0.75,
        max
    ];

    const longestLabel = Math.max(
        ...samples.map(value =>
            formatAdaptiveTickLabel(value, isLog).length
        )
    );

    // X labels need horizontal separation. Longer labels mean
    // fewer ticks across the available plot width.
    // Y labels need vertical separation, adjusted for long labels.
    const labelSpacing = orientation === "x"
        ? Math.max(55, longestLabel * 8 + 18)
        : Math.max(100, 48 + longestLabel * 4);
        //: Math.max(34, 26 + Math.max(0, longestLabel - 4) * 3);

    const targetIntervals = Math.max(
        1,
        Math.floor(axisPixels / labelSpacing)
    );

    // Logarithmic coordinates represent powers of ten.
    // Only place ticks at integer exponents.
    if (isLog) {
	const firstExponent = Math.ceil(min - 1e-10);
	const lastExponent = Math.floor(max + 1e-10);

	if (firstExponent > lastExponent) {
            return [];
	}

	const exponentStep = Math.max(
            1,
            Math.ceil(
		(lastExponent - firstExponent) /
		    Math.max(1, targetIntervals)
            )
	);

	const ticks = [];

	for (
            let exponent = firstExponent;
            exponent <= lastExponent;
            exponent += exponentStep
	) {
            ticks.push(exponent);
	}

	return ticks;
    }

    const rawStep = span / targetIntervals;
    const magnitude = Math.pow(
        10,
        Math.floor(Math.log10(rawStep))
    );
    const normalizedStep = rawStep / magnitude;

    // Use familiar 1, 2, 5, 10 intervals at different scales.
    const niceFactor = normalizedStep <= 1
        ? 1
        : normalizedStep <= 2
        ? 2
        : normalizedStep <= 5
        ? 5
        : 10;

    const step = niceFactor * magnitude;
    const first = Math.ceil(min / step - 1e-10) * step;
    const ticks = [];

    for (
        let value = first;
        value <= max + step * 1e-9 && ticks.length < 100;
        value += step
    ) {
        ticks.push(Number(value.toPrecision(12)));
    }

    return ticks.length ? ticks : [min, max];
}


function getHRangeForPkUnits() {
    const parameter = parameterDefinitions.find(parameter => {
        const name = String(parameter.name)
            .toLowerCase()
            .replace(/[^a-z0-9]/g, "");

        return [
            "h",
            "h0",
            "hubble",
            "hubbleconstant"
        ].includes(name);
    });

    if (!parameter) {
        // No H0/h input slider: the existing fixed fallback applies.
        return {
            min: FIXED_H_FOR_PK_UNITS,
            max: FIXED_H_FOR_PK_UNITS
        };
    }

    const name = String(parameter.name)
        .toLowerCase()
        .replace(/[^a-z0-9]/g, "");

    const convertToH = value =>
        name === "h" ? Number(value) : Number(value) / 100;

    const h1 = convertToH(parameter.min);
    const h2 = convertToH(parameter.max);

    if (
        !Number.isFinite(h1) ||
        !Number.isFinite(h2) ||
        h1 <= 0 ||
        h2 <= 0
    ) {
        return {
            min: FIXED_H_FOR_PK_UNITS,
            max: FIXED_H_FOR_PK_UNITS
        };
    }

    return {
        min: Math.min(h1, h2),
        max: Math.max(h1, h2)
    };
}


function getFixedPkXAxisBounds() {
    const kValues = (xValuesByObservable.Pk || [])
        .map(Number)
        .filter(k => Number.isFinite(k) && k > 0);

    if (kValues.length < 2) {
        return null;
    }

    const kMin = Math.min(...kValues);
    const kMax = Math.max(...kValues);
    const { min: hMin, max: hMax } = getHRangeForPkUnits();

    // Intersection of the model's transformed k-ranges across
    // the entire allowed h range. This stays fixed as the slider moves.
    const fixedKMin = kMin / hMin;
    const fixedKMax = kMax / hMax;

    if (
        !Number.isFinite(fixedKMin) ||
        !Number.isFinite(fixedKMax) ||
        fixedKMin <= 0 ||
        fixedKMax <= fixedKMin
    ) {
        return null;
    }

    // The Pk plot uses log10(k) as its chart coordinate.
    return [
        Math.log10(fixedKMin),
        Math.log10(fixedKMax)
    ];
}


function updateOtherObservableAxes() {
    const observable = selectedObservable.value;
    const chartElement = document.getElementById("outputChart");
    const compact = isCompactChart();
    const isDarkMode = document.body.classList.contains("dark-mode");
    const axisColor = isDarkMode ? "white" : "black";

    const labelFilter = isDarkMode
        ? "invert(100%) sepia(100%) saturate(0%) brightness(106%) contrast(106%)"
        : "invert(0%) sepia(97%) saturate(0%) brightness(89%) contrast(103%)";

    const xKey = xValueKeyByObservable[observable] || observable;
    const hForPk = observable === "Pk"
	  ? getCurrentHForPkUnits()
	  : 1;

    const xGrid = (xValuesByObservable[xKey] || [])
	  .map(Number)
	  .map(k => observable === "Pk" ? k / hForPk : k)
	  .filter(value =>
              Number.isFinite(value) &&
		  (currentXAxisMode !== "log" || value > 0)
	  );

    if (xGrid.length < 2) {
	return;
    }

    function transformX(value) {
	return currentXAxisMode === "log"
            ? Math.log10(value)
            : value;
    }

    const rawXMin = Math.min(...xGrid);
    const rawXMax = Math.max(...xGrid);

    const xBounds = observable === "Pk"
	  ? (
              getFixedPkXAxisBounds() || [
		  transformX(rawXMin),
		  transformX(rawXMax)
              ]
	  )
	  : [
              transformX(rawXMin),
              transformX(rawXMax)
	  ];

    if (!xBounds.every(Number.isFinite) || xBounds[1] <= xBounds[0]) {
	console.error(`Invalid x-grid bounds for ${observable}.`);
	return;
    }

    function logTicks(bounds) {
        const [min, max] = bounds;
        const step = max - min > 2 ? 1 : 0.5;
        const first = Math.ceil(min / step) * step;
        const ticks = [];

        for (let value = first; value <= max + 1e-9; value += step) {
            ticks.push(Number(value.toFixed(6)));
        }

        return ticks;
    }

    function tickFormatter(isLog) {
	return function () {
            return formatAdaptiveTickLabel(
		this.value,
		isLog,
		true
            );
	};
    }

    const boundsKey = `${observable}_${selectedOption.value}`;
    const ySettings = fixedYAxisSettings.get(boundsKey);

    if (!ySettings) {
	console.error(`No fixed y-axis settings for ${boundsKey}.`);
	return;
    }

    const yBounds = ySettings.bounds;

    const xIsLog = currentXAxisMode === "log";
    const yIsLog = ySettings.mode === "log";

    const chartWidth = chartElement?.clientWidth || 600;
    const chartHeight = chartElement?.clientHeight || 400;

    const availablePlotWidth = Math.max(
	150,
	chartWidth - 110 - 30
    );

    const availablePlotHeight = Math.max(
	120,
	chartHeight - 80
    );

    const xTickPositions = generateAdaptiveTickPositions(
	xBounds,
	xIsLog,
	availablePlotWidth,
	"x"
    );

    const yTickPositions = generateAdaptiveTickPositions(
	yBounds,
	yIsLog,
	availablePlotHeight,
	"y"
    );

    // Reserve enough space above the highest tick for its rotated label.
    // Keep yTickPositions based on the original bounds, so this does
    // not introduce any additional ticks.
    let yAxisMax = yBounds[1];

    if (yTickPositions.length > 0) {
	const topTick = Math.max(...yTickPositions);

	const topTickLabel = formatAdaptiveTickLabel(
            topTick,
            yIsLog,
            false
	);

	const labelCharacters = topTickLabel.length;

	// Estimate the vertical extent of a rotated 14px tick label.
	const estimatedLabelExtentPx =
              labelCharacters * 3.5 + 4;

	const originalSpan = yBounds[1] - yBounds[0];

	if (originalSpan > 0 && availablePlotHeight > 0) {
            const requiredFraction = Math.min(
		estimatedLabelExtentPx / availablePlotHeight,
		0.4
            );

            const currentGapFraction =
		  (yBounds[1] - topTick) / originalSpan;

            if (currentGapFraction < requiredFraction) {
		// Solve for a new upper bound that leaves the required
		// fraction of the plot height above the highest tick.
		const requiredMax =
                      (topTick - requiredFraction * yBounds[0]) /
                      (1 - requiredFraction);

		yAxisMax = Math.max(yBounds[1], requiredMax);
            }
	}
    }

    let xFile = null;
    let xFallback = observable;
    let xWidth = 100;
    let xHeight = 40;
    let xMargin = 18;

    if (observable === "Pk") {
        xFile = "k.svg";
        xFallback = "k";
	xWidth = compact ? 180 : 240;
	xHeight = compact ? 25 : 35;
    } else if (observable === "bg" || observable === "th") {
        xFile = "z.svg";
        xFallback = "z";
	xHeight = compact ? 25 : 40;
	xMargin = compact ? 10 : 18;
    }

    let yFile = null;
    let yFallback = selectedOption.value;
    let yWidth = 100;
    let yHeight = 70;

    if (observable === "Pk") {
        yFile = "Pk.svg";
        yFallback = "P(k)";
	yWidth = compact ? 120 : 160;
	yHeight = compact ? 87 : 110;
    }

    const xTitle = axisTitleHTML(
        xFile,
        xFallback,
        xWidth,
        xHeight,
        labelFilter,
	0
    );

    const yTitle = axisTitleHTML(
        yFile,
        yFallback,
        yWidth,
        yHeight,
        labelFilter,
	-90
    );

    const yTickLabelsForMargin = yIsLog
	  ? logTicks(yBounds).map(value => {
              const exponent = Number(value.toFixed(1)).toString();
              return `10^${exponent}`;
	  })
	  : [
              yBounds[0],
              (yBounds[0] + yBounds[1]) / 2,
              yBounds[1]
	  ].map(value => {
              if (value === 0) return "0";

              const absValue = Math.abs(value);

              if (absValue < 1e-3 || absValue >= 1e4) {
		  const [mantissa, exponent] =
			value.toExponential(2).split("e");

		  return `${mantissa}×10^${Number(exponent)}`;
              }

              return Number(value.toPrecision(4)).toString();
	  });


    axisConfig.chart = {
        type: "line",
        backgroundColor: isDarkMode ? "#000000" : "#FFFFFF",
        animation: false,
        marginLeft: compact ? 65 : 80,
        marginRight: 30,
        marginTop: 15,
	marginBottom: compact ? 80 : 110,
        plotBorderWidth: 2,
        plotBorderColor: axisColor
    };

    axisConfig.xAxis = {
        type: "linear",
        min: xBounds[0],
        max: xBounds[1],
        tickPositions: xTickPositions,
        labels: {
            useHTML: true,
            formatter: tickFormatter(xIsLog),
            style: {
                color: axisColor,
                fontSize: compact ? "14px" : "20px"
            }
        },
        title: {
            text: xTitle,
            useHTML: true,
            align: "middle",
            margin: xMargin
        },
        lineColor: axisColor,
        lineWidth: 2,
        tickColor: axisColor,
        tickWidth: 2,
        tickLength: 10,
	startOnTick: false,
	endOnTick: false
    };

    let yLabelMargin;
    let yLabelXOffset;
    let yLabelYOffset;
    if (yFile != null) {
	yLabelMargin = compact ? 60 : 80;
	yLabelXOffset = yWidth + 80 + 20 - 100;
	yLabelYOffset = compact ? -40 : -40;
    } else {
	yLabelMargin = compact ? 10 : 10;
	yLabelXOffset = 0;
	yLabelYOffset = 0;
    }

    axisConfig.yAxis = {
        min: yBounds[0],
        max: yAxisMax,
        tickPositions: yTickPositions,
        labels: {
            useHTML: true,
            formatter: tickFormatter(yIsLog),
	    rotation: 270,
            style: {
                color: axisColor,
                fontSize: compact ? "14px" : "20px"
            }
        },
        title: {
            text: yTitle,
            useHTML: true,
            rotation: 0,
            align: "middle",
            margin: yLabelMargin,
            x: yLabelXOffset,
	    y: yLabelYOffset
        },
        gridLineWidth: 0,
        lineColor: axisColor,
        lineWidth: 2,
        tickColor: axisColor,
        tickWidth: 2,
        tickLength: 10,
        alignTicks: false,
        startOnTick: false,
        endOnTick: false
    };
}


window.addEventListener('load', async () => {

    modelBundle = await loadModelBundleFromUrl(
	DEFAULT_MODEL_BUNDLE_URL
    );

    outputRanges = await readOutputRanges(modelBundle.archive);
    xValuesByObservable = await readXValues(modelBundle.archive);

    model = modelBundle.model;

    buildObservableOptions();
    populateObservableSelector();

    // Read this model's own parameter definitions and defaults.
    parameterDefinitions = await readParameterDefinitions(
	modelBundle.archive
    );

    await createParameterSliders(
	parameterDefinitions,
	modelBundle.archive
    );

    createRedshiftSlider();
    updateRedshiftSliderForSelection();

    resetSlidersToBestfit();

    await initializeFixedYAxisSettings();

    // Keep your existing initialization below:
    document.body.addEventListener("click", updateAxesAndChart);
    window.addEventListener("resize", updateAxesAndChart);

    updateAxes();
    updateDataAndChart();

});




// initialize dictionaries. One contains axis plot configurations, and the other contains the data to be plotted
const axisConfig = {
    chart: {
	type: 'spline',
	backgroundColor: '#FFFFFF',
	marginLeft: 80,
	marginRight: 30,
	marginTop: 15,
	plotBorderWidth: 2,
	plotBorderColor: 'black',
    },
    title: {
		text: ''
    },
    xAxis: {
	type: 'linear',
	tickPositions: [],
	labels: {
	    formatter: function() {
		return xTotalLabels[xTotalTicks.indexOf(this.value)] || '';
	    },
	    style: {
		color: 'black',
		fontSize: '20px'
	    },
	    rotation: 0,
	    overflow: 'allow',
	},
	title: {
            text: '',
	    useHTML: true,
	    align: 'middle',
	},
	lineColor: 'black',
	lineWidth: 2,
	tickWidth: 2,
	tickLength: 10,
	tickColor: 'black',
	min: 0,
	max: 1,
	},
    yAxis: {
	tickPositions: [0, 2000, 4000, 6000, 8000],
	labels: {
	    formatter: function() {
		return ['0', '2000', '4000', '6000', '8000'].includes(this.value.toString()) ? this.value : '';
	    },
	    style: {
		color: 'black',
		fontSize: '20px'
	    },
	    rotation: 270,
	},
	title: {
	    text: '',
	    useHTML: true,
	    rotation: 0,
	    align: 'middle',
	    margin: 100,
	    y: -70,
	},
	gridLineWidth: 0,
	lineWidth: 2,
	lineColor: 'black',
	tickWidth: 2,
	tickLength: 10,
	tickColor: 'black',
	alignTicks: false,
	endOnTick: false,
	startOnTick: false,
	min: 0,
	max: 8400,
    },
    series: [{
	data: [],
	lineWidth: 6,
	marker: {
	    enabled: false
	},
	color: 'rgb(38, 135, 242)'
    }, {
        type: 'scatter',
        data: [],
        marker: {
            symbol: 'circle',
            radius: OBSERVATION_MARKER_RADIUS,
            fillColor: CMB_OBSERVED_COLOR,
            lineWidth: 0
        }
    }, {
        type: 'line',
        data: [],
        linkedTo: ':previous',
        lineWidth: OBSERVATION_ERRORBAR_LINE_WIDTH,
        opacity: OBSERVATION_ERRORBAR_OPACITY,
        marker: { enabled: false },
        connectNulls: false
    }],
    legend: {
	enabled: false
    },
    tooltip: {
	enabled: false
    },
    credits: {
	enabled: false
    }
};


const logscale = 200.0;
const transition = 0.3;
const l_min = 2.0;
let currentXAxisMode = "mixedCl";
let currentYAxisMode = "linear";

function generateXTicks(fewTicks = false) {
    let majorTicks = [];
    let minorTicks = [];
    let logOrder = 1;
    var linMajor = 500;
    var linMinor = 100;
    var minorTicksLogFactor = 1;
    if (fewTicks) {
	linMajor = 1000;
	linMinor = 200;
	minorTicksLogFactor = 2;
	}
    
    for (let i = 2; i <= 2500; i++) {
        if (i < logscale) {
            if (Number.isInteger(Math.log10(i))) {
                majorTicks.push(logScale(i));
                logOrder *= 10;
            } else if (i/minorTicksLogFactor % logOrder === 0) {
                minorTicks.push(logScale(i));
            }
        } else {
            if (i % linMajor === 0) {
                majorTicks.push(linScale(i));
            } else if (i % linMinor === 0) {
                minorTicks.push(linScale(i));
            }
        }
    }
    
    return { majorTicks, minorTicks };
}

function logScale(x) {
    const logx = Math.log10(x);
    const logxnew = (logx / Math.log10(logscale)) * transition;
    
    const loglmin = (Math.log10(l_min) / Math.log10(logscale)) * transition;
    const logscaleVal = (Math.log10(logscale) / Math.log10(logscale)) * transition;
    const logdiff = logscaleVal - loglmin;
    
    return ((logxnew - loglmin) * transition / logdiff);
}

function linScale(x) {
    const linxnew = (x / (2500 - logscale)) - (logscale / (2500 - logscale));
    return ((linxnew * (1 - transition)) + transition);
}

function scaleGraphPoint(xValue) {
    if (xValue <= logscale) {
        return logScale(xValue);
    } else {
        return linScale(xValue);
    }
}

// Function to merge, sort arrays based on values, and synchronize labels
function mergeSortAndSyncArrays(arr1, labels1, arr2, labels2) {
    // Combine arrays and labels into array of objects
    let combinedData = [];
    
    // Combine data from arr1 with labels from labels1
    for (let i = 0; i < arr1.length; i++) {
        combinedData.push({ value: arr1[i], label: labels1[i] });
    }
    
    // Combine data from arr2 with labels from labels2
    for (let i = 0; i < arr2.length; i++) {
        combinedData.push({ value: arr2[i], label: labels2[i] });
    }

    // Sort combined data based on value
    combinedData.sort((a, b) => a.value - b.value);

    // Extract sorted arrays and labels from sorted combined data
    let sortedValues = [];
    let sortedLabels = [];

    for (let item of combinedData) {
        sortedValues.push(item.value);
        sortedLabels.push(item.label);
    }

    return {
        sortedValues: sortedValues,//.map(v => ({value: v})),
        sortedLabels: sortedLabels
    };
}



function updateAxesAndChart() {
    updateAxes();
    updateDataAndChart();
    displayChart();
}



function updateAxes() {
    if (selectedObservable.value !== "Cl") {
        updateOtherObservableAxes();
        return;
    }
    var xLabels = [];
    var outputTicks = {};
    var fontSize = '20px';
    var xSvgWidth = '120px';
    var xSvgHeight = '50px';
    var ySvgWidth = '150px';
    var ySvgHeight = '150px';
    var ySvgFile = 'Cl.svg';
    var marginLeft = 80;
    var marginRight = 30;
    var marginBottom = 110;
    var yLabelMargin = 100;
    var yLabelOffset = -70;
    var lineWidth = 6;
    var yMin = 0;
    var yMax = 7000;
    var yTicks = [0, 2000, 4000, 6000];
    var yLabels = ['0', '2000', '4000', '6000'];
    var yLabelsReduced = ['0', '2000', '4000', '6000'];
    if (selectedOption.value === "TE") {
	yMin = -230;
	yMax = 230;
	yTicks = [-200, -100, 0, 100, 200];
	yLabels = ['-200', '-100', '0', '100', '200'];
	yLabelsReduced = ['-200', '-100', '0', '100', '200'];
    } else if (selectedOption.value === "EE") {
	yMin = -2;
	yMax = 60;
	yTicks = [0, 20, 40, 60];
	yLabels = ['0', '20', '40', '60'];
	yLabelsReduced = ['0', '20', '40', '60'];
    } else if (selectedOption.value === "PP") {
	yMin = 0;
	yMax = 2.5;
	yTicks = [0, 0.5, 1, 1.5, 2, 2.5];
	yLabels = ['0', '0.5', '1', '1.5', '2', '2.5'];
	yLabelsReduced = ['0', '0.5', '1', '1.5', '2', '2.5'];
	ySvgFile = 'Cl_pp.svg';
	ySvgWidth = '260px';
	ySvgHeight = '260px';
	yLabelOffset = -140;
	yLabelMargin = 155;
    }
    const compact = isCompactChart();
    if (!compact) {
        xLabels = ['10¹', '10²', "500", "1000", "1500", "2000", "2500"];
        outputTicks = generateXTicks(false);
    } else {
        xLabels = ['10¹', '10²', "1000", "2000"];
        outputTicks = generateXTicks(true);
        fontSize = '14px';
        xSvgWidth = '80px';
        xSvgHeight = '30px';
        ySvgWidth = '100px';
        ySvgHeight = '100px';
        marginLeft = 65;
        marginRight = 20;
        marginBottom = 80;
        yLabelMargin = 65;
        yLabelOffset = -40;
        lineWidth = 3;
        yLabels = yLabelsReduced;

        if (selectedOption.value === "PP") {
            ySvgWidth = '170px';
            ySvgHeight = '170px';
            yLabelOffset = -80;
            yLabelMargin = 100;
        }
    }
    xMajorTicks = outputTicks.majorTicks;
    xMinorTicks = outputTicks.minorTicks;
    const xMinorLabels = [];
    for (let i = 0; i < xMinorTicks.length; i++) {
        xMinorLabels.push("");
    }
    const { sortedValues: xTotalTicks, sortedLabels: xTotalLabels } = mergeSortAndSyncArrays(xMajorTicks, xLabels, xMinorTicks, xMinorLabels);

    const clXGrid = (xValuesByObservable.Cl || [])
	  .map(Number)
	  .filter(Number.isFinite);

    const clXMin = clXGrid.length
	  ? scaleGraphPoint(Math.min(...clXGrid))
	  : 0;

    const clXMax = clXGrid.length
	  ? scaleGraphPoint(Math.max(...clXGrid))
	  : 1;
    
    const isDarkMode = document.body.classList.contains('dark-mode');
    const axisColor = isDarkMode ? 'white' : 'black';
    const labelFilter = isDarkMode ? "invert(100%) sepia(100%) saturate(0%) hue-rotate(201deg) brightness(106%) contrast(106%);" : "invert(0%) sepia(97%) saturate(0%) hue-rotate(34deg) brightness(89%) contrast(103%);";

    axisConfig.chart = {
        type: 'line',
        backgroundColor: isDarkMode ? '#000000' : '#FFFFFF',
        animation: false, // Disable animation                                                                     
        marginLeft: marginLeft,
        marginRight: marginRight,
        marginTop: 15,
	marginBottom: marginBottom,
        plotBorderWidth: 2,
        plotBorderColor: axisColor,
    };
    axisConfig.title = {
	text: ''
    };
    axisConfig.xAxis = {
        type: 'linear',
        tickPositions: xTotalTicks,
        labels: {
            formatter: function() {
                return xTotalLabels[xTotalTicks.indexOf(this.value)] || '';
            },
            style: {
                color: axisColor,
                fontSize: fontSize
            },
            rotation: 0,
            overflow: 'allow',
        },
        title: {
            text: axisTitleHTML(
		"ell.svg",
		"ℓ",
		parseFloat(xSvgWidth),
		parseFloat(xSvgHeight),
		labelFilter,
		0
	    ),
            useHTML: true,
            align: 'middle',
        },
        lineColor: axisColor,
        lineWidth: 2,
        tickWidth: 2,
        tickLength: 10,
        tickColor: axisColor,
        min: clXMin,
	max: clXMax,
	startOnTick: false,
	endOnTick: false,
    };
    axisConfig.yAxis = {
        tickPositions: yTicks,
        labels: {
            formatter: function() {
                return yLabels.includes(this.value.toString()) ? this.value : '';
            },
            style: {
                color: axisColor,
                fontSize: fontSize
            },
            rotation: 270,
        },
        title: {
            text: axisTitleHTML(
		ySvgFile,
		selectedOption.value === "PP"
		    ? "Cφφ"
		    : `Cℓ (${selectedOption.value})`,
		parseFloat(ySvgWidth),
		parseFloat(ySvgHeight),
		labelFilter,
		0
	    ),
            useHTML: true,
            rotation: 0,
            align: 'middle',
            margin: compact ? yLabelMargin - 10 : yLabelMargin,
	    x: parseFloat(ySvgWidth) + yLabelMargin - 40,
            y: yLabelOffset,
        },
        gridLineWidth: 0,
        lineWidth: 2,
        lineColor: axisColor,
        tickWidth: 2,
        tickLength: 10,
        tickColor: axisColor,
        alignTicks: false,
        endOnTick: false,
        startOnTick: false,
        min: yMin,
        max: yMax,
    };
    axisConfig.series[0].lineWidth = lineWidth;
    axisConfig.series[1].marker.radius = getObservationMarkerRadius();
    axisConfig.series[2].lineWidth = OBSERVATION_ERRORBAR_LINE_WIDTH;
    axisConfig.series[2].opacity = OBSERVATION_ERRORBAR_OPACITY;
};

function interpolatePchipAt(xValues, yValues, targetX) {
    const n = xValues.length;

    if (n !== yValues.length || n === 0) {
        throw new Error("Invalid PCHIP interpolation arrays.");
    }

    if (n === 1) {
        return yValues[0];
    }

    const h = [];
    const delta = [];

    for (let i = 0; i < n - 1; i++) {
        h[i] = xValues[i + 1] - xValues[i];

        if (!(h[i] > 0)) {
            throw new Error("Redshift values must be strictly increasing.");
        }

        delta[i] = (yValues[i + 1] - yValues[i]) / h[i];
    }

    if (n === 2) {
        return yValues[0] +
            (targetX - xValues[0]) * delta[0];
    }

    function endpointSlope(h0, h1, d0, d1) {
        let slope = (
            (2 * h0 + h1) * d0 - h0 * d1
        ) / (h0 + h1);

        if (Math.sign(slope) !== Math.sign(d0)) {
            slope = 0;
        } else if (
            Math.sign(d0) !== Math.sign(d1) &&
            Math.abs(slope) > Math.abs(3 * d0)
        ) {
            slope = 3 * d0;
        }

        return slope;
    }

    const slopes = new Array(n);

    slopes[0] = endpointSlope(
        h[0], h[1], delta[0], delta[1]
    );

    slopes[n - 1] = endpointSlope(
        h[n - 2],
        h[n - 3],
        delta[n - 2],
        delta[n - 3]
    );

    for (let i = 1; i < n - 1; i++) {
        const dLeft = delta[i - 1];
        const dRight = delta[i];

        if (
            dLeft === 0 ||
            dRight === 0 ||
            Math.sign(dLeft) !== Math.sign(dRight)
        ) {
            slopes[i] = 0;
        } else {
            const w1 = 2 * h[i] + h[i - 1];
            const w2 = h[i] + 2 * h[i - 1];

            slopes[i] = (w1 + w2) /
                (w1 / dLeft + w2 / dRight);
        }
    }

    // Find the interval containing the requested redshift.
    let i = 0;

    while (i < n - 2 && targetX > xValues[i + 1]) {
        i++;
    }

    const t = (targetX - xValues[i]) / h[i];
    const t2 = t * t;
    const t3 = t2 * t;

    return (
        (2 * t3 - 3 * t2 + 1) * yValues[i] +
        (t3 - 2 * t2 + t) * h[i] * slopes[i] +
        (-2 * t3 + 3 * t2) * yValues[i + 1] +
        (t3 - t2) * h[i] * slopes[i + 1]
    );
}

function getPkSpectrumAtRedshift(
    modelOutput,
    pkName,
    requestedRedshift,
    expectedLength
) {
    const familyKey = String(pkName)
	  .replace(/^Pk_/i, "")
	  .toLowerCase();

    const family = pkOutputsByName.get(familyKey);

    if (!family || family.redshifts.length === 0) {
        throw new Error(
            `No emulated redshifts found for Pk family "${pkName}".`
        );
    }

    const redshifts = family.redshifts;

    const spectra = redshifts.map(redshift => {
        const range = family.rangesByRedshift.get(redshift);

        if (!range) {
            throw new Error(
                `Missing output range for ${pkName}_${redshift}.`
            );
        }

        const values = modelOutput.slice(range.start, range.end);

        if (values.length !== expectedLength) {
            throw new Error(
                `Output length mismatch for ${pkName}_${redshift}: ` +
                `${values.length}, expected ${expectedLength}.`
            );
        }

        return values;
    });

    // Restrict the requested redshift to the supported range.
    const targetZ = Math.max(
        redshifts[0],
        Math.min(
            redshifts[redshifts.length - 1],
            Number(requestedRedshift)
        )
    );

    // Use the exact model output when the requested redshift
    // is already emulated.
    const exactIndex = redshifts.findIndex(
        z => Math.abs(z - targetZ) < 1e-8
    );

    if (exactIndex !== -1) {
        return spectra[exactIndex];
    }

    // A family with only one emulated redshift has no interval
    // over which to interpolate.
    if (redshifts.length === 1) {
        return spectra[0];
    }

    // Interpolate across redshift separately for every k-grid point.
    // interpolateData() is your existing cubic-spline interpolator.
    const interpolatedSpectrum = new Array(expectedLength);

    for (let kIndex = 0; kIndex < expectedLength; kIndex++) {
        const valuesAtK = spectra.map(
            spectrum => Number(spectrum[kIndex])
        );

        if (!valuesAtK.every(Number.isFinite)) {
            throw new Error(
                `Invalid Pk values at k-grid index ${kIndex}.`
            );
        }

        if (
	    valuesAtK.every(
		value => Number.isFinite(value) && value > 0
	    )
	) {
	    // Interpolate logarithmic power to preserve positivity and
	    // behave better when the power spans many orders of magnitude.
	    const logPower = valuesAtK.map(value => Math.log10(value));

	    interpolatedSpectrum[kIndex] = Math.pow(
		10,
		interpolatePchipAt(redshifts, logPower, targetZ)
	    );
	} else {
	    // Defensive fallback for non-positive or underflowed samples.
	    const interpolatedValue = interpolatePchipAt(
		redshifts,
		valuesAtK,
		targetZ
	    );

	    interpolatedSpectrum[kIndex] = Math.max(
		Number.MIN_VALUE,
		interpolatedValue
	    );
	}
    }

    return interpolatedSpectrum;
}

// Draw vertical error bars as plain line segments without end-cap whiskers.
function buildVerticalErrorLineData(errorBounds) {
    const data = [];

    errorBounds.forEach(([x, low, high]) => {
        if (
            !Number.isFinite(x) ||
            !Number.isFinite(low) ||
            !Number.isFinite(high) ||
            high < low
        ) {
            return;
        }

        data.push(
            { x, y: low },
            { x, y: high },
            { x, y: null }
        );
    });

    return data;
}

function buildPkObservationSeries(dataBundle) {
    const datasets = dataBundle?.datasets || [];
    const series = [];
    const markerRadius = getObservationMarkerRadius();

    datasets.forEach((dataset, datasetIndex) => {
        const color = PK_DATA_COLORS[
            datasetIndex % PK_DATA_COLORS.length
        ];

        const points = (dataset.points || []).filter(point =>
            Number.isFinite(Number(point.k)) &&
            Number(point.k) > 0 &&
            Number.isFinite(Number(point.pk)) &&
            Number(point.pk) > 0
        );

        if (points.length === 0) return;

        const seriesId = `pk-observed-${String(
            dataset.id || datasetIndex
        ).replace(/[^a-zA-Z0-9_-]/g, "-")}`;

        // Observational values are already h-scaled.
        // The chart's Pk axes use log10 coordinates.
        const pointData = points.map(point => [
            Math.log10(Number(point.k)),
            Math.log10(Number(point.pk))
        ]);

        series.push({
            id: seriesId,
            type: "scatter",
            name: dataset.label || dataset.id || `Dataset ${datasetIndex + 1}`,
            showInLegend: true,
            data: pointData,
            color,
            zIndex: 8,
            enableMouseTracking: false,
            marker: {
                enabled: true,
                symbol: "circle",
                radius: markerRadius,
                fillColor: color,
                lineWidth: 0
            }
        });

	// Vertical Pk uncertainties drawn as line segments.
	const yErrorBounds = points
	      .filter(point =>
		  Number.isFinite(Number(point.pkLow)) &&
		      Number(point.pkLow) > 0 &&
		      Number.isFinite(Number(point.pkHigh)) &&
		      Number(point.pkHigh) >= Number(point.pkLow)
	      )
	      .map(point => [
		  Math.log10(Number(point.k)),
		  Math.log10(Number(point.pkLow)),
		  Math.log10(Number(point.pkHigh))
	      ]);

	const yErrorData = buildVerticalErrorLineData(yErrorBounds);

	if (yErrorData.length > 0) {
	    series.push({
		type: "line",
		linkedTo: seriesId,
		showInLegend: false,
		data: yErrorData,
		color,
		lineWidth: OBSERVATION_ERRORBAR_LINE_WIDTH,
		opacity: OBSERVATION_ERRORBAR_OPACITY,
		marker: { enabled: false },
		connectNulls: false,
		enableMouseTracking: false,
		zIndex: 7
	    });
	}

	// Horizontal k uncertainty ranges, without end-cap whiskers.
	const xErrorData = [];

	points.forEach(point => {
	    const kLow = Number(point.kLow);
	    const kHigh = Number(point.kHigh);
	    const pk = Number(point.pk);

	    if (
		!Number.isFinite(kLow) ||
		    !Number.isFinite(kHigh) ||
		    kLow <= 0 ||
		    kHigh < kLow
	    ) {
		return;
	    }

	    const y = Math.log10(pk);
	    const xLow = Math.log10(kLow);
	    const xHigh = Math.log10(kHigh);

	    // Horizontal uncertainty stem.
	    xErrorData.push(
		{ x: xLow, y },
		{ x: xHigh, y },
		{ x: xHigh, y: null }
	    );
	});

	if (xErrorData.length > 0) {
	    series.push({
		type: "line",
		linkedTo: seriesId,
		showInLegend: false,
		data: xErrorData,
		color,
		lineWidth: OBSERVATION_ERRORBAR_LINE_WIDTH,
		opacity: OBSERVATION_ERRORBAR_OPACITY,
		marker: { enabled: false },
		connectNulls: false,
		enableMouseTracking: false,
		zIndex: 6
	    });
	}
    });

    return series;
}

async function updateDataAndChart() {
    // Get slider values
    const sliderValues = sliders.map(slider => parseFloat(slider.value));
    sliderValues.forEach((value, index) => {
        document.getElementById(`slider${index + 1}-value`).innerText = value;
    });

    // Create input tensor
    const inputData = tf.tensor([sliderValues]);
    const outputData = model.predict(inputData);
    const outputArray = await outputData.array();

    const observable = selectedObservable.value;
    const option = selectedOption.value;

    const xKey =
	  xValueKeyByObservable[observable] || observable;

    const xdata = xValuesByObservable[xKey];

    if (!Array.isArray(xdata)) {
	throw new Error(
            `No x-values found for observable "${observable}" ` +
		`(key "${xKey}").`
	);
    }

    let rawSpectrum;

    if (observable === "Pk") {
	// Assemble the spectrum for the currently selected redshift.
	rawSpectrum = getPkSpectrumAtRedshift(
            outputArray[0],
            option,
            selectedRedshift.value,
            xdata.length
	);
    } else {
	// Preserve the existing lookup for non-Pk observables.
	const range = findOutputRange(observable, option);

	if (!range) {
            throw new Error(
		`No model output found for ${observable}_${option}.`
            );
	}

	rawSpectrum = outputArray[0].slice(
            range.start,
            range.end
	);
    }

    if (rawSpectrum.length !== xdata.length) {
	throw new Error(
            `Output length (${rawSpectrum.length}) does not match ` +
		`x-value length (${xdata.length}) for ${observable}_${option}.`
	);
    }

    // Convert the model P(k) and k grid to h-scaled units.
    // Observational points in Pk_data.json are already in these units.
    const hForPk = observable === "Pk"
	  ? getCurrentHForPkUnits()
	  : 1;

    const plotXdata = observable === "Pk"
	  ? xdata.map(k => Number(k) / hForPk)
	  : xdata.map(Number);

    const factor = observable === "Cl"
	  ? (option === "PP" ? 1e7 : (2.7255e6) ** 2)
	  : observable === "Pk"
	  ? hForPk ** 3
	  : 1;

    const scaledSpectrum = rawSpectrum.map(
	value => value * factor
    );

    function spansDecade(values) {
	if (
            values.length === 0 ||
		values.some(value => !Number.isFinite(value) || value <= 0)
	) {
            return false;
	}

	const min = Math.min(...values);
	const max = Math.max(...values);

	return max > min && Math.log10(max / min) >= 1;
    }


    // Keep the CMB mixed x transformation intact. Matter power spectra
    // use log-log coordinates; other observables select a suitable scale.
    if (observable === "Cl") {
	currentXAxisMode = "mixedCl";
    } else if (["Pk", "bg", "th"].includes(observable)) {
	// These observables always use logarithmic x-coordinates.
	currentXAxisMode = "log";
    } else {
	// Other observables, including extra, retain automatic scaling.
	currentXAxisMode =
            spansDecade(plotXdata) && plotXdata.every(value => value > 0)
            ? "log"
            : "linear";
    }

    const ySettings = fixedYAxisSettings.get(
	`${observable}_${option}`
    );

    currentYAxisMode = observable === "Cl"
	? "linear"
	: (ySettings?.mode || "linear");

    const points = plotXdata
	  .map((x, index) => ({
              x: Number(x),
              y: Number(scaledSpectrum[index])
	  }))
	  .filter(point =>
              Number.isFinite(point.x) &&
		  Number.isFinite(point.y) &&
		  (currentXAxisMode === "linear" || point.x > 0) &&
		  (currentYAxisMode !== "log" || point.y > 0)
	  )
	  .sort((a, b) => a.x - b.x);

    if (points.length < 3) {
	throw new Error(
            `Not enough valid points to interpolate ${observable}_${option}.`
	);
    }

    const splineX = points.map(point => point.x);

    // Interpolate log(y) when the displayed y-axis is logarithmic.
    // This avoids negative overshoots for positive spectra.
    const splineY = points.map(point =>
	currentYAxisMode === "log"
            ? Math.log10(point.y)
            : point.y
    );

    for (let i = 1; i < splineX.length; i++) {
	if (splineX[i] <= splineX[i - 1]) {
            throw new Error(
		`Duplicate x-values near index ${i} for ${observable}_${option}.`
            );
	}
    }

    const numPoints = 800;
    const minX = splineX[0];
    const maxX = splineX[splineX.length - 1];

    const useLogGrid =
	  observable === "Cl" ||
	  currentXAxisMode === "log";

    const xGrid = Array.from({ length: numPoints }, (_, index) => {
	const fraction = index / (numPoints - 1);

	if (useLogGrid) {
            const startLog = Math.log10(minX);
            const endLog = Math.log10(maxX);

            return Math.pow(
		10,
		startLog + (endLog - startLog) * fraction
            );
	}

	return minX + (maxX - minX) * fraction;
    });

    const usePchip = observable === "th";

    const interpolatedSpectrum = usePchip
	  ? interpolatePchipData(splineX, splineY, xGrid)
	  : interpolateData(splineY, splineX, xGrid);

    // For logarithmic axes, keep y-values in log10 coordinates.
    // updateOtherObservableAxes() formats the ticks as powers of ten.
    const smoothSpectrum = interpolatedSpectrum;

    function plottedX(value) {
	if (observable === "Cl") {
            return scaleGraphPoint(value);
	}

	if (currentXAxisMode === "log") {
            return Math.log10(value);
	}

	return value;
    }


    axisConfig.series[0].data = smoothSpectrum
	.map((y, index) => [
            plottedX(xGrid[index]),
            y
	])
	.filter(point =>
            Number.isFinite(point[0]) &&
		Number.isFinite(point[1])
	);

    // Remove dynamic observational series left over from the last redraw.
    // Keep the model curve and the two existing CMB observation series.
    axisConfig.series = axisConfig.series.slice(0, 3);

    axisConfig.series[1].data = [];
    axisConfig.series[2].data = [];

    axisConfig.series[1].name = "";
    axisConfig.series[1].showInLegend = false;
    axisConfig.series[2].showInLegend = false;

    if (data && observable === "Cl") {
	let seriesData = [];
	let errorBounds = [];

	if (option === "PP") {
            seriesData = data.map(d => [
		scaleGraphPoint(d[0]),
		d[1] * 1e7
            ]);

            errorBounds = data.map(d => [
		scaleGraphPoint(d[0]),
		(d[1] - d[2]) * 1e7,
		(d[1] + d[3]) * 1e7
            ]);
	} else {
            seriesData = data.map(d => [
		scaleGraphPoint(d[0]),
		d[1]
            ]);

            errorBounds = data.map(d => [
		scaleGraphPoint(d[0]),
		d[1] - d[2],
		d[1] + d[3]
            ]);
	}

	const errorData = buildVerticalErrorLineData(errorBounds);

	axisConfig.series[1].data = seriesData;
	axisConfig.series[2].data = errorData;

	// Legend entry for the currently selected CMB spectrum.
	axisConfig.series[1].name = option === "PP"
            ? "SPT-SZ + Planck lensing map"
            : `Planck 2018 ${option}`;

	axisConfig.series[1].showInLegend = true;
	axisConfig.series[2].showInLegend = false;

    axisConfig.series[1].color = CMB_OBSERVED_COLOR;
    // Markers remain fully opaque; transparency applies only to error bars.
    axisConfig.series[1].marker = {
        symbol: 'circle',
        radius: getObservationMarkerRadius(),
        fillColor: CMB_OBSERVED_COLOR,
        lineWidth: 0
    };

    axisConfig.series[2].color = CMB_OBSERVED_COLOR;
    axisConfig.series[2].lineWidth = OBSERVATION_ERRORBAR_LINE_WIDTH;
    axisConfig.series[2].opacity = OBSERVATION_ERRORBAR_OPACITY;
    axisConfig.series[2].marker = { enabled: false };
    axisConfig.series[2].connectNulls = false;

    } else if (
	observable === "Pk" &&
	    checkboxChecked &&
	    pkCompilationData?.datasets
    ) {
	// These data are already in h-scaled units.
	axisConfig.series.push(
            ...buildPkObservationSeries(pkCompilationData)
	);
    }

    // Apply the observable-specific axes before creating the chart.
    updateAxes();

    // A semitransparent legend, enabled only when observations are visible.
    const darkMode = document.body.classList.contains("dark-mode");

    const hasLegendEntries = axisConfig.series.some(
	series => series.showInLegend && series.data?.length > 0
    );

    const compact = isCompactChart();

    axisConfig.series[0].showInLegend = false;

    let xLegendOffset
    let yLegendOffset
    if (compact && observable === "Cl") {
	xLegendOffset = -14;
	yLegendOffset = 8;
    } else if (compact && observable === "Pk") {
	xLegendOffset = 58;
	yLegendOffset = -70;
    } else {
	xLegendOffset = -23;
	yLegendOffset = 8;
    }

    axisConfig.legend = {
	enabled: hasLegendEntries,
	layout: "vertical",
	align: compact && observable === "Pk" ? "left" : "right",
	verticalAlign: compact && observable === "Pk" ? "bottom" : "top",
	floating: true,
	x: xLegendOffset,
	y: yLegendOffset,

	backgroundColor: darkMode
            ? "rgba(15, 15, 15, 0.82)"
            : "rgba(255, 255, 255, 0.82)",

	borderColor: darkMode ? "#BBBBBB" : "#666666",
	borderWidth: 1,
	borderRadius: 4,
	shadow: false,

	padding: 8,
	itemMarginTop: 2,
	itemMarginBottom: 3,
	symbolRadius: 3,

	itemStyle: {
            color: darkMode ? "#FFFFFF" : "#222222",
            fontSize: "12px",
            fontWeight: "normal"
	}
    };

    displayChart();
};

function displayChart() {
    Highcharts.setOptions({
	plotOptions: {
	    series: {
		animation: false,
		states: {
                    hover: {
			enabled: false
                    },
		    inactive: {
			enabled: false
		    }
		},
	    },
	}
    });
    const chart = Highcharts.chart('outputChart', axisConfig);

    // Attach error handlers after Highcharts creates the chart.
    chart.renderTo
	.querySelectorAll("img.axis-title-image")
	.forEach(img => {
            const fallback = img.parentElement.querySelector(
		".axis-title-fallback"
            );

            const showFallback = () => {
		img.style.display = "none";

		if (fallback) {
                    fallback.style.display = "inline-block";
		}
            };

            img.addEventListener("error", showFallback, { once: true });

            // Handle an image that failed before the listener was attached.
            if (img.complete && img.naturalWidth === 0) {
		showFallback();
            }
	});
}


// Cubic spline interpolation function
function interpolateData(yData, xData, xGrid) {
    const n = yData.length;
    if (n < 2) {
        throw new Error('Data array must have at least 2 points for interpolation.');
    }

    // Prepare arrays for the spline interpolation
    const x = [...xData]; // Copy xData array
    const y = [...yData]; // Copy yData array

    // Calculate coefficients for the cubic spline
    const a = [];
    const b = [];
    const c = [];
    const d = [];
    const h = [];
    const alpha = [];
    const l = [];
    const mu = [];
    const z = [];

    // Step 1: Compute h[i] = x[i+1] - x[i]
    for (let i = 0; i < n - 1; i++) {
        h[i] = x[i + 1] - x[i];
    }

    // Step 2: Compute alpha[i] = (3/h[i]) * (y[i+1] - y[i]) - (3/h[i-1]) * (y[i] - y[i-1])
    for (let i = 1; i < n - 1; i++) {
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1]);
    }

    // Step 3: Compute l and mu
    l[0] = 1;
    mu[0] = 0;
    z[0] = 0;

    for (let i = 1; i < n - 1; i++) {
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    // Step 4: Compute l[n], z[n], and c
    l[n - 1] = 1;
    z[n - 1] = 0;
    c[n - 1] = 0;

    for (let j = n - 2; j >= 0; j--) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
        a[j] = y[j];
    }

    // Step 5: Evaluate the spline at each desired point.
    // xGrid is sorted, so reuse the segment index instead of
    // searching from the start for every grid point.
    const interpolatedData = [];
    let j = 0;

    for (let i = 0; i < xGrid.length; i++) {
	const xValue = xGrid[i];

	while (j < n - 2 && x[j + 1] < xValue) {
            j++;
	}

	const dt = xValue - x[j];

	const interpolatedValue =
              a[j] +
              b[j] * dt +
              c[j] * dt ** 2 +
              d[j] * dt ** 3;

	interpolatedData.push(interpolatedValue);
    }

    return interpolatedData;
}

function interpolatePchipData(x, y, xGrid) {
    const n = x.length;

    if (n !== y.length || n < 2) {
        throw new Error("PCHIP requires matching arrays with at least 2 points.");
    }

    const h = [];
    const delta = [];

    for (let i = 0; i < n - 1; i++) {
        h[i] = x[i + 1] - x[i];

        if (!(h[i] > 0)) {
            throw new Error("PCHIP x-values must be strictly increasing.");
        }

        delta[i] = (y[i + 1] - y[i]) / h[i];
    }

    if (n === 2) {
        return xGrid.map(xq =>
            y[0] + (xq - x[0]) * delta[0]
        );
    }

    function endpointSlope(h0, h1, d0, d1) {
        let slope =
            ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);

        if (Math.sign(slope) !== Math.sign(d0)) {
            slope = 0;
        } else if (
            Math.sign(d0) !== Math.sign(d1) &&
            Math.abs(slope) > 3 * Math.abs(d0)
        ) {
            slope = 3 * d0;
        }

        return slope;
    }

    const m = new Array(n);

    m[0] = endpointSlope(h[0], h[1], delta[0], delta[1]);

    m[n - 1] = endpointSlope(
        h[n - 2],
        h[n - 3],
        delta[n - 2],
        delta[n - 3]
    );

    for (let i = 1; i < n - 1; i++) {
        const left = delta[i - 1];
        const right = delta[i];

        if (
            left === 0 ||
            right === 0 ||
            Math.sign(left) !== Math.sign(right)
        ) {
            m[i] = 0;
        } else {
            const w1 = 2 * h[i] + h[i - 1];
            const w2 = h[i] + 2 * h[i - 1];

            m[i] = (w1 + w2) /
                (w1 / left + w2 / right);
        }
    }

    const result = [];
    let j = 0;

    for (const xq of xGrid) {
        while (j < n - 2 && xq > x[j + 1]) {
            j++;
        }

        const t = (xq - x[j]) / h[j];
        const t2 = t * t;
        const t3 = t2 * t;

        result.push(
            (2 * t3 - 3 * t2 + 1) * y[j] +
            (t3 - 2 * t2 + t) * h[j] * m[j] +
            (-2 * t3 + 3 * t2) * y[j + 1] +
            (t3 - t2) * h[j] * m[j + 1]
        );
    }

    return result;
}
