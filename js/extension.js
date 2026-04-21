import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_WIDTH = 380;
const NODE_HEIGHT = 820;
const CANVAS_SIZE = 260;
const VISIBLE_SHAPE = 10;
const VISIBLE_EXPR = 10;
const POSE_LABELS = [
    "global_rx", "global_ry", "global_rz",
    "neck_rx", "neck_ry", "neck_rz",
    "jaw_rx", "jaw_ry", "jaw_rz",
    "eye_l_rx", "eye_l_ry", "eye_l_rz",
    "eye_r_rx", "eye_r_ry", "eye_r_rz",
];
const TRANS_LABELS = ["tx", "ty", "tz"];

const TOPOLOGY_CACHE = new Map();

function debounce(fn, delay) {
    let timeout = null;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), delay);
    };
}

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name) || null;
}

function getInputNode(node, inputName) {
    const input = node.inputs?.find((slot) => slot.name === inputName);
    const linkId = input?.link;
    if (!linkId) return null;
    const link = app.graph?.links?.[linkId];
    if (!link?.origin_id) return null;
    return app.graph.getNodeById?.(link.origin_id) || null;
}

function getLoadFlameModelConfig(node) {
    return {
        gender: getWidget(node, "gender")?.value || "generic",
        shape_dim: Number(getWidget(node, "shape_dim")?.value ?? 50),
        expr_dim: Number(getWidget(node, "expr_dim")?.value ?? 50),
    };
}

function defaultParams(shapeDim, exprDim) {
    return {
        shape: Array(shapeDim).fill(0),
        expr: Array(exprDim).fill(0),
        pose: Array(15).fill(0),
        trans: Array(3).fill(0),
    };
}

function normalizeParams(node, input) {
    const cfg = getLoadFlameModelConfig(node);
    const out = defaultParams(cfg.shape_dim, cfg.expr_dim);
    if (!input || typeof input !== "object") return out;
    for (const key of ["shape", "expr", "pose", "trans"]) {
        const dst = out[key];
        const src = Array.isArray(input[key]) ? input[key] : [];
        for (let i = 0; i < dst.length; i++) {
            dst[i] = Number(src[i] ?? 0);
        }
    }
    return out;
}

function parseParams(node) {
    const widget = getWidget(node, "params_json");
    if (!widget?.value) return normalizeParams(node, null);
    try {
        return normalizeParams(node, JSON.parse(widget.value));
    } catch {
        return normalizeParams(node, null);
    }
}

function writeParams(node, params) {
    const widget = getWidget(node, "params_json");
    if (!widget) return;
    widget.value = JSON.stringify(params);
    node._flameParams = params;
    app.graph.setDirtyCanvas(true, false);
}

function hideParamsWidget(node) {
    const widget = getWidget(node, "params_json");
    if (!widget) return;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.serializeValue = () => widget.value;
}

function decodeBase64ToTypedArray(b64, Type) {
    const raw = atob(b64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
        bytes[i] = raw.charCodeAt(i);
    }
    return new Type(bytes.buffer);
}

async function getTopology(cfg) {
    const cacheKey = cfg.gender;
    if (TOPOLOGY_CACHE.has(cacheKey)) {
        return TOPOLOGY_CACHE.get(cacheKey);
    }
    const query = new URLSearchParams({ gender: cfg.gender });
    const response = await api.fetchApi(`/flame/faces?${query.toString()}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data?.error || "Failed to load FLAME topology");
    const topology = {
        faces: new Int32Array(decodeBase64ToTypedArray(data.faces_b64, Int32Array)),
        template: new Float32Array(decodeBase64ToTypedArray(data.template_b64, Float32Array)),
        n_vertices: Number(data.n_vertices),
        n_faces: Number(data.n_faces),
        gender: data.gender,
    };
    TOPOLOGY_CACHE.set(cacheKey, topology);
    return topology;
}

function resolveUpstreamParams(node) {
    const upstream = getInputNode(node, "flame_params_in");
    if (!upstream) return null;
    const paramsWidget = getWidget(upstream, "params_json");
    if (paramsWidget?.value) {
        try {
            return normalizeParams(node, JSON.parse(paramsWidget.value));
        } catch {
            return null;
        }
    }
    return null;
}

function buildSlider({ label, min, max, step, value, onInput }) {
    const row = document.createElement("label");
    Object.assign(row.style, {
        display: "grid",
        gridTemplateColumns: "110px 1fr 52px",
        gap: "8px",
        alignItems: "center",
        fontSize: "11px",
        color: "#d6d6d6",
    });

    const title = document.createElement("span");
    title.textContent = label;

    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    input.style.width = "100%";

    const valueText = document.createElement("span");
    valueText.textContent = Number(value).toFixed(2);
    valueText.style.fontFamily = "monospace";

    input.addEventListener("input", () => {
        const next = Number(input.value);
        valueText.textContent = next.toFixed(2);
        onInput(next);
    });

    row.append(title, input, valueText);
    return row;
}

function rotateVertex(x, y, z, yaw, pitch) {
    const cy = Math.cos(yaw);
    const sy = Math.sin(yaw);
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);

    const x1 = cy * x + sy * z;
    const z1 = -sy * x + cy * z;
    const y2 = cp * y - sp * z1;
    const z2 = sp * y + cp * z1;
    return [x1, y2, z2];
}

function createViewer(canvas, statusEl) {
    const ctx = canvas.getContext("2d");
    const state = {
        verts: null,
        faces: null,
        yaw: 0.0,
        pitch: 0.0,
        zoom: 1.4,
        dragging: false,
        lastX: 0,
        lastY: 0,
    };

    function render() {
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#7d848c";
        ctx.fillRect(0, 0, width, height);
        if (!state.verts || !state.faces) {
            ctx.fillStyle = "#20252c";
            ctx.fillText("No FLAME preview yet", 16, 20);
            return;
        }

        const vertexCount = state.verts.length / 3;
        const projected = new Array(vertexCount);
        const light = [0.25, -0.4, 1.0];
        const ll = Math.hypot(...light);
        light[0] /= ll; light[1] /= ll; light[2] /= ll;

        let minZ = Infinity;
        let maxZ = -Infinity;
        for (let i = 0; i < vertexCount; i++) {
            const x = state.verts[i * 3 + 0];
            const y = state.verts[i * 3 + 1];
            const z = state.verts[i * 3 + 2];
            const [rx, ry, rz] = rotateVertex(x, y, z, state.yaw, state.pitch);
            minZ = Math.min(minZ, rz);
            maxZ = Math.max(maxZ, rz);
            const depth = rz + 2.2;
            const scale = (width * 0.42 * state.zoom) / Math.max(depth, 0.2);
            projected[i] = {
                x: width * 0.5 + rx * scale,
                y: height * 0.53 - ry * scale,
                z: rz,
                rx,
                ry,
                rz,
            };
        }

        const tris = [];
        for (let i = 0; i < state.faces.length; i += 3) {
            const ia = state.faces[i + 0];
            const ib = state.faces[i + 1];
            const ic = state.faces[i + 2];
            const a = projected[ia];
            const b = projected[ib];
            const c = projected[ic];
            const ux = b.rx - a.rx;
            const uy = b.ry - a.ry;
            const uz = b.rz - a.rz;
            const vx = c.rx - a.rx;
            const vy = c.ry - a.ry;
            const vz = c.rz - a.rz;
            const nx = uy * vz - uz * vy;
            const ny = uz * vx - ux * vz;
            const nz = ux * vy - uy * vx;
            const nl = Math.hypot(nx, ny, nz) || 1.0;
            const ndotl = Math.max(0.0, (nx * light[0] + ny * light[1] + nz * light[2]) / nl);
            if (nz >= 0) continue;
            tris.push({
                depth: (a.z + b.z + c.z) / 3,
                a, b, c,
                shade: 0.35 + ndotl * 0.65,
            });
        }

        tris.sort((lhs, rhs) => lhs.depth - rhs.depth);
        for (const tri of tris) {
            const shade = Math.max(0, Math.min(255, Math.round(190 * tri.shade)));
            ctx.beginPath();
            ctx.moveTo(tri.a.x, tri.a.y);
            ctx.lineTo(tri.b.x, tri.b.y);
            ctx.lineTo(tri.c.x, tri.c.y);
            ctx.closePath();
            ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
            ctx.fill();
        }
    }

    canvas.addEventListener("pointerdown", (event) => {
        state.dragging = true;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        canvas.setPointerCapture?.(event.pointerId);
        event.stopPropagation();
    });
    canvas.addEventListener("pointermove", (event) => {
        if (!state.dragging) return;
        const dx = event.clientX - state.lastX;
        const dy = event.clientY - state.lastY;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        state.yaw += dx * 0.01;
        state.pitch = Math.max(-1.2, Math.min(1.2, state.pitch + dy * 0.01));
        render();
        event.stopPropagation();
    });
    canvas.addEventListener("pointerup", (event) => {
        state.dragging = false;
        canvas.releasePointerCapture?.(event.pointerId);
        event.stopPropagation();
    });
    canvas.addEventListener("wheel", (event) => {
        state.zoom = Math.max(0.6, Math.min(3.0, state.zoom * (event.deltaY > 0 ? 0.92 : 1.08)));
        render();
        event.preventDefault();
        event.stopPropagation();
    }, { passive: false });

    render();
    return {
        setMesh(verts, faces) {
            state.verts = verts;
            state.faces = faces;
            render();
        },
        resetView() {
            state.yaw = 0.0;
            state.pitch = 0.0;
            state.zoom = 1.4;
            render();
            statusEl.textContent = "View reset.";
        },
    };
}

async function requestForward(node) {
    const cfg = getLoadFlameModelConfig(node);
        const response = await api.fetchApi("/flame/forward", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            gender: cfg.gender,
            shape_dim: cfg.shape_dim,
            expr_dim: cfg.expr_dim,
            params: node._flameParams || parseParams(node),
        }),
    });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data?.error || "FLAME forward failed");
    }
    return data;
}

function attachFlameEditor(node) {
    if (node._flameAttached) return;
    node._flameAttached = true;
    hideParamsWidget(node);

    const container = document.createElement("div");
    Object.assign(container.style, {
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        padding: "10px",
        background: "linear-gradient(180deg, #1f2328 0%, #171a1f 100%)",
        border: "1px solid #2e333b",
        borderRadius: "10px",
        color: "#f0f0f0",
    });

    const status = document.createElement("div");
    Object.assign(status.style, {
        fontSize: "11px",
        color: "#98a2ad",
        minHeight: "16px",
    });
    status.textContent = "Select a FLAME model variant to initialize the viewer.";

    const canvas = document.createElement("canvas");
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    Object.assign(canvas.style, {
        width: "100%",
        height: `${CANVAS_SIZE}px`,
        background: "#7d848c",
        borderRadius: "8px",
        border: "1px solid #424852",
        touchAction: "none",
    });

    const buttonRow = document.createElement("div");
    Object.assign(buttonRow.style, {
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: "8px",
    });

    function makeButton(label) {
        const button = document.createElement("button");
        button.textContent = label;
        Object.assign(button.style, {
            padding: "6px 8px",
            borderRadius: "6px",
            border: "1px solid #4a525e",
            background: "#252b33",
            color: "#f5f7fa",
            cursor: "pointer",
            fontSize: "11px",
        });
        return button;
    }

    const resetButton = makeButton("Reset");
    const copyButton = makeButton("Copy JSON");
    const reloadButton = makeButton("Reload");
    const viewButton = makeButton("View");

    const sliderWrap = document.createElement("div");
    Object.assign(sliderWrap.style, {
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        maxHeight: "430px",
        overflowY: "auto",
        paddingRight: "4px",
    });

    const viewer = createViewer(canvas, status);
    node._flameViewer = viewer;

    const widget = node.addDOMWidget("flame_preview", `flameeditor${node.id}`, container, {
        serialize: false,
        hideOnZoom: false,
        getValue() { return ""; },
        setValue() {},
    });
    widget.computeSize = (width) => [Math.max(width || NODE_WIDTH, NODE_WIDTH), NODE_HEIGHT];
    widget.element = container;

    async function ensureMesh() {
        const cfg = getLoadFlameModelConfig(node);
        const topology = await getTopology(cfg);
        node._flameTopology = topology;
        return topology;
    }

    async function updateMesh() {
        const token = (node._flamePreviewToken || 0) + 1;
        node._flamePreviewToken = token;
        const topology = await ensureMesh();
        const data = await requestForward(node);
        if (node._flamePreviewToken !== token) return;
        const verts = new Float32Array(decodeBase64ToTypedArray(data.verts_b64, Float32Array));
        viewer.setMesh(verts, topology.faces);
        status.textContent = `${data.gender} | ${topology.n_vertices} verts | drag to orbit`;
    }

    node._flameDebouncedPreview = debounce(async () => {
        try {
            await updateMesh();
        } catch (error) {
            status.textContent = error?.message || "FLAME preview failed";
        }
    }, 70);

    function rebuild() {
        const params = parseParams(node);
        node._flameParams = params;
        sliderWrap.replaceChildren();
        const sections = [
            {
                title: "Shape",
                labels: Array.from({ length: Math.min(params.shape.length, VISIBLE_SHAPE) }, (_, i) => `shape_${i + 1}`),
                values: params.shape,
                visible: Math.min(params.shape.length, VISIBLE_SHAPE),
                min: -3,
                max: 3,
                step: 0.01,
                apply: (index, value) => { params.shape[index] = value; },
            },
            {
                title: "Expression",
                labels: Array.from({ length: Math.min(params.expr.length, VISIBLE_EXPR) }, (_, i) => `expr_${i + 1}`),
                values: params.expr,
                visible: Math.min(params.expr.length, VISIBLE_EXPR),
                min: -3,
                max: 3,
                step: 0.01,
                apply: (index, value) => { params.expr[index] = value; },
            },
            {
                title: "Pose",
                labels: POSE_LABELS,
                values: params.pose,
                visible: params.pose.length,
                min: -0.8,
                max: 0.8,
                step: 0.01,
                apply: (index, value) => { params.pose[index] = value; },
            },
            {
                title: "Translation",
                labels: TRANS_LABELS,
                values: params.trans,
                visible: params.trans.length,
                min: -1,
                max: 1,
                step: 0.01,
                apply: (index, value) => { params.trans[index] = value; },
            },
        ];

        for (const section of sections) {
            const panel = document.createElement("div");
            Object.assign(panel.style, {
                display: "flex",
                flexDirection: "column",
                gap: "6px",
                padding: "8px",
                background: "#20252c",
                border: "1px solid #303741",
                borderRadius: "8px",
            });

            const heading = document.createElement("div");
            heading.textContent = section.title;
            heading.style.fontSize = "12px";
            heading.style.fontWeight = "600";
            panel.appendChild(heading);

            for (let i = 0; i < section.visible; i++) {
                panel.appendChild(buildSlider({
                    label: section.labels[i],
                    min: section.min,
                    max: section.max,
                    step: section.step,
                    value: section.values[i] ?? 0,
                    onInput: (value) => {
                        section.apply(i, value);
                        writeParams(node, params);
                        node._flameLocallyEdited = true;
                        node._flameDebouncedPreview();
                    },
                }));
            }

            if (section.visible < section.values.length) {
                const note = document.createElement("div");
                note.textContent = `Showing first ${section.visible} of ${section.values.length}`;
                note.style.fontSize = "10px";
                note.style.color = "#8994a3";
                panel.appendChild(note);
            }
            sliderWrap.appendChild(panel);
        }
    }

    async function reloadFromInput() {
        const seeded = resolveUpstreamParams(node);
        if (!seeded) {
            status.textContent = "No live upstream FLAME editor state available.";
            return;
        }
        node._flameLocallyEdited = false;
        writeParams(node, seeded);
        rebuild();
        node._flameDebouncedPreview();
        status.textContent = "Reloaded params from connected input editor.";
    }

    resetButton.addEventListener("click", () => {
        const cfg = getLoadFlameModelConfig(node);
        writeParams(node, defaultParams(cfg.shape_dim, cfg.expr_dim));
        node._flameLocallyEdited = true;
        rebuild();
        node._flameDebouncedPreview();
    });
    copyButton.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(getWidget(node, "params_json")?.value || "{}");
            status.textContent = "Copied FLAME params JSON.";
        } catch {
            status.textContent = "Clipboard write failed.";
        }
    });
    reloadButton.addEventListener("click", () => {
        reloadFromInput();
    });
    viewButton.addEventListener("click", () => {
        viewer.resetView();
    });

    container.append(status, canvas, buttonRow, sliderWrap);
    buttonRow.append(resetButton, copyButton, reloadButton, viewButton);

    const originalOnConfigure = node.onConfigure?.bind(node);
    node.onConfigure = function (info) {
        originalOnConfigure?.(info);
        hideParamsWidget(node);
        rebuild();
        node._flameDebouncedPreview();
    };

    const originalOnConnectionsChange = node.onConnectionsChange?.bind(node);
    node.onConnectionsChange = function (type, slotIndex, connected, linkInfo, ioSlot) {
        const result = originalOnConnectionsChange?.apply(this, arguments);
        if (type === LiteGraph.INPUT && ioSlot?.name === "flame_params_in" && connected) {
            const seeded = resolveUpstreamParams(node);
            if (seeded && !node._flameLocallyEdited) {
                writeParams(node, seeded);
                rebuild();
                node._flameDebouncedPreview();
                status.textContent = "Seeded from connected FLAME params input.";
            } else if (!seeded) {
                status.textContent = "Connected input; live reload works when the source is another FLAME editor.";
            }
        }
        return result;
    };

    const originalOnRemoved = node.onRemoved?.bind(node);
    node.onRemoved = function () {
        originalOnRemoved?.();
        node._flamePreviewToken = (node._flamePreviewToken || 0) + 1;
    };

    for (const widgetDef of ["gender", "shape_dim", "expr_dim"]) {
        const w = getWidget(node, widgetDef);
        if (w) {
            const orig = w.callback;
            w.callback = function () {
                orig?.apply(this, arguments);
                const next = normalizeParams(node, node._flameParams || parseParams(node));
                writeParams(node, next);
                rebuild();
                node._flameDebouncedPreview();
            };
        }
    }

    if (typeof node.setSize === "function") {
        node.setSize([
            Math.max(node.size?.[0] ?? 0, NODE_WIDTH),
            Math.max(node.size?.[1] ?? 0, NODE_HEIGHT + 20),
        ]);
    }

    rebuild();
    node._flameDebouncedPreview();
}

app.registerExtension({
    name: "comfyui.flame.editor",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FlameEditor") return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            attachFlameEditor(this);
        };
    },

    nodeCreated(node) {
        if (node.comfyClass === "FlameEditor") {
            attachFlameEditor(node);
        }
    },
});
