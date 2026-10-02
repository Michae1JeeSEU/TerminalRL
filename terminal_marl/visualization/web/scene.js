/*
 * Renderer-only Canvas2D scene.  The class deliberately exposes one default
 * export so the no-build offline player can inline this module with a simple
 * import/export rewrite.
 */
export default class Scene {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
    this.frame = null;
    this.referenceFrame = null;
    this.history = new Map();
    this.selectedUid = null;
    this.hoverUid = null;
    this.layers = {
      rings: true,
      targetRings: true,
      gates: true,
      landingZones: true,
      grid: true,
      labels: true,
      trails: true,
      committed: true,
      conflicts: true,
    };
    this.camera = {
      yaw: -0.72,
      pitch: -0.54,
      zoom: 1,
      panX: 0,
      panY: 0,
      projection: "perspective",
      heightScale: 1.6,
    };
    this.palette = ["#5aa8ff", "#25d3c6", "#f5b64e", "#c68bff", "#ec7592", "#6bdd9a", "#77d9e9", "#ff9d66"];
    this.metrics = { cssWidth: 0, cssHeight: 0, dpr: 1, scale: 1, extent: 100 };
    this.onSelect = typeof options.onSelect === "function" ? options.onSelect : () => {};
    this.onHover = typeof options.onHover === "function" ? options.onHover : () => {};
    this.drag = null;
    this.lastRender = 0;
    this._bindPointer();
    this._resizeObserver = typeof ResizeObserver !== "undefined"
      ? new ResizeObserver(() => this.render())
      : null;
    if (this._resizeObserver) this._resizeObserver.observe(canvas);
    this.render();
  }

  setFrame(frame, options = {}) {
    this.frame = frame || null;
    if (Object.prototype.hasOwnProperty.call(options, "referenceFrame")) this.referenceFrame = options.referenceFrame || null;
    if (options.history) this.history = options.history;
    if (options.layers) this.layers = { ...this.layers, ...options.layers };
    if (Object.prototype.hasOwnProperty.call(options, "selectedUid")) this.selectedUid = options.selectedUid;
    if (Object.prototype.hasOwnProperty.call(options, "hoverUid")) this.hoverUid = options.hoverUid;
    if (Number.isFinite(options.heightScale)) this.camera.heightScale = options.heightScale;
    this.render();
  }

  setLayers(layers = {}) { this.layers = { ...this.layers, ...layers }; this.render(); }

  setHeightScale(value) {
    if (Number.isFinite(Number(value))) this.camera.heightScale = Math.max(.2, Math.min(4, Number(value)));
    this.render();
  }

  setProjection(projection) {
    this.camera.projection = projection === "top" ? "top" : "perspective";
    if (this.camera.projection === "top") this.camera.pitch = -0.03;
    else if (Math.abs(this.camera.pitch) < 0.1) this.camera.pitch = -0.54;
    this.render();
  }

  resetCamera() {
    this.camera.yaw = -0.72;
    this.camera.pitch = -0.54;
    this.camera.zoom = 1;
    this.camera.panX = 0;
    this.camera.panY = 0;
    this.camera.projection = "perspective";
    this.render();
  }

  focusOn(uid, frame = this.frame) {
    const target = frame && Array.isArray(frame.uavs) ? frame.uavs.find((u) => Number(u.uid) === Number(uid)) : null;
    if (!target || !Array.isArray(target.position)) return false;
    const p = this._project(target.position, true);
    this.camera.panX += this.metrics.cssWidth * .5 - p.x;
    this.camera.panY += this.metrics.cssHeight * .5 - p.y;
    this.render();
    return true;
  }

  focusPoint(point) {
    if (!this._finitePoint(point)) return false;
    const projected = this._project(point, true);
    this.camera.panX += this.metrics.cssWidth * .5 - projected.x;
    this.camera.panY += this.metrics.cssHeight * .5 - projected.y;
    this.render();
    return true;
  }

  toDataURL() { return this.canvas.toDataURL("image/png"); }

  destroy() {
    if (this._resizeObserver) this._resizeObserver.disconnect();
    this.canvas.removeEventListener("pointerdown", this._pointerDown);
    this.canvas.removeEventListener("pointermove", this._pointerMove);
    this.canvas.removeEventListener("pointerup", this._pointerUp);
    this.canvas.removeEventListener("pointercancel", this._pointerUp);
    this.canvas.removeEventListener("pointerleave", this._pointerLeave);
    this.canvas.removeEventListener("wheel", this._wheel, { passive: false });
  }

  _bindPointer() {
    this._pointerDown = (event) => {
      if (event.button !== 0 && event.button !== 1) return;
      this.canvas.setPointerCapture?.(event.pointerId);
      this.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false, shift: event.shiftKey };
      this.canvas.classList.add("dragging");
    };
    this._pointerMove = (event) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const uid = this.pick(x, y);
      if (uid !== this.hoverUid) { this.hoverUid = uid; this.onHover(uid); this.render(); }
      if (!this.drag || this.drag.id !== event.pointerId) return;
      const dx = event.clientX - this.drag.x;
      const dy = event.clientY - this.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) this.drag.moved = true;
      if (this.drag.moved) {
        if (this.drag.shift || event.shiftKey) {
          this.camera.panX += dx;
          this.camera.panY += dy;
        } else {
          this.camera.yaw += dx * .008;
          this.camera.pitch = Math.max(-1.35, Math.min(.25, this.camera.pitch + dy * .006));
          this.camera.projection = "perspective";
        }
        this.drag.x = event.clientX;
        this.drag.y = event.clientY;
        this.render();
      }
    };
    this._pointerUp = (event) => {
      if (!this.drag || this.drag.id !== event.pointerId) return;
      const wasMoved = this.drag.moved;
      this.drag = null;
      this.canvas.classList.remove("dragging");
      if (!wasMoved) {
        const rect = this.canvas.getBoundingClientRect();
        this.onSelect(this.pick(event.clientX - rect.left, event.clientY - rect.top));
      }
    };
    this._pointerLeave = () => {
      if (this.hoverUid !== null) { this.hoverUid = null; this.onHover(null); this.render(); }
    };
    this._wheel = (event) => {
      event.preventDefault();
      const factor = event.deltaY > 0 ? .9 : 1.1;
      this.camera.zoom = Math.max(.35, Math.min(5, this.camera.zoom * factor));
      this.render();
    };
    this.canvas.addEventListener("pointerdown", this._pointerDown);
    this.canvas.addEventListener("pointermove", this._pointerMove);
    this.canvas.addEventListener("pointerup", this._pointerUp);
    this.canvas.addEventListener("pointercancel", this._pointerUp);
    this.canvas.addEventListener("pointerleave", this._pointerLeave);
    this.canvas.addEventListener("wheel", this._wheel, { passive: false });
  }

  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(2.5, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height || this.metrics.dpr !== dpr || this.metrics.cssWidth !== rect.width || this.metrics.cssHeight !== rect.height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.metrics = { ...this.metrics, cssWidth: rect.width, cssHeight: rect.height, dpr, width, height };
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width: Math.max(1, rect.width), height: Math.max(1, rect.height) };
  }

  _finitePoint(point) {
    return Array.isArray(point) && point.length >= 3 && point.slice(0, 3).every((v) => Number.isFinite(Number(v)));
  }

  _bounds(frame) {
    let ringExtent = 0;
    const points = [];
    const add = (point) => { if (this._finitePoint(point)) points.push(point); };
    for (const u of frame?.uavs || []) add(u.position);
    for (const g of frame?.gates || []) add(g.gate_position);
    for (const z of frame?.landing_zones || []) add(z.position);
    for (const r of frame?.rings || []) {
      ringExtent = Math.max(ringExtent, Math.abs(Number(r.radius) || 0), Math.abs(Number(r.target_radius) || 0));
    }
    const terminalExtent = Math.max(120, ringExtent * 1.4);
    let extent = terminalExtent;
    if (ringExtent > 0) return { extent };
    for (const p of points) {
      const horizontal = Math.max(Math.abs(Number(p[0]) || 0), Math.abs(Number(p[1]) || 0));
      // Gate spawn positions can be kilometres away from the terminal.  The
      // gate itself and active UAVs still contribute when they are near the
      // terminal envelope; a remote spawn marker must not zoom the scene out.
      if (horizontal <= terminalExtent * 2.8) extent = Math.max(extent, horizontal, Math.abs(Number(p[2]) || 0) * .5);
    }
    return { extent: Math.max(80, extent) };
  }

  _project(point, includeScale = false) {
    const [x, y, z] = this._finitePoint(point) ? point.map(Number) : [0, 0, 0];
    const yaw = this.camera.yaw;
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const rx = x * c - y * s;
    const ry = x * s + y * c;
    const rz = z * this.camera.heightScale;
    const pitch = this.camera.pitch;
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const projectedY = ry * cp - rz * sp;
    const depth = ry * sp + rz * cp;
    let perspective = 1;
    if (this.camera.projection === "perspective") perspective = Math.max(.72, Math.min(1.28, 1 + depth / (this.metrics.extent * 7)));
    const xOut = this.metrics.cssWidth * .5 + rx * this.metrics.scale * perspective + this.camera.panX;
    const yOut = this.metrics.cssHeight * .52 - projectedY * this.metrics.scale * perspective + this.camera.panY;
    return { x: xOut, y: yOut, depth, factor: perspective, worldX: rx, worldY: ry, worldZ: rz, ...(includeScale ? { scale: this.metrics.scale } : {}) };
  }

  _depth(point) { return this._project(point).depth; }

  _path(points, style, width = 1, closed = false, dash = []) {
    if (!Array.isArray(points) || points.length < 2) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = style;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.setLineDash(dash);
    ctx.beginPath();
    points.forEach((point, index) => { const p = this._project(point); index ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); });
    if (closed) ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }

  _circlePoints(radius, height, count = 96) {
    const points = [];
    for (let i = 0; i <= count; i++) { const a = (i / count) * Math.PI * 2; points.push([radius * Math.cos(a), radius * Math.sin(a), height]); }
    return points;
  }

  _drawGrid(width, height, extent) {
    const ctx = this.ctx;
    ctx.save();
    const step = this._gridStep(extent);
    const limit = Math.ceil(extent / step) * step;
    for (let x = -limit; x <= limit; x += step) this._path([[x, -limit, 0], [x, limit, 0]], x === 0 ? "rgba(103,159,201,.27)" : "rgba(79,125,162,.14)", x === 0 ? 1 : .65);
    for (let y = -limit; y <= limit; y += step) this._path([[-limit, y, 0], [limit, y, 0]], y === 0 ? "rgba(103,159,201,.27)" : "rgba(79,125,162,.14)", y === 0 ? 1 : .65);
    ctx.restore();
  }

  _gridStep(extent) {
    const raw = extent / 6;
    const power = Math.pow(10, Math.floor(Math.log10(Math.max(1, raw))));
    const normalized = raw / power;
    return (normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1) * power;
  }

  _drawRing(ring, index) {
    const radius = Math.max(0, Number(ring.radius) || 0);
    const height = Number(ring.height) || 0;
    const targetRadius = Math.max(0, Number(ring.target_radius) || 0);
    const targetHeight = Number(ring.target_height) || 0;
    const color = this.palette[index % this.palette.length];
    const outer = this._circlePoints(radius + Math.max(2, radius * .035), height);
    const inner = this._circlePoints(Math.max(0, radius - Math.max(2, radius * .035)), height);
    const ctx = this.ctx;
    if (this.layers.rings && radius > 0) {
      const projectedOuter = outer.map((p) => this._project(p));
      const projectedInner = inner.map((p) => this._project(p));
      ctx.save();
      ctx.beginPath();
      projectedOuter.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
      projectedInner.slice().reverse().forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.closePath();
      ctx.fillStyle = `${color}18`;
      ctx.fill();
      ctx.restore();
      this._path(this._circlePoints(radius, height), color, 1.35);
      this._path(this._circlePoints(radius, 0), `${color}48`, .75, false, [3, 5]);
      this._path([[radius, 0, 0], [radius, 0, height]], `${color}7a`, .75, false, [2, 4]);
    }
    if (this.layers.targetRings && targetRadius > 0 && (Math.abs(targetRadius - radius) > .01 || Math.abs(targetHeight - height) > .01)) {
      this._path(this._circlePoints(targetRadius, targetHeight), "#f5b64eaa", 1, false, [5, 4]);
      this._path(this._circlePoints(targetRadius, 0), "#f5b64e4a", .6, false, [2, 5]);
    }
    if (this.layers.labels) {
      const p = this._project([radius, 0, height]);
      ctx.save();
      ctx.font = "10px Inter, system-ui, sans-serif";
      ctx.fillStyle = color;
      ctx.globalAlpha = .95;
      ctx.fillText(`${String(ring.role ?? "ring")}  ${Number(ring.occupancy) || 0}/${Number(ring.capacity) || 0}`, p.x + 6, p.y - 6);
      ctx.restore();
    }
  }

  _drawGate(gate) {
    const a = gate.spawn_position, b = gate.gate_position;
    if (!this._finitePoint(a) || !this._finitePoint(b)) return;
    this._path([a, b], "#849bb0c9", 1.2, false, [4, 4]);
    const p = this._project(b);
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(Math.atan2(p.y - this._project(a).y, p.x - this._project(a).x));
    ctx.fillStyle = "#a7bfd3";
    ctx.beginPath(); ctx.moveTo(7, 0); ctx.lineTo(-4, -4); ctx.lineTo(-2, 0); ctx.lineTo(-4, 4); ctx.closePath(); ctx.fill();
    ctx.restore();
    if (this.layers.labels) { ctx.save(); ctx.font = "9px Inter, system-ui, sans-serif"; ctx.fillStyle = "#9db4c8"; ctx.fillText(`GATE ${gate.gate_id ?? "—"}`, p.x + 8, p.y - 6); ctx.restore(); }
  }

  _drawLandingZone(zone) {
    if (!this._finitePoint(zone.position)) return;
    const p = this._project(zone.position);
    const status = String(zone.status || "unknown").toLowerCase();
    const color = status === "free" ? "#52d99b" : status === "occupied" ? "#fc6f78" : "#f5b64e";
    const size = Math.max(4, Math.min(10, this.metrics.scale * 2.4));
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = `${color}26`;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25;
    ctx.fillRect(-size, -size, size * 2, size * 2);
    ctx.strokeRect(-size, -size, size * 2, size * 2);
    ctx.restore();
    if (this.layers.labels) { ctx.save(); ctx.font = "9px Inter, system-ui, sans-serif"; ctx.fillStyle = color; ctx.fillText(`LZ ${zone.lz_id ?? "—"} · ${status}`, p.x + 8, p.y + 3); ctx.restore(); }
  }

  _drawTrail(uav, points) {
    if (!Array.isArray(points) || points.length < 2) return;
    const color = `${this._colorFor(uav.uid)}88`;
    this._path(points, color, 1.2);
    const last = points[points.length - 1];
    if (this._finitePoint(last)) {
      const p = this._project(last);
      this.ctx.save(); this.ctx.fillStyle = color; this.ctx.globalAlpha = .35; this.ctx.beginPath(); this.ctx.arc(p.x, p.y, 2, 0, Math.PI * 2); this.ctx.fill(); this.ctx.restore();
    }
  }

  _drawCommitted(uav) {
    const points = Array.isArray(uav.trajectory_points) ? uav.trajectory_points.filter((p) => this._finitePoint(p)) : [];
    if (points.length < 2) return;
    const selected = this.selectedUid != null && Number(uav.uid) === Number(this.selectedUid);
    this._path(points, selected ? this._colorFor(uav.uid) : `${this._colorFor(uav.uid)}6e`, selected ? 1.7 : .95, false, [7, 4]);
    const end = this._project(points[points.length - 1]);
    this.ctx.save(); this.ctx.strokeStyle = this._colorFor(uav.uid); this.ctx.globalAlpha = .7; this.ctx.lineWidth = 1; this.ctx.beginPath(); this.ctx.arc(end.x, end.y, 3, 0, Math.PI * 2); this.ctx.stroke(); this.ctx.restore();
  }

  _drawReferenceUav(uav) {
    if (!this._finitePoint(uav.position)) return;
    const p = this._project(uav.position);
    const color = this._colorFor(uav.uid);
    const size = 6;
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.strokeStyle = `${color}b8`;
    ctx.fillStyle = `${color}12`;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(-size, 0); ctx.lineTo(size, 0); ctx.moveTo(0, -size); ctx.lineTo(0, size); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(0, 0, size * .72, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.restore();
    if (this.layers.labels) { ctx.save(); ctx.font = "9px Inter, system-ui, sans-serif"; ctx.fillStyle = `${color}bb`; ctx.fillText(`R${uav.uid}`, p.x + 9, p.y + 3); ctx.restore(); }
  }

  _drawReferenceFrame(frame) {
    if (!frame) return;
    const ctx = this.ctx;
    for (const uav of frame.uavs || []) {
      const points = Array.isArray(uav.trajectory_points) ? uav.trajectory_points.filter((p) => this._finitePoint(p)) : [];
      if (points.length >= 2) this._path(points, `${this._colorFor(uav.uid)}82`, 1.1, false, [3, 4]);
    }
    const refs = Array.isArray(frame.uavs) ? frame.uavs.slice().sort((a, b) => this._depth(a.position) - this._depth(b.position)) : [];
    refs.forEach((uav) => this._drawReferenceUav(uav));
    if (refs.length) {
      ctx.save(); ctx.fillStyle = "#f5b64e"; ctx.font = "8px Inter, system-ui, sans-serif"; ctx.fillText("REF", 12, 24); ctx.restore();
    }
  }

  _drawTerminalEvents(frame) {
    const events = Array.isArray(frame?.terminal_events) ? frame.terminal_events : [];
    for (const event of events) {
      const outcome = `${event?.outcome || ""} ${event?.reason || ""}`.toLowerCase();
      if (!outcome.includes("failed") || !this._finitePoint(event?.position)) continue;
      const p = this._project(event.position);
      const ctx = this.ctx;
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.strokeStyle = "#fc6f78";
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(-6, -6); ctx.lineTo(6, 6); ctx.moveTo(6, -6); ctx.lineTo(-6, 6); ctx.stroke();
      if (this.layers.labels) { ctx.font = "8px Inter, system-ui, sans-serif"; ctx.fillStyle = "#ffb1b6"; ctx.fillText(String(event.outcome || event.reason || "failed"), 9, 3); }
      ctx.restore();
    }
  }

  _drawUav(uav, conflict, showLabel = true) {
    if (!this._finitePoint(uav.position)) return;
    const p = this._project(uav.position);
    const color = this._colorFor(uav.uid);
    const selected = this.selectedUid != null && Number(uav.uid) === Number(this.selectedUid);
    const hovered = this.hoverUid != null && Number(uav.uid) === Number(this.hoverUid);
    const heading = this._heading(uav);
    const size = selected || hovered ? 9 : 7;
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(heading);
    if (selected) { ctx.strokeStyle = "#e9ffff"; ctx.globalAlpha = .95; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(0, 0, size + 6, 0, Math.PI * 2); ctx.stroke(); }
    if (conflict && this.layers.conflicts) { ctx.strokeStyle = "#fc6f78"; ctx.globalAlpha = .88; ctx.lineWidth = 1.4; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(0, 0, size + 4, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]); }
    ctx.strokeStyle = color; ctx.globalAlpha = .82; ctx.lineWidth = 1.5;
    const arm = size * .85;
    ctx.beginPath(); ctx.moveTo(-arm, -arm); ctx.lineTo(arm, arm); ctx.moveTo(-arm, arm); ctx.lineTo(arm, -arm); ctx.stroke();
    ctx.fillStyle = `${color}32`;
    for (const [x, y] of [[-arm, -arm], [arm, arm], [-arm, arm], [arm, -arm]]) { ctx.beginPath(); ctx.arc(x, y, size * .48, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(0, 0, 2.6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#eefaff"; ctx.beginPath(); ctx.moveTo(size + 5, 0); ctx.lineTo(size - 1, -3); ctx.lineTo(size - 1, 3); ctx.closePath(); ctx.fill();
    ctx.restore();
    if (showLabel || selected || hovered) {
      ctx.save(); ctx.font = `${selected ? "600 " : ""}10px Inter, system-ui, sans-serif`; ctx.fillStyle = selected ? "#f2ffff" : "#b3c8dc"; ctx.fillText(`U${uav.uid}`, p.x + 12, p.y - 9); ctx.restore();
    }
  }

  _heading(uav) {
    const v = Array.isArray(uav.velocity) ? uav.velocity : [0, 0, 0];
    const vx = Number(v[0]) || 0, vy = Number(v[1]) || 0;
    if (Math.abs(vx) + Math.abs(vy) < .001) return -this.camera.yaw;
    const c = Math.cos(this.camera.yaw), s = Math.sin(this.camera.yaw);
    return Math.atan2(-(vx * s + vy * c), vx * c - vy * s);
  }

  _colorFor(uid) { const n = Math.abs(Number(uid) || 0); return this.palette[n % this.palette.length]; }

  _drawAxis() {
    const baseX = 44, baseY = this.metrics.cssHeight - 45, len = 28;
    const ctx = this.ctx;
    ctx.save(); ctx.lineWidth = 1.1; ctx.font = "9px Inter, system-ui, sans-serif";
    const axis = [["X", this.camera.yaw, "#f27886"], ["Y", this.camera.yaw - Math.PI / 2, "#5fd9a0"], ["Z", -Math.PI / 2 - this.camera.pitch, "#64a9ff"]];
    axis.forEach(([label, angle, color]) => { ctx.strokeStyle = color; ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(baseX, baseY); ctx.lineTo(baseX + Math.cos(angle) * len, baseY + Math.sin(angle) * len); ctx.stroke(); ctx.fillText(label, baseX + Math.cos(angle) * (len + 6) - 2, baseY + Math.sin(angle) * (len + 6) + 3); });
    ctx.restore();
  }

  render() {
    const { width, height } = this._resize();
    const ctx = this.ctx;
    ctx.clearRect(0, 0, width, height);
    const gradient = ctx.createRadialGradient(width * .52, height * .4, 5, width * .52, height * .48, Math.max(width, height) * .76);
    gradient.addColorStop(0, "#112a42"); gradient.addColorStop(1, "#071420"); ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
    if (!this.frame) { this._drawAxis(); this.lastRender = performance.now(); return; }
    const bounds = this._bounds(this.frame);
    this.metrics.extent = bounds.extent;
    this.metrics.scale = Math.min(width, height) * .52 / bounds.extent * this.camera.zoom;
    if (this.layers.grid) this._drawGrid(width, height, bounds.extent);
    const rings = Array.isArray(this.frame.rings) ? this.frame.rings : [];
    if (this.layers.rings || this.layers.targetRings) rings.map((r, i) => ({ r, i, d: this._depth([Number(r.radius) || 0, 0, Number(r.height) || 0]) })).sort((a, b) => a.d - b.d).forEach(({ r, i }) => this._drawRing(r, i));
    if (this.layers.gates) (this.frame.gates || []).slice().sort((a, b) => this._depth(a.gate_position) - this._depth(b.gate_position)).forEach((gate) => this._drawGate(gate));
    if (this.layers.landingZones) (this.frame.landing_zones || []).slice().sort((a, b) => this._depth(a.position) - this._depth(b.position)).forEach((zone) => this._drawLandingZone(zone));
    this._drawReferenceFrame(this.referenceFrame);
    this._drawTerminalEvents(this.frame);
    const conflicts = new Set((this.frame.recent_conflict_uids || []).map((uid) => Number(uid)));
    const uavs = Array.isArray(this.frame.uavs) ? this.frame.uavs.slice() : [];
    if (this.layers.trails) uavs.forEach((uav) => this._drawTrail(uav, this.history instanceof Map ? this.history.get(`${uav.uid}:${uav.phase}`) || this.history.get(Number(uav.uid)) : null));
    if (this.layers.committed) uavs.forEach((uav) => this._drawCommitted(uav));
    const showUavLabels = this.layers.labels && uavs.length <= 8;
    uavs.sort((a, b) => this._depth(a.position) - this._depth(b.position)).forEach((uav) => this._drawUav(uav, conflicts.has(Number(uav.uid)), showUavLabels));
    if (this.layers.labels) this._drawAxis();
    this.lastRender = performance.now();
  }

  pick(x, y) {
    if (!this.frame || !Array.isArray(this.frame.uavs)) return null;
    let hit = null, best = 18;
    for (const uav of this.frame.uavs) {
      const p = this._project(uav.position);
      const distance = Math.hypot(x - p.x, y - p.y);
      if (distance <= best) { best = distance; hit = uav.uid; }
    }
    return hit;
  }
}
