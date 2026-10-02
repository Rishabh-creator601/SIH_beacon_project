"""Performance report generation: report.html (interactive) and report.pdf.

Both contain
  * run information (seed, scene, identification method, settings)
  * the performance table required by the problem statement: simulation
    duration, FPS, acquisition time, average / maximum tracking error, lock
    retention, processing time (plus time locked, false locks, handover ...)
  * a 3D VIEW of the run on the sky dome: the beacon's trajectory, where the
    camera looked (coloured by tracker state) and the tracker's estimate
  * a 3D azimuth / elevation / time view of beacon vs camera pointing
  * time charts: pointing error with lock periods, tracker state timeline,
    processing time per frame

The HTML is self-contained (the bundled plotly.js file is embedded, so it
works offline) and its 3D views can be rotated and zoomed. Figures are built
as plain plotly.js JSON, so the plotly Python package is not needed. The PDF (ReportLab) uses matplotlib renders
of the same views.
"""

import base64
import html
import io
from datetime import datetime

import numpy as np

R = 100.0
EL0 = 35.0            # same sky placement as the application's 3D view
STATE_ORDER = ["SEARCH", "ACQUIRING", "TRACKING", "COASTING"]
STATE_COLORS = {"SEARCH": "#e04848", "ACQUIRING": "#f08a24", "TRACKING": "#23a55a", "COASTING": "#d9b400"}
MAX_POINTS = 2500     # plots are down-sampled to keep files small


def _xyz(az, el, radius=R):
    a, e = np.radians(np.asarray(az, float)), np.radians(np.asarray(el, float) + EL0)
    return radius * np.sin(a) * np.cos(e), radius * np.cos(a) * np.cos(e), radius * np.sin(e)


def _columns(rows):
    step = max(1, len(rows) // MAX_POINTS)
    rows = rows[::step]
    col = lambda k: np.array([float(r[k]) for r in rows])
    return {
        "t": col("t"), "state": [r["state"] for r in rows], "locked": col("locked").astype(bool),
        "baz": col("beacon_az"), "bel": col("beacon_el"), "laz": col("los_az"), "lel": col("los_el"),
        "eaz": col("est_az"), "eel": col("est_el"), "err": col("err_px"), "proc": col("proc_ms"),
        "occl": col("beacon_occluded").astype(bool),
    }


def _kpis(s):
    f = lambda k, spec="{:.2f}", unit="": ("-" if s.get(k) is None else spec.format(s[k]) + unit)
    return [
        ("Simulation duration", f("simulation_duration_s", "{:.1f}", " s")),
        ("Frames simulated", f("frames", "{}")),
        ("Achieved frame rate", f("achieved_fps", "{:.1f}", " FPS")),
        ("Processing time (mean / max)", f"{f('mean_processing_ms', '{:.2f}')} / {f('max_processing_ms', '{:.2f}')} ms"),
        ("Processing throughput", f("processing_throughput_fps", "{:.0f}", " FPS")),
        ("Acquisition time", f("acquisition_time_s", "{:.2f}", " s") if s.get("acquired") else "not acquired"),
        ("Time locked on beacon (whole run)", f("time_locked_pct", "{:.1f}", " %")),
        ("Lock retention (after first lock)", f("lock_retention_pct", "{:.1f}", " %")),
        ("Lock retention while beacon visible", f("lock_retention_unoccluded_pct", "{:.1f}", " %")),
        ("Lock losses", f("lock_losses", "{}")),
        ("Tracking error mean / RMS / max", f"{f('tracking_error_mean_px')} / {f('tracking_error_rms_px')} / "
                                             f"{f('tracking_error_max_px')} px"),
        ("Tracking error mean / max", f"{f('tracking_error_mean_mrad', '{:.3f}')} / "
                                      f"{f('tracking_error_max_mrad', '{:.3f}')} mrad"),
        ("Ready for fine-pointing handover", f("handover_ready_pct_of_locked", "{:.1f}", " % of locked time")),
        ("False-lock frames", f("false_lock_frames", "{}")),
        ("Candidates rejected / locks dropped", f"{f('candidates_rejected', '{}')} / {f('locks_dropped_by_identity', '{}')}"),
        ("Beacon in field of view / occluded", f"{f('beacon_in_fov_pct', '{:.1f}')} / {f('beacon_occluded_pct', '{:.1f}')} %"),
    ]


def _run_info(s):
    info = [("Generated", s.get("generated", datetime.now().isoformat(timespec="seconds"))),
            ("Seed", str(s.get("seed", "-"))),
            ("Identification method", str(s.get("identification_method", "-")))]
    scene = s.get("random_scene")
    if isinstance(scene, dict):
        info += [("Beacon path", f"{scene['beacon_path']} at {scene['beacon_speed_dps']} deg/s"),
                 ("Decoys", f"{scene['decoys']} ({scene['blinking_decoys']} blinking)"),
                 ("Turbulence / vibration", f"{scene['turbulence_strength']} / {scene['vibration_strength']}"),
                 ("Clouds", "yes" if scene["clouds"] else "no"),
                 ("Reported-position error (GPS prior)", f"{scene['initial_pointing_error_deg']} deg")]
        if "camera_start_deg" in scene:
            info.append(("Camera start", f"az {scene['camera_start_deg'][0]}, el {scene['camera_start_deg'][1]} deg "
                                         f"({scene['camera_to_prior_deg']} deg from the reported position)"))
    else:
        info.append(("Scene", "fixed scenario"))
    return info


# ============================================================ matplotlib (PDF)
def _mpl_figures(c, bounds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

    figs = {}
    # 3D dome view.
    fig = plt.figure(figsize=(8.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    az_min, az_max, el_min, el_max = bounds
    for e in np.linspace(el_min, el_max, 5):
        ax.plot(*_xyz(np.linspace(az_min, az_max, 40), np.full(40, e)), color="#9fb3c8", lw=0.5)
    for a in np.linspace(az_min, az_max, 7):
        ax.plot(*_xyz(np.full(40, a), np.linspace(el_min, el_max, 40)), color="#9fb3c8", lw=0.5)
    ax.plot(*_xyz(c["baz"], c["bel"], R * 0.99), color="#d6284b", lw=2.0, label="beacon trajectory")
    lx, ly, lz = _xyz(c["laz"], c["lel"], R * 0.97)
    for st in STATE_ORDER:
        m = np.array([s == st for s in c["state"]])
        if m.any():
            ax.scatter(lx[m], ly[m], lz[m], s=3, color=STATE_COLORS[st], label=f"camera pointing - {st}")
    ok = ~np.isnan(c["eaz"])
    if ok.any():
        ax.scatter(*_xyz(c["eaz"][ok], c["eel"][ok], R * 0.98), s=1.5, color="#a13ad6", alpha=0.5,
                   label="tracker estimate")
    ax.scatter([0], [0], [0], color="#334", s=60, marker="^", label="PAT terminal")
    for (a0, e0) in ((c["laz"][-1], c["lel"][-1]),):
        ax.plot(*zip((0, 0, 0), _xyz(a0, e0)), color="#1b8fd1", lw=1.2)
    ax.set_title("3D view: beacon trajectory and camera line of sight on the sky dome")
    ax.set_xlabel("East"); ax.set_ylabel("North"); ax.set_zlabel("Up")
    ax.view_init(elev=22, azim=-115)
    ax.legend(loc="upper left", fontsize=7, markerscale=3)
    figs["dome"] = fig

    # 3D az / el / time.
    fig = plt.figure(figsize=(8.2, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(c["t"], c["baz"], c["bel"], color="#d6284b", lw=2, label="beacon")
    ax.plot(c["t"], c["laz"], c["lel"], color="#1b8fd1", lw=1, alpha=0.8, label="camera line of sight")
    lock = np.where(c["locked"], 1.0, np.nan)
    ax.plot(c["t"], c["laz"] * lock, c["lel"] * lock, color="#23a55a", lw=2.5, label="locked")
    ax.set_xlabel("time (s)"); ax.set_ylabel("azimuth (deg)"); ax.set_zlabel("elevation (deg)")
    ax.set_title("3D view: azimuth / elevation over time - beacon vs camera pointing")
    ax.view_init(elev=20, azim=-60)
    ax.legend(fontsize=7)
    figs["azel_time"] = fig

    # Time charts.
    fig, axs = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True,
                            gridspec_kw={"height_ratios": [3, 1, 2]})
    axs[0].semilogy(c["t"], np.maximum(c["err"], 0.1), color="#1b8fd1", lw=1)
    axs[0].fill_between(c["t"], 0.1, np.where(c["locked"], np.maximum(c["err"], 0.1), 0.1),
                        color="#23a55a", alpha=0.25, label="locked")
    axs[0].set_ylabel("pointing error (px)"); axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3)
    idx = np.array([STATE_ORDER.index(s) for s in c["state"]])
    for i, st in enumerate(STATE_ORDER):
        m = idx == i
        axs[1].scatter(c["t"][m], np.full(m.sum(), i), s=4, color=STATE_COLORS[st])
    axs[1].set_yticks(range(4), STATE_ORDER, fontsize=7); axs[1].set_ylabel("state")
    axs[2].plot(c["t"], c["proc"], color="#555", lw=0.8)
    axs[2].set_ylabel("processing (ms)"); axs[2].set_xlabel("time (s)"); axs[2].grid(alpha=0.3)
    fig.suptitle("Tracking performance over time")
    fig.tight_layout()
    figs["charts"] = fig
    return figs


def _fig_png(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def write_pdf(path, s, c, pngs):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            title="FSOC Coarse Tracking - Performance Report")
    width = A4[0] - 3.2 * cm

    def table(rows, header):
        t = Table([header] + [[a, b] for a, b in rows], colWidths=[0.48 * width, 0.52 * width])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2a40")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5d0dc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    def image(png, w):
        img = Image(io.BytesIO(png))
        img.drawWidth, img.drawHeight = w, w * img.imageHeight / img.imageWidth
        return img

    story = [Paragraph("FSOC Coarse Pointing, Acquisition &amp; Tracking - Performance Report", styles["Title"]),
             Spacer(1, 6), table(_run_info(s), ["Run", ""]), Spacer(1, 12),
             Paragraph("Performance summary", styles["Heading2"]), table(_kpis(s), ["Metric", "Value"]),
             PageBreak(),
             Paragraph("3D view of the run", styles["Heading2"]), image(pngs["dome"], width),
             Paragraph("Red: true beacon path. Dots: where the camera pointed, coloured by tracker state "
                       "(red search, orange acquiring, green tracking, yellow coasting). Purple: tracker estimate.",
                       styles["BodyText"]),
             PageBreak(),
             Paragraph("3D view: azimuth / elevation over time", styles["Heading2"]), image(pngs["azel_time"], width),
             Paragraph("The camera line of sight (blue, green while locked) converging onto and following the "
                       "beacon (red) shows acquisition and tracking in one picture.", styles["BodyText"]),
             PageBreak(),
             Paragraph("Time charts", styles["Heading2"]), image(pngs["charts"], width)]
    doc.build(story)


# ============================================================ plotly (HTML)
def _plotly_js():
    """The bundled plotly.js library (MIT licence), or None if missing."""
    from .config import PROJECT_ROOT
    path = PROJECT_ROOT / "fsoc" / "assets" / "plotly.min.js"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _l(a):
    """numpy -> JSON-safe list (NaN -> null, which plotly draws as a gap)."""
    return [None if (isinstance(v, float) and v != v) else v for v in np.asarray(a, float).round(4).tolist()]


def _figures(c, bounds):
    """Figure specs (plotly.js JSON: data + layout) for the HTML report.

    Built as plain dictionaries, so the report needs only the plotly.js file,
    not the (large) plotly Python package.
    """
    az_min, az_max, el_min, el_max = bounds
    dark = dict(paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14", font=dict(color="#d7e3f4"))
    grid_line = dict(color="#34506e", width=2)

    dome = []
    for e in np.linspace(el_min, el_max, 5):
        x, y, z = _xyz(np.linspace(az_min, az_max, 40), np.full(40, e))
        dome.append(dict(type="scatter3d", mode="lines", x=_l(x), y=_l(y), z=_l(z), line=grid_line,
                         hoverinfo="skip", showlegend=False))
    for a in np.linspace(az_min, az_max, 7):
        x, y, z = _xyz(np.full(40, a), np.linspace(el_min, el_max, 40))
        dome.append(dict(type="scatter3d", mode="lines", x=_l(x), y=_l(y), z=_l(z), line=grid_line,
                         hoverinfo="skip", showlegend=False))
    bx, by, bz = _xyz(c["baz"], c["bel"], R * 0.99)
    traces = dome + [dict(type="scatter3d", mode="lines", x=_l(bx), y=_l(by), z=_l(bz),
                          line=dict(color="#ff4d6d", width=6), name="beacon trajectory",
                          text=[f"t={t:.1f}s" for t in c["t"]])]
    lx, ly, lz = _xyz(c["laz"], c["lel"], R * 0.97)
    for st in STATE_ORDER:
        m = np.array([s == st for s in c["state"]])
        if m.any():
            traces.append(dict(type="scatter3d", mode="markers", x=_l(lx[m]), y=_l(ly[m]), z=_l(lz[m]),
                               marker=dict(size=2.2, color=STATE_COLORS[st]), name=f"camera - {st}",
                               text=[f"t={t:.1f}s" for t in c["t"][m]]))
    ok = ~np.isnan(c["eaz"])
    if ok.any():
        ex, ey, ez = _xyz(c["eaz"][ok], c["eel"][ok], R * 0.98)
        traces.append(dict(type="scatter3d", mode="markers", x=_l(ex), y=_l(ey), z=_l(ez),
                           marker=dict(size=1.5, color="#d65cff"), name="tracker estimate", visible="legendonly"))
    x, y, z = _xyz(c["laz"][-1], c["lel"][-1])
    traces.append(dict(type="scatter3d", mode="lines", x=[0, float(x)], y=[0, float(y)], z=[0, float(z)],
                       line=dict(color="#38c6ff", width=4), name="final line of sight"))
    traces.append(dict(type="scatter3d", mode="markers", x=[0], y=[0], z=[0], name="PAT terminal",
                       marker=dict(size=6, color="#9fb7d6", symbol="diamond")))
    base3d = dict(height=620, margin=dict(l=0, r=0, t=40, b=0), legend=dict(font=dict(size=11)), **dark)
    axis = lambda title: dict(title=title, backgroundcolor="#0a0e14", gridcolor="#22314a", zerolinecolor="#22314a")
    dome_fig = dict(data=traces, layout=dict(
        title=dict(text="3D view - beacon trajectory and camera line of sight on the sky dome (drag to rotate)"),
        scene=dict(xaxis=axis("East"), yaxis=axis("North"), zaxis=axis("Up"), aspectmode="data",
                   camera=dict(eye=dict(x=-1.1, y=-1.6, z=0.8))), **base3d))

    lock = np.where(c["locked"], 1.0, np.nan)
    azel = dict(data=[
        dict(type="scatter3d", mode="lines", x=_l(c["t"]), y=_l(c["baz"]), z=_l(c["bel"]),
             line=dict(color="#ff4d6d", width=6), name="beacon"),
        dict(type="scatter3d", mode="lines", x=_l(c["t"]), y=_l(c["laz"]), z=_l(c["lel"]),
             line=dict(color="#38c6ff", width=3), name="camera line of sight"),
        dict(type="scatter3d", mode="lines", x=_l(c["t"]), y=_l(c["laz"] * lock), z=_l(c["lel"] * lock),
             line=dict(color="#35d07f", width=8), name="locked"),
    ], layout=dict(title=dict(text="3D view - azimuth / elevation over time"),
                   scene=dict(xaxis=axis("time (s)"), yaxis=axis("azimuth (deg)"), zaxis=axis("elevation (deg)"),
                              aspectmode="manual", aspectratio=dict(x=2, y=1, z=0.7)), **base3d))

    err = np.maximum(c["err"], 0.1)
    ax2d = dict(gridcolor="#22314a", zerolinecolor="#22314a")
    charts = dict(data=[
        dict(type="scatter", mode="lines", x=_l(c["t"]), y=_l(err), line=dict(color="#38c6ff", width=1),
             name="pointing error"),
        dict(type="scatter", mode="lines", x=_l(c["t"]), y=_l(np.where(c["locked"], err, np.nan)),
             line=dict(color="#35d07f", width=2), name="locked"),
    ], layout=dict(title=dict(text="Pointing error (green = locked)"), height=420,
                   yaxis=dict(type="log", title=dict(text="px"), **ax2d),
                   xaxis=dict(title=dict(text="time (s)"), **ax2d), margin=dict(l=60, r=20, t=50, b=50), **dark))
    states = dict(data=[dict(type="scatter", mode="markers", x=_l(c["t"]), y=list(c["state"]),
                             marker=dict(size=4, color=[STATE_COLORS[s] for s in c["state"]]), name="state")],
                  layout=dict(title=dict(text="Tracker state timeline"), height=220,
                              margin=dict(l=90, r=20, t=50, b=40), xaxis=ax2d,
                              yaxis=dict(categoryorder="array", categoryarray=STATE_ORDER, **ax2d), **dark))
    return [dome_fig, azel, charts, states]


def _plotly_html(c, bounds):
    import json
    js = _plotly_js()
    if js is None:
        raise ImportError("plotly.min.js not bundled")
    divs = []
    for i, fig in enumerate(_figures(c, bounds)):
        spec = json.dumps(fig, separators=(",", ":"))
        divs.append(f'<div id="fig{i}"></div><script>(function(){{var f={spec};'
                    f'Plotly.newPlot("fig{i}",f.data,f.layout,{{displaylogo:false,responsive:true}});}})();</script>')
    return js, divs


def write_html(path, s, c, bounds, pngs):
    try:
        js, divs = _plotly_html(c, bounds)
        script = f"<script>{js}</script>"
    except ImportError:                                   # plotly missing: static images
        script = ""
        divs = [f'<img src="data:image/png;base64,{base64.b64encode(pngs[k]).decode()}">'
                for k in ("dome", "azel_time", "charts")]
    row = lambda a, b: f"<tr><td>{html.escape(a)}</td><td>{html.escape(str(b))}</td></tr>"
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>FSOC Performance Report</title>{script}
<style>
body {{ background:#0a0e14; color:#d7e3f4; font-family:'Segoe UI',sans-serif; margin:0 auto; max-width:1200px; padding:24px; }}
h1 {{ font-weight:600; font-size:24px; margin:0 0 4px; }} h2 {{ color:#38c6ff; font-size:15px; letter-spacing:1px;
text-transform:uppercase; margin:28px 0 10px; }} .sub {{ color:#7f93ad; }}
.grid {{ display:grid; grid-template-columns:1fr 1.4fr; gap:16px; }}
table {{ width:100%; border-collapse:collapse; background:#111823; border:1px solid #22314a; border-radius:8px; }}
td {{ padding:7px 12px; border-bottom:1px solid #1a2638; font-size:13px; }} td:first-child {{ color:#7f93ad; }}
td:last-child {{ font-family:Consolas,monospace; }}
.card {{ background:#111823; border:1px solid #22314a; border-radius:8px; padding:6px; margin-bottom:16px; }}
img {{ max-width:100%; }}
@media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style></head><body>
<h1>FSOC Coarse Pointing, Acquisition &amp; Tracking</h1>
<div class="sub">Performance report · generated {html.escape(str(s.get('generated', '')))}</div>
<div class="grid">
<div><h2>Run</h2><table>{''.join(row(a, b) for a, b in _run_info(s))}</table></div>
<div><h2>Performance summary</h2><table>{''.join(row(a, b) for a, b in _kpis(s))}</table></div>
</div>
<h2>3D view of the run</h2><div class="card">{divs[0]}</div>
<div class="card">{divs[1]}</div>
<h2>Time charts</h2>{''.join(f'<div class="card">{d}</div>' for d in divs[2:])}
</body></html>"""
    path.write_text(page, encoding="utf-8")


def generate(folder, rows, summary, bounds):
    """Write report.html and report.pdf into ``folder``; returns their paths."""
    if not rows:
        return []
    c = _columns(rows)
    pngs = {k: _fig_png(f) for k, f in _mpl_figures(c, bounds).items()}
    html_path, pdf_path = folder / "report.html", folder / "report.pdf"
    write_html(html_path, summary, c, bounds, pngs)
    write_pdf(pdf_path, summary, c, pngs)
    return [html_path, pdf_path]
