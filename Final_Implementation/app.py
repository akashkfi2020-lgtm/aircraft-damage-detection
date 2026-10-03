import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import json
from collections import defaultdict

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Aircraft Defect Detection",
    layout="wide"
)

# ──────────────────────────────────────────────
# Custom CSS — Aviation dark-panel aesthetic
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0d14;
    color: #c9d4e8;
}

/* Title strip */
.title-strip {
    background: linear-gradient(90deg, #0d1b2a 0%, #112240 100%);
    border-left: 4px solid #00c2ff;
    padding: 18px 24px;
    margin-bottom: 24px;
    border-radius: 0 6px 6px 0;
}
.title-strip h1 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e8f0fe;
    margin: 0;
    letter-spacing: 0.04em;
}
.title-strip p {
    font-size: 0.85rem;
    color: #7a93b8;
    margin: 4px 0 0 0;
    font-family: 'Share Tech Mono', monospace;
}

/* Status card */
.status-card {
    border-radius: 8px;
    padding: 20px 24px;
    margin: 16px 0;
    border: 1px solid;
}
.status-go {
    background: #051a0f;
    border-color: #00c97a;
    color: #00c97a;
}
.status-caution {
    background: #1a1200;
    border-color: #f0a500;
    color: #f0a500;
}
.status-nogo {
    background: #1a0505;
    border-color: #e83030;
    color: #e83030;
}
.status-card .verdict {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    font-family: 'Share Tech Mono', monospace;
}
.status-card .verdict-sub {
    font-size: 0.82rem;
    opacity: 0.75;
    margin-top: 4px;
}

/* Defect row */
.defect-row {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.defect-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    padding: 3px 10px;
    border-radius: 3px;
    font-weight: 600;
    white-space: nowrap;
}
.badge-dent        { background:#1a2a4a; color:#5fa8ff; border:1px solid #2a4a7a; }
.badge-fastener    { background:#2a1a00; color:#ffaa33; border:1px solid #7a4a00; }
.badge-rupture     { background:#2a0505; color:#ff5555; border:1px solid #7a1010; }

/* Metric tiles */
.metric-tile {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-tile .num {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Share Tech Mono', monospace;
    color: #00c2ff;
}
.metric-tile .lbl {
    font-size: 0.75rem;
    color: #7a93b8;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Recommendation block */
.rec-block {
    background: #0d1520;
    border-left: 3px solid #00c2ff;
    padding: 14px 18px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.88rem;
    line-height: 1.6;
}
.rec-block.warn { border-left-color: #f0a500; }
.rec-block.danger { border-left-color: #e83030; }

/* Research box */
.research-box {
    background: #060d1a;
    border: 1px dashed #2a4a7a;
    border-radius: 8px;
    padding: 18px 22px;
    margin-top: 12px;
    font-size: 0.84rem;
    line-height: 1.7;
    color: #8a9fc0;
}
.research-box strong { color: #c9d4e8; }

/* Section headers */
.sec-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #4a6888;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 20px 0 8px 0;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 4px;
}

/* Override Streamlit whites */
.stApp { background-color: #0a0d14; }
[data-testid="stSidebar"] { background-color: #080c12; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Title
# ──────────────────────────────────────────────
st.markdown("""
<div class="title-strip">
  <h1>Aircraft Surface Defect Detection</h1>
  <p>YOLOv8 · Deep Learning Inspection System · Classes: Dent / Fastener Damage / Rupture</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Airworthiness Inference Engine
# ──────────────────────────────────────────────

# Severity config per defect class
# Each entry: (base_severity 1-10, description, maintenance_action, grounds_aircraft)
DEFECT_CONFIG = {
    "Dent": {
        "severity_base": 4,
        "description": "Structural deformation of the fuselage or wing skin. Dents alter aerodynamic profile and may indicate sub-surface fatigue.",
        "action": "Log dent dimensions. If depth > 2mm or diameter > 50mm, defer for sheet-metal inspection before next flight.",
        "grounds_aircraft": False,
        "icon": "🔵",
    },
    "Fastener Damage": {
        "severity_base": 7,
        "description": "Compromised rivets, bolts, or screws. Fastener failure can lead to panel separation or structural failure at load.",
        "action": "Aircraft must NOT fly until damaged fasteners are replaced and torque specs verified by a licensed AME.",
        "grounds_aircraft": True,
        "icon": "🟠",
    },
    "Rupture": {
        "severity_base": 10,
        "description": "Tear, crack, or breach in aircraft skin/structure. This is a CRITICAL airworthiness finding — structural integrity is compromised.",
        "action": "IMMEDIATE GROUND. Do not attempt flight. Escalate to senior maintenance engineer. NDT (non-destructive testing) inspection required.",
        "grounds_aircraft": True,
        "icon": "🔴",
    },
}

def compute_airworthiness(detections):
    """
    Rule-based airworthiness inference engine.
    
    Input:  list of dicts [{label, conf, x1, y1, x2, y2}, ...]
    Output: dict with verdict, risk_score, findings, recommendations
    """
    if not detections:
        return {
            "verdict": "AIRWORTHY",
            "verdict_class": "go",
            "risk_score": 0,
            "summary": "No defects detected. Aircraft surface appears nominal.",
            "findings": [],
            "recommendations": [
                "Continue scheduled maintenance intervals.",
                "Next inspection per manufacturer's maintenance manual.",
            ],
            "grounded": False,
        }

    # Aggregate by class
    by_class = defaultdict(list)
    for d in detections:
        by_class[d["label"]].append(d["conf"])

    grounded = False
    findings = []
    max_severity = 0
    weighted_risk = 0

    for label, confs in by_class.items():
        cfg = DEFECT_CONFIG.get(label, {})
        count = len(confs)
        avg_conf = sum(confs) / count
        max_conf = max(confs)

        # Severity scales with count (up to ×1.5) and confidence
        multiplier = min(1.0 + (count - 1) * 0.15, 1.5)
        severity = min(cfg.get("severity_base", 5) * multiplier * avg_conf, 10)
        weighted_risk = max(weighted_risk, severity)

        if cfg.get("grounds_aircraft"):
            grounded = True

        findings.append({
            "label": label,
            "count": count,
            "avg_conf": avg_conf,
            "max_conf": max_conf,
            "severity": round(severity, 1),
            "description": cfg.get("description", ""),
            "action": cfg.get("action", "Inspect before next flight."),
            "grounds": cfg.get("grounds_aircraft", False),
            "icon": cfg.get("icon", "⚪"),
        })

    # Sort by severity desc
    findings.sort(key=lambda x: x["severity"], reverse=True)

    # Compound risk: multiple defect types add 15% each
    compound_bonus = (len(by_class) - 1) * 0.15
    final_risk = min(weighted_risk * (1 + compound_bonus), 10)
    risk_pct = round(final_risk * 10)

    # Verdict logic
    has_rupture = "Rupture" in by_class
    has_fastener = "Fastener Damage" in by_class
    has_dent = "Dent" in by_class

    if has_rupture:
        verdict = "NO-GO — CRITICAL"
        verdict_class = "nogo"
        summary = "Structural rupture detected. Aircraft is UNAIRWORTHY. Immediate grounding mandatory."
    elif has_fastener and has_dent:
        verdict = "NO-GO — GROUNDED"
        verdict_class = "nogo"
        summary = "Multiple defect types including fastener damage. Do not dispatch until inspected."
    elif has_fastener:
        verdict = "NO-GO — MAINTENANCE REQUIRED"
        verdict_class = "nogo"
        summary = "Fastener damage compromises structural integrity. Aircraft must not fly until repaired."
    elif has_dent and len(by_class["Dent"]) >= 3:
        verdict = "CAUTION — DEFERRED INSPECTION"
        verdict_class = "caution"
        summary = "Multiple dents detected. Airworthiness deferred pending dimensional measurement."
    elif has_dent:
        verdict = "CAUTION — MONITOR"
        verdict_class = "caution"
        summary = "Minor surface denting detected. Serviceable with logging; re-inspect at next interval."
    else:
        verdict = "CAUTION — REVIEW"
        verdict_class = "caution"
        summary = "Defect(s) detected. Review findings and apply corrective action before dispatch."

    # Recommendations
    recs = []
    for f in findings:
        recs.append(f["action"])
    if len(findings) > 1:
        recs.append("Document all findings in the aircraft technical log (Form CA 28 / maintenance record).")
    if grounded:
        recs.append("Notify the Airworthiness Inspector. Do not move aircraft until cleared.")

    return {
        "verdict": verdict,
        "verdict_class": verdict_class,
        "risk_score": risk_pct,
        "summary": summary,
        "findings": findings,
        "recommendations": recs,
        "grounded": grounded,
    }


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
MODEL_PATH = "runs/detect/runs/detect/yolov8_aircraft_safe/weights/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
    model_ok = True
except Exception as e:
    model_ok = False
    st.error(f"⚠️ Model not found at `{MODEL_PATH}`. Update the path. Error: {e}")

# ──────────────────────────────────────────────
# Layout: two columns
# ──────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="sec-header">Image Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload an aircraft inspection image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    conf_thresh = st.slider("Detection Confidence Threshold", 0.10, 0.90, 0.25, 0.05)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col_right:
    if uploaded_file and model_ok:
        image = Image.open(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model(tmp.name, conf=conf_thresh)

        result = results[0]
        annotated = result.plot()

        st.markdown('<div class="sec-header">Detection Result</div>', unsafe_allow_html=True)
        st.image(annotated, caption="YOLOv8 Detections", use_container_width=True)

        # ── Parse detections ──
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = model.names[cls_id]
            xyxy   = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "conf": conf,
                "x1": xyxy[0], "y1": xyxy[1],
                "x2": xyxy[2], "y2": xyxy[3],
            })

        # ── Run inference engine ──
        report = compute_airworthiness(detections)

    elif not uploaded_file:
        st.info("Upload an image on the left to begin analysis.")
        report = None
    else:
        report = None

# ──────────────────────────────────────────────
# Report Section (full width below)
# ──────────────────────────────────────────────
if uploaded_file and model_ok and report:

    st.markdown("---")
    st.markdown('<div class="sec-header">Airworthiness Report</div>', unsafe_allow_html=True)

    # ── Verdict card ──
    vc = report["verdict_class"]
    st.markdown(f"""
    <div class="status-card status-{vc}">
      <div class="verdict">{'⛔' if vc=='nogo' else '⚠️' if vc=='caution' else '✅'} &nbsp; {report['verdict']}</div>
      <div class="verdict-sub">{report['summary']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="num">{len(detections)}</div>
          <div class="lbl">Defects Found</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        defect_types = len(set(d["label"] for d in detections))
        st.markdown(f"""
        <div class="metric-tile">
          <div class="num">{defect_types}</div>
          <div class="lbl">Defect Types</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="num">{report['risk_score']}%</div>
          <div class="lbl">Risk Score</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        grounded_label = "YES" if report["grounded"] else "NO"
        grounded_color = "#e83030" if report["grounded"] else "#00c97a"
        st.markdown(f"""
        <div class="metric-tile">
          <div class="num" style="color:{grounded_color}">{grounded_label}</div>
          <div class="lbl">Aircraft Grounded</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Per-defect findings ──
    if report["findings"]:
        st.markdown('<div class="sec-header">Defect Analysis</div>', unsafe_allow_html=True)
        for f in report["findings"]:
            badge_cls = {
                "Dent": "badge-dent",
                "Fastener Damage": "badge-fastener",
                "Rupture": "badge-rupture",
            }.get(f["label"], "badge-dent")

            st.markdown(f"""
            <div class="defect-row">
              <span class="defect-badge {badge_cls}">{f['icon']} {f['label']}</span>
              <span style="font-size:0.85rem; flex:1">
                <strong>{f['count']}× detected</strong> &nbsp;·&nbsp;
                Avg confidence: <code>{f['avg_conf']:.0%}</code> &nbsp;·&nbsp;
                Severity: <code>{f['severity']}/10</code>
              </span>
            </div>
            <div style="font-size:0.82rem; color:#7a93b8; padding: 2px 16px 10px 16px;">{f['description']}</div>
            """, unsafe_allow_html=True)

    # ── Recommendations ──
    st.markdown('<div class="sec-header">Maintenance Recommendations</div>', unsafe_allow_html=True)
    for i, rec in enumerate(report["recommendations"]):
        is_critical = any(w in rec.upper() for w in ["NOT FLY", "GROUND", "IMMEDIATE", "DO NOT"])
        cls = "danger" if is_critical else ("warn" if "Defer" in rec or "before" in rec.lower() else "")
        st.markdown(f'<div class="rec-block {cls}">{"⛔ " if is_critical else "→ "}{rec}</div>', unsafe_allow_html=True)

    # ── Raw detection table ──
    with st.expander("🔬 Raw Detection Data (JSON)"):
        st.json(detections)