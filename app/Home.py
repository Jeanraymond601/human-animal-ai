import sys
from pathlib import Path

# Ajoute la racine du projet au chemin Python (nécessaire avec `streamlit run app/Home.py`)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hashlib
from datetime import datetime
from html import escape
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from app.services.detector import TARGET_CLASSES, detect_objects  # noqa: E402
from app.evaluation_page import (  # noqa: E402
    render_evaluation_metrics,
    render_evaluation_page,
    run_cached_evaluation,
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Human & Animal AI",
    page_icon=":material/visibility:",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "Human & Animal AI"
APP_VERSION = "1.0.0"
MAX_HISTORY = 20

# Dossier permanent contenant les images originales analysées
DETECTED_IMAGES_DIR = ROOT_DIR / "data" / "detected_images"
DETECTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Dossier permanent contenant les annotations YOLO de référence
DETECTED_LABELS_DIR = DETECTED_IMAGES_DIR / "labels"
DETECTED_LABELS_DIR.mkdir(parents=True, exist_ok=True)

PAGES = [
    "Dashboard",
    "Détection",
    "Évaluation",
    "Historique",
    "À propos",
]


# ============================================================
# SESSION STATE
# ============================================================

def init_state():
    defaults = {
        "theme": "Sombre",
        "page": "Dashboard",
        "history": [],
        "last_detection_key": None,
        "last_detection": None,
        # Cache des évaluations individuelles (évite les doublons MLflow)
        "image_evaluations": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def go_to_detection():
    """Callback du bouton : change de page avant le rechargement du script."""
    st.session_state.page = "Détection"


# ============================================================
# HTML HELPERS
# ============================================================

def clean_html(content: str) -> str:
    """
    Supprime l'indentation et les lignes vides du HTML.

    Sans cela, Markdown interprète les lignes indentées de 4 espaces
    ou plus comme du code, et une ligne vide interrompt le bloc HTML.
    """
    return "\n".join(
        line.strip() for line in content.splitlines() if line.strip()
    )


def render_html(content: str) -> None:
    st.html(clean_html(content))


# ============================================================
# FILE HELPERS
# ============================================================

def save_original_image(image_bytes: bytes, file_hash: str, filename: str) -> Path:
    """
    Sauvegarde l'image originale dans DETECTED_IMAGES_DIR.

    Le nom du fichier est le hash SHA-256 du contenu : la même image
    n'est écrite qu'une seule fois, même si le seuil de confiance change.
    """
    extension = Path(filename).suffix.lower()

    if extension not in {".jpg", ".jpeg", ".png"}:
        extension = ".png"

    image_path = DETECTED_IMAGES_DIR / f"{file_hash}{extension}"

    if not image_path.is_file():
        image_path.write_bytes(image_bytes)

    return image_path


# ============================================================
# SVG ICONS
# ============================================================

ICONS = {
    "scan": """
        <rect x="3" y="3" width="18" height="18" rx="4"/>
        <path d="M7 9V7h2M15 7h2v2M17 15v2h-2M9 17H7v-2"/>
    """,
    "dashboard": """
        <rect x="3" y="3" width="8" height="8" rx="2"/>
        <rect x="13" y="3" width="8" height="5" rx="2"/>
        <rect x="13" y="10" width="8" height="11" rx="2"/>
        <rect x="3" y="13" width="8" height="8" rx="2"/>
    """,
    "image": """
        <rect x="3" y="3" width="18" height="18" rx="3"/>
        <circle cx="8.5" cy="8.5" r="1.5"/>
        <path d="m21 15-5-5L5 21"/>
    """,
    "users": """
        <path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
        <circle cx="10" cy="7" r="4"/>
        <path d="M20 21v-2a4 4 0 0 0-3-3.87"/>
        <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
    """,
    "paw": """
        <circle cx="8" cy="7" r="2"/>
        <circle cx="16" cy="7" r="2"/>
        <circle cx="5" cy="13" r="2"/>
        <circle cx="19" cy="13" r="2"/>
        <path d="M12 12c-3 0-6 5-4 7 1 1 3 0 4 0s3 1 4 0c2-2-1-7-4-7z"/>
    """,
    "target": """
        <circle cx="12" cy="12" r="9"/>
        <circle cx="12" cy="12" r="5"/>
        <circle cx="12" cy="12" r="1"/>
    """,
    "clock": """
        <circle cx="12" cy="12" r="9"/>
        <path d="M12 7v5l3 2"/>
    """,
}


def icon(name, size=22, color="currentColor"):
    """Retourne une icône SVG sur une seule ligne (sans lignes vides)."""
    svg_path = " ".join(ICONS.get(name, ICONS["scan"]).split())

    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
        f'stroke-linejoin="round" '
        f'style="color:{color};display:inline-block;vertical-align:middle;flex-shrink:0;" '
        f'xmlns="http://www.w3.org/2000/svg">{svg_path}</svg>'
    )


# ============================================================
# THEME
# ============================================================

THEMES = {
    "Clair": {
        "bg": "#F5F7FB",
        "surface": "#FFFFFF",
        "surface2": "#F0F3F9",
        "text": "#172033",
        "muted": "#718096",
        "border": "#E4E9F1",
        "primary": "#356DF3",
        "primary-hover": "#2458D8",
        "sidebar": "#FFFFFF",
        "shadow": "0 8px 30px rgba(30,50,90,.06)",
    },
    "Sombre": {
        "bg": "#0B1020",
        "surface": "#131B2E",
        "surface2": "#1A2540",
        "text": "#F1F5FF",
        "muted": "#9AA9C2",
        "border": "#26334D",
        "primary": "#5B8CFF",
        "primary-hover": "#78A0FF",
        "sidebar": "#10182A",
        "shadow": "0 8px 30px rgba(0,0,0,.18)",
    },
}

BASE_CSS = """
.stApp { background: var(--bg); color: var(--text); }

[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child { background: var(--sidebar); }

[data-testid="stSidebar"] { border-right: 1px solid var(--border); }

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, h5, h6, p, label { color: var(--text); }

.app-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 4px 25px;
}
.brand-icon {
    width: 44px; height: 44px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #3978FF, #785BFF);
    box-shadow: 0 6px 20px rgba(70,110,255,.28);
}
.brand-name { color: var(--text); font-size: 16px; font-weight: 800; letter-spacing: -.5px; }
.brand-caption { color: var(--muted); font-size: 10px; letter-spacing: 1.2px; margin-top: 3px; }

.eyebrow {
    color: var(--primary); font-size: 11px; font-weight: 800;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px;
}
.hero-title {
    color: var(--text); font-size: clamp(29px, 4vw, 44px);
    font-weight: 850; line-height: 1.15; letter-spacing: -1.8px; margin-bottom: 12px;
}
.hero-description {
    color: var(--muted); font-size: 14px; line-height: 1.8; max-width: 720px;
}
.hero-panel {
    background:
        radial-gradient(circle at 85% 15%, rgba(91,140,255,.22), transparent 35%),
        linear-gradient(125deg, var(--surface), var(--surface2));
    border: 1px solid var(--border); border-radius: 22px;
    padding: 30px; margin-bottom: 25px; box-shadow: var(--shadow);
}
.hero-visual {
    display: flex; justify-content: center; align-items: center; min-height: 150px;
}
.hero-visual svg { filter: drop-shadow(0 8px 20px rgba(70,110,255,.2)); }

.section-title {
    font-size: 19px; font-weight: 750; letter-spacing: -.4px;
    margin: 12px 0 5px; color: var(--text);
}
.section-description { color: var(--muted); font-size: 13px; margin-bottom: 18px; }

.metric-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 17px; padding: 20px; min-height: 135px;
    box-shadow: var(--shadow);
    transition: transform .2s ease, border-color .2s ease;
}
.metric-card:hover { transform: translateY(-3px); border-color: var(--primary); }
.metric-icon { margin-bottom: 12px; }
.metric-label { color: var(--muted); font-size: 12px; font-weight: 600; }
.metric-value {
    color: var(--text); font-size: 31px; font-weight: 800;
    margin-top: 8px; letter-spacing: -1px;
}

.content-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 17px; padding: 20px;
    box-shadow: var(--shadow); margin-bottom: 15px;
}
.muted-text { color: var(--muted); font-size: 13px; line-height: 1.7; }

.status-pill {
    display: inline-block; color: #48C995;
    background: rgba(72,201,149,.12); border: 1px solid rgba(72,201,149,.25);
    border-radius: 30px; padding: 5px 11px; font-size: 11px; font-weight: 700;
}

.step-number {
    display: inline-flex; width: 30px; height: 30px; border-radius: 10px;
    align-items: center; justify-content: center;
    background: rgba(91,140,255,.15); color: var(--primary);
    font-weight: 800; margin-bottom: 10px;
}
.step-title { color: var(--text); font-size: 16px; font-weight: 700; margin-bottom: 6px; }
.icon-heading { display: flex; align-items: center; gap: 10px; }

.footer {
    color: var(--muted); border-top: 1px solid var(--border);
    padding-top: 18px; margin-top: 35px; font-size: 11px; text-align: center;
}

/* ---------- Widgets Streamlit ---------- */
[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border);
    padding: 16px; border-radius: 14px;
}
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--text); }

[data-testid="stExpander"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
}

[data-testid="stFileUploader"] {
    background: var(--surface); border: 1px dashed var(--primary);
    border-radius: 15px; padding: 15px;
}
[data-testid="stFileUploaderDropzone"] { background: var(--surface2); border-radius: 12px; }
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small { color: var(--text); }

button[kind="primary"],
button[data-testid="stBaseButton-primary"],
button[kind="secondary"],
button[data-testid="stBaseButton-secondary"] {
    border-radius: 10px; font-weight: 650; min-height: 42px;
}

button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: var(--primary); border: none; color: #FFFFFF;
}
button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover { background: var(--primary-hover); }
button[kind="primary"] p,
button[data-testid="stBaseButton-primary"] p { color: #FFFFFF; }

button[kind="secondary"],
button[data-testid="stBaseButton-secondary"] {
    background: var(--surface2); border: 1px solid var(--border); color: var(--text);
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
}

hr { border-color: var(--border); }
"""


def apply_theme(theme):
    colors = THEMES.get(theme, THEMES["Sombre"])
    variables = "".join(f"--{name}: {value};" for name, value in colors.items())
    render_html(f"<style>:root {{ {variables} }}\n{BASE_CSS}</style>")


apply_theme(st.session_state.theme)


# ============================================================
# UI HELPERS
# ============================================================

def section_title(title, description=None):
    render_html(f'<div class="section-title">{escape(title)}</div>')

    if description:
        render_html(
            f'<div class="section-description">{escape(description)}</div>'
        )


def metric_card(label, value, icon_name):
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-icon">
                {icon(icon_name, 24, "var(--primary)")}
            </div>
            <div class="metric-label">{escape(str(label))}</div>
            <div class="metric-value">{value}</div>
        </div>
        """
    )


def page_header(eyebrow, title, description):
    render_html(
        f"""
        <div class="eyebrow">{escape(eyebrow)}</div>
        <div class="hero-title">{escape(title)}</div>
        <div class="hero-description">{escape(description)}</div>
        """
    )


def empty_state(icon_name, title, text):
    render_html(
        f"""
        <div class="content-card" style="text-align:center;padding:45px 20px;">
            <div style="margin-bottom:15px;">
                {icon(icon_name, 55, "var(--primary)")}
            </div>
            <h3>{escape(title)}</h3>
            <div class="muted-text">{escape(text)}</div>
        </div>
        """
    )


def show_footer():
    render_html(
        f"""
        <div class="footer">
            {escape(APP_NAME)} · Version {APP_VERSION}
            <br>
            Système intelligent de détection et de classification
            des personnes et des animaux.
        </div>
        """
    )


def get_detection_rows(detections):
    rows = []

    for item in detections:
        coords = item.get("coordonnees", [])

        rows.append({
            "Classe": item.get("classe", "Inconnue"),
            "Catégorie": item.get("categorie", "-"),
            "Confiance": float(item.get("confiance", 0)),
            "Coordonnées": ", ".join(f"{value:.0f}" for value in coords),
        })

    return rows


def normalize_detection_result(annotated_image, detections, stats):
    """
    Regroupe les trois valeurs retournées par detect_objects()
    en une structure unique pour l'interface.
    """
    result_data = {
        **stats,
        "detections": detections,
    }

    return annotated_image, result_data


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        f"""
        <div class="app-brand">
            <div class="brand-icon">
                {icon("scan", 25, "#FFFFFF")}
            </div>
            <div>
                <div class="brand-name">{escape(APP_NAME)}</div>
                <div class="brand-caption">INTELLIGENT VISION SYSTEM</div>
            </div>
        </div>
        """
    )

    st.markdown("##### NAVIGATION")

    st.radio(
        "Navigation",
        PAGES,
        key="page",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("##### APPARENCE")

    st.radio(
        "Choisir le thème",
        ["Sombre", "Clair"],
        key="theme",
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("##### MODÈLE IA")

    render_html(
        f"""
        <div class="content-card">
            <div class="icon-heading">
                {icon("target", 21, "var(--primary)")}
                <strong>YOLO11n</strong>
            </div>
            <div class="muted-text">
                Modèle de détection pré-entraîné.
            </div>
            <br>
            <span class="status-pill">Disponible</span>
            <br><br>
            <div class="muted-text">
                Inférence : CPU<br>
                Classes ciblées : {len(TARGET_CLASSES)}<br>
                Entrée : Image
            </div>
        </div>
        """
    )

    st.caption("Human & Animal AI · 2026")


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard():

    render_html(
        """
        <div class="hero-panel">
            <div class="eyebrow">
                INTELLIGENCE ARTIFICIELLE · COMPUTER VISION
            </div>
            <div class="hero-title">
                Voyez plus.<br>
                Comprenez mieux.
            </div>
            <div class="hero-description">
                Une solution intelligente basée sur le Deep Learning
                pour détecter et classifier les personnes et les animaux
                à partir d'images.
            </div>
        </div>
        """
    )

    history = st.session_state.history

    total_images = len(history)
    total_objects = sum(item["total"] for item in history)
    total_people = sum(item["personnes"] for item in history)
    total_animals = sum(item["animaux"] for item in history)

    section_title("Vue d'ensemble", "Statistiques de cette session.")

    cols = st.columns(4)

    values = [
        ("Images analysées", total_images, "image"),
        ("Objets détectés", total_objects, "target"),
        ("Personnes", total_people, "users"),
        ("Animaux", total_animals, "paw"),
    ]

    for col, (label, value, icon_name) in zip(cols, values):
        with col:
            metric_card(label, value, icon_name)

    st.markdown("")

    section_title(
        "Commencer une analyse",
        "Importez une image pour lancer votre première détection.",
    )

    col1, col2 = st.columns([1.3, 1])

    with col1:
        render_html(
            """
            <div class="content-card">
                <div class="step-number">01</div>
                <div class="step-title">Importer une image</div>
                <div class="muted-text">
                    Choisissez une image au format JPG, JPEG ou PNG.
                </div>
                <br>
                <div class="step-number">02</div>
                <div class="step-title">Détection automatique</div>
                <div class="muted-text">
                    Le modèle YOLO analyse l'image et identifie
                    les personnes et les animaux.
                </div>
                <br>
                <div class="step-number">03</div>
                <div class="step-title">Consulter les résultats</div>
                <div class="muted-text">
                    Visualisez les objets détectés, les statistiques
                    et téléchargez l'image annotée.
                </div>
            </div>
            """
        )

    with col2:
        render_html(
            f"""
            <div class="content-card">
                <div class="hero-visual">
                    {icon("scan", 65, "var(--primary)")}
                </div>
                <h3 style="text-align:center;">YOLO11n</h3>
                <p class="muted-text" style="text-align:center;">
                    Modèle de vision par ordinateur léger.
                    Il permet de localiser plusieurs objets
                    dans une même image.
                </p>
            </div>
            """
        )

        st.button(
            "Ouvrir le module de détection",
            type="primary",
            icon=":material/arrow_forward:",
            use_container_width=True,
            on_click=go_to_detection,
        )

    if history:
        section_title("Analyses récentes")

        recent = history[-5:][::-1]

        df = pd.DataFrame([
            {
                "Date": item["date"],
                "Image": item["filename"],
                "Personnes": item["personnes"],
                "Animaux": item["animaux"],
                "Total": item["total"],
            }
            for item in recent
        ])

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

    show_footer()


# ============================================================
# DETECTION PAGE
# ============================================================

def render_detection():

    page_header(
        "MODULE D'ANALYSE",
        "Détection intelligente",
        "Importez une image et laissez le modèle YOLO11n "
        "détecter les personnes et les animaux.",
    )

    st.markdown("")

    col_upload, col_settings = st.columns([1.5, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Importer une image",
            type=["jpg", "jpeg", "png"],
            help="Formats acceptés : JPG, JPEG et PNG.",
        )

    with col_settings:
        confidence = st.slider(
            "Seuil de confiance",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help=(
                "Une valeur basse détecte davantage d'objets, "
                "mais peut augmenter les faux positifs."
            ),
        )

    if uploaded_file is None:

        empty_state(
            "image",
            "Votre image apparaîtra ici",
            "Importez une image pour commencer l'analyse.",
        )

        st.info(
            "Utilisez une image nette avec des objets bien visibles "
            "pour obtenir de meilleurs résultats.",
            icon=":material/lightbulb:",
        )

        show_footer()
        return

    try:

        image_bytes = uploaded_file.getvalue()

        original_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        file_hash = hashlib.sha256(image_bytes).hexdigest()

        detection_key = (file_hash, confidence)

        # ----------------------------------------------------
        # DETECTION (relancée seulement si l'image ou le seuil change)
        # ----------------------------------------------------

        if st.session_state.last_detection_key != detection_key:

            with st.spinner("Analyse de l'image par YOLO11n..."):

                # detect_objects retourne 3 éléments :
                # image annotée, liste des détections, statistiques.
                annotated_image, detections, stats = detect_objects(
                    original_image,
                    confidence=confidence,
                )

                annotated_image, result_data = normalize_detection_result(
                    annotated_image,
                    detections,
                    stats,
                )

            if not isinstance(annotated_image, Image.Image):
                annotated_image = Image.fromarray(annotated_image)

            image_buffer = BytesIO()

            annotated_image.convert("RGB").save(
                image_buffer,
                format="JPEG",
                quality=95,
            )

            annotated_bytes = image_buffer.getvalue()

            # Sauvegarde permanente de l'image originale sur disque
            original_path = save_original_image(
                image_bytes,
                file_hash,
                uploaded_file.name,
            )

            st.session_state.last_detection_key = detection_key

            st.session_state.last_detection = {
                "annotated_image": annotated_image,
                "annotated_bytes": annotated_bytes,
                "data": result_data,
                "image_path": str(original_path),
            }

            # ------------------------------------------------
            # HISTORIQUE
            # ------------------------------------------------

            record_id = f"{file_hash}_{confidence}"

            already_saved = any(
                item["id"] == record_id
                for item in st.session_state.history
            )

            if not already_saved:

                st.session_state.history.append({
                    "id": record_id,
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "filename": uploaded_file.name,
                    "personnes": result_data.get("personnes", 0),
                    "animaux": result_data.get("animaux", 0),
                    "total": result_data.get("total", 0),
                    "par_classe": result_data.get("par_classe", {}),
                    "annotated_bytes": annotated_bytes,
                    "image_path": str(original_path),
                })

                # Limite la mémoire utilisée par l'historique
                st.session_state.history = (
                    st.session_state.history[-MAX_HISTORY:]
                )

        # ----------------------------------------------------
        # RESULTATS
        # ----------------------------------------------------

        result = st.session_state.last_detection

        result_data = result["data"]

        annotated_image = result["annotated_image"]

        st.success("Analyse terminée.", icon=":material/check_circle:")

        section_title("Résultats de l'analyse")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Personnes détectées",
                result_data.get("personnes", 0),
                "users",
            )

        with c2:
            metric_card(
                "Animaux détectés",
                result_data.get("animaux", 0),
                "paw",
            )

        with c3:
            metric_card(
                "Total des objets",
                result_data.get("total", 0),
                "target",
            )

        # ----------------------------------------------------
        # COMPARAISON DES IMAGES
        # ----------------------------------------------------

        st.markdown("")

        section_title(
            "Comparaison visuelle",
            "Image originale et résultat après détection.",
        )

        left, right = st.columns(2)

        with left:
            st.markdown("**Image originale**")
            st.image(original_image, use_container_width=True)

        with right:
            st.markdown("**Image analysée**")
            st.image(annotated_image, use_container_width=True)

        # ----------------------------------------------------
        # REPARTITION PAR CLASSE
        # ----------------------------------------------------

        st.markdown("")

        section_title(
            "Répartition par classe",
            "Nombre d'objets détectés pour chaque classe.",
        )

        class_counts = result_data.get("par_classe", {})

        if class_counts:

            max_count = max(class_counts.values())

            for class_name, count in class_counts.items():

                col_name, col_bar, col_count = st.columns([1, 3, 0.4])

                with col_name:
                    st.write(class_name)

                with col_bar:
                    st.progress(count / max_count if max_count else 0)

                with col_count:
                    st.write(f"**{count}**")

        else:

            st.info(
                "Aucun objet ciblé n'a été détecté.",
                icon=":material/info:",
            )

        # ----------------------------------------------------
        # TABLEAU DES DETECTIONS
        # ----------------------------------------------------

        st.markdown("")

        section_title(
            "Détails des détections",
            "Informations sur les objets localisés par le modèle.",
        )

        rows = get_detection_rows(result_data.get("detections", []))

        if rows:

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confiance": st.column_config.ProgressColumn(
                        "Confiance",
                        format="%.1f %%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Coordonnées": st.column_config.TextColumn(
                        "Coordonnées (x1, y1, x2, y2)"
                    ),
                },
            )

        else:

            st.info(
                "Aucune personne ou aucun animal n'a été détecté "
                "dans cette image.",
                icon=":material/info:",
            )

        # ----------------------------------------------------
        # EVALUATION AUTOMATIQUE (après les résultats de détection)
        # ----------------------------------------------------

        st.markdown("")

        section_title(
            "Évaluation automatique",
            "Dès qu'une annotation de référence est disponible, "
            "les détections sont comparées à la vérité terrain.",
        )

        annotation_file = st.file_uploader(
            "Importer l'annotation de référence (.txt)",
            type=["txt"],
            help=(
                "Format YOLO : class_id x_center y_center width height. "
                "Coordonnées normalisées entre 0 et 1."
            ),
            key=f"annotation_{file_hash}",
        )

        label_path = DETECTED_LABELS_DIR / f"{file_hash}.txt"

        # Sauvegarde permanente de l'annotation importée
        if annotation_file is not None:

            annotation_bytes = annotation_file.getvalue()

            if (
                not label_path.is_file()
                or label_path.read_bytes() != annotation_bytes
            ):
                label_path.write_bytes(annotation_bytes)

        elif label_path.is_file():

            st.caption(
                "Annotation déjà enregistrée pour cette image : "
                "elle est utilisée automatiquement."
            )

        if not label_path.is_file():

            st.info(
                "Aucune annotation pour cette image. "
                "Importez-la : l'évaluation se lancera automatiquement.",
                icon=":material/info:",
            )

        else:

            try:

                with st.spinner("Évaluation automatique en cours..."):

                    evaluation_result, _ = run_cached_evaluation(
                        image_path=result["image_path"],
                        label_path=label_path,
                        detections=result_data.get("detections", []),
                        confidence=confidence,
                    )

                st.success(
                    "Évaluation terminée et enregistrée dans MLflow.",
                    icon=":material/check_circle:",
                )

                render_evaluation_metrics(evaluation_result)

            except Exception as evaluation_error:

                st.error(
                    "Impossible d'évaluer cette image. "
                    "Vérifiez le contenu du fichier d'annotation.",
                    icon=":material/error:",
                )

                st.exception(evaluation_error)

        # ----------------------------------------------------
        # TELECHARGEMENT
        # ----------------------------------------------------

        st.markdown("")

        st.download_button(
            label="Télécharger l'image annotée",
            data=result["annotated_bytes"],
            file_name="human_animal_detection.jpg",
            mime="image/jpeg",
            type="primary",
            icon=":material/download:",
            use_container_width=True,
        )

    except Exception as error:

        st.error(
            "Une erreur est survenue pendant l'analyse.",
            icon=":material/error:",
        )

        st.exception(error)

    show_footer()


# ============================================================
# HISTORY PAGE
# ============================================================

def render_history():

    page_header(
        "SUIVI DES ANALYSES",
        "Historique",
        "Retrouvez les images analysées pendant votre session.",
    )

    history = st.session_state.history

    if not history:

        empty_state(
            "clock",
            "Aucune analyse enregistrée",
            "Vos prochaines analyses apparaîtront ici.",
        )

    else:

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("### Analyses enregistrées")

        with col2:

            if st.button(
                "Effacer l'historique",
                icon=":material/delete:",
                use_container_width=True,
            ):

                st.session_state.history = []
                st.session_state.last_detection_key = None
                st.session_state.last_detection = None

                st.rerun()

        df = pd.DataFrame([
            {
                "Date": item["date"],
                "Fichier": item["filename"],
                "Personnes": item["personnes"],
                "Animaux": item["animaux"],
                "Total": item["total"],
            }
            for item in history[::-1]
        ])

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Détails des analyses")

        for index, item in enumerate(history[::-1]):

            with st.expander(f'{item["filename"]} · {item["date"]}'):

                c1, c2, c3 = st.columns(3)

                c1.metric("Personnes", item["personnes"])
                c2.metric("Animaux", item["animaux"])
                c3.metric("Total", item["total"])

                # Affichage de l'image originale sauvegardée
                image_path = item.get("image_path")

                if image_path and Path(image_path).is_file():
                    st.image(
                        image_path,
                        caption="Image originale",
                        use_container_width=True,
                    )

                if item.get("par_classe"):

                    st.write("**Répartition par classe**")

                    st.json(item["par_classe"])

                st.download_button(
                    "Télécharger le résultat",
                    data=item["annotated_bytes"],
                    file_name=f"resultat_{index + 1}.jpg",
                    mime="image/jpeg",
                    icon=":material/download:",
                    key=f"history_download_{item['id']}",
                )

    show_footer()


# ============================================================
# ABOUT PAGE
# ============================================================

def render_about():

    page_header(
        "INFORMATIONS DU PROJET",
        f"À propos de {APP_NAME}",
        "Un projet de vision par ordinateur qui utilise "
        "l'intelligence artificielle pour identifier et localiser "
        "des personnes et des animaux dans des images.",
    )

    st.markdown("")

    section_title("Objectif du projet")

    render_html(
        f"""
        <div class="content-card">
            <p class="muted-text">
                {escape(APP_NAME)} a pour objectif de concevoir un
                système intelligent capable de détecter et de
                classifier automatiquement des personnes et
                plusieurs espèces animales à partir d'images.
            </p>
            <p class="muted-text">
                Le système utilise un modèle de Deep Learning
                pré-entraîné et une interface web développée avec
                Streamlit.
            </p>
        </div>
        """
    )

    section_title("Technologies utilisées")

    technologies = [
        ("YOLO11n", "Détection d'objets par Deep Learning", "target"),
        ("Python", "Langage de programmation", "scan"),
        ("Streamlit", "Interface web interactive", "dashboard"),
        ("OpenCV", "Traitement d'images", "image"),
        ("Pillow", "Manipulation des images", "image"),
    ]

    cols = st.columns(2)

    for index, (name, description, icon_name) in enumerate(technologies):

        with cols[index % 2]:

            render_html(
                f"""
                <div class="content-card">
                    <div class="icon-heading">
                        {icon(icon_name, 22, "var(--primary)")}
                        <strong>{escape(name)}</strong>
                    </div>
                    <div class="muted-text">{escape(description)}</div>
                </div>
                """
            )

    section_title("Classes prises en charge")

    class_columns = st.columns(3)

    for index, (class_id, class_name) in enumerate(TARGET_CLASSES.items()):

        with class_columns[index % 3]:

            category = "Personne" if class_id == 0 else "Animal"
            class_icon = "users" if class_id == 0 else "paw"

            render_html(
                f"""
                <div class="content-card">
                    <div class="icon-heading">
                        {icon(class_icon, 20, "var(--primary)")}
                        <strong>{escape(class_name)}</strong>
                    </div>
                    <div class="muted-text">
                        {category} · Classe COCO {class_id}
                    </div>
                </div>
                """
            )

    show_footer()


# ============================================================
# MAIN ROUTER
# ============================================================

PAGE_RENDERERS = {
    "Dashboard": render_dashboard,
    "Détection": render_detection,
    "Évaluation": render_evaluation_page,
    "Historique": render_history,
    "À propos": render_about,
}

PAGE_RENDERERS[st.session_state.page]()