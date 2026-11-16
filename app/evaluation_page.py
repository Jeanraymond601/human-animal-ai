import hashlib
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st
from PIL import Image

from app.services.detector import detect_objects
from app.services.evaluation import evaluate_model, evaluate_single_image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATABASE_PATH = BASE_DIR / "mlflow.db"

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "evaluation"
    / "yolo11n_validation"
)

DETECTED_IMAGES_DIR = BASE_DIR / "data" / "detected_images"
DETECTED_LABELS_DIR = DETECTED_IMAGES_DIR / "labels"

EXPERIMENT_NAME = "Human-Animal-AI"

IOU_THRESHOLD = 0.50

mlflow.set_tracking_uri(
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)


# ============================================================
# RECUPERATION DES EVALUATIONS
# ============================================================

def get_all_evaluation_runs() -> pd.DataFrame:
    """Récupère les évaluations enregistrées dans MLflow."""

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        return pd.DataFrame()

    return mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )


def split_evaluation_runs(runs: pd.DataFrame):
    """Sépare les évaluations individuelles et globales."""

    if runs.empty:
        return pd.DataFrame(), pd.DataFrame()

    type_column = "tags.evaluation_type"

    if type_column in runs.columns:
        individual_runs = runs.loc[
            runs[type_column] == "single_image"
        ].copy()

        dataset_runs = runs.loc[
            runs[type_column] == "dataset"
        ].copy()

    else:
        individual_runs = pd.DataFrame()

        if "metrics.mAP50" in runs.columns:
            dataset_runs = runs.loc[
                runs["metrics.mAP50"].notna()
            ].copy()
        else:
            dataset_runs = pd.DataFrame()

    return individual_runs, dataset_runs


# ============================================================
# OUTILS D'AFFICHAGE
# ============================================================

def first_value(run, keys, default=None):
    """Retourne la première valeur non vide parmi plusieurs clés."""

    for key in keys:
        value = run.get(key)

        if value is not None and not pd.isna(value):
            return value

    return default


def metric_value(run, *keys):
    """Retourne une métrique numérique ou NaN (plusieurs clés possibles)."""

    value = first_value(run, keys)

    if value is None:
        return float("nan")

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def count_value(run, *keys):
    """Retourne un entier (TP, FP, FN) ou None s'il est absent."""

    value = metric_value(run, *keys)

    if pd.isna(value):
        return None

    return int(value)


def metric_column(runs: pd.DataFrame, key: str) -> pd.Series:
    """Retourne une colonne de métriques, ou des NaN si elle manque."""

    if key in runs.columns:
        return runs[key]

    return pd.Series(float("nan"), index=runs.index)


def display_metric(label: str, value) -> None:
    """Affiche une métrique en pourcentage."""

    if value is None or pd.isna(value):
        st.metric(label, "N/A")
    else:
        st.metric(
            label,
            f"{float(value) * 100:.2f} %",
        )


# ============================================================
# EVALUATION INDIVIDUELLE : FONCTIONS PARTAGEES AVEC Home.py
# ============================================================

def run_cached_evaluation(
    image_path,
    label_path,
    detections,
    confidence,
):
    """
    Évalue une image, sauf si la même évaluation existe déjà
    dans la session (même image, même seuil, même annotation).

    `detections` peut être une liste ou une fonction sans argument
    qui retourne la liste : elle n'est appelée que si nécessaire.

    Retourne (résultat, est_nouvelle).
    """

    image_path = Path(image_path)
    label_path = Path(label_path)

    annotation_hash = hashlib.sha256(
        label_path.read_bytes()
    ).hexdigest()

    key = (
        image_path.stem,
        round(float(confidence), 2),
        annotation_hash,
    )

    cache = st.session_state.setdefault("image_evaluations", {})

    if key in cache:
        return cache[key], False

    detection_list = detections() if callable(detections) else detections

    result = evaluate_single_image(
        image_path=image_path,
        label_path=label_path,
        detections=detection_list,
        iou_threshold=IOU_THRESHOLD,
        confidence=confidence,
    )

    cache[key] = result

    return result, True


def render_evaluation_metrics(evaluation_result) -> None:
    """Affiche les métriques d'une évaluation individuelle."""

    metrics = evaluation_result.get("metrics", evaluation_result)

    columns = st.columns(4)

    with columns[0]:
        display_metric("Précision", metrics.get("precision"))

    with columns[1]:
        display_metric("Rappel", metrics.get("recall"))

    with columns[2]:
        display_metric(
            "F1-score",
            metrics.get("f1", metrics.get("f1_score")),
        )

    with columns[3]:
        display_metric("IoU moyen", metrics.get("mean_iou"))

    tp = int(metrics.get("TP", 0))
    fp = int(metrics.get("FP", 0))
    fn = int(metrics.get("FN", 0))

    detail_columns = st.columns(3)

    with detail_columns[0]:
        st.metric("Vrais positifs (TP)", tp)

    with detail_columns[1]:
        st.metric("Faux positifs (FP)", fp)

    with detail_columns[2]:
        st.metric("Faux négatifs (FN)", fn)

    if tp + fp + fn == 0:
        st.info(
            "Aucun objet annoté et aucune détection : "
            "les métriques sont à 0 par convention."
        )

    run_id = evaluation_result.get("run_id")

    if run_id:
        st.caption(f"Run ID MLflow : {run_id}")

    st.caption(
        "Seuil IoU utilisé pour l'association des boîtes : "
        f"{evaluation_result.get('iou_threshold', IOU_THRESHOLD):.2f}"
    )


# ============================================================
# IMPORTATION ET EVALUATION INDIVIDUELLE
# ============================================================

def get_detected_images():
    """Récupère les images originales enregistrées."""

    if not DETECTED_IMAGES_DIR.exists():
        return []

    extensions = {".jpg", ".jpeg", ".png"}

    return sorted(
        [
            path
            for path in DETECTED_IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ],
        key=lambda path: path.name.lower(),
    )


def render_single_image_evaluation():
    """Évalue automatiquement une image dès que son annotation existe."""

    st.subheader("Nouvelle évaluation individuelle")

    st.write(
        "Sélectionnez une image précédemment utilisée dans le module "
        "Détection. Si son annotation YOLO est déjà enregistrée, "
        "l'évaluation se lance automatiquement ; sinon, importez-la."
    )

    images = get_detected_images()

    if not images:
        st.warning(
            "Aucune image enregistrée. "
            "Effectuez d'abord une détection dans le module Détection."
        )
        return

    selected_image = st.selectbox(
        "Sélectionner une image",
        options=images,
        format_func=lambda path: path.name,
        key="evaluation_selected_image",
    )

    if selected_image is None:
        return

    # Affichage de l'image sélectionnée
    try:
        original_image = Image.open(selected_image).convert("RGB")

        st.image(
            original_image,
            caption=f"Image sélectionnée : {selected_image.name}",
            width="stretch",
        )

    except Exception as error:
        st.error(f"Impossible de lire cette image : {error}")
        return

    # Seuil de confiance
    confidence = st.slider(
        "Seuil de confiance du modèle",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key=f"evaluation_confidence_{selected_image.stem}",
    )

    # Importation de l'annotation
    annotation_file = st.file_uploader(
        "Importer l'annotation de référence (.txt)",
        type=["txt"],
        help=(
            "Format YOLO : class_id x_center y_center width height. "
            "Les coordonnées doivent être normalisées entre 0 et 1."
        ),
        key=f"evaluation_annotation_{selected_image.stem}",
    )

    DETECTED_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    label_path = DETECTED_LABELS_DIR / f"{selected_image.stem}.txt"

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
            "Importez le fichier .txt correspondant à l'image "
            "pour lancer l'évaluation."
        )
        return

    if not label_path.read_bytes().strip():
        st.caption(
            "Annotation vide : aucun objet n'est attendu sur cette image."
        )

    st.caption(
        "L'annotation doit correspondre exactement à l'image sélectionnée."
    )

    # Évaluation automatique (la détection n'est lancée que si nécessaire)
    try:
        with st.spinner(
            "Détection et comparaison avec l'annotation..."
        ):
            evaluation_result, is_new = run_cached_evaluation(
                image_path=selected_image,
                label_path=label_path,
                detections=lambda: detect_objects(
                    original_image,
                    confidence=confidence,
                )[1],
                confidence=confidence,
            )

    except Exception as error:
        st.error(
            f"Erreur pendant l'évaluation individuelle : {error}"
        )
        st.exception(error)
        return

    # Recharge la page pour que l'historique MLflow inclue ce nouveau run
    if is_new:
        st.rerun()

    st.divider()
    st.subheader("Résultats de l'évaluation individuelle")

    st.success(
        "Évaluation terminée et enregistrée dans MLflow.",
        icon=":material/check_circle:",
    )

    render_evaluation_metrics(evaluation_result)


# ============================================================
# HISTORIQUE DES EVALUATIONS INDIVIDUELLES
# ============================================================

def render_individual_evaluations(
    runs: pd.DataFrame,
) -> None:

    st.subheader("Historique des évaluations individuelles")

    if runs.empty:
        st.info(
            "Aucune évaluation individuelle enregistrée. "
            "Sélectionnez une image ci-dessus et fournissez son "
            "annotation YOLO pour lancer une évaluation."
        )
        return

    rows = []

    for _, run in runs.iterrows():

        image_name = first_value(
            run,
            [
                "params.image_name",
                "params.image",
                "tags.image",
                "tags.mlflow.runName",
            ],
            "Image inconnue",
        )

        rows.append({
            "Image": image_name,
            "Date": run.get("start_time"),
            "Précision (%)": round(
                metric_value(run, "metrics.precision") * 100, 2
            ),
            "Rappel (%)": round(
                metric_value(run, "metrics.recall") * 100, 2
            ),
            "F1-score (%)": round(
                metric_value(run, "metrics.f1", "metrics.f1_score") * 100,
                2,
            ),
            "IoU moyen (%)": round(
                metric_value(run, "metrics.mean_iou") * 100, 2
            ),
            "TP": count_value(
                run, "metrics.TP", "metrics.true_positives"
            ),
            "FP": count_value(
                run, "metrics.FP", "metrics.false_positives"
            ),
            "FN": count_value(
                run, "metrics.FN", "metrics.false_negatives"
            ),
            "Run ID": run.get("run_id"),
        })

    history = pd.DataFrame(rows)

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"{len(history)} évaluation(s) individuelle(s) "
        "enregistrée(s) dans MLflow."
    )


# ============================================================
# GRAPHIQUES DE VALIDATION
# ============================================================

def render_evaluation_plots() -> None:

    st.subheader("Visualisations de l'évaluation globale")

    plots = [
        ("Courbe Precision-Recall", "BoxPR_curve.png"),
        ("Courbe F1", "BoxF1_curve.png"),
        ("Courbe Precision", "BoxP_curve.png"),
        ("Courbe Recall", "BoxR_curve.png"),
        ("Matrice de confusion", "confusion_matrix.png"),
        ("Matrice normalisée", "confusion_matrix_normalized.png"),
    ]

    available_plots = [
        (title, RESULTS_DIR / filename)
        for title, filename in plots
        if (RESULTS_DIR / filename).is_file()
    ]

    if not available_plots:
        st.info(
            "Aucun graphique disponible. "
            "Lancez une évaluation globale pour générer les visualisations."
        )
        return

    for index in range(0, len(available_plots), 2):

        columns = st.columns(2)

        for column, (title, image_path) in zip(
            columns,
            available_plots[index:index + 2],
        ):

            with column:
                st.markdown(f"**{title}**")
                st.image(
                    str(image_path),
                    width="stretch",
                )


# ============================================================
# EVALUATIONS GLOBALES
# ============================================================

def render_dataset_evaluations(
    runs: pd.DataFrame,
) -> None:

    st.subheader("Évaluation globale du modèle")

    if runs.empty:
        st.warning(
            "Aucune évaluation globale disponible. "
            "Lancez une évaluation du dataset."
        )
        return

    latest_run = runs.iloc[0]

    columns = st.columns(4)

    metrics = [
        ("Précision", "metrics.precision"),
        ("Rappel", "metrics.recall"),
        ("mAP@50", "metrics.mAP50"),
        ("mAP@50-95", "metrics.mAP50_95"),
    ]

    for column, (label, key) in zip(columns, metrics):

        with column:
            display_metric(
                label,
                metric_value(latest_run, key),
            )

    st.caption(
        f"Dernière évaluation : {latest_run.get('start_time', 'Date inconnue')} "
        f"| Run ID : {latest_run.get('run_id', 'Inconnu')}"
    )

    st.divider()

    st.subheader("Historique des évaluations globales")

    history = pd.DataFrame({
        "Run ID": runs["run_id"],
        "Début": runs["start_time"],
        "Précision (%)": metric_column(runs, "metrics.precision") * 100,
        "Rappel (%)": metric_column(runs, "metrics.recall") * 100,
        "mAP@50 (%)": metric_column(runs, "metrics.mAP50") * 100,
        "mAP@50-95 (%)": metric_column(runs, "metrics.mAP50_95") * 100,
    })

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )

    if len(history) > 1:

        st.markdown("**Évolution des performances**")

        chart_data = history[
            [
                "Précision (%)",
                "Rappel (%)",
                "mAP@50 (%)",
                "mAP@50-95 (%)",
            ]
        ].copy()

        chart_data.index = pd.RangeIndex(
            start=1,
            stop=len(chart_data) + 1,
            name="Évaluation",
        )

        st.line_chart(chart_data)


# ============================================================
# PAGE PRINCIPALE
# ============================================================

def render_evaluation_page():

    st.title("Évaluation du modèle")

    st.caption(
        "Analyse des performances de YOLO11n "
        "et suivi des expériences avec MLflow."
    )

    # Récupération des données MLflow
    try:
        all_runs = get_all_evaluation_runs()

        individual_runs, dataset_runs = split_evaluation_runs(
            all_runs
        )

    except Exception as error:
        st.error(
            f"Erreur lors de la récupération des données MLflow : {error}"
        )
        return

    # Nouvelle évaluation globale
    if st.button(
        "Lancer une nouvelle évaluation globale",
        type="primary",
        use_container_width=True,
    ):

        try:
            with st.spinner(
                "Évaluation globale de YOLO11n en cours. "
                "Cela peut prendre quelques minutes..."
            ):
                result = evaluate_model()

            st.success(
                f"Évaluation terminée. Run ID : {result['run_id']}"
            )

            st.rerun()

        except Exception as error:
            st.error(
                f"Erreur pendant l'évaluation globale : {error}"
            )
            st.exception(error)

    st.divider()

    # Évaluation individuelle automatique
    render_single_image_evaluation()

    st.divider()

    # Historique individuel
    render_individual_evaluations(individual_runs)

    st.divider()

    # Évaluation globale
    render_dataset_evaluations(dataset_runs)

    st.divider()

    # Graphiques
    render_evaluation_plots()

    st.divider()

    # Avertissement scientifique
    st.subheader("À propos des résultats")

    st.warning(
        "Le jeu de validation actuel contient seulement deux images "
        "et quatre objets annotés. Les métriques globales sont donc "
        "préliminaires et ne permettent pas de conclure sur les "
        "performances générales du modèle. Les annotations doivent "
        "également être vérifiées."
    )