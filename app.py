from __future__ import annotations

from io import BytesIO
from html import escape
from pathlib import Path

import numpy as np

import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler


st.set_page_config(
    page_title="Materials Intelligence Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #f4f1ea;
        --panel: rgba(255, 255, 255, 0.84);
        --panel-strong: #ffffff;
        --ink: #101828;
        --muted: #5b6472;
        --line: rgba(16, 24, 40, 0.10);
        --accent: #174ea6;
        --accent-2: #7c3aed;
        --accent-3: #0f766e;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(23, 78, 166, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(124, 58, 237, 0.10), transparent 24%),
            linear-gradient(180deg, #f7f6f1 0%, #eef2f7 100%);
        color: var(--ink);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    .hero {
        padding: 1.7rem 1.8rem;
        border-radius: 1.4rem;
        color: white;
        background:
            linear-gradient(135deg, rgba(10, 18, 36, 0.96), rgba(21, 74, 145, 0.96) 58%, rgba(14, 116, 144, 0.92)),
            radial-gradient(circle at top right, rgba(255, 255, 255, 0.16), transparent 25%);
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.20);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.55rem 0 0;
        max-width: 52rem;
        opacity: 0.94;
        font-size: 1.02rem;
        line-height: 1.5;
    }
    .card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 1rem;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        backdrop-filter: blur(10px);
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0 0 0.6rem 0;
    }
    .small-note {
        color: var(--muted);
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def load_many_data(uploaded_files) -> list[tuple[str, pd.DataFrame]]:
    loaded = []
    for uploaded_file in uploaded_files:
        loaded.append((uploaded_file.name, pd.read_csv(uploaded_file)))
    return loaded


def infer_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    preferred = ["sample_id", "sample", "id", "formula", "material", "batch", "run"]
    left_lower = {column.lower(): column for column in left.columns}
    right_lower = {column.lower(): column for column in right.columns}
    for key in preferred:
        if key in left_lower and key in right_lower:
            return [left_lower[key]]

    shared = [column for column in left.columns if column in right.columns]
    string_shared = [column for column in shared if left[column].dtype == "object" or right[column].dtype == "object"]
    if string_shared:
        return string_shared[:1]
    return shared[:1]


def merge_uploaded_tables(loaded_tables: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, dict]:
    report: dict = {"source_files": [name for name, _ in loaded_tables], "merge_steps": []}
    if not loaded_tables:
        raise ValueError("No tables to merge.")

    merged = loaded_tables[0][1].copy()
    for name, table in loaded_tables[1:]:
        join_keys = infer_join_keys(merged, table)
        if join_keys:
            merged = merged.merge(table, on=join_keys, how="left", suffixes=("", f"_{Path(name).stem}"))
            report["merge_steps"].append({"file": name, "keys": join_keys, "strategy": "left join"})
        else:
            table = table.copy()
            table.columns = [f"{column}_{Path(name).stem}" if column in merged.columns else column for column in table.columns]
            merged = pd.concat([merged, table], axis=1)
            report["merge_steps"].append({"file": name, "keys": [], "strategy": "column-wise concat"})
    report["merged_shape"] = merged.shape
    return merged, report


def guess_group_columns(df: pd.DataFrame) -> dict[str, str | None]:
    lower_map = {column.lower(): column for column in df.columns}
    candidates = {
        "sample_id": ["sample", "sample_id", "specimen", "id"],
        "batch": ["batch", "lot", "run", "series"],
        "material": ["material", "alloy", "composition", "sample_type"],
        "condition": ["condition", "temperature", "temp", "pressure", "process", "treatment"],
        "replicate": ["replicate", "rep", "trial", "repeat"],
    }
    selections = {}
    for label, options in candidates.items():
        selections[label] = next((lower_map[name] for name in lower_map if any(opt in name for opt in options)), None)
    return selections


def coerce_and_clean(df: pd.DataFrame, metadata_map: dict[str, str | None], log_transform: bool) -> tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    report: dict = {
        "original_shape": cleaned.shape,
        "dropped_duplicates": 0,
        "outlier_rows": 0,
    }

    cleaned = cleaned.drop_duplicates()
    report["dropped_duplicates"] = report["original_shape"][0] - cleaned.shape[0]

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].astype(str).replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    if metadata_map.get("sample_id") and metadata_map["sample_id"] in cleaned.columns:
        cleaned[metadata_map["sample_id"]] = cleaned[metadata_map["sample_id"]].astype(str)
    if metadata_map.get("batch") and metadata_map["batch"] in cleaned.columns:
        cleaned[metadata_map["batch"]] = cleaned[metadata_map["batch"]].astype(str)

    numeric_columns = cleaned.select_dtypes(include="number").columns.tolist()
    report["numeric_columns"] = numeric_columns
    report["missing_before"] = cleaned.isna().sum().to_dict()

    if numeric_columns:
        numeric_frame = cleaned[numeric_columns].copy()
        if log_transform:
            positive_columns = [column for column in numeric_columns if (numeric_frame[column].dropna() > 0).all()]
            for column in positive_columns:
                numeric_frame[column] = np.log1p(numeric_frame[column])
            report["log_transformed_columns"] = positive_columns
        imputer = SimpleImputer(strategy="median")
        cleaned[numeric_columns] = imputer.fit_transform(numeric_frame)
    else:
        report["log_transformed_columns"] = []

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].fillna("Unknown")

    report["missing_after"] = cleaned.isna().sum().to_dict()
    report["rows_after_cleaning"] = cleaned.shape[0]
    return cleaned, report


def robust_outlier_flags(df: pd.DataFrame, numeric_columns: list[str]) -> pd.Series | None:
    if len(numeric_columns) < 2 or len(df) < 5:
        return None

    numeric = df[numeric_columns]
    medians = numeric.median()
    mad = (numeric - medians).abs().median().replace(0, 1e-9)
    robust_z = ((numeric - medians).abs() / (1.4826 * mad)).fillna(0)
    robust_flag = robust_z.gt(3.5).any(axis=1)

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = (q3 - q1).replace(0, 1e-9)
    iqr_flag = ((numeric < (q1 - 1.5 * iqr)) | (numeric > (q3 + 1.5 * iqr))).any(axis=1)

    scaler = RobustScaler()
    scaled = scaler.fit_transform(numeric)
    forest = IsolationForest(contamination=min(0.1, max(0.02, 3 / len(df))), random_state=42)
    forest_flag = pd.Series(forest.fit_predict(scaled) == -1, index=df.index)

    return robust_flag | iqr_flag | forest_flag


def run_pca(df: pd.DataFrame, max_components: int = 2):
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None, None, None

    scaled = StandardScaler().fit_transform(numeric)
    n_components = min(max_components, numeric.shape[1], len(df))
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(scaled)
    columns = [f"PC{i + 1}" for i in range(coords.shape[1])]
    pca_df = pd.DataFrame(coords, columns=columns, index=df.index)
    return pca, pca_df, pca.explained_variance_ratio_


def run_clustering(df: pd.DataFrame, n_clusters: int):
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2 or len(df) < n_clusters:
        return None

    scaled = StandardScaler().fit_transform(numeric)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return pd.Series(model.fit_predict(scaled), index=df.index, name="cluster")


def build_dataset_summary(df: pd.DataFrame, metadata_map: dict[str, str | None]) -> str:
    lines = [
        f"<b>Rows:</b> {len(df)}",
        f"<b>Columns:</b> {len(df.columns)}",
    ]
    for label, column in metadata_map.items():
        if column:
            lines.append(f"<b>{label.replace('_', ' ').title()}:</b> {escape(column)}")
    return "<br>".join(lines)


def make_html_report(
    cleaned_df: pd.DataFrame,
    cleaning_report: dict,
    metadata_map: dict[str, str | None],
    pca_ratio,
    cluster_counts: pd.Series | None,
    anomaly_count: int | None,
    summary_stats: dict,
) -> str:
    numeric_columns = cleaning_report.get("numeric_columns", [])
    rows = [
        "<html><head><meta charset='utf-8'><title>Materials Data Analysis Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;line-height:1.5;color:#111827;margin:40px;background:#f7f8fb;}",
        ".panel{background:#fff;border:1px solid #dbe2ea;border-radius:14px;padding:22px 24px;margin-bottom:18px;box-shadow:0 10px 28px rgba(15,23,42,0.06)}",
        "h1,h2{margin:0 0 12px 0} h1{font-size:30px} h2{font-size:20px}",
        "table{border-collapse:collapse;width:100%;margin-top:10px} th,td{border:1px solid #dbe2ea;padding:8px;text-align:left;font-size:14px}",
        ".muted{color:#5b6472} .pill{display:inline-block;padding:4px 10px;border-radius:999px;background:#eaf1ff;color:#174ea6;font-size:12px;margin-right:8px}",
        "</style></head><body>",
        "<div class='panel'><h1>Materials Data Analysis Report</h1>",
        "<p class='muted'>Generated from the Materials Intelligence Lab workflow.</p>",
        f"<span class='pill'>Rows: {len(cleaned_df)}</span><span class='pill'>Columns: {len(cleaned_df.columns)}</span>",
        f"<p>{build_dataset_summary(cleaned_df, metadata_map)}</p></div>",
        "<div class='panel'><h2>Cleaning Summary</h2>",
        f"<p>Duplicate rows removed: {cleaning_report['dropped_duplicates']}</p>",
        f"<p>Missing values before cleaning: {sum(cleaning_report['missing_before'].values())}</p>",
        f"<p>Missing values after cleaning: {sum(cleaning_report['missing_after'].values())}</p>",
        f"<p>Numeric variables: {escape(', '.join(numeric_columns) if numeric_columns else 'None')}</p>",
        "</div>",
    ]
    if metadata_map:
        rows.extend([
            "<div class='panel'><h2>Materials Metadata</h2>",
            "<ul>",
        ])
        for key, value in metadata_map.items():
            rows.append(f"<li><b>{escape(key.replace('_', ' ').title())}:</b> {escape(value or 'Not mapped')}</li>")
        rows.extend(["</ul></div>"])
    if pca_ratio is not None:
        rows.extend([
            "<div class='panel'><h2>PCA</h2>",
            f"<p>Explained variance: {escape(', '.join(f'{value:.2%}' for value in pca_ratio))}</p>",
            "</div>",
        ])
    if cluster_counts is not None:
        rows.extend([
            "<div class='panel'><h2>Clustering</h2>",
            cluster_counts.to_frame("samples").to_html(index=True, border=0),
            "</div>",
        ])
    if anomaly_count is not None:
        rows.extend([
            "<div class='panel'><h2>Anomaly Detection</h2>",
            f"<p>Potential anomalies flagged: {anomaly_count}</p>",
            "</div>",
        ])
    rows.extend([
        "<div class='panel'><h2>Descriptive Statistics</h2>",
        pd.DataFrame(summary_stats).T.round(4).to_html(border=0),
        "</div>",
        "</body></html>",
    ])
    return "".join(rows)


def html_to_pdf_bytes(html_report: str) -> bytes:
    styles = getSampleStyleSheet()
    doc_buffer = BytesIO()
    doc = SimpleDocTemplate(doc_buffer, pagesize=A4, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = []
    story.append(Paragraph("Materials Data Analysis Report", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("This PDF summarizes the main findings from the uploaded dataset.", styles["BodyText"]))
    story.append(Spacer(1, 0.2 * inch))

    # Lightweight PDF summary derived from the HTML-ready content.
    text = html_report.replace("<br>", "\n")
    plain_sections = []
    for marker in ["Cleaning Summary", "Materials Metadata", "PCA", "Clustering", "Anomaly Detection", "Descriptive Statistics"]:
        if marker in text:
            plain_sections.append(marker)
    if plain_sections:
        story.append(Paragraph("Sections included: " + ", ".join(plain_sections), styles["BodyText"]))
        story.append(Spacer(1, 0.15 * inch))

    story.append(Table([["Report", "Generated by", "Format"], ["Materials Data Analysis", "Streamlit app", "PDF"]], colWidths=[2.4 * inch, 2.0 * inch, 1.0 * inch]))
    story[-1].setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174ea6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe2ea")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ]
        )
    )
    doc.build(story)
    return doc_buffer.getvalue()


st.markdown(
    """
    <div class="hero">
      <h1>Materials Intelligence Lab</h1>
      <p>A research-oriented materials analytics workspace for experimental CSVs, metadata-aware grouping, robust cleaning, PCA, clustering, anomaly detection, and exportable reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

uploaded_files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload a CSV file to begin. The app will map materials metadata, clean the data, run analysis, and export HTML/PDF reports.")
    st.stop()

loaded_tables = load_many_data(uploaded_files)
source_names = [name for name, _ in loaded_tables]

if len(loaded_tables) == 1:
    raw_df = loaded_tables[0][1]
    merge_report = {"source_files": source_names, "merge_steps": [], "merged_shape": raw_df.shape}
else:
    merge_mode = st.sidebar.radio("Multi-file handling", ["Auto-merge", "Use first file only"], index=0)
    if merge_mode == "Auto-merge":
        raw_df, merge_report = merge_uploaded_tables(loaded_tables)
    else:
        raw_df = loaded_tables[0][1]
        merge_report = {
            "source_files": source_names,
            "merge_steps": [{"file": source_names[0], "keys": [], "strategy": "first file only"}],
            "merged_shape": raw_df.shape,
        }

guessed = guess_group_columns(raw_df)

with st.sidebar:
    st.markdown("### Materials Metadata")
    st.caption("Map fields used for grouping and scientific reporting.")
    sample_id = st.selectbox("Sample ID", ["None"] + list(raw_df.columns), index=(["None"] + list(raw_df.columns)).index(guessed["sample_id"]) if guessed["sample_id"] in raw_df.columns else 0)
    batch = st.selectbox("Batch / Run", ["None"] + list(raw_df.columns), index=(["None"] + list(raw_df.columns)).index(guessed["batch"]) if guessed["batch"] in raw_df.columns else 0)
    material = st.selectbox("Material / Composition", ["None"] + list(raw_df.columns), index=(["None"] + list(raw_df.columns)).index(guessed["material"]) if guessed["material"] in raw_df.columns else 0)
    condition = st.selectbox("Condition / Process Field", ["None"] + list(raw_df.columns), index=(["None"] + list(raw_df.columns)).index(guessed["condition"]) if guessed["condition"] in raw_df.columns else 0)
    replicate = st.selectbox("Replicate / Trial", ["None"] + list(raw_df.columns), index=(["None"] + list(raw_df.columns)).index(guessed["replicate"]) if guessed["replicate"] in raw_df.columns else 0)

    st.markdown("### Preprocessing")
    log_transform = st.checkbox("Apply log1p to strictly positive numeric columns", value=False)
    remove_outliers = st.checkbox("Remove flagged outliers from analysis", value=False)

metadata_map = {
    "sample_id": None if sample_id == "None" else sample_id,
    "batch": None if batch == "None" else batch,
    "material": None if material == "None" else material,
    "condition": None if condition == "None" else condition,
    "replicate": None if replicate == "None" else replicate,
}

cleaned_df, cleaning_report = coerce_and_clean(raw_df, metadata_map, log_transform=log_transform)
numeric_columns = cleaned_df.select_dtypes(include="number").columns.tolist()
outlier_flags = robust_outlier_flags(cleaned_df, numeric_columns)
if outlier_flags is not None:
    cleaned_for_analysis = cleaned_df.loc[~outlier_flags].copy() if remove_outliers else cleaned_df.copy()
    outlier_count = int(outlier_flags.sum())
else:
    cleaned_for_analysis = cleaned_df.copy()
    outlier_count = None

sample_label = metadata_map.get("sample_id") or "Sample"
group_column = metadata_map.get("batch") or metadata_map.get("material") or metadata_map.get("condition")

st.markdown(
    f"""
    <div class="card">
      <div class="section-title">Dataset Overview</div>
      <div class="small-note">{build_dataset_summary(cleaned_df, metadata_map)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

with st.expander("File merge summary"):
    st.write(f"Loaded files: {', '.join(source_names)}")
    st.json(merge_report)

metric_cols = st.columns(4)
metric_cols[0].metric("Rows", f"{cleaned_df.shape[0]}")
metric_cols[1].metric("Columns", f"{cleaned_df.shape[1]}")
metric_cols[2].metric("Numeric columns", f"{len(numeric_columns)}")
metric_cols[3].metric("Missing cells", f"{int(raw_df.isna().sum().sum())}")

if outlier_count is not None:
    st.caption(f"Robust outlier scan flagged {outlier_count} rows using IQR, robust z-scores, and Isolation Forest.")

st.subheader("Preview")
st.dataframe(cleaned_df.head(20), use_container_width=True)

if group_column and group_column in cleaned_df.columns:
    group_counts = cleaned_df.groupby(group_column).size().sort_values(ascending=False)
    st.subheader("Sample Grouping")
    group_chart = px.bar(group_counts.reset_index(name="count"), x=group_column, y="count", title=f"Samples by {group_column}")
    st.plotly_chart(group_chart, use_container_width=True)

left, right = st.columns([1, 2])

with left:
    st.subheader("Analysis Controls")
    if len(numeric_columns) >= 2:
        pca_max = min(5, len(numeric_columns))
        pca_components = st.slider("PCA components", 2, pca_max, min(2, pca_max))
    else:
        pca_components = 2
        st.caption("PCA needs at least two numeric columns.")

    if len(cleaned_for_analysis) > 2:
        cluster_max = min(8, len(cleaned_for_analysis) - 1)
        n_clusters = st.slider("KMeans clusters", 2, cluster_max, min(3, cluster_max))
    else:
        n_clusters = 2
        st.caption("Clustering needs more rows.")

with right:
    st.subheader("Exploration")
    if numeric_columns:
        x_axis = st.selectbox("X axis", numeric_columns, index=0)
        y_axis = st.selectbox("Y axis", numeric_columns, index=min(1, len(numeric_columns) - 1))
        color_options = ["None"] + [column for column in [metadata_map.get("batch"), metadata_map.get("material"), metadata_map.get("condition")] if column and column in cleaned_df.columns]
        color_column = st.selectbox("Color by", color_options, index=0)
        plot_df = cleaned_df if color_column == "None" else cleaned_df
        fig = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color=None if color_column == "None" else color_column,
            hover_data=[column for column in [metadata_map.get("sample_id"), metadata_map.get("batch"), metadata_map.get("material")] if column and column in cleaned_df.columns],
            title=f"{y_axis} vs {x_axis}",
            template="plotly_white",
        )
        fig.update_layout(font=dict(family="Aptos, Arial, sans-serif"), title=dict(x=0.02))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns found for visualization.")

pca_model, pca_df, pca_ratio = run_pca(cleaned_for_analysis, max_components=pca_components)

if pca_df is not None:
    st.subheader("PCA Analysis")
    pca_plot = pca_df.copy()
    if len(pca_plot.columns) >= 2:
        pca_plot["group"] = cleaned_for_analysis[group_column] if group_column and group_column in cleaned_for_analysis.columns else "All samples"
        fig = px.scatter(
            pca_plot,
            x="PC1",
            y="PC2",
            color="group",
            title="PCA projection",
            template="plotly_white",
        )
        fig.update_layout(font=dict(family="Aptos, Arial, sans-serif"), title=dict(x=0.02))
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Explained variance: " + ", ".join(f"{value:.2%}" for value in pca_ratio))

cluster_labels = run_clustering(cleaned_for_analysis, n_clusters=n_clusters)
if cluster_labels is not None:
    st.subheader("Clustering")
    clustered = cleaned_for_analysis.copy()
    clustered["cluster"] = cluster_labels.astype(str)
    if numeric_columns:
        fig = px.scatter(
            clustered,
            x=numeric_columns[0],
            y=numeric_columns[min(1, len(numeric_columns) - 1)],
            color="cluster",
            title="Cluster assignment",
            template="plotly_white",
        )
        fig.update_layout(font=dict(family="Aptos, Arial, sans-serif"), title=dict(x=0.02))
        st.plotly_chart(fig, use_container_width=True)
    st.write(clustered["cluster"].value_counts().sort_index())
else:
    clustered = cleaned_for_analysis.copy()

anomaly_labels = robust_outlier_flags(cleaned_for_analysis, numeric_columns)
if anomaly_labels is not None:
    st.subheader("Anomaly Detection")
    anomalies = cleaned_for_analysis.copy()
    anomalies["anomaly_flag"] = anomaly_labels
    anomaly_count = int(anomalies["anomaly_flag"].sum())
    st.metric("Potential anomalies", anomaly_count)
    st.dataframe(anomalies[anomalies["anomaly_flag"]], use_container_width=True)
else:
    anomaly_count = None

summary_stats = cleaned_for_analysis.select_dtypes(include="number").describe().to_dict()
html_report = make_html_report(
    cleaned_for_analysis,
    cleaning_report,
    metadata_map,
    pca_ratio,
    clustered["cluster"].value_counts().sort_index() if "cluster" in clustered.columns else None,
    anomaly_count,
    summary_stats,
)
pdf_report = html_to_pdf_bytes(html_report)

st.subheader("Scientific Report")
report_col1, report_col2 = st.columns(2)
with report_col1:
    st.download_button(
        "Download HTML report",
        data=html_report.encode("utf-8"),
        file_name="materials_analysis_report.html",
        mime="text/html",
    )
with report_col2:
    st.download_button(
        "Download PDF report",
        data=pdf_report,
        file_name="materials_analysis_report.pdf",
        mime="application/pdf",
    )

with st.expander("Preview report HTML"):
    st.code(html_report[:5000], language="html")

export_df = cleaned_for_analysis.copy()
if "cluster" in clustered.columns:
    export_df["cluster"] = clustered["cluster"]
if anomaly_labels is not None:
    export_df["anomaly_flag"] = anomaly_labels

st.download_button(
    "Download processed CSV",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="processed_materials_data.csv",
    mime="text/csv",
)
