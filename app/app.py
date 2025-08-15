# app.py
# Streamlit dashboard for Simulacrum Lung Cancer (C34)

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lifelines import KaplanMeierFitter

st.set_page_config(page_title="C34 Simulacrum Explorer", page_icon="🫁", layout="wide")

# ------------------------ Branding / Logo ------------------------
# Put a logo file next to app.py (e.g., logo.png) or set an env var:
#   APP_LOGO_PATH=/content/mylogo.png
# LOGO_PATH = os.environ.get("APP_LOGO_PATH", "logo.png")
LOGO_PATH = "./assets/logo.png"

def show_header():
    left, right = st.columns([1, 6], vertical_alignment="center")
    with left:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, caption=None, use_container_width=True)
        else:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    with right:
        st.title("🫁 Simulacrum C34 (Lung) – Interactive Explorer")
        st.caption("Patient-centric master table (synthetic). Use filters on the left to subset in real time.")

# ------------------------ Data utils ------------------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer):
    """Load CSV and coerce common date columns to datetime."""
    df = pd.read_csv(path_or_buffer)
    date_cols = [
        "DIAGNOSISDATEBEST", "VITALSTATUSDATE",
        "START_DATE_OF_REGIMEN", "DATE_DECISION_TO_TREAT",
        "DATE_OF_FINAL_TREATMENT", "APPTDATE",
        "DECISIONTOTREATDATE", "EARLIESTCLINAPPROPDATE"
    ]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute survival fields and create fallbacks so the app is robust
    even if the CSV lacks some columns.
    """
    d = df.copy()

    # Stage group: prefer descriptive label, fall back to STAGE_BEST → "Stage X"
    if "Stage_Group" not in d.columns:
        if "stage_label" in d.columns:
            d["Stage_Group"] = d["stage_label"].fillna(
                d.get("STAGE_BEST", pd.Series(index=d.index)).astype(str).str[0].radd("Stage ")
            )
        elif "STAGE_BEST" in d.columns:
            d["Stage_Group"] = d["STAGE_BEST"].astype(str).str[0].radd("Stage ")
        else:
            d["Stage_Group"] = "Unknown"

    # Survival fields
    if "VITALSTATUS" in d.columns:
        d["event_observed"] = (d["VITALSTATUS"] == "D").astype(int)
    if {"DIAGNOSISDATEBEST", "VITALSTATUSDATE"}.issubset(d.columns):
        durations = (d["VITALSTATUSDATE"] - d["DIAGNOSISDATEBEST"]).dt.days
        d["followup_days"] = pd.to_numeric(durations, errors="coerce").clip(lower=0)
    else:
        d["followup_days"] = np.nan

    # Gene testing group
    if "num_genes_tested" in d.columns:
        d["tested_group"] = np.where(d["num_genes_tested"].fillna(0) > 0, "Tested", "Not Tested")
    else:
        d["tested_group"] = "Unknown"

    # Treatment flags (fallbacks)
    if "received_sact" not in d.columns:
        if "MERGED_REGIMEN_ID" in d.columns:
            d["received_sact"] = d["MERGED_REGIMEN_ID"].notna()
        elif "INTENT_OF_TREATMENT" in d.columns:
            d["received_sact"] = d["INTENT_OF_TREATMENT"].notna()
        else:
            d["received_sact"] = False

    if "received_rt" not in d.columns:
        if "RADIOTHERAPYINTENT" in d.columns:
            d["received_rt"] = d["RADIOTHERAPYINTENT"].notna()
        else:
            d["received_rt"] = False

    # Age groups
    if "AGE" in d.columns:
        d["age_group"] = pd.cut(
            d["AGE"], bins=[0, 64, 74, 120],
            labels=["<65", "65-74", "75+"], include_lowest=True
        )
    else:
        d["age_group"] = "Unknown"

    # Clean labels for display
    for col in ["gender_label", "ethnicity_label", "Stage_Group", "rt_label"]:
        if col in d.columns:
            d[col] = d[col].fillna("Unknown")

    return d

# ------------------------ KM helpers ------------------------
def _km_mask(df: pd.DataFrame) -> pd.Series:
    """Valid KM rows: finite, non-negative durations and non-null event flags."""
    if not {"followup_days", "event_observed"}.issubset(df.columns):
        return pd.Series(False, index=df.index)
    durations = pd.to_numeric(df["followup_days"], errors="coerce")
    events = pd.to_numeric(df["event_observed"], errors="coerce")
    return durations.notna() & events.notna() & (durations >= 0)

# ------------------------ UI helpers ------------------------
def kpi_tiles(df: pd.DataFrame):
    n_pat = df["PATIENTID"].nunique() if "PATIENTID" in df.columns else len(df)
    med_age = float(df["AGE"].median()) if "AGE" in df.columns else np.nan
    med_fu = float(df["followup_days"].median()) if "followup_days" in df.columns else np.nan

    one_year = np.nan
    mask = _km_mask(df)
    if mask.sum() >= 10:
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, "followup_days"], event_observed=df.loc[mask, "event_observed"])
        try:
            one_year = float(kmf.predict(365.0))
        except Exception:
            one_year = np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{n_pat:,}")
    c2.metric("Median age", f"{med_age:.1f}" if not np.isnan(med_age) else "—")
    c3.metric("Median follow-up (days)", f"{med_fu:.0f}" if not np.isnan(med_fu) else "—")
    c4.metric("KM @ 1 year", f"{one_year:.2%}" if not np.isnan(one_year) else "—")

def count_bar(df, x, title, color=None, order=None):
    if x not in df.columns:
        st.info(f"Column '{x}' missing")
        return
    vc = df[x].fillna("Unknown").value_counts().reset_index()
    vc.columns = [x, "count"]
    if order is None:
        order = sorted(vc[x].tolist())
    fig = px.bar(vc, x=x, y="count", color=color, title=title,
                 category_orders={x: order})
    st.plotly_chart(fig, use_container_width=True)

def stacked_bar(df, x, y, title):
    if x not in df.columns or y not in df.columns:
        st.info(f"Need columns '{x}' and '{y}'")
        return
    tbl = (df[[x, y]].fillna("Unknown")
           .groupby([x, y]).size().reset_index(name="count"))
    fig = px.bar(tbl, x=x, y="count", color=y, barmode="stack", title=title)
    st.plotly_chart(fig, use_container_width=True)

def plot_km(df, group_col=None, title="Kaplan–Meier survival"):
    mask_all = _km_mask(df)
    if mask_all.sum() < 10:
        st.info("Not enough valid rows for survival plot.")
        return

    kmf = KaplanMeierFitter()
    fig = go.Figure()

    if group_col and group_col in df.columns:
        for name, grp in df.groupby(group_col):
            m = _km_mask(grp)
            if m.sum() < 10:
                continue
            kmf.fit(grp.loc[m, "followup_days"], grp.loc[m, "event_observed"], label=str(name))
            sf = kmf.survival_function_.reset_index()
            fig.add_trace(go.Scatter(x=sf["timeline"], y=sf[kmf._label],
                                     mode="lines", name=str(name)))
    else:
        kmf.fit(df.loc[mask_all, "followup_days"], df.loc[mask_all, "event_observed"], label="All")
        sf = kmf.survival_function_.reset_index()
        fig.add_trace(go.Scatter(x=sf["timeline"], y=sf[kmf._label],
                                 mode="lines", name="All"))

    fig.update_layout(title=title, xaxis_title="Days since diagnosis",
                      yaxis_title="Survival probability")
    st.plotly_chart(fig, use_container_width=True)

def corr_heatmap(df, height=900, width=1400):
    """
    Bigger correlation heatmap.
    - Default height=900, width=1400 (you can tweak in the call).
    """
    drop_ids = [c for c in ["PATIENTID", "LINKNUMBER", "LINK_NUMBER",
                            "MERGED_REGIMEN_ID", "PRESCRIPTIONID",
                            "RADIOTHERAPYEPISODEID"] if c in df.columns]
    num = df.drop(columns=drop_ids, errors="ignore").select_dtypes(include=np.number)
    num = num.loc[:, num.nunique() >= 2]  # drop constants
    if num.shape[1] < 2:
        st.info("Not enough numeric columns with variance for correlation.")
        return
    corr = num.corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
        title="Correlation heatmap (variance-filtered)", height=height, width=width
    )
    # Use provided width/height → set use_container_width=False
    st.plotly_chart(fig, use_container_width=False)

# ------------------------ Load data with success banner ------------------------
st.sidebar.caption("Upload merged CSV (c34_merged.csv) or place it next to app.py.")
upl = st.sidebar.file_uploader("Upload CSV", type=["csv"])
default_path = "c34_merged.csv"

if upl is not None:
    df_raw = load_csv(upl)
    st.success(f"✅ Loaded uploaded file with {len(df_raw):,} rows.")
elif os.path.exists(default_path):
    df_raw = load_csv(default_path)
    st.success(f"✅ Loaded {default_path} with {len(df_raw):,} rows.")
else:
    st.error("❌ No CSV found. Upload in the sidebar or place ./c34_merged.csv next to app.py.")
    st.stop()

df = ensure_fields(df_raw)

# ------------------------ Sidebar filters ------------------------
with st.sidebar:
    st.header("Filters")

    if "Stage_Group" in df.columns:
        stages = sorted(df["Stage_Group"].dropna().unique().tolist())
        sel = st.multiselect("Stage", stages, default=stages)
        df = df[df["Stage_Group"].isin(sel)]

    if "age_group" in df.columns:
        ages = [x for x in df["age_group"].dropna().unique()]
        sel = st.multiselect("Age group", ages, default=ages)
        df = df[df["age_group"].isin(sel)]

    if "gender_label" in df.columns:
        gens = [x for x in df["gender_label"].dropna().unique()]
        sel = st.multiselect("Gender", gens, default=gens)
        df = df[df["gender_label"].isin(sel)]

    if "ethnicity_label" in df.columns:
        eth = [x for x in df["ethnicity_label"].dropna().unique()]
        sel = st.multiselect("Ethnicity", eth, default=eth)
        df = df[df["ethnicity_label"].isin(sel)]

    if "received_sact" in df.columns and st.checkbox("Only SACT recipients", value=False):
        df = df[df["received_sact"] == True]

    if "received_rt" in df.columns and st.checkbox("Only RT recipients", value=False):
        df = df[df["received_rt"] == True]

    q = st.text_input("Search PATIENTID contains", value="")
    if q and "PATIENTID" in df.columns:
        df = df[df["PATIENTID"].astype(str).str.contains(q, na=False)]

# ------------------------ Layout ------------------------
show_header()

# KPIs
kpi_tiles(df)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Treatments", "Survival", "Equity", "Quality"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        count_bar(df, "Stage_Group", "Stage distribution")
    with c2:
        count_bar(df, "gender_label", "Gender distribution")

    c3, c4 = st.columns(2)
    with c3:
        count_bar(df, "ethnicity_label", "Ethnicity distribution")
    with c4:
        if "AGE" in df.columns:
            st.plotly_chart(px.histogram(df, x="AGE", nbins=30, title="Age at diagnosis", marginal="rug"),
                            use_container_width=True)

with tab2:
    if "received_sact" in df.columns:
        stacked_bar(df, "Stage_Group", "received_sact", "SACT uptake by stage")
    if "received_rt" in df.columns:
        stacked_bar(df, "Stage_Group", "received_rt", "Radiotherapy uptake by stage")
    if "rt_label" in df.columns and df.get("received_rt", pd.Series(False, index=df.index)).any():
        count_bar(df[df["received_rt"] == True], "rt_label", "Radiotherapy intent mix")

    # Time on treatment proxy
    if {"START_DATE_OF_REGIMEN", "DATE_OF_FINAL_TREATMENT"}.issubset(df.columns):
        tmp = df.dropna(subset=["START_DATE_OF_REGIMEN", "DATE_OF_FINAL_TREATMENT"]).copy()
        if not tmp.empty:
            tmp["time_on_tx"] = (tmp["DATE_OF_FINAL_TREATMENT"] - tmp["START_DATE_OF_REGIMEN"]).dt.days
            st.plotly_chart(px.box(tmp, x="Stage_Group", y="time_on_tx", points=False,
                                   title="Time on treatment (days) by stage"),
                            use_container_width=True)

with tab3:
    plot_km(df, None, "Overall survival")
    plot_km(df, "Stage_Group", "Survival by stage")
    if "tested_group" in df.columns:
        plot_km(df, "tested_group", "Survival by gene-testing status")

with tab4:
    if {"ethnicity_label", "event_observed"}.issubset(df.columns):
        tbl = (df[["ethnicity_label", "event_observed"]]
               .groupby(["ethnicity_label", "event_observed"]).size()
               .reset_index(name="count"))
        st.plotly_chart(px.bar(tbl, x="ethnicity_label", y="count", color="event_observed",
                               barmode="stack", title="Outcome by ethnicity (Death=1)"),
                        use_container_width=True)
    if {"gender_label", "received_sact"}.issubset(df.columns):
        stacked_bar(df, "gender_label", "received_sact", "SACT uptake by gender")

with tab5:
    # Bigger heatmap by default (height=900, width=1400); adjust if you want
    corr_heatmap(df, height=900, width=1400)
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if not miss.empty:
        miss_fig = px.bar(miss.reset_index(), x="index", y=0,
                          title="Missingness by column",
                          labels={"index": "Column", "0": "Proportion missing"})
        miss_fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(miss_fig, use_container_width=True)

st.divider()
st.download_button("⬇️ Download filtered CSV",
                   df.to_csv(index=False).encode("utf-8"),
                   file_name="c34_filtered.csv", mime="text/csv")
