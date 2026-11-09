"""FairPrice Auto - Streamlit interface for the prototype.

Run from the project folder (the one that contains data/ and data/models/):
    streamlit run app.py
"""
import base64
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

warnings.filterwarnings("ignore")

DATA_FOLDER = "data"
MODEL_DIR = os.path.join(DATA_FOLDER, "models")
LOGO_PATH = os.path.join("assets", "logo2.png")
REFERENCE_YEAR = 2025
CATEGORY_LABELS = ["well_below_market", "below_market", "in_line", "above_market", "well_above_market"]
CATEGORY_TEXT = {
    "well_below_market": "Well below market", "below_market": "Below market", "in_line": "In line with market",
    "above_market": "Above market", "well_above_market": "Well above market",
}

st.set_page_config(page_title="FairPrice Auto", page_icon="🚗", layout="wide")

# --------------------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;700&family=Mulish:wght@400;600;700&display=swap');
:root { --navy:#0B1B2D; --card:#13263C; --copper:#D08259; --light:#E4E1DC; --cream:#F3EEE8; --soft:#C9D3DE; --muted:#A9B6C6; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *, [data-testid="stWidgetLabel"] *, .stButton button, input { font-family:'Mulish', Arial, sans-serif !important; }
.stApp { background: var(--navy); color: var(--cream); }
h1, h2, h3 { font-family:'Outfit', Arial, sans-serif !important; color: var(--copper) !important; }
section[data-testid="stSidebar"] { background: var(--card); }
section[data-testid="stSidebar"] * { color: var(--cream); }
.block-container { padding-top: 3.5rem; max-width: 1250px; }
.fp-header { display:flex; align-items:center; justify-content:space-between; padding: 8px 0 18px 0; border-bottom: 3px solid var(--copper); margin-bottom: 26px; }
.fp-title { font-family:'Outfit', Arial, sans-serif; font-size: 46px; font-weight:700; color: var(--cream); line-height:1; }
.fp-sub { color: var(--soft); font-size: 18px; margin-top: 8px; }
.fp-eyebrow { color: var(--copper); font-weight:700; letter-spacing:3px; font-size:13px; text-transform:uppercase; margin-bottom:6px; }
.fp-card { background: var(--card); border-radius: 10px; padding: 22px 26px; margin-bottom: 18px; }
.fp-card-light { background: #FBFAF7; color: var(--navy); border-radius: 10px; padding: 22px 26px; margin-bottom: 18px; }
.fp-label { color: var(--muted); font-size: 14px; }
.fp-label-dark { color: #4A5868; font-size: 14px; }
.fp-big { font-family:'Outfit', Arial, sans-serif; font-size: 34px; font-weight:700; }
.fp-pill { display:inline-block; padding: 6px 18px; border-radius: 30px; font-weight:700; font-size: 15px; }
.fp-signal { border-left: 5px solid var(--copper); background: var(--card); padding: 12px 16px; margin: 8px 0; border-radius: 0 8px 8px 0; color: var(--cream); }
.fp-ok { border-left: 5px solid #5D6F86; background: var(--card); padding: 12px 16px; margin: 8px 0; border-radius: 0 8px 8px 0; color: var(--soft); }
.fp-foot { color: var(--muted); font-size: 13px; margin-top: 30px; border-top: 1px solid #3A4B60; padding-top: 12px; }
div.stButton > button { background: var(--copper); color: var(--navy); border: none; font-weight: 700; }
div.stButton > button:hover { background: #E3A07D; color: var(--navy); }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    price_model = joblib.load(os.path.join(MODEL_DIR, "lgbm_price_model.joblib"))
    q10 = joblib.load(os.path.join(MODEL_DIR, "lgbm_price_q10.joblib"))
    q90 = joblib.load(os.path.join(MODEL_DIR, "lgbm_price_q90.joblib"))
    iso_forest = joblib.load(os.path.join(MODEL_DIR, "iso_forest.joblib"))
    with open(os.path.join(MODEL_DIR, "price_model_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    return price_model, q10, q90, iso_forest, shap.TreeExplainer(iso_forest), config


@st.cache_data(show_spinner="Loading data...")
def load_data():
    train = pd.read_csv(os.path.join(DATA_FOLDER, "train_with_price.csv"))
    test = pd.read_csv(os.path.join(DATA_FOLDER, "test_with_price.csv"))
    return train, test


def add_derived(df):
    """Derived features, recomputed after any edit (same formulas as the notebook)."""
    df = df.copy()
    df["car_age"] = (REFERENCE_YEAR - df["registration_year"]).clip(lower=0)
    df["km_per_year"] = df["mileage_km"] / np.maximum(df["car_age"], 1)
    df["price_per_year"] = df["price"] / np.maximum(df["car_age"], 1)
    vol = df["cylinders_volume_cc"].replace(0, np.nan)
    df["hp_per_cc"] = (df["power_hp"] / vol).fillna(0)
    return df


@st.cache_resource(show_spinner="Preparing reference statistics...")
def prepare_reference(_train, _config, _iso_forest):
    train = add_derived(_train)
    excluded = ["price_negotiable", "price_tax_deductible", "seller_is_dealer", "n_missing_fields"]
    ordinal = [c for c in ["fuel_category", "transmission", "body_type"] if c not in excluded]
    freq = [c for c in ["make", "model"] if c not in excluded]
    numerical = [c for c in _config["numerical_cols"] + ["price", "price_per_year", "hp_per_cc"] if c not in excluded]

    categories = {c: train[c].astype("category").cat.categories for c in ordinal}
    frequencies = {c: train[c].value_counts(normalize=True) for c in freq}

    X_train = train[numerical].copy()
    for c in ordinal:
        X_train[c] = pd.Categorical(train[c], categories=categories[c]).codes
    for c in freq:
        X_train[f"{c}_freq"] = train[c].map(frequencies[c]).fillna(0)
    scores = _iso_forest.decision_function(X_train.fillna(0))
    min_score = float(np.min(scores))

    train["age_group"] = pd.cut(train["car_age"], bins=[-1, 3, 7, 12, 100], labels=["0-3", "4-7", "8-12", "12+"])
    medians = train.groupby(["make", "model", "age_group"], observed=False)[numerical].median()

    def robust_mad(x):
        mad = np.median(np.abs(x - np.median(x)))
        return mad * 1.4826 if mad > 0 else 1.0

    peers = train.groupby(["make", "model", "registration_year"]).agg(
        peer_median=("price", "median"), peer_mad=("price", robust_mad), peer_n=("price", "size")).reset_index()
    return dict(ordinal=ordinal, freq=freq, numerical=numerical, categories=categories, frequencies=frequencies,
                thresholds=(min_score * 0.66, min_score * 0.33, 0.0), medians=medians, peers=peers,
                train_categories={c: train[c].astype("category").cat.categories for c in _config["categorical_cols"]})


# --------------------------------------------------------------------------- pipeline
def price_check(ad, config, ref, price_model, q10, q90):
    X = ad[config["features"]].copy()
    for c in config["categorical_cols"]:
        X[c] = pd.Categorical(X[c], categories=ref["train_categories"][c])
    ideal = float(np.exp(price_model.predict(X)[0]))
    low, high = float(np.exp(q10.predict(X)[0])), float(np.exp(q90.predict(X)[0]))
    residual = float(np.log(ad["price"].iloc[0]) - np.log(ideal))
    thresholds = list(config["residual_thresholds"].values())
    category = str(pd.cut([residual], bins=[-np.inf, *thresholds, np.inf], labels=CATEGORY_LABELS)[0])
    return dict(ideal=ideal, low=low, high=high, residual=residual, category=category, thresholds=thresholds)


def coherence_rules(row):
    fuel = str(row.get("fuel_category", "")).strip().lower()
    get = lambda k, d=0: d if pd.isna(row.get(k, d)) else row.get(k, d)
    rules = {
        "A new car with more than 1,000 km": bool(get("is_new", False)) and get("mileage_km") > 1000,
        "Declared both new and used": bool(get("is_new", False)) and bool(get("is_used", False)),
        "Power in hp does not match power in kW": abs(get("power_hp") - get("power_kw") * 1.35962) > 5,
        "Electric car with cylinders or engine displacement": fuel == "electric" and (get("cylinders") > 0 or get("cylinders_volume_cc") > 0),
        "Combustion car with an electric range": fuel in ["diesel", "gasoline", "lpg", "cng"] and get("electric_range_km") > 0,
        "Registration date in the future": get("registration_year") > REFERENCE_YEAR,
        "A new car with previous owners": bool(get("is_new", False)) and get("nr_prev_owners") > 0,
        "A new car with a declared accident": bool(get("is_new", False)) and bool(get("had_accident", False)),
        "More than 50,000 km per year": get("km_per_year") > 50000,
        "Used car, 3+ years old, under 1,000 km per year": bool(get("is_used", False)) and get("car_age") >= 3 and get("km_per_year") < 1000,
    }
    return [name for name, hit in rules.items() if hit]


def anomaly_check(ad, ref, iso_forest, explainer):
    X = pd.DataFrame(index=[0])
    for c in ref["numerical"]:
        X[c] = ad[c].values[0] if c in ad.columns else 0
    for c in ref["ordinal"]:
        X[c] = pd.Categorical(ad[c], categories=ref["categories"][c]).codes
    for c in ref["freq"]:
        X[f"{c}_freq"] = ad[c].map(ref["frequencies"][c]).fillna(0).values
    X = X.fillna(0)
    score = float(iso_forest.decision_function(X)[0])
    t_high, t_med, t_low = ref["thresholds"]
    risk = "high" if score < t_high else "medium" if score < t_med else "low" if score < t_low else "none"

    features = []
    if risk != "none":
        shap_values = explainer.shap_values(X)[0]
        age = ad["car_age"].iloc[0]
        group = "0-3" if age <= 3 else "4-7" if age <= 7 else "8-12" if age <= 12 else "12+"
        try:
            med = ref["medians"].loc[(ad["make"].iloc[0], ad["model"].iloc[0], group)]
        except KeyError:
            med = None
        for idx in np.argsort(shap_values)[:3]:
            name = X.columns[idx]
            m = med[name] if (med is not None and name in med.index and pd.notna(med[name])) else None
            features.append(dict(feature=name, value=float(X.iloc[0, idx]), median=None if m is None else float(m),
                                 shap=float(shap_values[idx])))

    peer = ref["peers"][(ref["peers"]["make"] == ad["make"].iloc[0]) & (ref["peers"]["model"] == ad["model"].iloc[0])
                        & (ref["peers"]["registration_year"] == ad["registration_year"].iloc[0])]
    # need at least 3 peers with some spread, otherwise the z-score is meaningless
    if peer.empty or peer["peer_n"].iloc[0] < 3 or peer["peer_mad"].iloc[0] <= 1.0:
        z = None
    else:
        z = float((ad["price"].iloc[0] - peer["peer_median"].iloc[0]) / peer["peer_mad"].iloc[0])
    return dict(score=score, risk=risk, features=features, rules=coherence_rules(ad.iloc[0]), z=z)


# --------------------------------------------------------------------------- helpers for display
def euro(x):
    return f"{x:,.0f} €"


def nice(feature):
    return feature.replace("_freq", " (frequency)").replace("_", " ")


def scale_html(residual, thresholds):
    t = [-0.35] + thresholds + [0.35]
    widths = [12, 16, 44, 16, 12]
    colors = ["#7A3B1E", "#B86A42", "#0B1B2D", "#B86A42", "#7A3B1E"]
    labels = ["well below", "below", "in line", "above", "well above"]
    # marker position: linear inside the segment the residual falls in
    r = min(max(residual, t[0]), t[-1])
    pos, acc = 0.0, 0.0
    for i in range(5):
        if r <= t[i + 1] or i == 4:
            pos = acc + widths[i] * (r - t[i]) / (t[i + 1] - t[i])
            break
        acc += widths[i]
    segs = "".join(f'<div style="width:{w}%;background:{c};height:26px"></div>' for w, c in zip(widths, colors))
    labs = "".join(f'<div style="width:{w}%;text-align:center;font-size:13px;color:#0B1B2D">{l}</div>' for w, l in zip(widths, labels))
    return (f'<div style="position:relative;padding-top:16px">'
            f'<div style="position:absolute;left:calc({pos:.1f}% - 8px);top:0;width:0;height:0;border-left:8px solid transparent;'
            f'border-right:8px solid transparent;border-top:13px solid #D08259"></div>'
            f'<div style="display:flex">{segs}</div><div style="display:flex;margin-top:4px">{labs}</div></div>')


# --------------------------------------------------------------------------- app
try:
    price_model, q10, q90, iso_forest, explainer, config = load_models()
    train_df, test_df = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run the main notebook first, so that data/ and data/models/ exist.")
    st.stop()
ref = prepare_reference(train_df, config, iso_forest)

logo = ""
if os.path.exists(LOGO_PATH):
    logo = f'<img src="data:image/png;base64,{base64.b64encode(open(LOGO_PATH, "rb").read()).decode()}" style="height:130px">'
st.markdown(f'<div class="fp-header"><div><div class="fp-eyebrow">Prototype</div><div class="fp-title">FairPrice Auto</div>'
            f'<div class="fp-sub">Explainable price and anomaly screening for used-car listings</div></div>{logo}</div>',
            unsafe_allow_html=True)

# ---- sidebar: choose and edit a listing
with st.sidebar:
    st.markdown("### Choose a listing")
    if "idx" not in st.session_state:
        st.session_state.idx = int(test_df.sample(1, random_state=42).index[0])
    if st.button("Random test listing", use_container_width=True):
        st.session_state.idx = int(test_df.sample(1).index[0])
    st.session_state.idx = st.number_input("Row of the test set", 0, len(test_df) - 1, st.session_state.idx)
    base = test_df.loc[[st.session_state.idx]].copy()

    st.markdown("### Edit the listing")
    st.caption("Change a value to see how price and signals react, e.g. triple the power or multiply the mileage.")
    price = st.number_input("Asking price (€)", 100.0, 5_000_000.0, float(base["price"].iloc[0]), step=500.0, key=f"edit_price_{st.session_state.idx}")
    km = st.number_input("Mileage (km)", 0.0, 2_000_000.0, float(base["mileage_km"].iloc[0]), step=1000.0, key=f"edit_mileage_km_{st.session_state.idx}")
    year = st.number_input("Registration year", 1950, 2030, int(base["registration_year"].iloc[0]), key=f"edit_registration_year_{st.session_state.idx}")
    kw = st.number_input("Power (kW)", 1.0, 1500.0, float(base["power_kw"].iloc[0]), step=5.0, key=f"edit_power_kw_{st.session_state.idx}")
    st.caption("Power in hp is left as listed, so changing kW alone also triggers the kW/hp coherence rule.")

ad = base.copy()
ad["price"], ad["mileage_km"], ad["registration_year"], ad["power_kw"] = price, km, year, kw
ad = add_derived(ad)

p = price_check(ad, config, ref, price_model, q10, q90)
a = anomaly_check(ad, ref, iso_forest, explainer)

# ---- listing summary
r0 = ad.iloc[0]
st.markdown(f'<div class="fp-eyebrow">Listing</div><h2 style="margin-top:0">{str(r0["make"]).upper()} {str(r0["model"]).title()} · '
            f'{int(r0["registration_year"])} · {r0["mileage_km"]:,.0f} km</h2>'
            f'<div class="fp-sub" style="margin-top:-10px;margin-bottom:22px">{r0.get("fuel_category", "")} · {r0.get("transmission", "")} · '
            f'{r0["power_kw"]:.0f} kW · {r0.get("body_type", "")} · seller: {r0.get("seller_type", "")}</div>', unsafe_allow_html=True)

left, right = st.columns([1.05, 1], gap="large")

with left:
    pill_bg = "#0B1B2D" if p["category"] == "in_line" else "#D08259"
    pill_fg = "#F3EEE8" if p["category"] == "in_line" else "#0B1B2D"
    st.markdown(f"""
<div class="fp-card-light">
  <div class="fp-eyebrow" style="color:#9A5530">Price check</div>
  <div style="display:flex;gap:28px;flex-wrap:wrap;margin:8px 0 14px 0">
    <div><div class="fp-label-dark">Asking</div><div class="fp-big">{euro(price)}</div></div>
    <div><div class="fp-label-dark">Expected</div><div class="fp-big">{euro(p['ideal'])}</div></div>
    <div><div class="fp-label-dark">Range p10–p90</div><div class="fp-big" style="font-size:24px;padding-top:8px">{euro(p['low'])} – {euro(p['high'])}</div></div>
  </div>
  <span class="fp-pill" style="background:{pill_bg};color:{pill_fg}">{CATEGORY_TEXT[p['category']]}</span>
  <span style="margin-left:12px;color:#4A5868">{(np.exp(p['residual']) - 1) * 100:+.1f}% vs expected price</span>
  <div style="margin-top:18px">{scale_html(p['residual'], p['thresholds'])}</div>
</div>""", unsafe_allow_html=True)
    z_text = ("too few comparable cars of the same make, model and year" if a["z"] is None
              else f"robust z-score {a['z']:+.1f} against cars of the same make, model and year"
              + (" — <b style='color:#D08259'>unusual price</b>" if abs(a["z"]) > 3 else ""))
    st.markdown(f'<div class="fp-ok">Peer comparison: {z_text}</div>', unsafe_allow_html=True)

with right:
    risk_colors = {"high": "#D08259", "medium": "#B86A42", "low": "#5D6F86", "none": "#3A4B60"}
    st.markdown(f"""
<div class="fp-card">
  <div class="fp-eyebrow">Consistency check</div>
  <div style="display:flex;align-items:center;gap:16px;margin:6px 0 4px 0">
    <span class="fp-pill" style="background:{risk_colors[a['risk']]};color:#0B1B2D">Risk: {a['risk'].upper()}</span>
    <span class="fp-label">anomaly score {a['score']:.3f} (negative = unusual)</span>
  </div>
</div>""", unsafe_allow_html=True)
    if a["features"]:
        st.markdown("**Attention signals** — the three features that make this listing unusual")
        for f in a["features"]:
            med = "no comparable cars" if f["median"] is None else f"median of comparable cars = {f['median']:,.2f}"
            st.markdown(f'<div class="fp-signal"><b>{nice(f["feature"])} = {f["value"]:,.2f}</b><br>'
                        f'<span style="color:#C9D3DE">{med} · SHAP {f["shap"]:+.2f}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="fp-ok">No attention signal from the anomaly model.</div>', unsafe_allow_html=True)
    st.markdown("**Coherence rules**")
    if a["rules"]:
        for r in a["rules"]:
            st.markdown(f'<div class="fp-signal">{r}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="fp-ok">No rule violated.</div>', unsafe_allow_html=True)

with st.expander("Raw output (JSON)"):
    st.json({"asking_price": price, "ideal_price": round(p["ideal"], 2), "range_p10_p90": [round(p["low"], 2), round(p["high"], 2)],
             "price_category": p["category"], "anomaly_score": round(a["score"], 4), "risk_category": a["risk"],
             "top_3_anomalous_features": a["features"], "coherence_signals": a["rules"],
             "robust_z_score": None if a["z"] is None else round(a["z"], 2)})

st.markdown('<div class="fp-foot">FairPrice Auto flags statistical inconsistencies in the declared data. It does not detect hidden '
            'defects and is not an appraisal or a guarantee: when something does not add up, check a vehicle history report '
            'or have the car inspected. Prototype trained on a November 2025 snapshot of Italian listings.</div>',
            unsafe_allow_html=True)
