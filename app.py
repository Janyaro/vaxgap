from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="VaxGap", page_icon="💉", layout="wide")

BASE = Path(__file__).parent
# Asli Pakistan DHS data par chalane ke baad isay False kar do
DATA_IS_PRACTICE = True

FEATURES = ["v012", "bord_c", "v190", "v106", "v025", "v024", "b4",
            "v151", "v467d", "anc_cat", "place"]
CAT = ["v025", "v024", "b4", "v151", "v467d", "anc_cat", "place"]

# --- Labels: codebook dekh kar yahan theek kar lena ---
AREA = {1: "Urban", 2: "Rural"}
REGION = {1: "Region 1", 2: "Region 2", 3: "Region 3", 4: "Region 4"}
WEALTH = {1: "1 Poorest", 2: "2 Poorer", 3: "3 Middle", 4: "4 Richer", 5: "5 Richest"}
EDU = {0: "No education", 1: "Primary", 2: "Secondary", 3: "Higher"}
SEX = {1: "Male", 2: "Female"}
HEAD = {1: "Male", 2: "Female"}
DIST = {0.0: "Code 0", 1.0: "Code 1", 2.0: "Code 2"}
ANC = {"0": "0 visits", "1-3": "1-3 visits", "4+": "4+ visits", "unknown": "Unknown"}
PLACE = {"home": "Home", "public_facility": "Public facility",
         "private_facility": "Private facility", "other_unknown": "Other/unknown"}
PRETTY = {"v012": "Mother's age", "bord_c": "Birth order", "v190": "Wealth",
          "v106": "Mother's education", "v025": "Area", "v024": "Region",
          "b4": "Child sex", "v151": "Household head sex",
          "v467d": "Distance problem", "anc_cat": "Antenatal visits", "place": "Delivery place"}
VALMAP = {"v025": AREA, "v024": REGION, "b4": SEX, "v151": HEAD,
          "v467d": DIST, "anc_cat": ANC, "place": PLACE}


@st.cache_resource
def load_model():
    return joblib.load(BASE / "vaxgap_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(BASE / "vaxgap_clean.csv")


def dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


@st.cache_data
def scores_and_mean():
    m, d = load_model(), load_data()
    s = m.predict_proba(d[FEATURES])[:, 1]
    mean_x = dense(m.named_steps["pre"].transform(d[FEATURES])).mean(axis=0)
    return s, mean_x


def coverage(df, col, labels=None):
    rows = []
    for k, x in df.groupby(col):
        rows.append({col: labels.get(k, k) if labels else k, "Children": len(x),
                     "Coverage %": round(100 * (1 - np.average(x["missed"], weights=x["w"])), 1)})
    return pd.DataFrame(rows)


def bar(df, col, title):
    fig = px.bar(df, x=col, y="Coverage %", text="Coverage %", title=title,
                 hover_data=["Children"], range_y=[0, 100])
    fig.update_layout(xaxis_title="", margin=dict(t=50, b=10))
    return fig


def pretty_feature(name):
    name = name.split("__", 1)[1]
    if name in PRETTY:
        return PRETTY[name]
    for col in sorted(CAT, key=len, reverse=True):
        if name.startswith(col + "_"):
            val = name[len(col) + 1:]
            try:
                val = float(val) if col == "v467d" else (int(val) if val.isdigit() else val)
            except ValueError:
                pass
            return f"{PRETTY[col]}: {VALMAP[col].get(val, val)}"
    return name


model, data = load_model(), load_data()

st.title("💉 VaxGap")
st.caption("Predicting missed basic-vaccination risk in children aged 12-23 months")
if DATA_IS_PRACTICE:
    st.warning("This dashboard currently runs on the DHS Program **model (practice) dataset**, "
               "not real Pakistan data. Numbers shown are for testing the pipeline only.")

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
f_area = st.sidebar.multiselect("Area", list(AREA), list(AREA), format_func=AREA.get)
f_reg = st.sidebar.multiselect("Region", list(REGION), list(REGION), format_func=REGION.get)
f_wealth = st.sidebar.multiselect("Wealth", list(WEALTH), list(WEALTH), format_func=WEALTH.get)
f_edu = st.sidebar.multiselect("Mother's education", list(EDU), list(EDU), format_func=EDU.get)
flt = data[data["v025"].isin(f_area) & data["v024"].isin(f_reg)
           & data["v190"].isin(f_wealth) & data["v106"].isin(f_edu)]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Drivers", "Risk Predictor", "Model"])

# ---------------- Tab 1: Overview ----------------
with tab1:
    if len(flt) < 30:
        st.info("Too few children match these filters (fewer than 30). Widen the filters.")
    else:
        cov = 100 * (1 - np.average(flt["missed"], weights=flt["w"]))
        by_area = coverage(flt, "v025", AREA)
        by_reg = coverage(flt, "v024", REGION)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Children analysed", f"{len(flt):,}")
        c2.metric("Basic vaccination coverage", f"{cov:.1f}%")
        if set(by_area["v025"]) == {"Urban", "Rural"}:
            r = by_area.set_index("v025")["Coverage %"]
            c3.metric("Rural vs urban", f"{r['Rural'] - r['Urban']:+.1f} pts")
        else:
            c3.metric("Rural vs urban", "n/a")
        low = by_reg.sort_values("Coverage %").iloc[0]
        c4.metric("Lowest-coverage region", str(low["v024"]), f"{low['Coverage %']}%")

        a, b = st.columns(2)
        a.plotly_chart(bar(by_reg, "v024", "Coverage by region"), width="stretch")
        b.plotly_chart(bar(by_area, "v025", "Coverage by area"), width="stretch")
        a.plotly_chart(bar(coverage(flt, "v190", WEALTH), "v190", "Coverage by wealth"),
                       width="stretch")
        b.plotly_chart(bar(coverage(flt, "v106", EDU), "v106", "Coverage by mother's education"),
                       width="stretch")
        st.caption("Coverage uses DHS survey weights. Groups with few children are unreliable; "
                   "hover a bar to see the count.")

# ---------------- Tab 2: Drivers ----------------
with tab2:
    st.subheader("What the model relies on")
    st.image(str(BASE / "shap_summary.png"),
             caption="SHAP summary. Red = high feature value, blue = low. "
                     "Right of 0 = pushes predicted risk up, left = down.")
    a, b = st.columns(2)
    a.plotly_chart(bar(coverage(data, "anc_cat", ANC), "anc_cat",
                       "Coverage by antenatal visits"), width="stretch")
    b.plotly_chart(bar(coverage(data, "place", PLACE), "place",
                       "Coverage by delivery place"), width="stretch")

    names = model.named_steps["pre"].get_feature_names_out()
    coef = model.named_steps["model"].coef_[0]
    counts = []
    for n in names:
        raw = n.split("__", 1)[1]
        cnt = len(data)
        for col in sorted(CAT, key=len, reverse=True):
            if raw.startswith(col + "_"):
                cnt = int((data[col].astype(str) == raw[len(col) + 1:]).sum())
                break
        counts.append(cnt)
    tbl = pd.DataFrame({"Feature": [pretty_feature(n) for n in names],
                        "Coefficient": coef.round(3),
                        "Odds ratio": np.exp(coef).round(2),
                        "Children in group": counts})
    tbl["Reliability"] = np.where(tbl["Children in group"] < 30, "⚠ small group", "ok")
    tbl = tbl.reindex(tbl["Coefficient"].abs().sort_values(ascending=False).index)
    st.markdown("**Logistic regression coefficients** (positive = higher missed-vaccination risk)")
    st.dataframe(tbl.head(15), width="stretch", hide_index=True)
    st.caption("These show association, not cause. Coefficients from groups with fewer than "
               "30 children should not be interpreted.")

# ---------------- Tab 3: Risk predictor ----------------
with tab3:
    st.subheader("Risk predictor (research demo)")
    c1, c2, c3 = st.columns(3)
    area = c1.selectbox("Area", list(AREA), format_func=AREA.get)
    region = c1.selectbox("Region", list(REGION), format_func=REGION.get)
    wealth = c1.selectbox("Wealth", list(WEALTH), index=2, format_func=WEALTH.get)
    edu = c2.selectbox("Mother's education", list(EDU), format_func=EDU.get)
    age = c2.slider("Mother's age", 15, 49, 28)
    bord = c2.slider("Birth order (5 = 5th or later)", 1, 5, 2)
    anc = c3.selectbox("Antenatal visits", list(ANC), index=2, format_func=ANC.get)
    place = c3.selectbox("Delivery place", list(PLACE), index=1, format_func=PLACE.get)
    sex = c3.selectbox("Child sex", list(SEX), format_func=SEX.get)
    head = c1.selectbox("Household head sex", list(HEAD), format_func=HEAD.get)
    dist = c2.selectbox("Distance to facility (code)", list(DIST), index=1, format_func=DIST.get)

    if st.button("Estimate risk", type="primary"):
        row = pd.DataFrame([{"v012": age, "bord_c": bord, "v190": wealth, "v106": edu,
                             "v025": area, "v024": region, "b4": sex, "v151": head,
                             "v467d": dist, "anc_cat": anc, "place": place}])[FEATURES]
        p = float(model.predict_proba(row)[:, 1][0])
        s, mean_x = scores_and_mean()
        pct = float((s < p).mean() * 100)
        band = "Higher than most" if pct >= 67 else ("Around average" if pct >= 33 else "Lower than most")
        m1, m2 = st.columns(2)
        m1.metric("Model risk score", f"{p:.0%}")
        m2.metric("Compared with the dataset", band, f"{pct:.0f}th percentile")
        st.caption("The model is trained with class weights, so this score is not a literal "
                   "probability. Read it relative to other children (the percentile).")

        xr = dense(model.named_steps["pre"].transform(row))[0]
        contrib = (xr - mean_x) * model.named_steps["model"].coef_[0]
        order = np.argsort(contrib)
        fnames = [pretty_feature(n) for n in model.named_steps["pre"].get_feature_names_out()]
        up = [f"{fnames[i]} ({contrib[i]:+.2f})" for i in order[::-1][:3] if contrib[i] > 0]
        down = [f"{fnames[i]} ({contrib[i]:+.2f})" for i in order[:2] if contrib[i] < 0]
        st.markdown("**Factors raising the score:** " + (", ".join(up) or "none"))
        st.markdown("**Factors lowering the score:** " + (", ".join(down) or "none"))
    st.info("Research demo only. Not for clinical or programme decisions.")

# ---------------- Tab 4: Model ----------------
with tab4:
    st.subheader("Model comparison (5-fold stratified cross-validation)")
    st.dataframe(pd.DataFrame({
        "Model": ["Dummy (baseline)", "Logistic Regression", "Random Forest"],
        "ROC-AUC": [0.500, 0.642, 0.626], "PR-AUC": [0.316, 0.480, 0.422],
        "Recall": [0.000, 0.596, 0.494], "Precision": [0.000, 0.433, 0.408],
        "F1": [0.000, 0.499, 0.446]}), width="stretch", hide_index=True)

    st.markdown("**Selected model: Logistic Regression** (out-of-fold predictions)")
    st.write("ROC-AUC 0.637 | PR-AUC 0.460 | recall 0.596 | precision 0.428")
    cm = pd.DataFrame([[441, 257], [130, 192]],
                      index=["Actually vaccinated", "Actually missed"],
                      columns=["Predicted vaccinated", "Predicted missed"])
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                              title="Confusion matrix"), width="stretch")

    st.markdown("**Performance by group**")
    st.dataframe(pd.DataFrame({
        "Group": ["Urban", "Rural", "Wealth 1", "Wealth 2", "Wealth 3", "Wealth 4", "Wealth 5"],
        "Children": [322, 698, 283, 216, 202, 215, 104],
        "AUC": [0.583, 0.661, 0.538, 0.701, 0.625, 0.672, 0.638],
        "Recall": [0.535, 0.624, 0.451, 0.653, 0.573, 0.662, 0.697]}),
        width="stretch", hide_index=True)
    st.markdown("**Limitations**")
    st.markdown("- Survey data; the model was never tested in a real vaccination campaign.\n"
                "- Results show association, not cause.\n"
                "- Missing and \"don't know\" vaccine answers are treated as not vaccinated (DHS convention).\n"
                "- Small dataset; some groups have too few children for reliable estimates.\n"
                "- Performance differs between groups, so it should be checked before any real use.")
