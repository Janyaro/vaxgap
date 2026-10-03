import pandas as pd 


df = pd.read_stata("zzkr62dt/ZZKR62FL.DTA", convert_categoricals=False).copy()
# agar tumhari file ka naam alag hai, to wahi likho

print(df.shape)
cols = ["b5", "b19", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9",
        "v025", "v024", "v190", "v106", "v012", "b4", "bord",
        "m14", "m15", "v005"]
# print([c for c in cols if c not in df.columns])   # jo columns nahi mile
# print(df[[c for c in cols if c in df.columns]].head())
# print(df.info())


vacc_cols = ["h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"]

# 1. umar mahino mein
df["age_months"] = df["v008"] - df["b3"]

# 2. sirf zinda bache, 12 se 23 mahine
d = df[(df["b5"] == 1) & (df["age_months"].between(12, 23))].copy()
print("Filtered rows:", d.shape[0])

# 3. missing values check (target banane se pehle)
print(d[vacc_cols].isna().sum())
print(d[vacc_cols].apply(lambda s: s.value_counts(dropna=False)).T)

# 4. target: har vaccine ki value 1, 2 ya 3 honi chahiye
d["basic_vaccinated"] = d[vacc_cols].isin([1, 2, 3]).all(axis=1).astype(int)

print(d["basic_vaccinated"].value_counts())
print(d["basic_vaccinated"].value_counts(normalize=True).round(3))


feat = ["v025","v024","v190","v106","v012","b4","bord",
        "m14","m15","v467d","v151"]

print("Missing columns:", [c for c in feat if c not in d.columns])
print(d[[c for c in feat if c in d.columns]].isna().sum())

print(d["m14"].value_counts(dropna=False).head(8))
print(d["m15"].value_counts(dropna=False).head(8))

# survey weights ke saath coverage
d["w"] = d["v005"] / 1_000_000
d["bw"] = d["basic_vaccinated"] * d["w"]

for col in ["v025", "v190", "v106", "v024"]:
    g = d.groupby(col).agg(n=("w", "size"), bw=("bw", "sum"), w=("w", "sum"))
    g["coverage_%"] = (g["bw"] / g["w"] * 100).round(1)
    print(f"\n--- {col} ---")
    print(g[["n", "coverage_%"]])




    import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

# --- cleaning ---
d["m14c"] = d["m14"].replace({98: np.nan, 99: np.nan})
d["anc_cat"] = pd.cut(d["m14c"], bins=[-1, 0, 3, 100],
                      labels=["0", "1-3", "4+"]).astype(object).fillna("unknown")

def place(x):
    if x in (11, 12): return "home"
    if x in (21, 22, 23): return "public_facility"
    if x in (31, 32): return "private_facility"
    return "other_unknown"
d["place"] = d["m15"].apply(place)

d["bord_c"] = d["bord"].clip(upper=5)     # 5 ya zyada ek hi group
d["missed"] = 1 - d["basic_vaccinated"]   # 1 = risk wala bacha

num = ["v012", "bord_c", "v190", "v106"]
cat = ["v025", "v024", "b4", "v151", "v467d", "anc_cat", "place"]
X = d[num + cat]
y = d["missed"]

print(d["anc_cat"].value_counts())
print(d["place"].value_counts())
print(d["v467d"].value_counts(dropna=False))

# --- models ---
pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])
models = {
    "Dummy (baseline)": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced",
                                            random_state=42, n_jobs=-1),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision",
           "recall": "recall", "precision": "precision", "f1": "f1"}

for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)])
    r = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    print(f"\n{name}")
    for k in scoring:
        s = r[f"test_{k}"]
        print(f"  {k:10s} {s.mean():.3f} (+/- {s.std():.3f})")

import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             recall_score, precision_score, confusion_matrix)

best = Pipeline([("pre", pre),
                 ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])

# 1. out-of-fold predictions (har bache ki prediction us model se jisne usay dekha nahi)
proba = cross_val_predict(best, X, y, cv=cv, method="predict_proba")[:, 1]
pred = (proba >= 0.5).astype(int)
yv = y.values

print("Overall: ROC-AUC %.3f | PR-AUC %.3f | recall %.3f | precision %.3f" % (
    roc_auc_score(yv, proba), average_precision_score(yv, proba),
    recall_score(yv, pred), precision_score(yv, pred)))
print(confusion_matrix(yv, pred))

# 2. group-wise performance (fairness check)
def group_report(col, labels=None):
    print(f"\n--- performance by {col} ---")
    for g in sorted(X[col].unique()):
        m = (X[col] == g).values
        if len(set(yv[m])) < 2:
            continue
        name = labels.get(g, g) if labels else g
        print(f"{name}: n={m.sum()}, AUC={roc_auc_score(yv[m], proba[m]):.3f}, "
              f"recall={recall_score(yv[m], pred[m]):.3f}")

group_report("v025", {1: "Urban", 2: "Rural"})
group_report("v190")

# 3. final model aur coefficients
best.fit(X, y)
names = best.named_steps["pre"].get_feature_names_out()
coef = best.named_steps["model"].coef_[0]
imp = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
imp = imp.reindex(imp["coef"].abs().sort_values(ascending=False).index)
print("\nTop 15 features:")
print(imp.head(15).round(3))

# 4. SHAP
Xt = best.named_steps["pre"].transform(X)
if hasattr(Xt, "toarray"):
    Xt = Xt.toarray()
explainer = shap.LinearExplainer(best.named_steps["model"], Xt)
sv = explainer.shap_values(Xt)
shap.summary_plot(sv, Xt, feature_names=names, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.close()

# 5. dashboard ke liye save
joblib.dump(best, "vaxgap_model.joblib")
d[num + cat + ["missed", "w"]].to_csv("vaxgap_clean.csv", index=False)
print("Saved: model, clean CSV, shap_summary.png")