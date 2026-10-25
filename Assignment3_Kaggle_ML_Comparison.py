# %% [markdown]
# # 🔗 LIVE DEPLOYMENT LINK: https://PASTE-YOUR-STREAMLIT-LINK-HERE.streamlit.app
# 
# # Assignment 3 – Price Prediction System (Kaggle Dataset)
# **Course:** Machine Learning Fundamentals (AIC354)  
# **Dataset:** *(write dataset name + Kaggle link here)* – FULL dataset  
# **Evaluation:** 10-Fold Cross-Validation, metric = RMSE (lower is better)  
# **Algorithms:** Ridge Regression, Extra Trees, AdaBoost, HistGradientBoosting, MLP Neural Network

# %% [markdown]
# ## Step 1: Import Libraries

# %%
# --- Core Python Libraries ---
import os
import time
import joblib

import numpy as np
import pandas as pd

# --- Machine Learning (scikit-learn) ---
import sklearn
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- The 5 Regression Algorithms (none are used in the Car Price notebook) ---
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# --- Visualization ---
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# --- Progress bar (shows elapsed and remaining time) ---
from tqdm import tqdm

pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

print("Library versions used in this run:")
print(f"  NumPy        : {np.__version__}")
print(f"  Pandas       : {pd.__version__}")
print(f"  scikit-learn : {sklearn.__version__}")
print(f"  Matplotlib   : {matplotlib.__version__}")

# %% [markdown]
# ## Step 2: Load the FULL Kaggle Dataset
# **Only change these 3 lines for your dataset.**

# %%
DATA_FILE = "diamonds.csv"   # <-- name of the CSV you downloaded from Kaggle
TARGET    = "price"          # <-- the column you want to predict
UNIT      = "$"              # <-- currency symbol of the target ("" if none)

# google.colab is only available inside Google Colab
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB and not os.path.exists(DATA_FILE):
    print(f"Please click 'Choose Files' and select '{DATA_FILE}' ...")
    uploaded = files.upload()

    # If the browser renamed the file (e.g. 'diamonds (1).csv'), rename it back
    uploaded_name = list(uploaded.keys())[0]
    if uploaded_name != DATA_FILE:
        os.rename(uploaded_name, DATA_FILE)

if os.path.exists(DATA_FILE):
    print(f"'{DATA_FILE}' is ready to use.")
else:
    raise FileNotFoundError(f"'{DATA_FILE}' not found. Please run this cell again and upload it.")

# %%
# Load the FULL dataset (no sampling)
data = pd.read_csv(DATA_FILE)

print(f"Full dataset: {data.shape[0]:,} rows, {data.shape[1]} columns")
data.head()

# %% [markdown]
# ## Step 3: Explore the Data

# %%
print("\nAttributes in Data:")
print("===================\n")
print(list(data.columns))

print("\nNumber of Instances in Data:")
print("============================\n")
print(len(data))

print("\nMissing Values in Each Attribute:")
print("=================================\n")
missing = data.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

# %%
# Summary statistics of the target
data[TARGET].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).to_frame(name=TARGET)

# %% [markdown]
# ## Step 4: Pre-processing

# %%
"""
Pre-processing steps:
---------------------
1. Remove duplicate rows (a duplicate could be in BOTH the training folds and the
   test fold, which makes the RMSE look better than it really is).
2. Remove rows where the target is missing (we cannot learn from them).
3. Remove useless columns: ID / index columns and text columns with too many
   different values (e.g. names) that cannot help a model.
4. Missing values in the remaining features are NOT filled here. They are filled
   inside the pipeline in Step 5, using only the training folds (no data leakage).
"""

df = data.copy()

rows_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows removed       : {rows_before - len(df):,}")

rows_before = len(df)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
print(f"Rows with missing target     : {rows_before - len(df):,}")

features = df.drop(columns=TARGET)
dropped = []
for col in features.columns:
    is_id = pd.api.types.is_numeric_dtype(features[col]) and features[col].is_unique and \
            (col.lower().startswith("unnamed") or "id" in col.lower() or features[col].is_monotonic_increasing)
    too_many_categories = (not pd.api.types.is_numeric_dtype(features[col])) and features[col].nunique() > 50
    if is_id or too_many_categories:
        dropped.append(col)
features = features.drop(columns=dropped)

print(f"Useless columns removed      : {dropped if dropped else 'none'}")
print(f"Instances remaining          : {len(df):,}")

# %%
numeric_features = features.select_dtypes(include="number").columns.tolist()
categorical_features = [c for c in features.columns if c not in numeric_features]

X = features
y = df[TARGET]

print(f"Input features (X): {X.shape[1]} columns")
print(f"  Numeric     : {numeric_features}")
print(f"  Categorical : {categorical_features}")
print(f"Output (y)        : {TARGET}")
print(f"Number of rows    : {len(X):,}")

X.head()

# %% [markdown]
# ## Step 5: Pre-processing Pipeline

# %%
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

transformers = [("num", numeric_pipeline, numeric_features)]
if categorical_features:
    transformers.append(("cat", categorical_pipeline, categorical_features))

preprocessor = ColumnTransformer(transformers)

print("Pre-processing pipeline created:")
preprocessor

# %% [markdown]
# ## Step 6: The 5 Algorithms

# %%
algorithms = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Extra Trees": ExtraTreesRegressor(n_estimators=50, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "MLP Neural Network": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200,
                                       early_stopping=True, random_state=42),
}

print("Algorithms to compare:")
for number, name in enumerate(algorithms, start=1):
    print(f"  {number}. {name}")

# %% [markdown]
# ## Step 7: 10-Fold Cross Validation

# %%
K = 10
kfold = KFold(n_splits=K, shuffle=True, random_state=42)

print(f"{K}-Fold Cross Validation created (shuffle=True, random_state=42)")

# %%
fold_sizes = []
for fold_number, (train_index, test_index) in enumerate(kfold.split(X), start=1):
    fold_sizes.append({
        "Fold (Round)": fold_number,
        "Training Rows": len(train_index),
        "Testing Rows": len(test_index),
        f"Avg {TARGET} of Test Rows": y.iloc[test_index].mean(),
        f"Max {TARGET} of Test Rows": y.iloc[test_index].max(),
    })

fold_sizes_table = pd.DataFrame(fold_sizes).set_index("Fold (Round)")
fold_sizes_table

# %% [markdown]
# ## Step 8: Train and Test Every Algorithm on Every Fold

# %%
fold_rmse = {}        # RMSE on each fold, for each algorithm
training_time = {}    # total time taken for all folds

folds = list(kfold.split(X))     # the same folds for every algorithm
total_steps = len(algorithms) * K

with tqdm(total=total_steps, unit="fold") as progress_bar:
    for name, algorithm in algorithms.items():
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", algorithm),
        ])

        scores = []
        start = time.time()
        for fold_number, (train_index, test_index) in enumerate(folds, start=1):
            progress_bar.set_description(f"{name}, fold {fold_number}/{K}")

            fold_model = clone(model)                                   # fresh, untrained copy
            fold_model.fit(X.iloc[train_index], y.iloc[train_index])    # train on K-1 folds
            predictions = fold_model.predict(X.iloc[test_index])        # test on 1 fold
            rmse = np.sqrt(mean_squared_error(y.iloc[test_index], predictions))
            scores.append(rmse)

            progress_bar.update(1)

        training_time[name] = time.time() - start
        fold_rmse[name] = np.array(scores)
        tqdm.write(f"{name:<22} -> mean RMSE = {UNIT}{fold_rmse[name].mean():>12,.2f}   ({training_time[name]:.1f} s)")

    progress_bar.set_description("Done")

print(f"\n{K}-fold cross validation completed for all {len(algorithms)} algorithms.")

# %% [markdown]
# ## Step 9: Results

# %%
per_fold_table = pd.DataFrame(fold_rmse)
per_fold_table.index = [f"Fold {i}" for i in range(1, K + 1)]
per_fold_table.index.name = "Fold"

money = lambda v: f"{UNIT}{v:,.0f}"

(per_fold_table.style
    .format(money)
    .highlight_min(axis=1, props="background-color: #c8ebd2; color: #0b0b0b; font-weight: bold;"))

# %%
average_value = y.mean()

summary = pd.DataFrame({
    "Mean RMSE": per_fold_table.mean(),
    "Std RMSE": per_fold_table.std(),
    "Best Fold RMSE": per_fold_table.min(),
    "Worst Fold RMSE": per_fold_table.max(),
    "RMSE as % of Avg Target": per_fold_table.mean() / average_value * 100,
    "Time (s)": pd.Series(training_time),
})
summary = summary.sort_values("Mean RMSE")
summary.insert(0, "Rank", range(1, len(summary) + 1))
summary.index.name = "Algorithm"

print(f"Average {TARGET} in this data: {UNIT}{average_value:,.2f}\n")

rmse_columns = ["Mean RMSE", "Std RMSE", "Best Fold RMSE", "Worst Fold RMSE"]
(summary.style
    .format({**{c: money for c in rmse_columns},
             "RMSE as % of Avg Target": "{:.1f}%",
             "Time (s)": "{:.2f}"})
    .highlight_min(subset=["Mean RMSE"], props="background-color: #c8ebd2; color: #0b0b0b; font-weight: bold;"))

# %% [markdown]
# ## Step 10: Charts

# %%
# Chart colours: one blue for the data, dark/grey text, light grid
BAR_COLOR = "#2a78d6"
TEXT_COLOR = "#0b0b0b"
MUTED_TEXT = "#52514e"
GRID_COLOR = "#e4e3df"
units = FuncFormatter(lambda value, _: f"{UNIT}{value:,.0f}")

def style_axis(ax):
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED_TEXT, length=0)
    ax.xaxis.set_major_formatter(units)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)

order = summary.index.tolist()            # best algorithm first
positions = np.arange(len(order))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

# ---- Chart 1: mean RMSE with +/- 1 std ----
means = summary["Mean RMSE"]
stds = summary["Std RMSE"]
ax1.barh(positions, means, height=0.5, color=BAR_COLOR,
         xerr=stds, error_kw=dict(ecolor=TEXT_COLOR, elinewidth=1.2, capsize=4))
for i, name in enumerate(order):
    label = f"{UNIT}{means[name]:,.0f}" + ("   (best)" if i == 0 else "")
    ax1.text(means[name] + stds[name] + means.max() * 0.02, i, label,
             va="center", color=TEXT_COLOR, fontsize=10)
ax1.set_yticks(positions, order, color=TEXT_COLOR)
ax1.invert_yaxis()
ax1.set_xlim(0, (means + stds).max() * 1.3)
style_axis(ax1)
ax1.set_title(f"Mean RMSE across {K} folds (lower is better)", loc="left", color=TEXT_COLOR, fontsize=12)
ax1.set_xlabel(f"RMSE (same unit as {TARGET})", color=MUTED_TEXT)

# ---- Chart 2: RMSE of every fold ----
for i, name in enumerate(order):
    ax2.scatter(per_fold_table[name], np.full(K, i), s=60, color=BAR_COLOR,
                edgecolor="white", linewidth=1.5, zorder=3,
                label="RMSE of one fold" if i == 0 else None)
    ax2.scatter(means[name], i, marker="|", s=500, color=TEXT_COLOR, linewidth=2.5, zorder=4,
                label="Mean RMSE" if i == 0 else None)
ax2.set_yticks(positions, order, color=TEXT_COLOR)
ax2.invert_yaxis()
ax2.set_xlim(0, per_fold_table.values.max() * 1.1)
style_axis(ax2)
ax2.set_title("RMSE of each fold (spread = consistency)", loc="left", color=TEXT_COLOR, fontsize=12)
ax2.set_xlabel(f"RMSE (same unit as {TARGET})", color=MUTED_TEXT)
ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=MUTED_TEXT)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 11: Train the Best Model on the Full Data and Save It

# %%
best_name = summary.index[0]
print(f"Best algorithm: {best_name}  (mean RMSE = {UNIT}{summary.loc[best_name, 'Mean RMSE']:,.0f})")

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", clone(algorithms[best_name])),
])
final_model.fit(X, y)

# Information the web app needs to build its input form
meta = {"numeric": {}, "categorical": {}}
for col in numeric_features:
    lo, med, hi = float(X[col].min()), float(X[col].median()), float(X[col].max())
    meta["numeric"][col] = {"min": lo, "median": med, "max": hi if hi > lo else lo + 1,
                            "is_int": bool(pd.api.types.is_integer_dtype(X[col]))}
for col in categorical_features:
    meta["categorical"][col] = X[col].dropna().value_counts().index.tolist()

joblib.dump({"model": final_model, "features": list(X.columns), "name": best_name,
             "target": TARGET, "unit": UNIT, "meta": meta}, "model.pkl", compress=3)

size_mb = os.path.getsize("model.pkl") / 1e6
print(f"Saved: model.pkl ({size_mb:.1f} MB)")
if size_mb > 90:
    print("WARNING: file is over GitHub's 100 MB limit. Reduce n_estimators in Step 6 and run again.")

# %%
# Test the saved model
saved = joblib.load("model.pkl")

sample = X.iloc[[0]]
print(f"Predicted: {UNIT}{saved['model'].predict(sample)[0]:,.0f}   |   Actual: {UNIT}{y.iloc[0]:,.0f}")

# %%
# Download model.pkl (Colab only)
try:
    from google.colab import files
    files.download("model.pkl")
except ImportError:
    print("Not running in Colab - model.pkl is in the current folder.")

# %% [markdown]
# ## Step 12: Conclusion
# *(Write 3–4 lines in your own words: which algorithm had the lowest mean RMSE, why it beat the others, and what the Std RMSE says about its stability across the 10 folds.)*
# 
# ## Step 13: Deployment
# `model.pkl` is served by a Streamlit web app (`app.py`) on Streamlit Community Cloud. The live link is at the top of this notebook.
