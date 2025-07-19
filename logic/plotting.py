import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from io import BytesIO
from typing import Protocol, Callable
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import shap
import base64
import numpy as np

# --- Plotting Interface and Registry ---


class PlotFunction(Protocol):
    def __call__(self, df: pd.DataFrame, feature: str) -> str:
        ...


class PlotAllFunction(Protocol):
    def __call__(self, df: pd.DataFrame) -> str:
        ...


class PlotRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, plot_type: str, func: PlotFunction, all: PlotAllFunction | None = None):
        self._registry[plot_type] = {"feature": func, "all": all}

    def get(self, plot_type: str) -> PlotFunction:
        return self._registry.get(plot_type)


def apply_operators(df: pd.DataFrame, operators: dict[str,Callable]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        for op_name, func in operators.items():
            new_col_name = f"{op_name}_of_{col}"
            df[new_col_name] = func(df[col])
    return df
# Show Feature Force with Shap after Random Forest
def plot_shap_feature_force(df: pd.DataFrame) -> str:
    max_rows = 15
    window = int(len(df) / 100) if len(df) > 100 else 1
    w_estimators = {
        "mov_avg": lambda x: x.rolling(window=100, center=True).mean(),
        "mov_std": lambda x: x.rolling(window=100, center=True).std(),
        "mov_min": lambda x: x.rolling(window=100, center=True).min(),
        "mov_max": lambda x: x.rolling(window=100, center=True).max(),
        "mov_median": lambda x: x.rolling(window=100, center=True).median(),
        "skew": lambda x: x.rolling(window=100, center=True).apply(lambda y: y.skew(), raw=False)
    }
    feature_columns = [col for col in df.columns if col != "label"]
    html = ""
    # train a Random Forest
    df = df.copy()
    X = df[feature_columns]
    X = (X - X.mean()) / X.std()  # Standardize features
    X = apply_operators(X, w_estimators)
    label_map = {"OK": 1, "KO": 0}
    y = df["label"].map(label_map)

    abs_max_depth = 1  # Set a max depth for the Random Forest
    # 10 is an heuristic to avoid overfitting
    max_depth = min(abs_max_depth, int(np.floor(np.log2(df.shape[0] / 10))))
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=max_depth, random_state=42)
    rf.fit(X, y)

    # use SHAP to explain the feature importance with a force plot
    X_scatter = X.sample(max_rows, random_state=42) if len(X) > max_rows else X
    explainer = shap.Explainer(rf, X)
    shap_values = explainer(X_scatter)
    print("SHAP values calculated. ", shap_values.shape)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    shap.plots.bar(shap_values[..., 0], show=False, ax=axs[0])
    axs[0].set_title("SHAP Feature Importance: Class KO")

    shap.plots.bar(shap_values[..., 1], show=False, ax=axs[1])
    axs[1].set_title("SHAP Feature Importance: Class OK")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()

    buf.seek(0)
    img_bytes = buf.read()
    buf.close()

    # return the HTML string with the force plot
    analysis = f"<p>Shapley Force Plot</p>"
    b64_data = base64.b64encode(img_bytes).decode('utf-8')
    img_tag = f'<br><img src="data:image/png;base64,{b64_data}" style="max-width: 400px; max-height: 300px;"/>'
    html += analysis + img_tag

    return html

# Plot All Histograms


def plot_all_histograms(df: pd.DataFrame) -> str:
    feature_columns = [col for col in df.columns if col != "label"]
    html = ""
    for col in feature_columns:
        html += histogram_plot(df, col)
    return html

# Plot Histogram


def histogram_plot(df: pd.DataFrame, feature: str) -> str:
    # feature_columns = [col for col in df.columns if col != "label"]
    df_ko = df[df["label"] == "KO"]
    df_ok = df[df["label"] == "OK"]

    html = ""
    # HTML string with embedded image
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_ok = colors[0 % len(colors)]
    color_ko = colors[1 % len(colors)]

    # for col in feature_columns:
    plt.figure(figsize=(6, 4))

    # Plot histogram
    plt.hist(df_ok[feature], bins=20, alpha=0.4, label=f'OK', color=color_ok)
    plt.hist(df_ko[feature], bins=20, alpha=0.4, label=f'KO', color=color_ko)

    # Plot mean line
    mean_ok = df_ok[feature].mean()
    std_ok = df_ok[feature].std()
    mean_ko = df_ko[feature].mean()
    std_ko = df_ko[feature].std()

    plt.axvline(mean_ok, color=color_ok, alpha=0.6, linestyle='-', linewidth=2)
    plt.axvline(mean_ok + std_ok, color=color_ok,
                alpha=0.6, linestyle=':', linewidth=2)
    plt.axvline(mean_ko, color=color_ko, alpha=0.6, linestyle='-', linewidth=2)
    plt.axvline(mean_ko + std_ko, color=color_ko,
                alpha=0.6, linestyle=':', linewidth=2)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    legend_lines = [
        Line2D([0], [0], color='black', linestyle='-',
               linewidth=2, label='Mean'),
        Line2D([0], [0], color='black', linestyle=':',
               linewidth=2, label='Mean + 1 Std')
    ]
    plt.legend(handles=legend_lines, loc='upper right')

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()

    buf.seek(0)
    img_bytes = buf.read()
    buf.close()

    analysis = f"<p>Histogram of {feature}</p>"
    b64_data = base64.b64encode(img_bytes).decode('utf-8')
    img_tag = f'<br><img src="data:image/png;base64,{b64_data}" style="max-width: 400px; max-height: 300px;"/>'
    html += analysis + img_tag

    return html

# Plot All Scatter Plots


def plot_all_scatter_plots(df: pd.DataFrame) -> str:
    feature_columns = [col for col in df.columns if col != "label"]
    html = ""
    for idx, col in enumerate(feature_columns):
        for in_idx, in_col in enumerate(feature_columns):
            if idx > in_idx:
                html += scatter_plot(df, col, in_col)
    return html

# Plot Scatterplot


def scatter_plot(df: pd.DataFrame, feature1: str, feature2: str) -> str:
    max_rows = 1500
    # feature_columns = [col for col in df.columns if col != "label"]
    df.loc[:, df.columns != "label"] = (df.loc[:, df.columns != "label"] - df.loc[:,
                                        df.columns != "label"].mean()) / df.loc[:, df.columns != "label"].std()
    df_ko = df[df["label"] == "KO"]
    df_ok = df[df["label"] == "OK"]

    html = ""
    # HTML string with embedded image
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_ok = colors[0 % len(colors)]
    color_ko = colors[1 % len(colors)]

    # for idx, col in enumerate(feature_columns):
    #     for in_idx, in_col in enumerate(feature_columns):
    #         if idx > in_idx:
    plt.figure(figsize=(6, 4))
    df_ko_scatter = df_ko.sample(max_rows, random_state=42) if len(
        df_ko) > max_rows else df_ko
    df_ok_scatter = df_ok.sample(max_rows, random_state=42) if len(
        df_ok) > max_rows else df_ok

    plt.scatter(df_ko_scatter[feature1], df_ko_scatter[feature2],
                label='KO', alpha=0.3, color=color_ko, s=5)
    plt.scatter(df_ok_scatter[feature1], df_ok_scatter[feature2],
                label='OK', alpha=0.3, color=color_ok, s=5)

    # Linear regression for KO
    model_ko = LinearRegression()
    X_ko = df_ko[feature1].values.reshape(-1, 1)
    y_ko = df_ko[feature2].values
    model_ko.fit(X_ko, y_ko)
    ko_min = df_ko[feature1].min()
    ko_max = df_ko[feature1].max()
    ko_margin = (ko_max - ko_min) * .05  # 5% margin
    x_vals_ko = np.linspace(ko_min - ko_margin, ko_max + ko_margin, 100).reshape(-1, 1)
    y_vals_ko = model_ko.predict(x_vals_ko)
    plt.plot(x_vals_ko, y_vals_ko, color=color_ko,
             alpha=0.6, linestyle='-', linewidth=1.5)

    # Linear regression for OK
    model_ok = LinearRegression()
    X_ok = df_ok[feature1].values.reshape(-1, 1)
    y_ok = df_ok[feature2].values
    model_ok.fit(X_ok, y_ok)
    ok_min = df_ok[feature1].min()
    ok_max = df_ok[feature1].max()
    ok_margin = (ok_max - ok_min) * .05  # 5% margin
    x_vals_ok = np.linspace(ok_min - ok_margin, ok_max + ok_margin, 100).reshape(-1, 1)
    y_vals_ok = model_ok.predict(x_vals_ok)
    plt.plot(x_vals_ok, y_vals_ok, color=color_ok,
             alpha=0.6, linestyle='-', linewidth=1.5)

    plt.title(f"Scatter Plot: {feature1} vs {feature2}")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()

    analysis = f"<p>Scatter plot of {feature1} vs {feature2}</p>"
    b64_data = base64.b64encode(img_bytes).decode('utf-8')
    img_tag = f'<br><img src="data:image/png;base64,{b64_data}" style="max-width: 400px; max-height: 300px;"/>'
    html += analysis + img_tag

    return html

# Plot All Time Series


def plot_all_timeseries(df: pd.DataFrame, time_feature: str | None = None) -> str:
    feature_columns = [col for col in df.columns if col != "label"]
    html = ""
    for col in feature_columns:
        html += timeseries_plot(df, col, time_feature)
    return html

# Plot Time Series


def timeseries_plot(df: pd.DataFrame, feature: str, time_feature: str | None = None) -> str:
    max_rows = 1500
    window = int(len(df) / 100) if len(df) > 100 else 1
    # feature_columns = [col for col in df.columns if col != "label"]
    if time_feature is None:
        time_feature = "__time__"
        df = df.copy()
        df[time_feature] = np.arange(len(df))
    df_ko = df[df["label"] == "KO"]
    df_ok = df[df["label"] == "OK"]

    html = ""
    # HTML string with embedded image
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_ok = colors[0 % len(colors)]
    color_ko = colors[1 % len(colors)]

    # for col in feature_columns:
    plt.figure(figsize=(12, 6))
    x_ok = df_ok[time_feature]
    x_ko = df_ko[time_feature]
    y_ok = df_ok[feature]
    y_ko = df_ko[feature]

    x_ok_scatter = x_ok.sample(max_rows, random_state=42) if len(
        x_ok) > max_rows else x_ok
    x_ko_scatter = x_ko.sample(max_rows, random_state=42) if len(
        x_ko) > max_rows else x_ko

    y_ok_scatter = df_ok.loc[x_ok_scatter.index, feature]
    y_ko_scatter = df_ko.loc[x_ko_scatter.index, feature]

    plt.scatter(x_ok_scatter, y_ok_scatter, label="OK",
                color=color_ok, alpha=0.4, s=5)
    plt.scatter(x_ko_scatter, y_ko_scatter, label="KO",
                color=color_ko, alpha=0.4, s=5)

    # OK moving average
    y_ok_ma = y_ok.rolling(window=window, center=True).mean()
    plt.plot(x_ok, y_ok_ma, color=color_ok,
             linewidth=2, alpha=0.8, label="OK Mov Avg")

    # KO moving average
    y_ko_ma = y_ko.rolling(window=window, center=True).mean()
    plt.plot(x_ko, y_ko_ma, color=color_ko,
             linewidth=2, alpha=0.8, label="KO Mov Avg")

    # Linear regression for OK
    if len(x_ok) > 1:
        model_ok = LinearRegression().fit(x_ok.values.reshape(-1, 1), y_ok.values)
        y_pred_ok = model_ok.predict(x_ok.values.reshape(-1, 1))
        plt.plot(x_ok, y_pred_ok, color=color_ok,
                 alpha=.7, linestyle="--", linewidth=2)

    # Linear regression for KO
    if len(x_ko) > 1:
        model_ko = LinearRegression().fit(x_ko.values.reshape(-1, 1), y_ko.values)
        y_pred_ko = model_ko.predict(x_ko.values.reshape(-1, 1))
        plt.plot(x_ko, y_pred_ko, color=color_ko,
                 alpha=.7, linestyle="--", linewidth=2)
    plt.title(f"Time Series Plot of {feature}")
    plt.xlabel("Index")
    plt.ylabel(feature)
    plt.legend()

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()

    buf.seek(0)
    img_bytes = buf.read()
    buf.close()

    analysis = f"<p>Time series plot of {feature}</p>"
    b64_data = base64.b64encode(img_bytes).decode('utf-8')
    img_tag = f'<br><img src="data:image/png;base64,{b64_data}" style="max-width: 400px; max-height: 300px;"/>'
    html += analysis + img_tag

    return html

# Plot All FFT Histograms


def plot_all_ffthist(df: pd.DataFrame, time_feature: str) -> str:
    feature_columns = [col for col in df.columns if col != "label"]
    html = ""
    for col in feature_columns:
        html += ffthist_plot(df, col, time_feature)
    return html

# Plot FFT Histogram


def ffthist_plot(df: pd.DataFrame, feature: str, time_feature: str) -> str:
    # feature_columns = [col for col in df.columns if col != "label"]
    if time_feature is None:
        time_feature = "__time__"
        df = df.copy()
        df[time_feature] = np.arange(len(df))
    df_ko = df[df["label"] == "KO"]
    df_ok = df[df["label"] == "OK"]

    html = ""
    # HTML string with embedded image

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_ok = colors[0 % len(colors)]
    color_ko = colors[1 % len(colors)]

    # for col in feature_columns:
    plt.figure(figsize=(6, 4))
    fft_ok = np.abs(np.fft.fft(df_ok[feature]))
    fft_ko = np.abs(np.fft.fft(df_ko[feature]))

    # Compute histograms manually
    counts_ok, bins_ok = np.histogram(np.log(fft_ok), bins=20)
    counts_ko, bins_ko = np.histogram(np.log(fft_ko), bins=20)

    # Get most frequent bins (could be multiple)
    max_count_ok = counts_ok.max()
    dominant_bins_ok = [i for i, c in enumerate(
        counts_ok) if c == max_count_ok]

    max_count_ko = counts_ko.max()
    dominant_bins_ko = [i for i, c in enumerate(
        counts_ko) if c == max_count_ko]

    # Plot OK histogram with highlights
    for i, (count, left, right) in enumerate(zip(counts_ok, bins_ok[:-1], bins_ok[1:])):
        alpha = .7 if i in dominant_bins_ok else .4
        plt.bar((left + right)/2, count, width=right - left, alpha=alpha,
                color=color_ok, align='center', label='OK' if i == 0 else None)

    # Plot KO histogram with highlights
    for i, (count, left, right) in enumerate(zip(counts_ko, bins_ko[:-1], bins_ko[1:])):
        alpha = .7 if i in dominant_bins_ko else .4
        plt.bar((left + right)/2, count, width=right - left, alpha=alpha,
                color=color_ko, align='center', label='KO' if i == 0 else None)

    plt.title(f"FFT Histogram of {feature}")
    plt.xlabel("Frequency")
    plt.ylabel("Log Magnitude")
    plt.legend()

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()

    buf.seek(0)
    img_bytes = buf.read()
    buf.close()

    analysis = f"<p>FFT Histogram of {feature}</p>"
    b64_data = base64.b64encode(img_bytes).decode('utf-8')
    img_tag = f'<br><img src="data:image/png;base64,{b64_data}" style="max-width: 400px; max-height: 300px;"/>'
    html += analysis + img_tag

    return html
