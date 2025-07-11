import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from io import BytesIO
from typing import Protocol
from sklearn.linear_model import LinearRegression
import base64
import numpy as np

# --- Plotting Interface and Registry ---


class PlotFunction(Protocol):
    def __call__(self, df: pd.DataFrame, feature: str) -> str:
        ...


class PlotRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, plot_type: str, func: PlotFunction):
        self._registry[plot_type] = func

    def get(self, plot_type: str) -> PlotFunction:
        return self._registry.get(plot_type)


# Plot Histogram
def histogram_plot(df: pd.DataFrame, feature: str) -> str:
    feature_columns = [col for col in df.columns if col != "label"]
    df_ko = df[df["label"] == "KO"]
    df_ok = df[df["label"] == "OK"]

    html = ""
    # HTML string with embedded image
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_ok = colors[0 % len(colors)]
    color_ko = colors[1 % len(colors)]
    for col in feature_columns:
        plt.figure(figsize=(6, 4))

        # Plot histogram
        plt.hist(df_ok[col], bins=20, alpha=0.4, label=f'OK', color=color_ok)
        plt.hist(df_ko[col], bins=20, alpha=0.4, label=f'KO', color=color_ko)

        # Plot mean line
        mean_ok = df_ok[col].mean()
        std_ok = df_ok[col].std()
        mean_ko = df_ko[col].mean()
        std_ko = df_ko[col].std()

        plt.axvline(mean_ok, color=color_ok, alpha=0.6,
                    linestyle='-', linewidth=2)
        plt.axvline(mean_ok + std_ok, color=color_ok,
                    alpha=0.6, linestyle=':', linewidth=2)
        plt.axvline(mean_ko, color=color_ko, alpha=0.6,
                    linestyle='-', linewidth=2)
        plt.axvline(mean_ko + std_ko, color=color_ko,
                    alpha=0.6, linestyle=':', linewidth=2)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
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

        analysis = f"<p>Histogram of {col}</p>"
        b64_data = base64.b64encode(img_bytes).decode('utf-8')
        img_tag = f'<br><img src="data:image/png;base64,{b64_data}" style="max-width: 400px; max-height: 300px;"/>'
        html += analysis + img_tag
    return html


# Plot Scatterplots
def scatter_plot(df: pd.DataFrame, feature: str) -> str:
    max_rows = 1500
    feature_columns = [col for col in df.columns if col != "label"]
    df_ko = df[df["label"] == "KO"]
    df_ok = df[df["label"] == "OK"]

    html = ""
    # HTML string with embedded image
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_ok = colors[0 % len(colors)]
    color_ko = colors[1 % len(colors)]

    for idx, col in enumerate(feature_columns):
        for in_idx, in_col in enumerate(feature_columns):
            if idx > in_idx:
                plt.figure(figsize=(6, 4))
                df_ko_scatter = df_ko.sample(max_rows, random_state=42) if len(
                    df_ko) > max_rows else df_ko
                df_ok_scatter = df_ok.sample(max_rows, random_state=42) if len(
                    df_ok) > max_rows else df_ok

                plt.scatter(df_ko_scatter[col], df_ko_scatter[in_col],
                            label='KO', alpha=0.3, color=color_ko, s=5)
                plt.scatter(df_ok_scatter[col], df_ok_scatter[in_col],
                            label='OK', alpha=0.3, color=color_ok, s=5)

                # Linear regression for KO
                model_ko = LinearRegression()
                X_ko = df_ko[col].values.reshape(-1, 1)
                y_ko = df_ko[in_col].values
                model_ko.fit(X_ko, y_ko)
                x_vals_ko = np.linspace(
                    df_ko[col].min(), df_ko[col].max(), 100).reshape(-1, 1)
                y_vals_ko = model_ko.predict(x_vals_ko)
                plt.plot(x_vals_ko, y_vals_ko, color=color_ko,
                         alpha=0.6, linestyle='-', linewidth=1.5)

                # Linear regression for OK
                model_ok = LinearRegression()
                X_ok = df_ok[col].values.reshape(-1, 1)
                y_ok = df_ok[in_col].values
                model_ok.fit(X_ok, y_ok)
                x_vals_ok = np.linspace(
                    df_ok[col].min(), df_ok[col].max(), 100).reshape(-1, 1)
                y_vals_ok = model_ok.predict(x_vals_ok)
                plt.plot(x_vals_ok, y_vals_ok, color=color_ok,
                         alpha=0.6, linestyle='-', linewidth=1.5)

                plt.title(f"Scatter Plot: {col} vs {in_col}")
                plt.xlabel(col)
                plt.ylabel(in_col)
                plt.legend()
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                img_bytes = buf.read()
                buf.close()

                analysis = f"<p>Scatter plot of {col} vs {in_col}</p>"
                b64_data = base64.b64encode(img_bytes).decode('utf-8')
                img_tag = f'<br><img src="data:image/png;base64,{b64_data}" style="max-width: 400px; max-height: 300px;"/>'
                html += analysis + img_tag
    return html


# Plot Time Series
def timeseries_plot(df: pd.DataFrame, feature: str) -> str:
    max_rows = 1500
    window = int(len(df) / 100)
    feature_columns = [col for col in df.columns if col != "label"]
    if feature is None:
        feature = "__time__"
        df = df.copy()
        df[feature] = np.arange(len(df))
    df_ko = df[df["label"] == "KO"]
    df_ok = df[df["label"] == "OK"]

    html = ""
    # HTML string with embedded image
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_ok = colors[0 % len(colors)]
    color_ko = colors[1 % len(colors)]

    for col in feature_columns:
        plt.figure(figsize=(12, 6))
        x_ok = df_ok[feature]
        x_ko = df_ko[feature]
        y_ok = df_ok[col]
        y_ko = df_ko[col]

        x_ok_scatter = x_ok.sample(max_rows, random_state=42) if len(
            x_ok) > max_rows else x_ok
        x_ko_scatter = x_ko.sample(max_rows, random_state=42) if len(
            x_ko) > max_rows else x_ko

        y_ok_scatter = df_ok.loc[x_ok_scatter.index, col]
        y_ko_scatter = df_ko.loc[x_ko_scatter.index, col]

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
        plt.title(f"Time Series Plot of {col}")
        plt.xlabel("Index")
        plt.ylabel(col)
        plt.legend()

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()

        buf.seek(0)
        img_bytes = buf.read()
        buf.close()

        analysis = f"<p>Time series plot of {col}</p>"
        b64_data = base64.b64encode(img_bytes).decode('utf-8')
        img_tag = f'<br><img src="data:image/png;base64,{b64_data}" style="max-width: 400px; max-height: 300px;"/>'
        html += analysis + img_tag
    return html


# Plot FFT Histogram
def ffthist_plot(df: pd.DataFrame, feature: str) -> str:
    feature_columns = [col for col in df.columns if col != "label"]
    if feature is None:
        feature = "__time__"
        df = df.copy()
        df[feature] = np.arange(len(df))
    df_ko = df[df["label"] == "KO"]
    df_ok = df[df["label"] == "OK"]

    html = ""
    # HTML string with embedded image

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_ok = colors[0 % len(colors)]
    color_ko = colors[1 % len(colors)]

    for col in feature_columns:
        plt.figure(figsize=(6, 4))
        fft_ok = np.abs(np.fft.fft(df_ok[col]))
        fft_ko = np.abs(np.fft.fft(df_ko[col]))

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

        plt.title(f"FFT Histogram of {col}")
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

        analysis = f"<p>FFT Histogram of {col}</p>"
        b64_data = base64.b64encode(img_bytes).decode('utf-8')
        img_tag = f'<br><img src="data:image/png;base64,{b64_data}" style="max-width: 400px; max-height: 300px;"/>'
        html += analysis + img_tag
    return html
