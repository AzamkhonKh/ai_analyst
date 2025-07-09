import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Protocol

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


# Example plot function
def histogram_plot(df: pd.DataFrame, feature: str) ->  str:
    plt.figure(figsize=(6, 4))
    df[feature].hist(bins=20)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    analysis = f"Histogram of {feature}"
    # HTML string with embedded image
    import base64
    b64_data = base64.b64encode(img_bytes).decode('utf-8')
    img_tag = f'<br><img src="data:image/png;base64,{b64_data}" style="max-width: 400px; max-height: 300px;"/>'
    html = analysis + img_tag
    return analysis, img_bytes, html

