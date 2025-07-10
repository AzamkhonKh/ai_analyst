import pandas as pd
from logic.plotting import *

if __name__ == "__main__":
    df = pd.read_csv("dataset/combined_labeled.csv")
    html = ""
    html += histogram_plot(df)
    html += scatter_plot(df)
    html += timeseries_plot(df, feature=None)
    html += ffthist_plot(df, feature=None)

    # Save to file
    with open("plot_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Report saved to plot_report.html. Open it in a browser to view the plots.")