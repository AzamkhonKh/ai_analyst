import pandas as pd
from logic.plotting import *

if __name__ == "__main__":
    df = pd.read_csv("dataset/combined_labeled.csv")
    html = ""
    html += plot_shap_feature_force(df)
    #html += plot_all_histograms(df)
    #html += plot_all_scatter_plots(df)
    #html += plot_all_timeseries(df, time_feature=None)
    #html += plot_all_ffthist(df, time_feature=None)

    # Save to file
    with open("plot_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Report saved to plot_report.html. Open it in a browser to view the plots.")