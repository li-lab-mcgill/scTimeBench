import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("full.csv")

# 1. Configuration & Skip List
metrics_to_skip = []

# 2. Preprocessing (Assumes 'merged_df' from previous steps)
# If starting from raw 'df', perform the AUROC vs AUPRC merge first:
df["result"] = pd.to_numeric(df["result"], errors="coerce")
df_filtered = df[~df["metric"].isin(metrics_to_skip)].copy()

df_false = df_filtered[df_filtered["prc_threshold"] == False].rename(
    columns={"result": "AUROC_Result"}
)
df_true = df_filtered[df_filtered["prc_threshold"] == True].rename(
    columns={"result": "AUPRC_Result"}
)

merge_cols = ["method", "dataset", "step_setting", "metric", "time_type"]
merged_df = pd.merge(df_false, df_true, on=merge_cols).dropna(
    subset=["AUROC_Result", "AUPRC_Result"]
)

# 3. Visualization using Seaborn FacetGrid
# We create a grid where columns are 'dataset' and rows are 'method'
# 2. Use relplot (The 'Figure-Level' function)
# This replaces FacetGrid + map_dataframe and avoids the TypeError
g = sns.relplot(
    data=merged_df,
    x="AUROC_Result",
    y="AUPRC_Result",
    col="metric",
    hue="step_setting",
    kind="scatter",
    s=150,
    alpha=0.8,
    edgecolor="black",
    facet_kws={"margin_titles": True},
)


# 3. Add the y=x reference line to each subplot
def add_diagonal_line(data, **kwargs):
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5, zorder=0)
    plt.xlim(0, 1)
    plt.ylim(0, 1)


# Apply the line to every facet
g.map_dataframe(add_diagonal_line)

# 4. Final Formatting
g.set_axis_labels("AUROC Threshold Metric", "AUPRC Threshold Metric")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Performance Comparison: AUPRC vs AUROC Thresholding", fontsize=16)

plt.savefig("split_views.png")
# plt.show()
