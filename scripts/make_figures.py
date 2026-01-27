import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

def style_ax(ax):
    ax.grid(True, alpha=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)
    print("Saved", OUT / name)

def prompting_first_vs_second(df, metric, out_name):
    # expects columns: model, prompt, first_metric, second_metric
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    style_ax(ax)

    # If your CSV differs, adjust mapping below:
    # Try to be robust: if it has source_lang/prompt_technique columns, use those.
    if "prompt_technique" in df.columns:
        group_cols = ["source_lang", "prompt_technique"]
    else:
        group_cols = ["prompt"]

    g = df.groupby(group_cols)[[f"first_{metric}", f"second_{metric}"]].mean().reset_index()

    # Create a readable x label
    g["label"] = g[group_cols].astype(str).agg(" | ".join, axis=1)

    x = range(len(g))
    ax.plot(x, g[f"first_{metric}"], marker="o", linewidth=2.2, label="First attempt")
    ax.plot(x, g[f"second_{metric}"], marker="s", linewidth=2.2, label="Second attempt")

    ax.set_xticks(list(x))
    ax.set_xticklabels(g["label"], rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()}: First vs Second by Prompting Strategy")
    ax.legend(frameon=False)

    save(fig, out_name)

def threshold_ablation(bleu_csv="outputs/csv/bleu_threshold_ablation.csv",
                       comet_csv="outputs/csv/comet_threshold_ablation.csv",
                       out_name="threshold_ablation_bleu_comet.png"):
    bleu = pd.read_csv(bleu_csv)
    comet = pd.read_csv(comet_csv)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax in axes:
        style_ax(ax)

    # Expect columns: threshold, first, second  (rename if needed)
    def plot_panel(ax, df, title, ylab):
        cols = df.columns.tolist()
        # heuristics
        th = [c for c in cols if "th" in c.lower()][0]
        first = [c for c in cols if "first" in c.lower()][0]
        second = [c for c in cols if "second" in c.lower()][0]

        ax.plot(df[th], df[first], marker="o", linewidth=2.2, label="First attempt")
        ax.plot(df[th], df[second], marker="s", linewidth=2.2, label="Second attempt")
        ax.set_title(title)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False)

    plot_panel(axes[0], bleu, "BLEU Threshold Ablation", "BLEU")
    plot_panel(axes[1], comet, "COMET Threshold Ablation", "COMET")

    save(fig, out_name)

def delta_hist(df, metric, out_name):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    style_ax(ax)
    delta = df[f"second_{metric}"] - df[f"first_{metric}"]
    ax.hist(delta.dropna(), bins=35)
    ax.set_title(f"Δ{metric.upper()} (Second − First) distribution")
    ax.set_xlabel(f"Δ{metric.upper()}")
    ax.set_ylabel("Count")
    save(fig, out_name)

def main():
    # Main results
    main_csv = "outputs/csv/all_translation_results.csv"
    df = pd.read_csv(main_csv)

    # Prompting comparison (you may have a separate ablation CSV)
    # If you have average_scores_other_techniques.csv, you can use that instead.
    prompting_first_vs_second(df, "bleu", "bleu_prompting_first_vs_second.png")
    prompting_first_vs_second(df, "comet", "comet_prompting_first_vs_second.png")

    # Threshold ablations (expects you saved these)
    # If your filenames differ, rename them or change the args.
    if Path("outputs/csv/bleu_threshold_ablation.csv").exists() and Path("outputs/csv/comet_threshold_ablation.csv").exists():
        threshold_ablation()

    # Delta hists
    delta_hist(df, "bleu", "delta_bleu_hist.png")
    delta_hist(df, "comet", "delta_comet_hist.png")

if __name__ == "__main__":
    main()
