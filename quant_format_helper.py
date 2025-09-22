# Just to help format the inter and intra distance dictionaries into a TSV for Excel
from datetime import datetime
from pathlib import Path

def write_cluster_quant_txt(
    inter_dists_tr: dict,
    intra_dists_tr: dict,
    inter_dists_te: dict,
    intra_dists_te: dict,
    task: int,
    split: int,
    filename: str | None = None,
    title: str | None = None,
):
    """
    Writes a sorted, tab-separated table combining inter/intra distances
    for train/test into a single TSV that can be pasted into Excel.

    Columns = Class, Inter_Train, Intra_Train, Inter_Test, Intra_Test
    Includes a MEAN row at the bottom.
    """

    # Collect and sort all observed class labels
    all_labels = sorted(
        set(inter_dists_tr.keys())
        | set(intra_dists_tr.keys())
        | set(inter_dists_te.keys())
        | set(intra_dists_te.keys())
    )

    # File path (use .tsv for Excel-friendliness)
    if filename is None:
        file_path = Path(f"results/default/split{split}/cluster_quant/task{task}_cluster_quant.tsv")
    else:
        file_path = Path(filename)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Header/meta lines (commented with # so they don't interfere in Excel if opened)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = title or "CLUSTER QUANT INFO (sorted by class label)"

    # Column headers
    headers = ["Class", "Inter_Train", "Intra_Train", "Inter_Test", "Intra_Test"]

    def fmt_num(val):
        if isinstance(val, (int, float)):
            return f"{val:.6f}"
        return ""  # blank if missing

    # Build lines (TSV)
    lines = []
    lines.append(f"# {title}")
    lines.append(f"# Task\t{task}")
    lines.append(f"# Generated\t{now}")
    lines.append("\t".join(headers))

    # Collect values for means
    col_vals = [[], [], [], []]  # for the 4 numeric columns in order

    for c in all_labels:
        vals = [
            inter_dists_tr.get(c, None),
            intra_dists_tr.get(c, None),
            inter_dists_te.get(c, None),
            intra_dists_te.get(c, None),
        ]
        # Track means (only numeric present values)
        for i, v in enumerate(vals):
            if isinstance(v, (int, float)):
                col_vals[i].append(v)

        row = [str(c)] + [fmt_num(v) for v in vals]
        lines.append("\t".join(row))

    # Means row
    means = []
    for vals in col_vals:
        means.append(sum(vals) / len(vals) if vals else None)

    mean_row = ["MEAN"] + [fmt_num(v) for v in means]
    lines.append("\t".join(mean_row))

    # End marker (commented)
    lines.append("# END CLUSTER INFO")

    # Write file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also print the TSV part (no comments) so you can copy/paste directly from the console if you want
    print("\n".join([ln for ln in lines if not ln.startswith("#")]))
