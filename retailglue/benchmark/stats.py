import glob
import json
import os

import pandas as pd

from retailglue.benchmark.constants import BENCHMARK_DATA_ROOT
from retailglue.benchmark.drawer import save_heatmap_image


def recall(TP, FN):
    return round(TP / (TP + FN), 4) if (TP + FN) > 0 else 0.0


def precision(TP, FP):
    return round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0


def f1_score(TP, FN, FP):
    r, p = recall(TP, FN), precision(TP, FP)
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def tp_total_rate(TP, FP, FN):
    total = TP + FP + FN
    return round(TP / total, 4) if total > 0 else 0.0


def accuracy(TP, FN, TN, FP):
    total = TP + FN + TN + FP
    return round((TP + TN) / total, 4) if total > 0 else 0.0


def calculate_stats(matches, unmatched_single, unmatched_stitched,
                    box_count_of_stitched, box_count_of_single_pruned, box_count_of_single):
    diff = box_count_of_stitched - box_count_of_single
    diff_pct = round((diff / box_count_of_single * 100) if box_count_of_single > 0 else 0, 2)
    diff_pruned = box_count_of_stitched - box_count_of_single_pruned
    diff_pruned_pct = round((diff_pruned / box_count_of_single_pruned * 100) if box_count_of_single_pruned > 0 else 0, 2)

    return {
        "matches_tp": matches,
        "unmatched_single_fn": unmatched_single,
        "unmatched_stitched_fp": unmatched_stitched,
        "recall": recall(matches, unmatched_single),
        "precision": precision(matches, unmatched_stitched),
        "f1_score": f1_score(matches, unmatched_single, unmatched_stitched),
        "tp_total_rate": tp_total_rate(matches, unmatched_stitched, unmatched_single),
        "accuracy": accuracy(matches, unmatched_single, 0, unmatched_stitched),
        "total_stitched_tp_fp": box_count_of_stitched,
        "total_single_pruned_tp_fn": box_count_of_single_pruned,
        "total_single_tp_fp_fn": box_count_of_single,
        "detection_diff": diff,
        "detection_diff_percent": diff_pct,
        "detection_diff_pruned": diff_pruned,
        "detection_diff_pruned_percent": diff_pruned_pct,
    }


def save_stats(stats, save_path, suffix=""):
    save_heatmap_image(stats['matches_tp'], stats['unmatched_single_fn'], 0,
                       stats['unmatched_stitched_fp'], save_path, suffix=suffix)
    with open(os.path.join(save_path, f"Stats{suffix}.json"), "w") as f:
        json.dump(stats, f, indent=1)


def wrap_results(data_root=None):
    data_root = data_root or BENCHMARK_DATA_ROOT
    cols = {
        "Combination": [], "Sequence": [], "Panorama": [],
        "#Input Images": [], "#Panoramas": [], "Fully Stitched": [],
        "#Matches": [], "#Unmatched Single": [], "#Unmatched Stitched": [],
        "#Total Stitched": [], "#Total Single": [], "#Total Single Pruned": [],
        "Detection Diff": [], "Detection Diff %": [],
        "Detection Diff (Pruned)": [], "Detection Diff % (Pruned)": [],
        "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "TP_Total_Rate": [],
    }

    stat_files = glob.glob(os.path.join(data_root, "stitching_results", "*", "*", "Stats*.json"))
    for path in stat_files:
        stats = json.load(open(path, "r"))
        parts = path.split(os.sep)
        combination = parts[-3]
        sequence = parts[-2]
        filename = os.path.basename(path)
        pano_idx = filename.replace("Stats_pano", "").replace("Stats", "").replace(".json", "") or "0"

        cols["Combination"].append(combination)
        cols["Sequence"].append(sequence)
        cols["Panorama"].append(pano_idx)
        cols["#Input Images"].append(stats.get('num_input_images'))
        cols["#Panoramas"].append(stats.get('num_panoramas'))
        cols["Fully Stitched"].append(stats.get('fully_stitched'))
        cols["#Matches"].append(stats.get('matches_tp'))
        cols["#Unmatched Single"].append(stats.get('unmatched_single_fn'))
        cols["#Unmatched Stitched"].append(stats.get('unmatched_stitched_fp'))
        cols["#Total Single"].append(stats.get('total_single_tp_fp_fn'))
        cols["#Total Single Pruned"].append(stats.get('total_single_pruned_tp_fn'))
        cols["#Total Stitched"].append(stats.get('total_stitched_tp_fp'))
        cols["Detection Diff"].append(stats.get('detection_diff'))
        cols["Detection Diff %"].append(stats.get('detection_diff_percent'))
        cols["Detection Diff (Pruned)"].append(stats.get('detection_diff_pruned'))
        cols["Detection Diff % (Pruned)"].append(stats.get('detection_diff_pruned_percent'))
        cols["Accuracy"].append(stats.get('accuracy'))
        cols["Precision"].append(stats.get('precision'))
        cols["Recall"].append(stats.get('recall'))
        cols["F1"].append(stats.get('f1_score'))
        cols["TP_Total_Rate"].append(stats.get('tp_total_rate'))

    df = pd.DataFrame(cols)
    out_path = os.path.join(data_root, "ResultsAll.xlsx")
    with pd.ExcelWriter(out_path, engine='xlsxwriter', mode='w') as writer:
        for combination, group in df.groupby('Combination'):
            sheet = str(combination)[:31]
            group.drop('Combination', axis=1).sort_values('Sequence').reset_index(drop=True).to_excel(
                writer, sheet_name=sheet)

        summary = df.groupby('Combination')[["Accuracy", "Precision", "Recall", "F1", "TP_Total_Rate"]].mean()
        sizes = df.groupby('Combination').size()
        summary["TP_Total_Real_Rate"] = summary["TP_Total_Rate"] * (sizes / 100)

        stitch_ratio = df.drop_duplicates(subset=['Combination', 'Sequence']).groupby('Combination')['Fully Stitched'].mean()
        summary["Stitch_Ratio"] = (stitch_ratio * 100).round(2)
        summary.to_excel(writer, sheet_name="summary")
    return out_path
