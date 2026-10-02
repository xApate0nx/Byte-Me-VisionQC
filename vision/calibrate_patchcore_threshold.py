from pathlib import Path
import numpy as np


# These are the RAW PatchCore scores from the current evaluation.
# They are entered directly from evaluate_patchcore_v2_raw.py.

NORMAL_SCORES = np.array([
    10.6038351059,
    10.5259437561,
    12.3268375397,
    10.3874931335,
    10.3513994217,
    10.2965764999,
    10.0223283768,
    10.1671590805,
    12.3185834885,
    10.8552074432,
])

DEFECT_SCORES = np.array([
    13.6144571304,
    15.2854337692,
    15.8911066055,
    17.0424365997,
    19.3954544067,
    17.6308212280,
    15.8781328201,
    20.6811027527,
    19.5679626465,
    19.8419513702,
    21.0927181244,
    15.4058275223,
    12.4302072525,
    12.6430301666,
    18.7178421021,
    16.8834953308,
    15.1761331558,
    14.1757097244,
    14.6547718048,
    15.9990863800,
    14.3473281860,
    15.4683752060,
    12.4302072525,
])

# Remove accidental duplicate if present.
# Keep every evaluated defect sample.
# Duplicate scores are valid because they may belong to different images.


def evaluate_threshold(threshold):
    """
    Score >= threshold -> DEFECT
    Score < threshold  -> NORMAL
    """

    false_positives = int(np.sum(NORMAL_SCORES >= threshold))
    true_negatives = int(np.sum(NORMAL_SCORES < threshold))

    true_positives = int(np.sum(DEFECT_SCORES >= threshold))
    false_negatives = int(np.sum(DEFECT_SCORES < threshold))

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    specificity = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (true_positives + true_negatives)
        / (
            true_positives
            + true_negatives
            + false_positives
            + false_negatives
        )
    )

    return {
        "threshold": threshold,
        "tp": true_positives,
        "tn": true_negatives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def main():

    print("=" * 80)
    print("VisionQC — PatchCore Threshold Calibration")
    print("=" * 80)

    print(f"\nNormal samples : {len(NORMAL_SCORES)}")
    print(f"Defect samples : {len(DEFECT_SCORES)}")

    print(
        f"\nNormal range : "
        f"{NORMAL_SCORES.min():.4f} → "
        f"{NORMAL_SCORES.max():.4f}"
    )

    print(
        f"Defect range : "
        f"{DEFECT_SCORES.min():.4f} → "
        f"{DEFECT_SCORES.max():.4f}"
    )

    # Candidate thresholds between adjacent unique scores.
    all_scores = np.sort(
        np.unique(
            np.concatenate(
                [NORMAL_SCORES, DEFECT_SCORES]
            )
        )
    )

    thresholds = []

    for a, b in zip(all_scores[:-1], all_scores[1:]):
        thresholds.append((a + b) / 2)

    results = [
        evaluate_threshold(t)
        for t in thresholds
    ]

    # Sort by F1 for inspection.
    best_f1 = sorted(
        results,
        key=lambda x: x["f1"],
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("TOP THRESHOLDS BY F1")
    print("=" * 80)

    print(
        f"{'Threshold':>12} "
        f"{'TP':>4} "
        f"{'TN':>4} "
        f"{'FP':>4} "
        f"{'FN':>4} "
        f"{'Precision':>10} "
        f"{'Recall':>8} "
        f"{'Specificity':>12} "
        f"{'F1':>8}"
    )

    for r in best_f1[:10]:

        print(
            f"{r['threshold']:12.4f} "
            f"{r['tp']:4d} "
            f"{r['tn']:4d} "
            f"{r['fp']:4d} "
            f"{r['fn']:4d} "
            f"{r['precision']:10.3f} "
            f"{r['recall']:8.3f} "
            f"{r['specificity']:12.3f} "
            f"{r['f1']:8.3f}"
        )

    # Find thresholds with zero FP and zero FN.
    perfect = [
        r
        for r in results
        if r["fp"] == 0 and r["fn"] == 0
    ]

    print("\n" + "=" * 80)
    print("ZERO-FP / ZERO-FN THRESHOLDS")
    print("=" * 80)

    if perfect:
        for r in perfect:
            print(
                f"Threshold={r['threshold']:.6f} "
                f"TP={r['tp']} "
                f"TN={r['tn']} "
                f"FP={r['fp']} "
                f"FN={r['fn']}"
            )
    else:
        print("No threshold produced zero FP and zero FN.")

    # Midpoint between maximum normal and minimum defect.
    max_normal = NORMAL_SCORES.max()
    min_defect = DEFECT_SCORES.min()

    midpoint = (max_normal + min_defect) / 2

    midpoint_result = evaluate_threshold(midpoint)

    print("\n" + "=" * 80)
    print("OBSERVED CLASS-SEPARATION MIDPOINT")
    print("=" * 80)

    print(f"\nMaximum normal score : {max_normal:.10f}")
    print(f"Minimum defect score : {min_defect:.10f}")
    print(f"Observed gap         : {min_defect - max_normal:.10f}")

    print(f"\nMidpoint threshold   : {midpoint:.10f}")

    print("\nPerformance on this evaluation split:")

    for key, value in midpoint_result.items():
        if key != "threshold":
            print(f"{key:12s}: {value}")

    print("\n" + "=" * 80)
    print("IMPORTANT")
    print("=" * 80)

    print(
        "\nThese thresholds describe this current evaluation dataset."
    )

    print(
        "They should NOT yet be treated as a production threshold."
    )

    print(
        "A final supervisor threshold should be calibrated using "
        "a larger and independently collected validation set."
    )


if __name__ == "__main__":
    main()