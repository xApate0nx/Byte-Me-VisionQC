from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class InspectionResult:
    """
    Structured result produced by the VisionQC decision engine.
    """

    anomaly_score: Optional[float]
    threshold: float
    decision: str
    confidence: float
    reason: str

    image_quality_valid: bool
    quality_reason: str

    product_detected: bool = True
    alignment_valid: bool = True

    def to_dict(self):
        return asdict(self)


class VisionQCDecisionEngine:
    """
    Converts PatchCore anomaly scores and image-quality information
    into a VisionQC inspection decision.

    Decision states:
        PASS
        REVIEW
        FAIL
        INSPECTION_INVALID
    """

    def __init__(
        self,
        threshold: float = 12.3785223961,
        review_margin: float = 0.10,
    ):
        self.threshold = float(threshold)
        self.review_margin = float(review_margin)

    def calculate_confidence(self, anomaly_score: float) -> float:
        """
        Calculates decision confidence based on the distance of the
        anomaly score from the calibrated threshold.

        This is NOT a probability of defect.

        Confidence increases as the score moves farther away from
        the decision boundary.

        The confidence is intentionally capped at 99%.
        """

        distance = abs(anomaly_score - self.threshold)

        # A score exactly on the threshold is maximally uncertain.
        # 1.0 is used as the scale reference for the prototype.
        confidence = distance / 1.0

        confidence = max(0.0, min(confidence, 0.99))

        return confidence * 100.0

    def inspect(
        self,
        anomaly_score: Optional[float],
        image_quality_valid: bool = True,
        quality_reason: str = "Image quality acceptable",
        product_detected: bool = True,
        alignment_valid: bool = True,
    ) -> InspectionResult:

        # ---------------------------------------------------------
        # 1. IMAGE QUALITY GATE
        # ---------------------------------------------------------

        if not image_quality_valid:
            return InspectionResult(
                anomaly_score=None,
                threshold=self.threshold,
                decision="INSPECTION_INVALID",
                confidence=0.0,
                reason="Image quality / position issue",
                image_quality_valid=False,
                quality_reason=quality_reason,
                product_detected=product_detected,
                alignment_valid=alignment_valid,
            )

        # ---------------------------------------------------------
        # 2. PRODUCT DETECTION
        # ---------------------------------------------------------

        if not product_detected:
            return InspectionResult(
                anomaly_score=None,
                threshold=self.threshold,
                decision="INSPECTION_INVALID",
                confidence=0.0,
                reason="Product not detected",
                image_quality_valid=True,
                quality_reason=quality_reason,
                product_detected=False,
                alignment_valid=alignment_valid,
            )

        # ---------------------------------------------------------
        # 3. ALIGNMENT
        # ---------------------------------------------------------

        if not alignment_valid:
            return InspectionResult(
                anomaly_score=None,
                threshold=self.threshold,
                decision="INSPECTION_INVALID",
                confidence=0.0,
                reason="Product alignment / position issue",
                image_quality_valid=True,
                quality_reason=quality_reason,
                product_detected=True,
                alignment_valid=False,
            )

        # ---------------------------------------------------------
        # 4. SCORE VALIDATION
        # ---------------------------------------------------------

        if anomaly_score is None:
            return InspectionResult(
                anomaly_score=None,
                threshold=self.threshold,
                decision="INSPECTION_INVALID",
                confidence=0.0,
                reason="Anomaly score unavailable",
                image_quality_valid=True,
                quality_reason=quality_reason,
                product_detected=True,
                alignment_valid=True,
            )

        anomaly_score = float(anomaly_score)

        # ---------------------------------------------------------
        # 5. CONFIDENCE
        # ---------------------------------------------------------

        confidence = self.calculate_confidence(anomaly_score)

        # ---------------------------------------------------------
        # 6. PASS / REVIEW / FAIL
        # ---------------------------------------------------------

        distance_from_threshold = abs(
            anomaly_score - self.threshold
        )

        # REVIEW is used only when the score is close to the
        # decision boundary.
        if distance_from_threshold <= self.review_margin:

            decision = "REVIEW"
            reason = "Score close to supervisor threshold"

        elif anomaly_score < self.threshold:

            decision = "PASS"
            reason = "No significant deviation detected"

        else:

            decision = "FAIL"
            reason = "Significant deviation from learned normal"

        return InspectionResult(
            anomaly_score=round(anomaly_score, 6),
            threshold=round(self.threshold, 6),
            decision=decision,
            confidence=round(confidence, 2),
            reason=reason,
            image_quality_valid=True,
            quality_reason=quality_reason,
            product_detected=True,
            alignment_valid=True,
        )


def print_result(result: InspectionResult):
    """
    Human-readable console output.
    """

    print()
    print("=" * 60)
    print("VisionQC Inspection Result")
    print("=" * 60)

    print(f"Decision          : {result.decision}")
    print(f"Anomaly score     : {result.anomaly_score}")
    print(f"Threshold         : {result.threshold}")
    print(f"Confidence        : {result.confidence:.2f}%")
    print(f"Reason            : {result.reason}")
    print(f"Image quality     : {result.image_quality_valid}")
    print(f"Quality reason    : {result.quality_reason}")
    print(f"Product detected  : {result.product_detected}")
    print(f"Alignment valid   : {result.alignment_valid}")

    print("=" * 60)


if __name__ == "__main__":

    engine = VisionQCDecisionEngine(
        threshold=12.3785223961,
        review_margin=0.10,
    )

    # ---------------------------------------------------------
    # TEST 1 — CLEAR PASS
    # ---------------------------------------------------------

    result = engine.inspect(
        anomaly_score=10.60
    )

    print_result(result)

    # ---------------------------------------------------------
    # TEST 2 — CLEAR FAIL
    # ---------------------------------------------------------

    result = engine.inspect(
        anomaly_score=20.00
    )

    print_result(result)

    # ---------------------------------------------------------
    # TEST 3 — NEAR THRESHOLD
    # ---------------------------------------------------------

    result = engine.inspect(
        anomaly_score=12.35
    )

    print_result(result)

    # ---------------------------------------------------------
    # TEST 4 — INVALID IMAGE
    # ---------------------------------------------------------

    result = engine.inspect(
        anomaly_score=None,
        image_quality_valid=False,
        quality_reason="Image too blurry",
    )

    print_result(result)