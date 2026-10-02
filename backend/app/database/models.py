from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from backend.app.database.database import Base


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)

    inspection_id = Column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )

    timestamp = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    filename = Column(
        String(255),
        nullable=False,
    )

    product = Column(
        String(100),
        default="Water Bottle Cap",
        nullable=False,
    )

    sku = Column(
        String(100),
        default="water_cap_v1",
        nullable=False,
    )

    model_name = Column(
        String(100),
        default="PatchCore",
        nullable=False,
    )

    model_version = Column(
        String(100),
        default="patchcore_water_cap_v2",
        nullable=False,
    )

    anomaly_score = Column(
        Float,
        nullable=False,
    )

    threshold = Column(
        Float,
        nullable=False,
    )

    confidence = Column(
        Float,
        nullable=False,
    )

    decision = Column(
        String(30),
        nullable=False,
    )

    reason = Column(
        Text,
        nullable=True,
    )

    image_quality = Column(
        String(30),
        default="VALID",
        nullable=False,
    )

    alignment = Column(
        String(30),
        default="VALID",
        nullable=False,
    )

    original_image_path = Column(
        Text,
        nullable=True,
    )

    roi_image_path = Column(
        Text,
        nullable=True,
    )

    heatmap_path = Column(
        Text,
        nullable=True,
    )

    supervisor_verdict = Column(
        String(30),
        nullable=True,
    )

    supervisor_feedback = Column(
        Text,
        nullable=True,
    )