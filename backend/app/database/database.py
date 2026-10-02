import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DB_PATH = PROJECT_ROOT / "storage" / "visionqc.db"


def get_connection() -> sqlite3.Connection:
    """Create a SQLite connection with row access by column name."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database() -> None:
    """Create database tables and default settings if they don't exist."""
    connection = get_connection()

    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT,
                anomaly_score REAL NOT NULL,
                threshold REAL NOT NULL,
                result TEXT NOT NULL,
                image_path TEXT,
                heatmap_path TEXT
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO settings (key, value)
            VALUES ('threshold', '12.3785223961');
            """
        )

        connection.commit()

    finally:
        connection.close()


def get_setting(key: str) -> str | None:
    """Return a setting value or None if it doesn't exist."""
    connection = get_connection()

    try:
        row = connection.execute(
            "SELECT value FROM settings WHERE key = ?",
            (key,),
        ).fetchone()

        return row["value"] if row else None

    finally:
        connection.close()


def set_setting(key: str, value: Any) -> None:
    """Create or update a setting."""
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key)
            DO UPDATE SET value = excluded.value
            """,
            (key, str(value)),
        )

        connection.commit()

    finally:
        connection.close()


def create_inspection(
    *,
    timestamp: str,
    filename: str | None,
    anomaly_score: float,
    threshold: float,
    result: str,
    image_path: str | None = None,
    heatmap_path: str | None = None,
) -> int:
    """Insert an inspection and return its generated ID."""
    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO inspections (
                timestamp,
                filename,
                anomaly_score,
                threshold,
                result,
                image_path,
                heatmap_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                filename,
                anomaly_score,
                threshold,
                result,
                image_path,
                heatmap_path,
            ),
        )

        connection.commit()

        return int(cursor.lastrowid)

    finally:
        connection.close()


def get_inspections(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent inspections."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                timestamp,
                filename,
                anomaly_score,
                threshold,
                result,
                image_path,
                heatmap_path
            FROM inspections
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()


def get_today_stats() -> dict[str, Any]:
    """Return today's inspection statistics."""
    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(
                    SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END),
                    0
                ) AS passed,
                COALESCE(
                    SUM(CASE WHEN result = 'FAIL' THEN 1 ELSE 0 END),
                    0
                ) AS failed
            FROM inspections
            WHERE date(timestamp) = date('now')
            """
        ).fetchone()

        total = int(row["total"])
        passed = int(row["passed"])
        failed = int(row["failed"])

        rejection_rate = (
            (failed / total) * 100
            if total > 0
            else 0.0
        )

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "rejection_rate": rejection_rate,
        }

    finally:
        connection.close()