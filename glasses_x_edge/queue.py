from pathlib import Path

from persistqueue import SQLiteAckQueue


def create_persistent_queue(db_path: Path) -> SQLiteAckQueue:
    return SQLiteAckQueue(
        path=str(db_path),
        auto_commit=True,
        multithreading=True,
    )
