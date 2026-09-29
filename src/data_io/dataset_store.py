"""Read a single breath from the supplied benchmark CSV without loading it all."""

from __future__ import annotations

import csv
import json
import math
import threading
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {"train": ROOT / "train.csv", "test": ROOT / "test.csv"}
CACHE_DIR = ROOT / ".twinvent" / "cache"
EXPECTED_ROWS_PER_BREATH = 80
COMMON_COLUMNS = {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out"}


class DatasetError(ValueError):
    """A source file or selected breath is not usable by the replay app."""


class DatasetStore:
    def __init__(self) -> None:
        self._indices: dict[str, dict[str, Any]] = {}
        self._locks = {name: threading.Lock() for name in DATASETS}

    def clear_cached_indexes(self) -> None:
        for source, lock in self._locks.items():
            with lock:
                self._indices.pop(source, None)

    def _path(self, source: str) -> Path:
        if source not in DATASETS:
            raise DatasetError("Choose train.csv or test.csv as the data source.")
        path = DATASETS[source]
        if not path.is_file():
            raise DatasetError(f"{path.name} is missing from the project folder.")
        return path

    def _index_path(self, source: str, path: Path) -> Path:
        stat = path.stat()
        return CACHE_DIR / f"{source}-{stat.st_size}-{stat.st_mtime_ns}.json"

    def metadata(self, source: str) -> dict[str, Any]:
        info = self._get_index(source)
        breaths = info["breaths"]
        ids = [item["breath_id"] for item in breaths]
        return {
            "source": source,
            "file_name": DATASETS[source].name,
            "row_count": info["row_count"],
            "breath_count": len(breaths),
            "rows_per_breath": EXPECTED_ROWS_PER_BREATH,
            "has_pressure": info["has_pressure"],
            "first_breath_id": ids[0] if ids else None,
            "last_breath_id": ids[-1] if ids else None,
            "r_values": info["r_values"],
            "c_values": info["c_values"],
            "index_cached": info["cached"],
        }

    def adjacent_id(self, source: str, breath_id: str, direction: int) -> str:
        info = self._get_index(source)
        ids = [str(item["breath_id"]) for item in info["breaths"]]
        try:
            position = ids.index(str(int(breath_id)))
        except (ValueError, TypeError):
            raise DatasetError(f"Breath ID {breath_id} is not present in {DATASETS[source].name}.")
        next_position = max(0, min(len(ids) - 1, position + (1 if direction > 0 else -1)))
        return ids[next_position]

    def load_breath(self, source: str, breath_id: str) -> dict[str, Any]:
        path = self._path(source)
        info = self._get_index(source)
        key = str(int(breath_id))
        index = info["by_id"].get(key)
        if index is None:
            raise DatasetError(f"Breath ID {key} is not present in {path.name}.")

        rows: list[dict[str, Any]] = []
        with path.open("rb") as handle:
            handle.seek(index["offset"])
            for _ in range(index["count"]):
                raw = handle.readline()
                if not raw:
                    raise DatasetError(f"Breath {key} ends unexpectedly in {path.name}.")
                fields = raw.rstrip(b"\r\n").decode("utf-8").split(",")
                if len(fields) != len(info["columns"]):
                    raise DatasetError(f"A row in breath {key} has the wrong number of columns.")
                values = dict(zip(info["columns"], fields))
                rows.append(self._convert_row(values, path.name))

        rows.sort(key=lambda row: row["time_step"])
        self._validate_breath(rows, key, path.name)
        return {
            "source": source,
            "file_name": path.name,
            "breath_id": int(key),
            "has_pressure": info["has_pressure"],
            "rows": rows,
            "warnings": [],
        }

    @staticmethod
    def _convert_row(values: dict[str, str], file_name: str) -> dict[str, Any]:
        try:
            row: dict[str, Any] = {
                "id": int(values["id"]),
                "breath_id": int(values["breath_id"]),
                "time_step": float(values["time_step"]),
                "u_in": float(values["u_in"]),
                "u_out": int(values["u_out"]),
                "R": float(values["R"]) if values.get("R", "") != "" else None,
                "C": float(values["C"]) if values.get("C", "") != "" else None,
                "pressure": float(values["pressure"]) if values.get("pressure", "") != "" else None,
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(f"A row in {file_name} contains a missing or invalid number.") from exc
        for key in ("time_step", "u_in", "R", "C", "pressure"):
            if row[key] is not None and not math.isfinite(row[key]):
                raise DatasetError(f"A row in {file_name} contains a non-finite {key} value.")
        return row

    @staticmethod
    def _validate_breath(rows: list[dict[str, Any]], breath_id: str, file_name: str) -> None:
        if len(rows) != EXPECTED_ROWS_PER_BREATH:
            raise DatasetError(
                f"Breath {breath_id} has {len(rows)} rows; the benchmark expects "
                f"{EXPECTED_ROWS_PER_BREATH}. This breath was not plotted."
            )
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise DatasetError(f"Breath {breath_id} contains duplicate row IDs.")
        if any(row["breath_id"] != int(breath_id) for row in rows):
            raise DatasetError(f"Rows from more than one breath were found in {file_name}.")
        if any(row["u_out"] not in (0, 1) for row in rows):
            raise DatasetError(f"Breath {breath_id} has an invalid u_out value; expected 0 or 1.")
        times = [row["time_step"] for row in rows]
        if any(current <= previous for previous, current in zip(times, times[1:])):
            raise DatasetError(f"Breath {breath_id} has repeated or unordered time_step values.")
        for key in ("R", "C"):
            present = [row[key] for row in rows if row[key] is not None]
            if present and (len(present) != len(rows) or len(set(present)) != 1):
                raise DatasetError(f"Breath {breath_id} has inconsistent {key} values.")
        pressure_present = [row["pressure"] is not None for row in rows]
        if any(pressure_present) and not all(pressure_present):
            raise DatasetError(f"Breath {breath_id} has incomplete measured pressure values.")

    def _get_index(self, source: str) -> dict[str, Any]:
        path = self._path(source)
        with self._locks[source]:
            if source in self._indices:
                current = self._indices[source]
                stat = path.stat()
                if current["file_size"] == stat.st_size and current["mtime_ns"] == stat.st_mtime_ns:
                    return current
            cache_path = self._index_path(source, path)
            if cache_path.is_file():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    if cached["file_size"] == path.stat().st_size and cached["mtime_ns"] == path.stat().st_mtime_ns:
                        cached["by_id"] = {str(item["breath_id"]): item for item in cached["breaths"]}
                        cached["cached"] = True
                        self._indices[source] = cached
                        return cached
                except (OSError, KeyError, json.JSONDecodeError, TypeError):
                    pass
            info = self._build_index(path)
            self._indices[source] = info
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            to_save = {key: value for key, value in info.items() if key not in ("by_id", "cached")}
            temp_path = cache_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(to_save, separators=(",", ":")), encoding="utf-8")
            temp_path.replace(cache_path)
            info["cached"] = True
            return info

    @staticmethod
    def _build_index(path: Path) -> dict[str, Any]:
        with path.open("rb") as handle:
            header_line = handle.readline().decode("utf-8-sig").rstrip("\r\n")
            columns = next(csv.reader([header_line]))
            missing = sorted(COMMON_COLUMNS - set(columns))
            if missing:
                raise DatasetError(f"{path.name} is missing required columns: {', '.join(missing)}.")
            has_pressure = "pressure" in columns
            breath_idx = columns.index("breath_id")
            r_idx = columns.index("R")
            c_idx = columns.index("C")
            breaths: list[dict[str, Any]] = []
            seen: set[str] = set()
            r_values: set[float] = set()
            c_values: set[float] = set()
            current_id: str | None = None
            current_offset = handle.tell()
            current_count = 0
            row_count = 0
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                fields = raw.rstrip(b"\r\n").split(b",")
                if len(fields) != len(columns):
                    raise DatasetError(f"{path.name} row {row_count + 2} has the wrong number of columns.")
                try:
                    breath = fields[breath_idx].decode("ascii")
                    r_value = float(fields[r_idx]) if fields[r_idx] else None
                    c_value = float(fields[c_idx]) if fields[c_idx] else None
                except (ValueError, UnicodeDecodeError) as exc:
                    raise DatasetError(f"{path.name} row {row_count + 2} has an invalid breath ID or R/C value.") from exc
                if current_id is None:
                    current_id, current_offset = breath, offset
                elif breath != current_id:
                    breaths.append({"breath_id": int(current_id), "offset": current_offset, "count": current_count})
                    seen.add(current_id)
                    if breath in seen:
                        raise DatasetError(f"Rows for breath {breath} are separated in {path.name}; cannot index safely.")
                    current_id, current_offset, current_count = breath, offset, 0
                current_count += 1
                row_count += 1
                if r_value is not None:
                    r_values.add(r_value)
                if c_value is not None:
                    c_values.add(c_value)
            if current_id is not None:
                breaths.append({"breath_id": int(current_id), "offset": current_offset, "count": current_count})
            stat = path.stat()
        if not breaths:
            raise DatasetError(f"{path.name} has no breath rows.")
        return {
            "columns": columns,
            "has_pressure": has_pressure,
            "row_count": row_count,
            "breaths": breaths,
            "by_id": {str(item["breath_id"]): item for item in breaths},
            "r_values": sorted(r_values),
            "c_values": sorted(c_values),
            "file_size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "cached": False,
        }


STORE = DatasetStore()
