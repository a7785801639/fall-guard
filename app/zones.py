from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path


# Расчётный четырёхугольник пола (200×120 см) перед бортом кровати под стационарный ракурс 192.168.1.50
FALLBACK_ROOM_FLOOR_POLYGON: list[list[float]] = [
    [0.28, 0.79],
    [0.60, 0.79],
    [0.64, 0.98],
    [0.22, 0.98],
]


def get_default_floor_polygon() -> list[list[float]]:
    try:
        from . import homography as _hg  # type: ignore[import-not-found]

        for attr_name in (
            "DEFAULT_FLOOR_POLYGON",
            "DEFAULT_ROOM_FLOOR_POLYGON",
            "DEFAULT_CAMERA_FLOOR_POLYGON",
        ):
            cand = getattr(_hg, attr_name, None)
            if isinstance(cand, (list, tuple)) and len(cand) >= 4:
                return [[round(float(p[0]), 5), round(float(p[1]), 5)] for p in cand[:4]]
        cls_hg = getattr(_hg, "FloorHomography", None)
        if cls_hg is not None:
            for m_name in ("default_floor_polygon", "get_default_polygon"):
                fn = getattr(cls_hg, m_name, None)
                if callable(fn):
                    cand = fn()
                    if isinstance(cand, (list, tuple)) and len(cand) >= 4:
                        return [[round(float(p[0]), 5), round(float(p[1]), 5)] for p in cand[:4]]
    except Exception:
        pass
    return [list(pt) for pt in FALLBACK_ROOM_FLOOR_POLYGON]


@dataclass(frozen=True)
class ZonesConfig:
    """Полигоны в нормализованных координатах [0..1]: кровать, дверь, пол 3D и зоны игнорирования."""

    bed_polygon: list[list[float]]
    door_polygon: list[list[float]]
    ignore_polygons: list[list[list[float]]] = field(default_factory=list)
    floor_polygon: list[list[float]] = field(default_factory=list)
    floor_width_cm: float = 200.0
    floor_depth_cm: float = 120.0
    mattress_height_cm: float = 55.0

    @property
    def has_bed(self) -> bool:
        return len(self.bed_polygon) >= 3

    @property
    def has_door(self) -> bool:
        return len(self.door_polygon) >= 3

    @property
    def has_ignore(self) -> bool:
        return len(self.ignore_polygons) > 0

    @property
    def has_floor(self) -> bool:
        return len(self.floor_polygon) >= 4

    def to_dict(self) -> dict:
        return {
            "bed_polygon": self.bed_polygon,
            "door_polygon": self.door_polygon,
            "ignore_polygons": self.ignore_polygons,
            "floor_polygon": self.floor_polygon,
            "floor_width_cm": self.floor_width_cm,
            "floor_depth_cm": self.floor_depth_cm,
            "mattress_height_cm": self.mattress_height_cm,
        }


def _sanitize_polygon(raw: object) -> list[list[float]]:
    if not isinstance(raw, list):
        return []
    pts: list[list[float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x = max(0.0, min(1.0, float(item[0])))
            y = max(0.0, min(1.0, float(item[1])))
            pts.append([round(x, 5), round(y, 5)])
    return pts if len(pts) >= 3 else []


def _sanitize_floor_polygon(raw: object) -> list[list[float]]:
    if not isinstance(raw, list):
        return []
    pts: list[list[float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x = max(0.0, min(1.0, float(item[0])))
            y = max(0.0, min(1.0, float(item[1])))
            pts.append([round(x, 5), round(y, 5)])
    return pts[:4] if len(pts) >= 4 else []


def _sanitize_cm(raw: object, default: float, lo: float = 10.0, hi: float = 1000.0) -> float:
    if raw is None or isinstance(raw, bool):
        return default
    try:
        val = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return round(max(lo, min(hi, val)), 1)


def _point_in_poly_2d(px: float, py: float, poly: list[list[float]]) -> bool:
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / ((yj - yi) if abs(yj - yi) > 1e-12 else 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def polygon_coverage_ratio(
    cover_poly: list[list[float]],
    target_poly: list[list[float]],
    grid_n: int = 24,
) -> float:
    """Возвращает долю площади target_poly [0..1], накрытую полигоном cover_poly."""
    if len(cover_poly) < 3 or len(target_poly) < 3:
        return 0.0
    txs = [float(p[0]) for p in target_poly]
    tys = [float(p[1]) for p in target_poly]
    min_x, max_x = min(txs), max(txs)
    min_y, max_y = min(tys), max(tys)
    tw = max_x - min_x
    th = max_y - min_y
    if tw <= 1e-5 or th <= 1e-5:
        return 0.0

    cxs = [float(p[0]) for p in cover_poly]
    cys = [float(p[1]) for p in cover_poly]
    if max(cxs) < min_x or min(cxs) > max_x or max(cys) < min_y or min(cys) > max_y:
        return 0.0

    target_hits = 0
    both_hits = 0
    for iy in range(grid_n):
        py = min_y + th * ((iy + 0.5) / grid_n)
        for ix in range(grid_n):
            px = min_x + tw * ((ix + 0.5) / grid_n)
            if _point_in_poly_2d(px, py, target_poly):
                target_hits += 1
                if _point_in_poly_2d(px, py, cover_poly):
                    both_hits += 1
    return (both_hits / float(target_hits)) if target_hits > 0 else 0.0


def _filter_bed_covering_ignores(
    ignore_polys: list[list[list[float]]],
    bed_poly: list[list[float]],
    max_bed_coverage: float = 0.50,
) -> list[list[list[float]]]:
    if len(bed_poly) < 3 or not ignore_polys:
        return ignore_polys
    return [
        poly
        for poly in ignore_polys
        if polygon_coverage_ratio(poly, bed_poly) < max_bed_coverage
    ]


def _sanitize_polygon_list(raw: object) -> list[list[list[float]]]:
    if not isinstance(raw, list):
        return []
    polys: list[list[list[float]]] = []
    for item in raw:
        poly = _sanitize_polygon(item)
        if len(poly) >= 3:
            polys.append(poly)
    return polys


class ZoneStore:
    """Потокобезопасное хранилище размеченных зон (data/zones.json)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._zones = ZonesConfig(bed_polygon=[], door_polygon=[], ignore_polygons=[])
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self._path.exists():
                return
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                bed_poly = _sanitize_polygon(data.get("bed_polygon", []))
                door_poly = _sanitize_polygon(data.get("door_polygon", []))
                ignore_polys = _sanitize_polygon_list(data.get("ignore_polygons", []))
                floor_poly = _sanitize_floor_polygon(data.get("floor_polygon", []))
                needs_migration_save = False
                if len(bed_poly) >= 3 and len(floor_poly) < 4 and "floor_polygon" not in data:
                    floor_poly = get_default_floor_polygon()
                    needs_migration_save = True
                if len(bed_poly) >= 3 and "mattress_height_cm" not in data:
                    needs_migration_save = True
                self._zones = ZonesConfig(
                    bed_polygon=bed_poly,
                    door_polygon=door_poly,
                    ignore_polygons=ignore_polys,
                    floor_polygon=floor_poly,
                    floor_width_cm=_sanitize_cm(data.get("floor_width_cm"), 200.0, lo=40.0, hi=1000.0),
                    floor_depth_cm=_sanitize_cm(data.get("floor_depth_cm"), 120.0, lo=40.0, hi=1000.0),
                    mattress_height_cm=_sanitize_cm(data.get("mattress_height_cm"), 55.0, lo=10.0, hi=150.0),
                )
                if needs_migration_save:
                    try:
                        self._save_locked()
                    except OSError:
                        pass
            except Exception:
                self._zones = ZonesConfig(bed_polygon=[], door_polygon=[], ignore_polygons=[])

    def _save_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._zones.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def get(self) -> ZonesConfig:
        with self._lock:
            return self._zones

    def update(
        self,
        bed_polygon: list[list[float]] | None = None,
        door_polygon: list[list[float]] | None = None,
        ignore_polygons: list[list[list[float]]] | None = None,
        floor_polygon: list[list[float]] | None = None,
        floor_width_cm: float | None = None,
        floor_depth_cm: float | None = None,
        mattress_height_cm: float | None = None,
    ) -> ZonesConfig:
        with self._lock:
            new_bed = (
                _sanitize_polygon(bed_polygon)
                if bed_polygon is not None
                else self._zones.bed_polygon
            )
            new_door = (
                _sanitize_polygon(door_polygon)
                if door_polygon is not None
                else self._zones.door_polygon
            )
            new_ignore = (
                _sanitize_polygon_list(ignore_polygons)
                if ignore_polygons is not None
                else self._zones.ignore_polygons
            )
            if ignore_polygons is not None and len(new_bed) >= 3:
                new_ignore = _filter_bed_covering_ignores(new_ignore, new_bed, max_bed_coverage=0.50)
            new_floor = (
                _sanitize_floor_polygon(floor_polygon)
                if floor_polygon is not None
                else self._zones.floor_polygon
            )
            new_fw = (
                _sanitize_cm(floor_width_cm, self._zones.floor_width_cm, lo=40.0, hi=1000.0)
                if floor_width_cm is not None
                else self._zones.floor_width_cm
            )
            new_fd = (
                _sanitize_cm(floor_depth_cm, self._zones.floor_depth_cm, lo=40.0, hi=1000.0)
                if floor_depth_cm is not None
                else self._zones.floor_depth_cm
            )
            new_mh = (
                _sanitize_cm(mattress_height_cm, self._zones.mattress_height_cm, lo=10.0, hi=150.0)
                if mattress_height_cm is not None
                else self._zones.mattress_height_cm
            )
            self._zones = ZonesConfig(
                bed_polygon=new_bed,
                door_polygon=new_door,
                ignore_polygons=new_ignore,
                floor_polygon=new_floor,
                floor_width_cm=new_fw,
                floor_depth_cm=new_fd,
                mattress_height_cm=new_mh,
            )
            self._save_locked()
            return self._zones

    def add_ignore_polygon(self, polygon: list[list[float]]) -> ZonesConfig:
        clean = _sanitize_polygon(polygon)
        with self._lock:
            if len(clean) >= 3:
                if len(self._zones.bed_polygon) >= 3 and (
                    polygon_coverage_ratio(clean, self._zones.bed_polygon) >= 0.50
                ):
                    return self._zones
                updated = [*self._zones.ignore_polygons, clean]
                self._zones = ZonesConfig(
                    bed_polygon=self._zones.bed_polygon,
                    door_polygon=self._zones.door_polygon,
                    ignore_polygons=updated,
                    floor_polygon=self._zones.floor_polygon,
                    floor_width_cm=self._zones.floor_width_cm,
                    floor_depth_cm=self._zones.floor_depth_cm,
                    mattress_height_cm=self._zones.mattress_height_cm,
                )
                self._save_locked()
            return self._zones

    def clear_ignore(self) -> ZonesConfig:
        with self._lock:
            self._zones = ZonesConfig(
                bed_polygon=self._zones.bed_polygon,
                door_polygon=self._zones.door_polygon,
                ignore_polygons=[],
                floor_polygon=self._zones.floor_polygon,
                floor_width_cm=self._zones.floor_width_cm,
                floor_depth_cm=self._zones.floor_depth_cm,
                mattress_height_cm=self._zones.mattress_height_cm,
            )
            self._save_locked()
            return self._zones
