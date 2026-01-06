#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 09:26:39 2026

@author: rkotynski
"""
# planer_wydruku_3d_fast19.py
# Wymaga: pip install numpy pyqt5 numpy-stl matplotlib

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

import numpy as np
from stl import mesh as stlmesh

from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QRunnable, QThreadPool, QPoint, QEvent
from PyQt5.QtGui import QImage, QPixmap, QColor, QBrush, QPainter, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QDoubleSpinBox, QGroupBox, QFormLayout, QSlider, QMessageBox,
    QCheckBox, QComboBox, QInputDialog, QAction, QMenu
)

# =========================
# Ustawienia globalne
# =========================

CELL_MM = 0.2
WORK_MM = 200.0
WORK_W = int(round(WORK_MM / CELL_MM))
WORK_H = int(round(WORK_MM / CELL_MM))

AUTO_FIT_MARGIN_MM = 4.0

# Kratownica
LATTICE_PITCH_MM = 5.0
LATTICE_RIB_MM = 0.4
LATTICE_BASE_H_MM = 0.9      # minimalna wysokość (bazowa), potem * factor
LATTICE_SEG_MM = 5.0

# „ostrze” na górze
LATTICE_TAPER_TOP_H_MM = 0.2
LATTICE_TAPER_TOP_W_MM = 0.5

COLOR_PALETTE = [
    (255, 80, 80),
    (80, 200, 120),
    (80, 140, 255),
    (255, 200, 80),
    (220, 120, 255),
    (80, 220, 220),
    (200, 200, 200),
]

ARRANGE_ANGLES = [0, 90, 180, -90, 45, -45, 135, -135]
ORIENT_ANGLES = list(range(0, 91, 5))

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".planer_wydruku_3d_config.json")

# =========================
# i18n (PL/EN) — bez dodatkowych plików
# =========================

I18N = {
    "pl": {
        "app_title": "Planer Wydruku 3D",
        "btn_add": "Wczytaj obiekt (stl)…",
        "btn_copy": "Kopia",
        "btn_auto": "Uplasuj",
        "btn_arrange": "Rozmieść",
        "btn_del": "Usuń",
        "btn_save_layout": "Zapisz układ…",
        "btn_load_layout": "Wczytaj układ…",
        "btn_export": "Eksportuj scenę (stl)…",
        "btn_preview3d": "Podgląd 3D (z uproszczoną kratownicą)",
        "btn_lang_to_en": "English",
        "btn_lang_to_pl": "Polski",
        "gb_preview": "Podgląd",
        "chk_exact": "Podgląd dokładny (zgodny z eksportem)",
        "gb_export": "Eksport",
        "lbl_mode": "Tryb:",
        "chk_lattice_full": "Kratownica na całej siatce (oknie)",
        "lbl_lattice_strength": "Moc kratownicy (1..2):",
        "gb_grid": "Okno wyświetlania (mm) — 0.2mm; robocze 200×200mm",
        "lbl_grid_w": "Szerokość okna (mm):",
        "lbl_grid_h": "Wysokość okna (mm):",
        "chk_border": "Rysuj ramkę okna",
        "gb_transform": "Transformacje (wybrany obiekt)",
        "lbl_scale": "Skala:",
        "lbl_rx": "Obrót w pionie (oś x):",
        "lbl_ry": "Obrót w pionie (oś y):",
        "lbl_rz": "Obrót:",
        "lbl_tx": "Przesuń X:",
        "lbl_ty": "Przesuń Y:",
        "export_items": [
            "Bez kratownicy (szybko)",
            "Kratownica stałej wysokości (szybko)",
            "Kratownica zmiennej wysokości (wolniej)",
        ],
        "no_data": "Brak danych",
"overlap_title": "Uwaga: obiekty nachodzą na siebie",
"overlap_msg": "Wykryto nachodzenie obiektów na siebie w rzucie XY.\nObiekty mogą się przenikać w wyeksportowanym STL.\n\nPary nachodzących obiektów:",
"overlap_continue": "Kontynuować eksport?",
    },
    "en": {
        "app_title": "3D Print Planner",
        "btn_add": "Load object (stl)…",
        "btn_copy": "Duplicate",
        "btn_auto": "Auto place",
        "btn_arrange": "Arrange",
        "btn_del": "Remove",
        "btn_save_layout": "Save layout…",
        "btn_load_layout": "Load layout…",
        "btn_export": "Export scene (stl)…",
        "btn_preview3d": "3D Preview (with simplified lattice)",
        "btn_lang_to_en": "English",
        "btn_lang_to_pl": "Polski",
        "gb_preview": "Preview",
        "chk_exact": "Accurate preview (matches export)",
        "gb_export": "Export",
        "lbl_mode": "Mode:",
        "chk_lattice_full": "Lattice across entire window",
        "lbl_lattice_strength": "Lattice strength (1..2):",
        "gb_grid": "View window (mm) — 0.2mm; workspace 200×200mm",
        "lbl_grid_w": "Window width (mm):",
        "lbl_grid_h": "Window height (mm):",
        "chk_border": "Draw window border",
        "gb_transform": "Transforms (selected object)",
        "lbl_scale": "Scale:",
        "lbl_rx": "Tilt (x axis):",
        "lbl_ry": "Tilt (y axis):",
        "lbl_rz": "Rotate:",
        "lbl_tx": "Move X:",
        "lbl_ty": "Move Y:",
        "export_items": [
            "No lattice (fast)",
            "Fixed-height lattice (fast)",
            "Variable-height lattice (slower)",
        ],
        "no_data": "No data",
"overlap_title": "Warning: objects overlap",
"overlap_msg": "Some objects overlap in the XY projection.\nThey may intersect in the exported STL.\n\nOverlapping pairs:",
"overlap_continue": "Continue export?",
    },
}

def tr(lang: str, key: str):
    d = I18N.get(lang) or I18N["pl"]
    return d.get(key, I18N["pl"].get(key, key))


def set_half_screen_geometry(win, *, fraction: float = 0.5, min_size=(900, 650), center=True):
    """
    Ustawia okno na ułamek dostępnego obszaru ekranu (DPI-aware, działa na 4K).
    """
    screen = win.screen() or QApplication.primaryScreen()
    geo = screen.availableGeometry()

    w = max(int(geo.width() * fraction), int(min_size[0]))
    h = max(int(geo.height() * fraction), int(min_size[1]))

    win.resize(w, h)
    if center:
        win.move(
            geo.x() + (geo.width() - w) // 2,
            geo.y() + (geo.height() - h) // 2
        )


# =========================
# Rasteryzacja 2D
# =========================

def _point_in_tri(px: np.ndarray, py: np.ndarray,
                  ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> np.ndarray:
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay

    den = v0x * v1y - v1x * v0y
    if abs(den) < 1e-12:
        return np.zeros_like(px, dtype=bool)

    inv_den = 1.0 / den
    u = (v2x * v1y - v1x * v2y) * inv_den
    v = (v0x * v2y - v2x * v0y) * inv_den
    eps = 1e-9
    return (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps)


def rasterize_triangles_xy_mm_to_work_mask(tris_xy_mm: np.ndarray) -> np.ndarray:
    grid = np.zeros((WORK_H, WORK_W), dtype=bool)

    origin_x = -WORK_MM / 2.0
    origin_y = -WORK_MM / 2.0
    cell = CELL_MM

    for t in tris_xy_mm:
        (ax, ay), (bx, by), (cx, cy) = t

        minx = min(ax, bx, cx)
        maxx = max(ax, bx, cx)
        miny = min(ay, by, cy)
        maxy = max(ay, by, cy)

        ix0 = int(np.floor((minx - origin_x) / cell - 0.5))
        ix1 = int(np.ceil ((maxx - origin_x) / cell - 0.5))
        iy0 = int(np.floor((miny - origin_y) / cell - 0.5))
        iy1 = int(np.ceil ((maxy - origin_y) / cell - 0.5))

        ix0 = max(ix0, 0); iy0 = max(iy0, 0)
        ix1 = min(ix1, WORK_W - 1); iy1 = min(iy1, WORK_H - 1)
        if ix0 > ix1 or iy0 > iy1:
            continue

        xs = origin_x + (np.arange(ix0, ix1 + 1) + 0.5) * cell
        ys = origin_y + (np.arange(iy0, iy1 + 1) + 0.5) * cell
        X, Y = np.meshgrid(xs, ys)

        inside = _point_in_tri(X, Y, ax, ay, bx, by, cx, cy)
        grid[iy0:iy1 + 1, ix0:ix1 + 1] |= inside

    return grid


def mask_bbox_cells(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# =========================
# Transformacje 3D
# =========================

def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)

def rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)

# Ekranowe Z: dodatni obrót zgodnie z tym, co widzisz na podglądzie 2D
def rot_z_screen(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c,  s, 0],
                     [-s,  c, 0],
                     [ 0,  0, 1]], dtype=np.float64)


# =========================
# Operacje na masce
# =========================

def rotate_mask_nn_precomputed(mask: np.ndarray,
                               src_xi: np.ndarray, src_yi: np.ndarray,
                               inside: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    out[inside] = mask[src_yi[inside], src_xi[inside]]
    return out

def shift_mask(mask: np.ndarray, dx_cells: int, dy_cells: int) -> np.ndarray:
    if dx_cells == 0 and dy_cells == 0:
        return mask
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    x0_src = max(0, -dx_cells)
    x1_src = min(w, w - dx_cells)
    y0_src = max(0, -dy_cells)
    y1_src = min(h, h - dy_cells)

    x0_dst = max(0, dx_cells)
    y0_dst = max(0, dy_cells)

    if x1_src > x0_src and y1_src > y0_src:
        out[y0_dst:y0_dst + (y1_src - y0_src),
            x0_dst:x0_dst + (x1_src - x0_src)] = mask[y0_src:y1_src, x0_src:x1_src]
    return out

def crop_center(mask: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = mask.shape
    cx = w // 2
    cy = h // 2

    x0 = cx - out_w // 2
    y0 = cy - out_h // 2
    x1 = x0 + out_w
    y1 = y0 + out_h

    out = np.zeros((out_h, out_w), dtype=bool)

    sx0 = max(0, x0); sy0 = max(0, y0)
    sx1 = min(w, x1); sy1 = min(h, y1)

    dx0 = sx0 - x0
    dy0 = sy0 - y0
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    if sx1 > sx0 and sy1 > sy0:
        out[dy0:dy1, dx0:dx1] = mask[sy0:sy1, sx0:sx1]
    return out


# prosta morfologia (halo/obrys) bez scipy
def dilate4(mask: np.ndarray, r: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(1, r)):
        out = out | shift_mask(out, 1, 0) | shift_mask(out, -1, 0) | shift_mask(out, 0, 1) | shift_mask(out, 0, -1)
    return out

def erode4(mask: np.ndarray, r: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(1, r)):
        out = out & shift_mask(out, 1, 0) & shift_mask(out, -1, 0) & shift_mask(out, 0, 1) & shift_mask(out, 0, -1)
    return out

def outline_from_mask(mask: np.ndarray) -> np.ndarray:
    return mask & (~erode4(mask, 1))

def halo_from_mask(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    return dilate4(mask, thickness) & (~mask)


# =========================
# Kratownica 3D (jak przy eksporcie)
# =========================

def add_box_triangles(tris: List[np.ndarray], x0: float, x1: float, y0: float, y1: float, z0: float, z1: float):
    v000 = np.array([x0, y0, z0], dtype=np.float64)
    v100 = np.array([x1, y0, z0], dtype=np.float64)
    v110 = np.array([x1, y1, z0], dtype=np.float64)
    v010 = np.array([x0, y1, z0], dtype=np.float64)

    v001 = np.array([x0, y0, z1], dtype=np.float64)
    v101 = np.array([x1, y0, z1], dtype=np.float64)
    v111 = np.array([x1, y1, z1], dtype=np.float64)
    v011 = np.array([x0, y1, z1], dtype=np.float64)

    tris.append(np.stack([v000, v110, v100])); tris.append(np.stack([v000, v010, v110]))
    tris.append(np.stack([v001, v101, v111])); tris.append(np.stack([v001, v111, v011]))
    tris.append(np.stack([v000, v100, v101])); tris.append(np.stack([v000, v101, v001]))
    tris.append(np.stack([v010, v111, v110])); tris.append(np.stack([v010, v011, v111]))
    tris.append(np.stack([v000, v001, v011])); tris.append(np.stack([v000, v011, v010]))
    tris.append(np.stack([v100, v110, v111])); tris.append(np.stack([v100, v111, v101]))


def add_tapered_rib_box(tris: List[np.ndarray],
                        x0: float, x1: float, y0: float, y1: float,
                        total_h: float,
                        top_h: float = LATTICE_TAPER_TOP_H_MM) -> None:
    total_h = float(total_h)
    if total_h <= 1e-9:
        return

    top_h = float(min(top_h, total_h))
    z_mid = total_h - top_h

    if z_mid > 1e-9:
        add_box_triangles(tris, x0, x1, y0, y1, 0.0, z_mid)

    w = abs(x1 - x0)
    h = abs(y1 - y0)
    if w <= 1e-9 or h <= 1e-9:
        return

    if w <= h:
        cx = (x0 + x1) / 2.0
        rib_w = min(w, h)
        top_w = min(rib_w, LATTICE_TAPER_TOP_W_MM)  # nigdy nie cieńsze niż 0.5mm, ale też nie szersze niż żebro
        half = top_w / 2.0
        tx0, tx1 = cx - half, cx + half
        ty0, ty1 = y0, y1
    else:
        cy = (y0 + y1) / 2.0
        # szerokość żebra (w lub h) – dla pionowego żebra rib_w = w, dla poziomego rib_w = h
        rib_w = min(w, h)
        top_w = min(rib_w, LATTICE_TAPER_TOP_W_MM)  # nigdy nie cieńsze niż 0.5mm, ale też nie szersze niż żebro
        half = top_w / 2.0
        ty0, ty1 = cy - half, cy + half
        tx0, tx1 = x0, x1

    add_box_triangles(tris, tx0, tx1, ty0, ty1, z_mid, total_h)


def underside_z_at_xy_candidates(tris: np.ndarray, cand_idx: List[int], x: float, y: float) -> Optional[float]:
    best = None
    for i in cand_idx:
        t = tris[i]
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = t
        if x < min(ax, bx, cx) or x > max(ax, bx, cx) or y < min(ay, by, cy) or y > max(ay, by, cy):
            continue

        v0x, v0y = cx - ax, cy - ay
        v1x, v1y = bx - ax, by - ay
        v2x, v2y = x - ax, y - ay
        den = v0x * v1y - v1x * v0y
        if abs(den) < 1e-12:
            continue
        inv = 1.0 / den
        u = (v2x * v1y - v1x * v2y) * inv
        v = (v0x * v2y - v2x * v0y) * inv
        eps = 1e-9
        if u >= -eps and v >= -eps and (u + v) <= 1.0 + eps:
            z = az + u * (cz - az) + v * (bz - az)
            if z >= -1e-6:
                if best is None or z < best:
                    best = z
    return best


def build_triangle_bins_xy(tris: np.ndarray, bin_size: float) -> Tuple[Dict[Tuple[int, int], List[int]], float, float]:
    pts = tris.reshape(-1, 3)
    xmin = float(pts[:, 0].min()); ymin = float(pts[:, 1].min())
    origin_x = np.floor(xmin / bin_size) * bin_size
    origin_y = np.floor(ymin / bin_size) * bin_size

    bins: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, t in enumerate(tris):
        xs = t[:, 0]; ys = t[:, 1]
        x0 = float(xs.min()); x1 = float(xs.max())
        y0 = float(ys.min()); y1 = float(ys.max())

        ix0 = int(np.floor((x0 - origin_x) / bin_size))
        ix1 = int(np.floor((x1 - origin_x) / bin_size))
        iy0 = int(np.floor((y0 - origin_y) / bin_size))
        iy1 = int(np.floor((y1 - origin_y) / bin_size))

        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                bins[(ix, iy)].append(i)

    return bins, origin_x, origin_y


def bins_candidates(bins: Dict[Tuple[int, int], List[int]], origin_x: float, origin_y: float,
                    bin_size: float, x: float, y: float, radius: int = 0) -> List[int]:
    ix = int(np.floor((x - origin_x) / bin_size))
    iy = int(np.floor((y - origin_y) / bin_size))
    out: List[int] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            out.extend(bins.get((ix + dx, iy + dy), []))
    return out


# =========================
# Model
# =========================

def compute_bbox_center(points: np.ndarray) -> np.ndarray:
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return (mn + mx) / 2.0

def mesh_signature(path: str, mesh: Optional[stlmesh.Mesh] = None) -> dict:
    sig = {}
    try:
        st = os.stat(path)
        sig["size"] = int(st.st_size)
        sig["mtime"] = float(st.st_mtime)
    except Exception:
        sig["size"] = None
        sig["mtime"] = None

    if mesh is not None:
        try:
            vec = mesh.vectors.astype(np.float64)
            pts = vec.reshape(-1, 3)
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            sig["triangles"] = int(vec.shape[0])
            sig["bbox_min"] = [float(mn[0]), float(mn[1]), float(mn[2])]
            sig["bbox_max"] = [float(mx[0]), float(mx[1]), float(mx[2])]
        except Exception:
            pass
    return sig

def signature_diff(a: dict, b: dict) -> bool:
    keys = ["size", "triangles"]
    for k in keys:
        if (k in a) and (k in b) and (a.get(k) is not None) and (b.get(k) is not None):
            if a.get(k) != b.get(k):
                return True
    for k in ["bbox_min", "bbox_max"]:
        if a.get(k) and b.get(k):
            av = np.array(a[k], dtype=np.float64)
            bv = np.array(b[k], dtype=np.float64)
            if np.linalg.norm(av - bv) > 0.5:
                return True
    return False

@dataclass
class StlObject:
    path: str
    name: str
    mesh: stlmesh.Mesh
    pivot: np.ndarray
    color: Tuple[int, int, int] = (255, 255, 255)

    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    s: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    mask_fast: Optional[np.ndarray] = None
    mask_exact: Optional[np.ndarray] = None

    fast_dirty: bool = True
    exact_dirty: bool = True

    fast_pending: bool = False
    exact_pending: bool = False

    fast_ver: int = 0
    exact_ver: int = 0

    pending_auto: bool = False


# =========================
# Worker
# =========================

class WorkerSignals(QObject):
    done_fast = pyqtSignal(int, int, object)
    done_exact = pyqtSignal(int, int, object)
    error = pyqtSignal(int, str)

class ProjectionWorker(QRunnable):
    def __init__(self, obj_index: int, obj: StlObject, ver: int, mode: str):
        super().__init__()
        self.obj_index = obj_index
        self.obj = obj
        self.ver = ver
        self.mode = mode
        self.signals = WorkerSignals()

    def run(self):
        try:
            vecs = self.obj.mesh.vectors.astype(np.float64)
            P = self.obj.pivot.astype(np.float64)
            centered = vecs - P[None, None, :]

            if self.mode == "fast":
                A = (rot_y(self.obj.ry) @ rot_x(self.obj.rx)) * float(self.obj.s)
                pts = centered.reshape(-1, 3) @ A.T
                tri = pts.reshape(centered.shape)
                mask = rasterize_triangles_xy_mm_to_work_mask(tri[:, :, :2])
                self.signals.done_fast.emit(self.obj_index, self.ver, mask)
            else:
                R = rot_z_screen(self.obj.rz) @ rot_y(self.obj.ry) @ rot_x(self.obj.rx)
                A = R * float(self.obj.s)
                pts = centered.reshape(-1, 3) @ A.T
                tri = pts.reshape(centered.shape)
                tri[:, :, 0] += float(self.obj.tx)
                tri[:, :, 1] -= float(self.obj.ty)
                mask = rasterize_triangles_xy_mm_to_work_mask(tri[:, :, :2])
                self.signals.done_exact.emit(self.obj_index, self.ver, mask)

        except Exception as e:
            self.signals.error.emit(self.obj_index, str(e))


# =========================
# Podgląd z myszą
# =========================

class PreviewLabel(QLabel):
    clicked = pyqtSignal(int, int)
    dragged = pyqtSignal(int, int, int, int)
    wheeled = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dragging = False
        self._last = QPoint()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._dragging = True
            self._last = ev.pos()
            self.clicked.emit(ev.x(), ev.y())

    def mouseMoveEvent(self, ev):
        if self._dragging:
            p = ev.pos()
            self.dragged.emit(self._last.x(), self._last.y(), p.x(), p.y())
            self._last = p

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._dragging = False

    def wheelEvent(self, ev):
        if self._dragging:
            self.wheeled.emit(int(ev.angleDelta().y()))
        ev.accept()


# =========================
# Podgląd 3D (matplotlib) — okno responsywne + większe
# =========================

class Scene3DWindow(QMainWindow):
    def _install_fullscreen_toggle(self):
        # F11: przełącz pełny ekran (tak samo jak w oknie głównym)
        self._fs_prev_geom = None
        act = QAction("Pełny ekran", self)
        act.setShortcut(Qt.Key_F11)
        act.triggered.connect(self.toggle_fullscreen)
        self.addAction(act)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            if self._fs_prev_geom is not None:
                self.setGeometry(self._fs_prev_geom)
        else:
            self._fs_prev_geom = self.geometry()
            self.showFullScreen()

    def __init__(self, parent, triangles: np.ndarray, colors: List[Tuple[int, int, int]], title: str):
        super().__init__(parent)
        self._install_fullscreen_toggle()
        self.setWindowTitle(title)

        try:
            import matplotlib
            matplotlib.use("Qt5Agg")  # noqa
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            from matplotlib.figure import Figure
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except Exception as e:
            QMessageBox.critical(parent, "Błąd", f"Nie udało się uruchomić podglądu 3D (matplotlib):\n{e}")
            self.close()
            return

        # Central widget
        root = QWidget(self)
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # DPI daje ostrość; figsize NIE steruje rozmiarem okna Qt
        self.fig = Figure(dpi=200)
        self.fig.patch.set_facecolor((1, 1, 1, 1))

        self.canvas = FigureCanvas(self.fig)

        # KLUCZ: pozwól canvasowi rosnąć wraz z oknem
        from PyQt5.QtWidgets import QSizePolicy
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.toolbar = NavigationToolbar(self.canvas, self)

        lay.addWidget(self.toolbar, 0)
        lay.addWidget(self.canvas, 1)

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor((1, 1, 1, 1))
        self.ax.grid(False)

        # Decymacja dla szybkości
        T = triangles
        if T.shape[0] > 80000:
            step = int(np.ceil(T.shape[0] / 80000))
            T = T[::step]

        # Kolekcja trójkątów – kontrastowo
        poly = Poly3DCollection(T, linewidths=0.25, alpha=0.98, antialiased=True)
        poly.set_edgecolor((0.10, 0.10, 0.10, 0.70))
        poly.set_facecolor((0.82, 0.82, 0.86, 1.0))
        try:
            poly.set_shade(True)
        except Exception:
            pass
        self.ax.add_collection3d(poly)

        pts = T.reshape(-1, 3)
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        center = (mn + mx) / 2.0
        span = float(np.max(mx - mn))
        if span < 1e-6:
            span = 1.0

        self.ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        self.ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        self.ax.set_zlim(max(0.0, center[2] - span / 2), center[2] + span / 2)

        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")
        self.ax.view_init(elev=28, azim=-55)

        # Wypełnij obszar
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

        # Ustaw “sensowny” startowy rozmiar okna i minimalny rozmiar
        #self.resize(1400, 1000)
        set_half_screen_geometry(self, fraction=0.5, min_size=(1100, 750), center=True)

        self.setMinimumSize(900, 650)

        self.canvas.draw()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Przy resize Matplotlib sam się dopasuje do canvasa; draw_idle jest lekkie
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()


# =========================
# GUI
# =========================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._install_fullscreen_toggle()
        self.base_title = "Planer Wydruku 3D"
        self.setWindowTitle(self.base_title)

        self.objects: List[StlObject] = []
        self._updating_controls = False
        self.pool = QThreadPool.globalInstance()
        self._precompute_rotation_grids_for_work()
        self._last_render_context = None

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.recompute_preview)

        self._settings_save_timer = QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.timeout.connect(self.save_config)

        cfg = self.load_config()
        self.language = str(cfg.get("language", "pl")).lower() if isinstance(cfg, dict) else "pl"
        if self.language not in ("pl", "en"):
            self.language = "pl"

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        # Left
        left = QVBoxLayout()
        main.addLayout(left, 0)

        btn_row = QHBoxLayout()
        left.addLayout(btn_row)

        self.btn_add = QPushButton("Wczytaj obiekt (stl)…")
        self.btn_add.clicked.connect(self.load_stl)
        btn_row.addWidget(self.btn_add)

        self.btn_copy = QPushButton("Kopiuj")
        self.btn_copy.clicked.connect(self.copy_selected)
        btn_row.addWidget(self.btn_copy)

        self.btn_auto = QPushButton("Uplasuj")
        self.btn_auto.clicked.connect(self.auto_place_selected)
        btn_row.addWidget(self.btn_auto)

        self.btn_arrange = QPushButton("Rozmieść")
        self.btn_arrange.clicked.connect(self.arrange_all)
        btn_row.addWidget(self.btn_arrange)

        self.btn_del = QPushButton("Usuń")
        self.btn_del.clicked.connect(self.delete_selected)
        btn_row.addWidget(self.btn_del)

        state_row = QHBoxLayout()
        left.addLayout(state_row)

        self.btn_save_state = QPushButton("Zapisz układ…")
        self.btn_save_state.clicked.connect(self.save_state)
        state_row.addWidget(self.btn_save_state)

        self.btn_load_state = QPushButton("Wczytaj układ…")
        self.btn_load_state.clicked.connect(self.load_state)
        state_row.addWidget(self.btn_load_state)

        # Language toggle
        self.btn_lang = QPushButton("English")
        self.btn_lang.clicked.connect(self.toggle_language)
        state_row.addWidget(self.btn_lang)

        self.listw = QListWidget()
        self.listw.currentRowChanged.connect(self.on_select_object)
        self.listw.setContextMenuPolicy(Qt.CustomContextMenu)
        self.listw.customContextMenuRequested.connect(self.on_list_context_menu)
        left.addWidget(self.listw, 1)

        self.gb_preview = QGroupBox("Podgląd")
        self.prev_form = QFormLayout(self.gb_preview)
        self.chk_exact_preview = QCheckBox("Podgląd dokładny (zgodny z eksportem)")
        self.chk_exact_preview.setChecked(False)
        self.chk_exact_preview.stateChanged.connect(self.on_preview_mode_changed)
        self.prev_form.addRow("", self.chk_exact_preview)
        left.addWidget(self.gb_preview)

        # Export options
        self.gb_export = QGroupBox("Eksport")
        self.exp_form = QFormLayout(self.gb_export)

        self.cmb_export = QComboBox()
        self.cmb_export.addItems([
            "Bez kratownicy (szybko)",
            "Kratownica stałej wysokości (szybko)",
            "Kratownica zmiennej wysokości (wolniej)",
        ])
        self.cmb_export.setCurrentIndex(1)
        self.cmb_export.currentIndexChanged.connect(lambda: self.schedule_preview(40))

        self.chk_lattice_full = QCheckBox("Kratownica na całej siatce (oknie)")
        self.chk_lattice_full.setChecked(True)
        self.chk_lattice_full.stateChanged.connect(lambda: self.schedule_preview(40))

        self.lattice_factor = QDoubleSpinBox()
        self.lattice_factor.setRange(1.0, 2.0)
        self.lattice_factor.setSingleStep(0.1)
        self.lattice_factor.setValue(float(cfg.get("lattice_factor", 1.0)))
        self.lattice_factor.valueChanged.connect(lambda: (self.schedule_settings_save(), self.schedule_preview(40)))

        self.exp_form.addRow("Tryb:", self.cmb_export)
        self.exp_form.addRow("", self.chk_lattice_full)
        self.exp_form.addRow("Moc kratownicy (1..2):", self.lattice_factor)
        left.addWidget(self.gb_export)

        self.btn_save = QPushButton("Eksportuj scenę (stl)…")
        self.btn_save.clicked.connect(self.save_stl)
        left.addWidget(self.btn_save)

        self.btn_preview3d = QPushButton("Podgląd 3D (z uproszczoną kratownicą)")
        self.btn_preview3d.clicked.connect(self.open_preview_3d)
        left.addWidget(self.btn_preview3d)

        # Right
        right = QVBoxLayout()
        main.addLayout(right, 1)

        self.preview = PreviewLabel("Brak danych")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(760, 380)
        self.preview.setStyleSheet("QLabel { background: #111; color: #ddd; }")
        self.preview.clicked.connect(self.on_preview_clicked)
        self.preview.dragged.connect(self.on_preview_dragged)
        self.preview.wheeled.connect(self.on_preview_wheeled)
        self.preview.installEventFilter(self)  # resize -> refresh
        right.addWidget(self.preview, 1)

        self.gb_grid = QGroupBox("Okno wyświetlania (mm) — 0.2mm; robocze 200×200mm")
        self.grid_form = QFormLayout(self.gb_grid)

        self.grid_w_mm = QDoubleSpinBox()
        self.grid_w_mm.setRange(1.0, 500.0)
        self.grid_w_mm.setValue(float(cfg.get("grid_w_mm", 153.0)))

        self.grid_h_mm = QDoubleSpinBox()
        self.grid_h_mm.setRange(1.0, 500.0)
        self.grid_h_mm.setValue(float(cfg.get("grid_h_mm", 77.0)))

        self.show_grid_border = QCheckBox("Rysuj ramkę okna")
        self.show_grid_border.setChecked(True)

        self.grid_w_mm.valueChanged.connect(lambda: (self.schedule_preview(80), self.schedule_settings_save()))
        self.grid_h_mm.valueChanged.connect(lambda: (self.schedule_preview(80), self.schedule_settings_save()))
        self.show_grid_border.stateChanged.connect(lambda: self.schedule_preview(10))  # szybciej

        self.grid_form.addRow("Szerokość okna (mm):", self.grid_w_mm)
        self.grid_form.addRow("Wysokość okna (mm):", self.grid_h_mm)
        self.grid_form.addRow("", self.show_grid_border)

        info = QLabel(f"WORK: {WORK_MM:.0f}×{WORK_MM:.0f}mm @ {CELL_MM}mm → {WORK_W}×{WORK_H}")
        info.setStyleSheet("color: #666;")
        self.grid_form.addRow(info)
        right.addWidget(self.gb_grid, 0)

        # Transform controls
        self.gb_transform = QGroupBox("Transformacje (wybrany obiekt)")
        self.tr_form = QFormLayout(self.gb_transform)

        def slider_with_spin_percent(minv, maxv, init, cb):
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)

            sld = QSlider(Qt.Horizontal)
            sld.setRange(minv, maxv)
            sld.setValue(init)

            spn = QDoubleSpinBox()
            spn.setRange(minv, maxv)
            spn.setDecimals(1)
            spn.setSingleStep(1.0)
            spn.setValue(float(init))
            spn.setSuffix(" %")
            spn.setMinimumWidth(110)

            def on_sld(val):
                if self._updating_controls:
                    return
                self._updating_controls = True
                try:
                    spn.setValue(float(val))
                finally:
                    self._updating_controls = False
                cb(int(round(val)))

            def on_spn(valf):
                if self._updating_controls:
                    return
                v = int(round(float(valf)))
                self._updating_controls = True
                try:
                    sld.setValue(v)
                finally:
                    self._updating_controls = False
                cb(v)

            sld.valueChanged.connect(on_sld)
            spn.valueChanged.connect(on_spn)

            lay.addWidget(sld, 1)
            lay.addWidget(spn, 0)
            return w, sld, spn

        def slider_row(minv, maxv, init, suffix, cb):
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            sld = QSlider(Qt.Horizontal)
            sld.setRange(minv, maxv)
            sld.setValue(init)
            lbl = QLabel(f"{init}{suffix}")
            lbl.setMinimumWidth(95)
            lay.addWidget(sld, 1)
            lay.addWidget(lbl, 0)

            def on_change(val):
                lbl.setText(f"{val}{suffix}")
                cb(val)

            sld.valueChanged.connect(on_change)
            return w, sld

        self.w_scale, self.s_scale, self.scale_spin = slider_with_spin_percent(1, 400, 100, self.on_scale)
        self.w_rx, self.s_rx = slider_row(-180, 180, 0, "°", self.on_rx)
        self.w_ry, self.s_ry = slider_row(-180, 180, 0, "°", self.on_ry)
        self.w_rz, self.s_rz = slider_row(-180, 180, 0, "°", self.on_rz)
        self.w_tx, self.s_tx = slider_row(-250, 250, 0, " mm", self.on_tx)
        self.w_ty, self.s_ty = slider_row(-250, 250, 0, " mm", self.on_ty)

        self.tr_form.addRow("Skala:", self.w_scale)
        self.tr_form.addRow("Obrót w pionie (oś x):", self.w_rx)
        self.tr_form.addRow("Obrót w pionie (oś y):", self.w_ry)
        self.tr_form.addRow("Obrót:", self.w_rz)
        self.tr_form.addRow("Przesuń X:", self.w_tx)
        self.tr_form.addRow("Przesuń Y:", self.w_ty)

        right.addWidget(self.gb_transform, 0)
        self.enable_transform_controls(False)

        # apply language (after UI creation)
        self.apply_language()

        self.schedule_preview(20)
    def _install_fullscreen_toggle(self):
        self._fs_prev_geom = None
        act = QAction("Pełny ekran", self)
        act.setShortcut(Qt.Key_F11)
        act.triggered.connect(self.toggle_fullscreen)
        self.addAction(act)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            if self._fs_prev_geom is not None:
                self.setGeometry(self._fs_prev_geom)
        else:
            self._fs_prev_geom = self.geometry()
            self.showFullScreen()


    # ---------- refresh on resize ----------
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # po zmianie rozmiaru całego okna, przeliczamy skalowanie/proporcje podglądu
        self.schedule_preview(0)

    def eventFilter(self, obj, event):
        if obj is self.preview and event.type() == QEvent.Resize:
            self.schedule_preview(0)
        return super().eventFilter(obj, event)

    # ---------- config ----------
    def load_config(self) -> dict:
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def schedule_settings_save(self):
        self._settings_save_timer.start(200)

    def save_config(self):
        cfg = {
            "grid_w_mm": float(self.grid_w_mm.value()),
            "grid_h_mm": float(self.grid_h_mm.value()),
            "lattice_factor": float(self.lattice_factor.value()),
            "language": self.language,
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    
    # ---------- language ----------
    def toggle_language(self):
        self.language = "en" if self.language == "pl" else "pl"
        self.apply_language()
        self.save_config()
        # podgląd potrafi zależeć od tekstów (np. 'Brak danych')
        self.schedule_preview(0)

    def apply_language(self):
        lang = self.language
        # tytuł
        self.base_title = tr(lang, "app_title")
        self._update_title_progress()

        # przyciski
        self.btn_add.setText(tr(lang, "btn_add"))
        self.btn_copy.setText(tr(lang, "btn_copy"))
        self.btn_auto.setText(tr(lang, "btn_auto"))
        self.btn_arrange.setText(tr(lang, "btn_arrange"))
        self.btn_del.setText(tr(lang, "btn_del"))
        self.btn_save_state.setText(tr(lang, "btn_save_layout"))
        self.btn_load_state.setText(tr(lang, "btn_load_layout"))
        self.btn_save.setText(tr(lang, "btn_export"))
        self.btn_preview3d.setText(tr(lang, "btn_preview3d"))

        # przycisk języka pokazuje docelowy język
        if lang == "pl":
            self.btn_lang.setText(tr(lang, "btn_lang_to_en"))
        else:
            self.btn_lang.setText(tr(lang, "btn_lang_to_pl"))

        # groupboxy / checkboxy
        self.gb_preview.setTitle(tr(lang, "gb_preview"))
        self.chk_exact_preview.setText(tr(lang, "chk_exact"))

        self.gb_export.setTitle(tr(lang, "gb_export"))
        self.chk_lattice_full.setText(tr(lang, "chk_lattice_full"))

        self.gb_grid.setTitle(tr(lang, "gb_grid"))
        self.show_grid_border.setText(tr(lang, "chk_border"))

        self.gb_transform.setTitle(tr(lang, "gb_transform"))

        # QFormLayout etykiety (labelForField)
        def set_label(form, field_widget, text):
            lbl = form.labelForField(field_widget)
            if lbl is not None:
                lbl.setText(text)

        set_label(self.exp_form, self.cmb_export, tr(lang, "lbl_mode"))
        set_label(self.exp_form, self.lattice_factor, tr(lang, "lbl_lattice_strength"))

        set_label(self.grid_form, self.grid_w_mm, tr(lang, "lbl_grid_w"))
        set_label(self.grid_form, self.grid_h_mm, tr(lang, "lbl_grid_h"))

        set_label(self.tr_form, self.w_scale, tr(lang, "lbl_scale"))
        set_label(self.tr_form, self.w_rx, tr(lang, "lbl_rx"))
        set_label(self.tr_form, self.w_ry, tr(lang, "lbl_ry"))
        set_label(self.tr_form, self.w_rz, tr(lang, "lbl_rz"))
        set_label(self.tr_form, self.w_tx, tr(lang, "lbl_tx"))
        set_label(self.tr_form, self.w_ty, tr(lang, "lbl_ty"))

        # combo export items
        idx = self.cmb_export.currentIndex()
        self.cmb_export.blockSignals(True)
        try:
            self.cmb_export.clear()
            self.cmb_export.addItems(tr(lang, "export_items"))
            self.cmb_export.setCurrentIndex(min(idx, self.cmb_export.count() - 1))
        finally:
            self.cmb_export.blockSignals(False)

        # placeholder podglądu (gdy brak danych)
        if not self.objects and (self.preview.pixmap() is None or self.preview.pixmap().isNull()):
            self.preview.setText(tr(lang, "no_data"))

    def closeEvent(self, ev):
        self.save_config()
        super().closeEvent(ev)

    # ---------- rename menu ----------
    def on_list_context_menu(self, pos):
        idx = self.listw.indexAt(pos).row()
        if not (0 <= idx < len(self.objects)):
            return
        menu = QMenu(self.listw)
        act = QAction("Zmień nazwę…", self.listw)
        act.triggered.connect(lambda: self.rename_object(idx))
        menu.addAction(act)
        menu.exec_(self.listw.mapToGlobal(pos))

    def rename_object(self, idx: int):
        obj = self.objects[idx]
        new_name, ok = QInputDialog.getText(self, "Zmień nazwę", "Nowa nazwa obiektu:", text=obj.name)
        if not ok:
            return
        new_name = (new_name or "").strip()
        if not new_name:
            return
        obj.name = new_name
        item = self.listw.item(idx)
        if item is not None:
            item.setText(new_name)
        self.schedule_preview(10)

    # ---------- rot maps ----------
    def _precompute_rotation_grids_for_work(self):
        h, w = WORK_H, WORK_W
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        yy, xx = np.indices((h, w))
        self._work_rot_x = (xx - cx).astype(np.float32)
        self._work_rot_y = (yy - cy).astype(np.float32)
        self._work_cx = cx
        self._work_cy = cy

    def _rotation_maps_work(self, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.deg2rad(angle_deg)
        c, s = float(np.cos(a)), float(np.sin(a))
        x = self._work_rot_x
        y = self._work_rot_y
        cx, cy = self._work_cx, self._work_cy
        src_x =  c * x - s * y + cx
        src_y =  s * x + c * y + cy
        src_xi = np.rint(src_x).astype(np.int32)
        src_yi = np.rint(src_y).astype(np.int32)
        inside = (src_xi >= 0) & (src_xi < WORK_W) & (src_yi >= 0) & (src_yi < WORK_H)
        return src_xi, src_yi, inside

    @staticmethod
    def rotation_maps_for_shape(h: int, w: int, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.deg2rad(angle_deg)
        c, s = float(np.cos(a)), float(np.sin(a))
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        yy, xx = np.indices((h, w))
        x = (xx - cx).astype(np.float32)
        y = (yy - cy).astype(np.float32)
        src_x =  c * x - s * y + cx
        src_y =  s * x + c * y + cy
        src_xi = np.rint(src_x).astype(np.int32)
        src_yi = np.rint(src_y).astype(np.int32)
        inside = (src_xi >= 0) & (src_xi < w) & (src_yi >= 0) & (src_yi < h)
        return src_xi, src_yi, inside

    # ---------- helpers ----------
    def enable_transform_controls(self, enabled: bool):
        for w in (self.s_scale, self.scale_spin, self.s_rx, self.s_ry, self.s_rz, self.s_tx, self.s_ty,
                  self.btn_del, self.btn_copy, self.btn_auto, self.btn_arrange):
            w.setEnabled(enabled)

    def selected_index(self) -> int:
        return self.listw.currentRow()

    def selected_object(self) -> Optional[StlObject]:
        idx = self.selected_index()
        if 0 <= idx < len(self.objects):
            return self.objects[idx]
        return None

    def schedule_preview(self, delay_ms: int = 60):
        self._debounce_timer.start(delay_ms)

    def _current_window_mm(self) -> Tuple[float, float]:
        return float(self.grid_w_mm.value()), float(self.grid_h_mm.value())

    # =========================
    # Best Rz + scale (<=1.0) for NEW objects
    # =========================

    def _choose_best_rz_and_scale_to_window(self, obj: StlObject) -> None:
        out_w_mm, out_h_mm = self._current_window_mm()
        avail_w = max(1.0, out_w_mm - 2.0 * AUTO_FIT_MARGIN_MM)
        avail_h = max(1.0, out_h_mm - 2.0 * AUTO_FIT_MARGIN_MM)

        vecs = obj.mesh.vectors.reshape(-1, 3).astype(np.float64)
        centered = vecs - obj.pivot[None, :]

        A = (rot_y(obj.ry) @ rot_x(obj.rx))
        pts = centered @ A.T
        xy = pts[:, :2]

        best = None  # (s_fit, -abs(angle), angle)
        for ang in ORIENT_ANGLES:
            a = np.deg2rad(float(ang))
            c, s = float(np.cos(a)), float(np.sin(a))
            R2 = np.array([[c, s], [-s, c]], dtype=np.float64)
            xy_r = xy @ R2.T

            mn = xy_r.min(axis=0)
            mx = xy_r.max(axis=0)
            w0 = float(mx[0] - mn[0]); h0 = float(mx[1] - mn[1])
            w0 = max(w0, 1e-9); h0 = max(h0, 1e-9)

            s_fit = min(avail_w / w0, avail_h / h0)
            s_fit = min(1.0, s_fit)  # nie powiększaj
            cand = (s_fit, -abs(float(ang)), float(ang))
            if best is None or cand > best:
                best = cand

        if best is None:
            obj.s = max(0.01, min(1.0, obj.s))
            return

        s_fit, _, ang = best
        obj.rz = float(ang)
        obj.s = max(0.01, float(s_fit))

    # =========================
    # List ops
    # =========================

    def _add_object_to_list(self, obj: StlObject):
        item = QListWidgetItem(obj.name)
        item.setForeground(QBrush(QColor(*obj.color)))
        self.listw.addItem(item)

    def load_stl(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Wybierz pliki STL", "", "STL (*.stl);;Wszystkie (*.*)")
        if not paths:
            return

        for p in paths:
            try:
                m = stlmesh.Mesh.from_file(p)
            except Exception as e:
                QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać:\n{p}\n\n{e}")
                continue

            name = os.path.splitext(os.path.basename(p))[0]
            pts = m.vectors.reshape(-1, 3).astype(np.float64)
            pivot = compute_bbox_center(pts)

            color = COLOR_PALETTE[len(self.objects) % len(COLOR_PALETTE)]
            obj = StlObject(path=p, name=name, mesh=m, pivot=pivot, color=color)

            self._choose_best_rz_and_scale_to_window(obj)

            obj.fast_ver += 1
            obj.fast_dirty = True
            obj.mask_fast = None

            obj.exact_ver += 1
            obj.exact_dirty = True
            obj.mask_exact = None

            obj.pending_auto = True

            self.objects.append(obj)
            self._add_object_to_list(obj)
            self._kick_fast(len(self.objects) - 1)
            if self.chk_exact_preview.isChecked():
                self._kick_exact(len(self.objects) - 1)

        if self.objects:
            self.listw.setCurrentRow(len(self.objects) - 1)
        self.schedule_preview(20)


    def _next_numbered_copy_name(self, src_name: str) -> str:
        """Generate a nice copy name: Base (n). If src is already Base (k), next is Base (k+1)."""
        # Strip trailing ' (number)' if present
        m = re.match(r"^(.*) \((\d+)\)\s*$", src_name)
        base = m.group(1) if m else src_name

        # Find current max number for this base among existing objects
        max_n = 0
        for o in self.objects:
            m2 = re.match(rf"^{re.escape(base)} \((\d+)\)\s*$", o.name)
            if m2:
                max_n = max(max_n, int(m2.group(1)))
            elif o.name == base:
                max_n = max(max_n, 0)

        return f"{base} ({max_n + 1})"


    def copy_selected(self):
        src = self.selected_object()
        if src is None:
            return

        color = COLOR_PALETTE[len(self.objects) % len(COLOR_PALETTE)]
        obj = StlObject(
            path=src.path,
            name=self._next_numbered_copy_name(src.name),
            mesh=src.mesh,
            pivot=src.pivot.copy(),
            color=color,
            rx=src.rx, ry=src.ry, rz=src.rz,
            s=src.s,
            tx=src.tx + 10.0,
            ty=src.ty
        )

        obj.fast_ver += 1
        obj.fast_dirty = True
        obj.mask_fast = None

        obj.exact_ver += 1
        obj.exact_dirty = True
        obj.mask_exact = None

        obj.pending_auto = True

        self.objects.append(obj)
        self._add_object_to_list(obj)
        self._kick_fast(len(self.objects) - 1)
        if self.chk_exact_preview.isChecked():
            self._kick_exact(len(self.objects) - 1)

        self.listw.setCurrentRow(len(self.objects) - 1)
        self.schedule_preview(20)

    def delete_selected(self):
        idx = self.selected_index()
        if not (0 <= idx < len(self.objects)):
            return
        del self.objects[idx]
        self.listw.takeItem(idx)
        self.enable_transform_controls(bool(self.objects))
        self.schedule_preview(20)

    # =========================
    # Save/Load layout
    # =========================

    def save_state(self):
        if not self.objects:
            QMessageBox.information(self, "Układ", "Brak obiektów do zapisania.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Zapisz układ", "uklad.json", "JSON (*.json)")
        if not path:
            return

        data = {
            "version": 2,
            "grid_w_mm": float(self.grid_w_mm.value()),
            "grid_h_mm": float(self.grid_h_mm.value()),
            "lattice_factor": float(self.lattice_factor.value()),
            "objects": []
        }

        for o in self.objects:
            sig = mesh_signature(o.path, o.mesh)
            data["objects"].append({
                "path": o.path,
                "name": o.name,
                "color": list(o.color),
                "rx": float(o.rx), "ry": float(o.ry), "rz": float(o.rz),
                "s": float(o.s),
                "tx": float(o.tx), "ty": float(o.ty),
                "file_sig": sig
            })

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się zapisać układu:\n{e}")
            return

        QMessageBox.information(self, "OK", f"Zapisano układ:\n{path}")

    def load_state(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wczytaj układ", "", "JSON (*.json);;Wszystkie (*.*)")
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać układu:\n{e}")
            return

        if "grid_w_mm" in data:
            self.grid_w_mm.setValue(float(data["grid_w_mm"]))
        if "grid_h_mm" in data:
            self.grid_h_mm.setValue(float(data["grid_h_mm"]))
        if "lattice_factor" in data:
            self.lattice_factor.setValue(float(data["lattice_factor"]))

        self.objects.clear()
        self.listw.clear()

        missing = []
        changed = []

        for i, od in enumerate(data.get("objects", [])):
            p = od.get("path", "")
            if not p or not os.path.exists(p):
                missing.append(p)
                continue
            try:
                m = stlmesh.Mesh.from_file(p)
            except Exception:
                missing.append(p)
                continue

            saved_sig = od.get("file_sig", None)
            if saved_sig:
                now_sig = mesh_signature(p, m)
                if signature_diff(saved_sig, now_sig):
                    changed.append(p)

            pts = m.vectors.reshape(-1, 3).astype(np.float64)
            pivot = compute_bbox_center(pts)

            color = tuple(od.get("color", COLOR_PALETTE[i % len(COLOR_PALETTE)]))
            obj = StlObject(
                path=p,
                name=str(od.get("name", os.path.basename(p))),
                mesh=m,
                pivot=pivot,
                color=color,
                rx=float(od.get("rx", 0.0)),
                ry=float(od.get("ry", 0.0)),
                rz=float(od.get("rz", 0.0)),
                s=float(od.get("s", 1.0)),
                tx=float(od.get("tx", 0.0)),
                ty=float(od.get("ty", 0.0)),
            )

            obj.fast_ver += 1
            obj.fast_dirty = True
            obj.mask_fast = None

            obj.exact_ver += 1
            obj.exact_dirty = True
            obj.mask_exact = None

            obj.pending_auto = False  # odtwarzamy 1:1

            self.objects.append(obj)
            self._add_object_to_list(obj)
            self._kick_fast(len(self.objects) - 1)
            if self.chk_exact_preview.isChecked():
                self._kick_exact(len(self.objects) - 1)

        if self.objects:
            self.listw.setCurrentRow(0)
        self.schedule_preview(20)

        if missing:
            msg = "\n".join([m for m in missing if m])
            QMessageBox.warning(self, "Uwaga", f"Nie wczytano części plików (brak/błąd):\n{msg}")

        if changed:
            msg = "\n".join(changed[:20])
            more = "" if len(changed) <= 20 else f"\n… i {len(changed) - 20} więcej"
            QMessageBox.warning(
                self,
                "Uwaga: zmienione pliki STL",
                "Wykryto, że część plików STL zmieniła się od momentu zapisu układu.\n"
                "To często oznacza, że plik obiektu został nadpisany eksportem sceny (np. z kratownicą).\n\n"
                f"Zmienione:\n{msg}{more}"
            )

    # =========================
    # Selection / sliders
    # =========================

    def _sync_sliders_from_obj(self, obj: StlObject, include_scale: bool = True):
        self._updating_controls = True
        try:
            if include_scale:
                self.s_scale.setValue(int(round(obj.s * 100)))
                self.scale_spin.setValue(float(obj.s * 100.0))
            self.s_rx.setValue(int(round(obj.rx)))
            self.s_ry.setValue(int(round(obj.ry)))
            self.s_rz.setValue(int(round(obj.rz)))
            self.s_tx.setValue(int(round(obj.tx)))
            self.s_ty.setValue(int(round(obj.ty)))
        finally:
            self._updating_controls = False

    def on_select_object(self, row: int):
        obj = self.selected_object()
        self.enable_transform_controls(obj is not None)
        if obj is None:
            return
        self._sync_sliders_from_obj(obj, include_scale=True)
        self.schedule_preview(20)

    def _update_selected(self, *, dirty_fast: bool = False, dirty_exact: bool = False, **kwargs):
        if self._updating_controls:
            return
        obj = self.selected_object()
        if obj is None:
            return

        for k, v in kwargs.items():
            setattr(obj, k, v)

        if dirty_fast:
            obj.fast_ver += 1
            obj.fast_dirty = True
            obj.mask_fast = None
            self._kick_fast(self.selected_index())

            obj.exact_ver += 1
            obj.exact_dirty = True
            obj.mask_exact = None
            if self.chk_exact_preview.isChecked():
                self._kick_exact(self.selected_index())
        elif dirty_exact:
            obj.exact_ver += 1
            obj.exact_dirty = True
            obj.mask_exact = None
            if self.chk_exact_preview.isChecked():
                self._kick_exact(self.selected_index())

        self.schedule_preview(40)

    def on_scale(self, val_percent: int):
        self._update_selected(s=float(val_percent) / 100.0, dirty_fast=True)

    def on_rx(self, val: int):
        self._update_selected(rx=float(val), dirty_fast=True)

    def on_ry(self, val: int):
        self._update_selected(ry=float(val), dirty_fast=True)

    def on_rz(self, val: int):
        self._update_selected(rz=float(val), dirty_exact=True)

    def on_tx(self, val: int):
        self._update_selected(tx=float(val), dirty_exact=True)

    def on_ty(self, val: int):
        self._update_selected(ty=float(val), dirty_exact=True)

    # =========================
    # Preview mode / mouse
    # =========================

    def on_preview_mode_changed(self):
        if self.chk_exact_preview.isChecked():
            for i, o in enumerate(self.objects):
                o.exact_ver += 1
                o.exact_dirty = True
                o.mask_exact = None
                self._kick_exact(i)
        self.schedule_preview(20)

    def on_preview_clicked(self, x_pix: int, y_pix: int):
        ctx = self._last_render_context
        if ctx is None:
            return
        img_w, img_h, out_w, out_h, ox, oy = ctx
        xp = x_pix - ox
        yp = y_pix - oy
        if xp < 0 or yp < 0 or xp >= img_w or yp >= img_h:
            return
        ix = int(xp * out_w / img_w)
        iy = int(yp * out_h / img_h)

        exact = self.chk_exact_preview.isChecked()
        for idx in range(len(self.objects) - 1, -1, -1):
            o = self.objects[idx]
            m = self._object_view_mask(o, out_w, out_h, exact)
            if m[iy, ix]:
                self.listw.setCurrentRow(idx)
                return

    def on_preview_dragged(self, x0: int, y0: int, x1: int, y1: int):
        obj = self.selected_object()
        if obj is None:
            return
        ctx = self._last_render_context
        if ctx is None:
            return
        img_w, img_h, out_w, out_h, ox, oy = ctx

        dx_pix = x1 - x0
        dy_pix = y1 - y0

        dx_cells = dx_pix * out_w / max(1, img_w)
        dy_cells = dy_pix * out_h / max(1, img_h)

        obj.tx += float(dx_cells) * CELL_MM
        obj.ty -= float(dy_cells) * CELL_MM

        self._sync_sliders_from_obj(obj, include_scale=False)

        if self.chk_exact_preview.isChecked():
            obj.exact_ver += 1
            obj.exact_dirty = True
            obj.mask_exact = None
            self._kick_exact(self.selected_index())

        self.schedule_preview(10)

    def on_preview_wheeled(self, delta: int):
        obj = self.selected_object()
        if obj is None:
            return
        steps = int(np.sign(delta)) * max(1, abs(delta) // 120) if delta != 0 else 0
        if steps == 0:
            return

        obj.rz += steps * 5.0
        while obj.rz > 180.0:
            obj.rz -= 360.0
        while obj.rz < -180.0:
            obj.rz += 360.0

        self._sync_sliders_from_obj(obj, include_scale=False)

        if self.chk_exact_preview.isChecked():
            obj.exact_ver += 1
            obj.exact_dirty = True
            obj.mask_exact = None
            self._kick_exact(self.selected_index())

        self.schedule_preview(10)

    # =========================
    # Workers
    # =========================

    def _kick_fast(self, obj_index: int):
        if not (0 <= obj_index < len(self.objects)):
            return
        o = self.objects[obj_index]
        if o.fast_pending or (not o.fast_dirty and o.mask_fast is not None):
            return
        o.fast_pending = True
        ver = o.fast_ver
        w = ProjectionWorker(obj_index, o, ver, mode="fast")
        w.signals.done_fast.connect(self._on_done_fast)
        w.signals.error.connect(self._on_worker_error)
        self.pool.start(w)
        self._update_title_progress()

    def _kick_exact(self, obj_index: int):
        if not (0 <= obj_index < len(self.objects)):
            return
        o = self.objects[obj_index]
        if o.exact_pending or (not o.exact_dirty and o.mask_exact is not None):
            return
        if o.mask_fast is None and not o.fast_pending:
            self._kick_fast(obj_index)
        o.exact_pending = True
        ver = o.exact_ver
        w = ProjectionWorker(obj_index, o, ver, mode="exact")
        w.signals.done_exact.connect(self._on_done_exact)
        w.signals.error.connect(self._on_worker_error)
        self.pool.start(w)
        self._update_title_progress()

    def _on_done_fast(self, obj_index: int, ver: int, mask: np.ndarray):
        if 0 <= obj_index < len(self.objects):
            o = self.objects[obj_index]
            o.fast_pending = False
            if ver != o.fast_ver:
                o.fast_dirty = True
                self._kick_fast(obj_index)
            else:
                o.mask_fast = mask
                o.fast_dirty = False

                if o.pending_auto:
                    o.pending_auto = False
                    self.listw.setCurrentRow(obj_index)
                    QTimer.singleShot(0, self.auto_place_selected)

        self._update_title_progress()
        self.schedule_preview(20)

    def _on_done_exact(self, obj_index: int, ver: int, mask: np.ndarray):
        if 0 <= obj_index < len(self.objects):
            o = self.objects[obj_index]
            o.exact_pending = False
            if ver != o.exact_ver:
                o.exact_dirty = True
                self._kick_exact(obj_index)
            else:
                o.mask_exact = mask
                o.exact_dirty = False
        self._update_title_progress()
        self.schedule_preview(20)

    def _on_worker_error(self, obj_index: int, msg: str):
        if 0 <= obj_index < len(self.objects):
            o = self.objects[obj_index]
            o.fast_pending = False
            o.exact_pending = False
        self.setWindowTitle(f"{self.base_title} | błąd: {msg}")
        self._update_title_progress()
        self.schedule_preview(20)

    def _update_title_progress(self):
        total = len(self.objects)
        if total == 0:
            self.setWindowTitle(self.base_title)
            return
        running = sum(1 for o in self.objects if (o.fast_pending or o.exact_pending))
        if running > 0:
            self.setWindowTitle(f"{self.base_title} | liczenie: {running}/{total}")
        else:
            self.setWindowTitle(self.base_title)

    # =========================
    # Maski / render 2D + kratownica w podglądzie
    # =========================

    def _object_view_mask(self, obj: StlObject, out_w: int, out_h: int, exact: bool) -> np.ndarray:
        if exact:
            if obj.mask_exact is None:
                return np.zeros((out_h, out_w), dtype=bool)
            return crop_center(obj.mask_exact, out_w=out_w, out_h=out_h)

        if obj.mask_fast is None:
            return np.zeros((out_h, out_w), dtype=bool)

        m = obj.mask_fast
        if abs(obj.rz) > 1e-6:
            src_xi, src_yi, inside = self._rotation_maps_work(obj.rz)
            m = rotate_mask_nn_precomputed(m, src_xi, src_yi, inside)

        dx = int(round(obj.tx / CELL_MM))
        dy = -int(round(obj.ty / CELL_MM))
        m = shift_mask(m, dx, dy)

        return crop_center(m, out_w=out_w, out_h=out_h)

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        ys, xs = np.where(mask)
        if xs.size == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    def _make_lattice_overlay_mask(self, out_w: int, out_h: int, footprint: Optional[np.ndarray]) -> np.ndarray:
        """2D overlay kratownicy w oknie (pitch/rib)."""
        if self.cmb_export.currentIndex() == 0:
            return np.zeros((out_h, out_w), dtype=bool)

        factor = float(self.lattice_factor.value())
        rib_mm = LATTICE_RIB_MM * factor

        pitch_cells = max(1, int(round(LATTICE_PITCH_MM / CELL_MM)))
        rib_cells = max(1, int(round(rib_mm / CELL_MM)))
        half = max(1, rib_cells // 2)

        # obszar: albo cała siatka, albo bbox obiektów
        x0 = 0; y0 = 0; x1 = out_w - 1; y1 = out_h - 1
        if not self.chk_lattice_full.isChecked() and footprint is not None:
            bb = mask_bbox_cells(footprint)
            if bb is not None:
                x0, y0, x1, y1 = bb
                # rozszerz do siatki pitch (snap)
                x0 = (x0 // pitch_cells) * pitch_cells
                y0 = (y0 // pitch_cells) * pitch_cells
                x1 = min(out_w - 1, ((x1 + pitch_cells - 1) // pitch_cells) * pitch_cells)
                y1 = min(out_h - 1, ((y1 + pitch_cells - 1) // pitch_cells) * pitch_cells)

        mask = np.zeros((out_h, out_w), dtype=bool)

        # pionowe żebra
        for x in range(x0, x1 + 1, pitch_cells):
            xa = max(0, x - half)
            xb = min(out_w, x + half + 1)
            mask[y0:y1+1, xa:xb] = True

        # poziome żebra
        for y in range(y0, y1 + 1, pitch_cells):
            ya = max(0, y - half)
            yb = min(out_h, y + half + 1)
            mask[ya:yb, x0:x1+1] = True

        return mask

    def recompute_preview(self):
        if not self.objects:
            self.preview.setText("Brak danych")
            self.preview.setPixmap(QPixmap())
            self._last_render_context = None
            self._update_title_progress()
            return

        out_w_mm, out_h_mm = self._current_window_mm()
        out_w = int(max(1, round(out_w_mm / CELL_MM)))
        out_h = int(max(1, round(out_h_mm / CELL_MM)))

        for i, o in enumerate(self.objects):
            if o.mask_fast is None and (not o.fast_pending):
                self._kick_fast(i)
            if self.chk_exact_preview.isChecked() and o.mask_exact is None and (not o.exact_pending) and o.exact_dirty:
                self._kick_exact(i)

        self._update_title_progress()
        exact = self.chk_exact_preview.isChecked()
        active_idx = self.selected_index()

        img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        masks_view: List[np.ndarray] = []

        # obiekty
        for o in self.objects:
            m = self._object_view_mask(o, out_w, out_h, exact)
            masks_view.append(m)
            r, g, b = o.color
            img[m, 0] = r
            img[m, 1] = g
            img[m, 2] = b

        # kratownica (overlay)
        footprint = None
        if len(masks_view) > 0:
            footprint = np.zeros((out_h, out_w), dtype=bool)
            for m in masks_view:
                footprint |= m

        lattice_mask = self._make_lattice_overlay_mask(out_w, out_h, footprint)
        if lattice_mask.any():
            # delikatny jasnoszary (bez psucia kolorów obiektów)
            img[lattice_mask, 0] = np.maximum(img[lattice_mask, 0], 90)
            img[lattice_mask, 1] = np.maximum(img[lattice_mask, 1], 90)
            img[lattice_mask, 2] = np.maximum(img[lattice_mask, 2], 90)

        # halo + obrys aktywnego obiektu
        if 0 <= active_idx < len(self.objects):
            m_act = masks_view[active_idx]
            if m_act.any():
                halo = halo_from_mask(m_act, thickness=2)
                outl = outline_from_mask(m_act)

                img[halo, 0] = np.maximum(img[halo, 0], 200)
                img[halo, 1] = np.maximum(img[halo, 1], 200)
                img[halo, 2] = np.maximum(img[halo, 2], 200)

                img[outl, :] = 255

        if self.show_grid_border.isChecked():
            img[0, :, :] = 64
            img[-1, :, :] = 64
            img[:, 0, :] = 64
            img[:, -1, :] = 64

        qimg = QImage(img.data, out_w, out_h, 3 * out_w, QImage.Format_RGB888).copy()

        # podpisy (małe)
        painter = QPainter(qimg)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        base_font = QFont()
        base_font.setPointSize(5)
        bold_font = QFont(base_font)
        bold_font.setBold(True)

        for i, o in enumerate(self.objects):
            c = self._mask_centroid(masks_view[i])
            if c is None:
                continue
            cx, cy = c
            painter.setFont(bold_font if i == active_idx else base_font)

            text = o.name
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(int(cx) + 1, int(cy) + 1, text)
            painter.setPen(QColor(245, 245, 245))
            painter.drawText(int(cx), int(cy), text)

        painter.end()

        # skalowanie do widgetu (proporcje aktualne po resize)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.width(), self.preview.height(),
            Qt.KeepAspectRatio, Qt.FastTransformation
        )

        pw, ph = pix.width(), pix.height()
        ox = (self.preview.width() - pw) // 2
        oy = (self.preview.height() - ph) // 2
        self._last_render_context = (pw, ph, out_w, out_h, ox, oy)

        self.preview.setPixmap(pix)

    # =========================
    # Auto / Rozmieść
    # =========================

    def auto_place_selected(self):
        idx = self.selected_index()
        obj = self.selected_object()
        if obj is None:
            return

        out_w_mm, out_h_mm = self._current_window_mm()
        out_w = int(max(1, round(out_w_mm / CELL_MM)))
        out_h = int(max(1, round(out_h_mm / CELL_MM)))

        if obj.mask_fast is None:
            self._kick_fast(idx)
            return

        # Szybki test: czy obiekt w ogóle może się zmieścić w oknie (dla obrotów 0/90)?
        bb = mask_bbox_cells(obj.mask_fast)
        if bb is not None:
            x0b, y0b, x1b, y1b = bb
            w0 = x1b - x0b + 1
            h0 = y1b - y0b + 1
            w90, h90 = h0, w0
            fits0 = (w0 <= out_w) and (h0 <= out_h)
            fits90 = (w90 <= out_w) and (h90 <= out_h)
            if not (fits0 or fits90):
                # Nie ma sensu próbować – obiekt jest za duży dla tego okna
                self.setWindowTitle(f"{self.base_title} | Uplasuj: obiekt za duży")
                QTimer.singleShot(900, lambda: self.setWindowTitle(self.base_title))
                return

        old_title = self.windowTitle()
        self.setWindowTitle(f"{self.base_title} | Uplasuj: liczenie…")
        QApplication.processEvents()

        # zajętość przez inne obiekty (w trybie "fast" – wystarczające do uplasowania)
        others = np.zeros((out_h, out_w), dtype=bool)
        for j, o in enumerate(self.objects):
            if j == idx:
                continue
            others |= self._object_view_mask(o, out_w, out_h, exact=False)


        # Szybki test: jeśli wolnych pól jest mniej niż pole obiektu, nie ma sensu szukać.
        # (To jest warunek konieczny i bardzo szybki.)
        free_cells = int(out_w * out_h - others.sum())
        obj_area_cells = int(obj.mask_fast.sum())
        if obj_area_cells > free_cells:
            self.setWindowTitle(f"{self.base_title} | Uplasuj: brak miejsca (pole)")
            QApplication.processEvents()
            QTimer.singleShot(900, lambda: self.setWindowTitle(old_title))
            return

        step_cells = max(1, int(round(1.0 / CELL_MM)))  # 1mm kroku wyszukiwania
        max_r = max(out_w, out_h)

        # aktualne przesunięcie obiektu w komórkach (żeby testować obroty "w miejscu")
        dx0 = int(round(obj.tx / CELL_MM))
        dy0 = -int(round(obj.ty / CELL_MM))

        def spiral_find(mask_view: np.ndarray) -> Optional[Tuple[int, int]]:
            area0 = int(mask_view.sum())
            if area0 == 0:
                return None

            def ok_at(dx: int, dy: int) -> bool:
                shifted = shift_mask(mask_view, dx, dy)
                if int(shifted.sum()) != area0:
                    return False
                if np.any(shifted & others):
                    return False
                return True

            for r in range(0, max_r, step_cells):
                for dx in range(-r, r + 1, step_cells):
                    for dy in (-r, r):
                        if ok_at(dx, dy):
                            return dx, dy
                for dy in range(-r, r + 1, step_cells):
                    for dx in (-r, r):
                        if ok_at(dx, dy):
                            return dx, dy
            return None

        def normalize_deg(a: float) -> float:
            # dopasuj do zakresu [-180, 180] dla suwaka
            a = float(a) % 360.0
            if a > 180.0:
                a -= 360.0
            return a

        # Szukamy miejsca: tylko obroty co 90° (szybko)
        angle_sets = [
            [0, 90, 180, 270],
        ]

        best = None  # (set_index, score, ang, dx, dy)
        for set_i, angles in enumerate(angle_sets):
            for ang in angles:
                # informacja w tytule (przy dużych obiektach wyszukiwanie może trwać)
                self.setWindowTitle(f"{self.base_title} | Uplasuj: kąt {int(ang)%360}°…")
                QApplication.processEvents()
                if bb is not None:
                    if (int(ang) % 180) != 0:
                        if not fits90:
                            continue
                    else:
                        if not fits0:
                            continue
                # zbuduj maskę obiektu w WORK, obróć wokół środka, zastosuj aktualne przesunięcie
                m_work = obj.mask_fast
                if m_work is None:
                    continue
                if (ang % 360) != 0:
                    src_xi, src_yi, inside = self._rotation_maps_work(float(ang))
                    m_work = rotate_mask_nn_precomputed(m_work, src_xi, src_yi, inside)

                # aktualne przesunięcie i przycięcie do okna podglądu
                m_view = crop_center(shift_mask(m_work, dx0, dy0), out_w=out_w, out_h=out_h)
                found = spiral_find(m_view)
                if found is None:
                    continue

                dx, dy = found
                score = -(abs(dx) + abs(dy))
                cand = (set_i, score, int(ang), int(dx), int(dy))
                if best is None or cand > best:
                    best = cand

            if best is not None and best[0] == set_i:
                # znaleźliśmy w aktualnym (bardziej preferowanym) zestawie kątów
                break

        if best is None:
            self.setWindowTitle(old_title)
            return

        _, _, ang, dx, dy = best

        # ustaw obrót i przesunięcie
        obj.rz = normalize_deg(float(ang))
        obj.tx += float(dx) * CELL_MM
        obj.ty -= float(dy) * CELL_MM

        obj.exact_ver += 1
        obj.exact_dirty = True
        obj.mask_exact = None
        if self.chk_exact_preview.isChecked():
            self._kick_exact(idx)

        self._sync_sliders_from_obj(obj, include_scale=False)
        self.setWindowTitle(old_title)
        self.schedule_preview(10)



    def arrange_all(self):
        if not self.objects:
            return

        out_w_mm, out_h_mm = self._current_window_mm()
        out_w = int(max(1, round(out_w_mm / CELL_MM)))
        out_h = int(max(1, round(out_h_mm / CELL_MM)))

        not_ready = [i for i, o in enumerate(self.objects) if o.mask_fast is None]
        if not_ready:
            for i in not_ready:
                self._kick_fast(i)
            QMessageBox.information(self, "Rozmieść", "Część obiektów jeszcze się liczy. Spróbuj za chwilę.")
            return

        for o in self.objects:
            o.tx = 0.0
            o.ty = 0.0
            o.rz = 0.0
            o.exact_ver += 1
            o.exact_dirty = True
            o.mask_exact = None

        order = sorted(range(len(self.objects)),
                       key=lambda i: int(self.objects[i].mask_fast.sum()),
                       reverse=True)

        occ = np.zeros((out_h, out_w), dtype=bool)
        step_cells = max(1, int(round(1.0 / CELL_MM)))  # 1mm
        max_r = max(out_w, out_h)

        def try_place(mask_view: np.ndarray) -> Optional[Tuple[int, int, np.ndarray]]:
            area0 = int(mask_view.sum())
            if area0 == 0:
                return None

            def ok_at(dx: int, dy: int) -> Optional[np.ndarray]:
                shifted = shift_mask(mask_view, dx, dy)
                if int(shifted.sum()) != area0:
                    return None
                if np.any(shifted & occ):
                    return None
                return shifted

            for r in range(0, max_r, step_cells):
                for dx in range(-r, r + 1, step_cells):
                    for dy in (-r, r):
                        placed = ok_at(dx, dy)
                        if placed is not None:
                            return dx, dy, placed
                for dy in range(-r, r + 1, step_cells):
                    for dx in (-r, r):
                        placed = ok_at(dx, dy)
                        if placed is not None:
                            return dx, dy, placed
            return None

        rot_maps_view = {ang: self.rotation_maps_for_shape(out_h, out_w, float(ang)) for ang in ARRANGE_ANGLES}

        self.setWindowTitle(f"{self.base_title} | rozmieść: układanie…")
        QApplication.processEvents()

        for k, idx in enumerate(order, start=1):
            o = self.objects[idx]
            base_view = crop_center(o.mask_fast, out_w=out_w, out_h=out_h)

            best = None
            for ang in ARRANGE_ANGLES:
                m = base_view
                if ang != 0:
                    src_xi, src_yi, inside = rot_maps_view[ang]
                    m = rotate_mask_nn_precomputed(m, src_xi, src_yi, inside)

                res = try_place(m)
                if res is None:
                    continue
                dx, dy, placed = res
                score = -(abs(dx) + abs(dy))
                cand = (score, ang, dx, dy, placed)
                if best is None or cand > best:
                    best = cand

            if best is None:
                continue

            _, ang, dx, dy, placed = best
            o.rz = float(ang)
            o.tx = float(dx) * CELL_MM
            o.ty = -float(dy) * CELL_MM
            occ |= placed

            self.setWindowTitle(f"{self.base_title} | rozmieść: {k}/{len(order)}")
            QApplication.processEvents()

        self.setWindowTitle(self.base_title)

        if self.chk_exact_preview.isChecked():
            for i in range(len(self.objects)):
                self._kick_exact(i)

        sel = self.selected_object()
        if sel is not None:
            self._sync_sliders_from_obj(sel, include_scale=False)

        self.schedule_preview(20)

    # =========================
    # Export (pełny) + ostrzeżenie przed nadpisaniem źródła
    # =========================

    def _any_object_has_xy_tilt(self) -> bool:
        return any(abs(o.rx) > 1e-6 or abs(o.ry) > 1e-6 for o in self.objects)

    def _transform_object_triangles_for_export(self, obj: StlObject, z_target_min: float) -> np.ndarray:
        vecs = obj.mesh.vectors.astype(np.float64)
        P = obj.pivot.astype(np.float64)
        centered = vecs - P[None, None, :]

        R = rot_z_screen(obj.rz) @ rot_y(obj.ry) @ rot_x(obj.rx)
        A = R * float(obj.s)

        pts = centered.reshape(-1, 3) @ A.T
        tri = pts.reshape(centered.shape)

        tri[:, :, 0] += float(obj.tx)
        tri[:, :, 1] -= float(obj.ty)

        min_z = float(tri[:, :, 2].min())
        lift = (z_target_min - min_z) if min_z < z_target_min else 0.0
        tri[:, :, 2] += lift
        return tri

    def _lattice_range_mm(self, full_window: bool, footprint: Optional[np.ndarray],
                          out_w_mm: float, out_h_mm: float) -> Optional[Tuple[float, float, float, float]]:
        if full_window:
            return (-out_w_mm / 2.0, out_w_mm / 2.0, -out_h_mm / 2.0, out_h_mm / 2.0)
        if footprint is None:
            return None
        bb = mask_bbox_cells(footprint)
        if bb is None:
            return None
        x0c, y0c, x1c, y1c = bb
        cx = WORK_W // 2
        cy = WORK_H // 2
        x0 = (x0c - cx) * CELL_MM
        x1 = (x1c - cx) * CELL_MM
        y0 = (y0c - cy) * CELL_MM
        y1 = (y1c - cy) * CELL_MM
        return (x0, x1, y0, y1)

    def _make_lattice_fixed(self, full_window: bool, footprint: Optional[np.ndarray],
                            out_w_mm: float, out_h_mm: float,
                            rib_mm: float, base_h_mm: float) -> List[np.ndarray]:
        r = self._lattice_range_mm(full_window, footprint, out_w_mm, out_h_mm)
        if r is None:
            return []
        x0, x1, y0, y1 = r

        def snap_down(v): return np.floor(v / LATTICE_PITCH_MM) * LATTICE_PITCH_MM
        def snap_up(v): return np.ceil(v / LATTICE_PITCH_MM) * LATTICE_PITCH_MM
        x0 = snap_down(x0); x1 = snap_up(x1)
        y0 = snap_down(y0); y1 = snap_up(y1)

        tris: List[np.ndarray] = []
        half_r = rib_mm / 2.0
        seg = LATTICE_SEG_MM

        xs = np.arange(x0, x1 + 1e-9, LATTICE_PITCH_MM)
        ys = np.arange(y0, y1 + 1e-9, LATTICE_PITCH_MM)

        for x in xs:
            y = y0
            while y < y1 - 1e-9:
                y_next = min(y + seg, y1)
                add_tapered_rib_box(tris, x - half_r, x + half_r, y, y_next, total_h=base_h_mm)
                y = y_next

        for y in ys:
            x = x0
            while x < x1 - 1e-9:
                x_next = min(x + seg, x1)
                add_tapered_rib_box(tris, x, x_next, y - half_r, y + half_r, total_h=base_h_mm)
                x = x_next

        return tris

    def _make_lattice_variable(self, full_window: bool, footprint: Optional[np.ndarray],
                               out_w_mm: float, out_h_mm: float,
                               object_tris_world: List[np.ndarray],
                               rib_mm: float, base_h_mm: float) -> List[np.ndarray]:
        r = self._lattice_range_mm(full_window, footprint, out_w_mm, out_h_mm)
        if r is None:
            return []
        x0, x1, y0, y1 = r

        def snap_down(v): return np.floor(v / LATTICE_PITCH_MM) * LATTICE_PITCH_MM
        def snap_up(v): return np.ceil(v / LATTICE_PITCH_MM) * LATTICE_PITCH_MM
        x0 = snap_down(x0); x1 = snap_up(x1)
        y0 = snap_down(y0); y1 = snap_up(y1)

        bin_size = float(LATTICE_PITCH_MM)
        tri_bins = []
        for tris in object_tris_world:
            bins, ox, oy = build_triangle_bins_xy(tris, bin_size=bin_size)
            tri_bins.append((tris, bins, ox, oy))

        def underside_global(x: float, y: float) -> Optional[float]:
            best = None
            for tris, bins, ox, oy in tri_bins:
                cand = bins_candidates(bins, ox, oy, bin_size, x, y, radius=1)
                if not cand:
                    continue
                z = underside_z_at_xy_candidates(tris, cand, x, y)
                if z is None:
                    continue
                if best is None or z < best:
                    best = z
            return best

        tris_out: List[np.ndarray] = []
        half_r = rib_mm / 2.0
        seg = LATTICE_SEG_MM

        xs = np.arange(x0, x1 + 1e-9, LATTICE_PITCH_MM)
        ys = np.arange(y0, y1 + 1e-9, LATTICE_PITCH_MM)

        self.setWindowTitle(f"{self.base_title} | eksport: kratownica variable…")
        QApplication.processEvents()

        for x in xs:
            y = y0
            while y < y1 - 1e-9:
                y_next = min(y + seg, y1)
                y_mid = (y + y_next) / 2.0
                z_under = underside_global(x, y_mid)
                total_h = base_h_mm if z_under is None else max(base_h_mm, float(z_under))
                add_tapered_rib_box(tris_out, x - half_r, x + half_r, y, y_next, total_h=total_h)
                y = y_next

        for y in ys:
            x = x0
            while x < x1 - 1e-9:
                x_next = min(x + seg, x1)
                x_mid = (x + x_next) / 2.0
                z_under = underside_global(x_mid, y)
                total_h = base_h_mm if z_under is None else max(base_h_mm, float(z_under))
                add_tapered_rib_box(tris_out, x, x_next, y - half_r, y + half_r, total_h=total_h)
                x = x_next

        return tris_out

    def save_stl(self):
        if not self.objects:
            QMessageBox.information(self, "Info", "Brak obiektów do eksportu.")
            return

        if self.cmb_export.currentIndex() == 1 and self._any_object_has_xy_tilt():
            res = QMessageBox.question(
                self,
                "Sugestia",
                "Wykryto niezerowy obrót w pionie (oś x/y) w co najmniej jednym obiekcie.\n"
                "Dla kratownicy lepsza bywa wersja ZMIENNEJ wysokości (wolniejsza).\n\n"
                "Przełączyć eksport na 'kratownica zmiennej wysokości'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if res == QMessageBox.Yes:
                self.cmb_export.setCurrentIndex(2)

        mode = self.cmb_export.currentIndex()
        use_lattice = (mode != 0)
        lattice_variable = (mode == 2)

        out_path, _ = QFileDialog.getSaveFileName(self, "Eksportuj scenę (stl)", "scena.stl", "STL (*.stl)")
        if not out_path:
            return

        # ostrzeż przed nadpisaniem pliku źródłowego obiektu
        src_paths = {os.path.abspath(o.path) for o in self.objects}
        if os.path.abspath(out_path) in src_paths:
            r = QMessageBox.warning(
                self,
                "Uwaga: nadpisanie pliku źródłowego",
                "Wybrany plik wyjściowy jest jednocześnie plikiem jednego z obiektów.\n"
                "Jeśli zapiszesz, nadpiszesz oryginalny STL (a potem w układzie może pojawić się kratownica!).\n\n"
                "Czy na pewno kontynuować?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if r != QMessageBox.Yes:
                return

        out_w_mm, out_h_mm = self._current_window_mm()

        factor = float(self.lattice_factor.value())
        rib_mm = LATTICE_RIB_MM * factor
        base_h_mm = LATTICE_BASE_H_MM * factor
        z_target = (base_h_mm if use_lattice else 0.0)

        object_tris_world: List[np.ndarray] = []
        tris_out: List[np.ndarray] = []

        self.setWindowTitle(f"{self.base_title} | eksport: transformacje…")
        QApplication.processEvents()

        for obj in self.objects:
            tri = self._transform_object_triangles_for_export(obj, z_target_min=z_target)
            object_tris_world.append(tri)
            tris_out.extend([t for t in tri])


        # --- sprawdź nakładanie obiektów (w rzucie XY) ---
        try:
            out_w = int(max(1, round(out_w_mm / CELL_MM)))
            out_h = int(max(1, round(out_h_mm / CELL_MM)))
            masks_view = []
            for tri in object_tris_world:
                m_work = rasterize_triangles_xy_mm_to_work_mask(tri[:, :, :2])
                masks_view.append(crop_center(m_work, out_w=out_w, out_h=out_h))

            overlaps = []
            for i in range(len(masks_view)):
                for j in range(i + 1, len(masks_view)):
                    if np.any(masks_view[i] & masks_view[j]):
                        overlaps.append((self.objects[i].name, self.objects[j].name))

            if overlaps:
                # pokaż max kilka par, żeby nie zalać okna
                show_pairs = overlaps[:12]
                pairs_txt = "\n".join([f"• {a}  ×  {b}" for a, b in show_pairs])
                if len(overlaps) > 12:
                    pairs_txt += f"\n… (+{len(overlaps) - 12})"

                title = tr(self.language, "overlap_title") if "overlap_title" in I18N.get(self.language, {}) else "Uwaga"
                msg = tr(self.language, "overlap_msg") if "overlap_msg" in I18N.get(self.language, {}) else (
                    "Wykryto nachodzenie obiektów na siebie w rzucie XY.\n"
                    "Obiekty mogą się przenikać w wyeksportowanym STL.\n\n"
                    "Pary nachodzących obiektów:"
                )
                msg2 = tr(self.language, "overlap_continue") if "overlap_continue" in I18N.get(self.language, {}) else "Kontynuować eksport?"
                r = QMessageBox.warning(self, title, f"{msg}\n\n{pairs_txt}\n\n{msg2}",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if r != QMessageBox.Yes:
                    self.setWindowTitle(self.base_title)
                    return
        except Exception:
            # w razie problemu nie blokuj eksportu (to tylko ostrzeżenie)
            pass

        # footprint: jeśli kratownica nie na całym oknie -> pod obiektami (z podglądu exact/export zgodnego)
        footprint = None
        if not self.chk_lattice_full.isChecked():
            out_w = int(max(1, round(out_w_mm / CELL_MM)))
            out_h = int(max(1, round(out_h_mm / CELL_MM)))
            footprint = np.zeros((out_h, out_w), dtype=bool)
            # używamy dokładnego maskowania (zgodnego z exportem): jeśli brak, to awaryjnie fast
            for o in self.objects:
                m = self._object_view_mask(o, out_w, out_h, exact=self.chk_exact_preview.isChecked())
                footprint |= m

        if use_lattice:
            if lattice_variable:
                lattice_tris = self._make_lattice_variable(
                    full_window=self.chk_lattice_full.isChecked(),
                    footprint=footprint,
                    out_w_mm=out_w_mm,
                    out_h_mm=out_h_mm,
                    object_tris_world=object_tris_world,
                    rib_mm=rib_mm,
                    base_h_mm=base_h_mm
                )
            else:
                self.setWindowTitle(f"{self.base_title} | eksport: kratownica fixed…")
                QApplication.processEvents()
                lattice_tris = self._make_lattice_fixed(
                    full_window=self.chk_lattice_full.isChecked(),
                    footprint=footprint,
                    out_w_mm=out_w_mm,
                    out_h_mm=out_h_mm,
                    rib_mm=rib_mm,
                    base_h_mm=base_h_mm
                )
            tris_out.extend(lattice_tris)

        self.setWindowTitle(f"{self.base_title} | eksport: STL…")
        QApplication.processEvents()

        vectors = np.stack(tris_out, axis=0).astype(np.float32)
        merged = stlmesh.Mesh(np.zeros(vectors.shape[0], dtype=stlmesh.Mesh.dtype))
        merged.vectors[:] = vectors

        try:
            merged.save(out_path)
        except Exception as e:
            self.setWindowTitle(self.base_title)
            QMessageBox.critical(self, "Błąd", f"Nie udało się wyeksportować STL:\n{e}")
            return

        self.setWindowTitle(self.base_title)
        QMessageBox.information(self, "OK", f"Wyeksportowano:\n{out_path}")

    # =========================
    # Podgląd 3D
    # =========================

    def open_preview_3d(self):
        if not self.objects:
            QMessageBox.information(self, "Podgląd 3D", "Brak obiektów.")
            return

        mode = self.cmb_export.currentIndex()
        use_lattice = (mode != 0)

        out_w_mm, out_h_mm = self._current_window_mm()

        factor = float(self.lattice_factor.value())
        rib_mm = LATTICE_RIB_MM * factor
        base_h_mm = LATTICE_BASE_H_MM * factor

        # Zbieramy trójkąty obiektów tak jak do eksportu (ale: dla podglądu 3D lattice robimy tylko fixed - szybkie)
        tris_all: List[np.ndarray] = []
        z_target = base_h_mm if use_lattice else 0.0

        self.setWindowTitle(f"{self.base_title} | podgląd 3D: transformacje…")
        QApplication.processEvents()

        object_tris_world: List[np.ndarray] = []
        for o in self.objects:
            tri = self._transform_object_triangles_for_export(o, z_target_min=z_target)
            object_tris_world.append(tri)
            tris_all.append(tri)

        # Lattice do 3D podglądu: tylko fixed (żeby było responsywne)
        if use_lattice:
            footprint = None
            if not self.chk_lattice_full.isChecked():
                out_w = int(max(1, round(out_w_mm / CELL_MM)))
                out_h = int(max(1, round(out_h_mm / CELL_MM)))
                footprint = np.zeros((out_h, out_w), dtype=bool)
                for o in self.objects:
                    footprint |= self._object_view_mask(o, out_w, out_h, exact=False)

            self.setWindowTitle(f"{self.base_title} | podgląd 3D: kratownica…")
            QApplication.processEvents()
            lattice_tris = self._make_lattice_fixed(
                full_window=self.chk_lattice_full.isChecked(),
                footprint=footprint,
                out_w_mm=out_w_mm,
                out_h_mm=out_h_mm,
                rib_mm=rib_mm,
                base_h_mm=base_h_mm
            )
            if lattice_tris:
                tris_all.append(np.stack(lattice_tris, axis=0).astype(np.float64))

        self.setWindowTitle(self.base_title)

        tri_cat = np.concatenate(tris_all, axis=0) if len(tris_all) > 1 else tris_all[0]
        colors = [o.color for o in self.objects]
        w = Scene3DWindow(self, tri_cat, colors, f"{self.base_title} | Podgląd 3D")
        
        screen = self.screen() or QApplication.primaryScreen()
        geo = screen.availableGeometry()
        w.resize(geo.width() // 2, geo.height() // 2)
        w.show()

    # =========================
    # Pozostałe: preview mode + klik/drag
    # =========================


# =========================
# main
# =========================
def main():
    app = QApplication([])
    w = MainWindow()
    set_half_screen_geometry(w, fraction=0.5, min_size=(1100, 750), center=True)
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
