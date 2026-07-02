"""
Cortical Depth Intensity Profile
=================================
Napari widget that measures average image intensity along the long axis
of a rectangle drawn in a Shapes layer. Designed for Nissl-stained brightfield
cortex images to quantify density changes across cortical depth.

Usage:
    1. Open your image in napari.
    2. Run this script.
    3. Draw a rectangle on the Shapes layer (long axis = cortical depth).
    4. Click "Plot Profile".
    5. Optionally export to CSV.
"""

import numpy as np
import napari
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QLabel,
    QSpinBox, QCheckBox, QPushButton, QFileDialog, QGroupBox,
)
from scipy.ndimage import map_coordinates


# ── Geometry helpers ──────────────────────────────────────────────────

def get_rectangle_axes(corners: np.ndarray):
    """Return origin, long/short vectors and their lengths from 4 corners."""
    edge1 = corners[1] - corners[0]
    edge2 = corners[3] - corners[0]
    len1, len2 = np.linalg.norm(edge1), np.linalg.norm(edge2)

    if len1 >= len2:
        long_vec, short_vec = edge1, edge2
        long_len, short_len = len1, len2
    else:
        long_vec, short_vec = edge2, edge1
        long_len, short_len = len2, len1

    return corners[0], long_vec, short_vec, long_len, short_len


def sample_rectangle_profile(image, corners, n_long=None, n_short=None):
    """
    Sample intensity along the long axis of a rectangle, averaging across
    the short axis at each position.

    Returns (distances, profile) arrays.
    """
    origin, long_vec, short_vec, long_len, short_len = get_rectangle_axes(corners)

    if n_long is None:
        n_long = max(int(np.round(long_len)), 2)
    if n_short is None:
        n_short = max(int(np.round(short_len)), 3)

    long_dir = long_vec / long_len
    short_dir = short_vec / short_len

    t_vals = np.linspace(0, long_len, n_long)
    s_vals = np.linspace(0, short_len, n_short)

    tt, ss = np.meshgrid(t_vals, s_vals, indexing='ij')
    coords = (origin[np.newaxis, np.newaxis, :]
              + tt[..., np.newaxis] * long_dir[np.newaxis, np.newaxis, :]
              + ss[..., np.newaxis] * short_dir[np.newaxis, np.newaxis, :])

    flat_coords = coords.reshape(-1, 2).T

    # Convert to 2D grayscale
    if image.ndim == 3:
        if image.shape[2] <= 4:  # (Y, X, C)
            gray = np.mean(image, axis=2).astype(np.float64)
        else:  # (C, Y, X)
            gray = np.mean(image, axis=0).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    sampled = map_coordinates(gray, flat_coords, order=1, mode='nearest')
    sampled = sampled.reshape(n_long, n_short)

    return t_vals, np.mean(sampled, axis=1)


# ── Widget ────────────────────────────────────────────────────────────

class ProfileWidget(QWidget):
    """Qt widget with manually-managed layer combo boxes."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        # ── Layer selectors (plain QComboBox) ──
        self.image_combo = QComboBox()
        self.shapes_combo = QComboBox()

        img_row = self._labelled_row("Image layer:", self.image_combo)
        shp_row = self._labelled_row("Shapes layer:", self.shapes_combo)

        # ── Options ──
        self.shape_index = QSpinBox()
        self.shape_index.setRange(0, 999)
        self.shape_index.setValue(0)
        idx_row = self._labelled_row("Rectangle index:", self.shape_index)

        self.invert_check = QCheckBox("Invert intensity (dark stain → high)")
        self.invert_check.setChecked(True)
        self.normalize_check = QCheckBox("Normalize 0–1")

        # ── Buttons ──
        self.plot_btn = QPushButton("Plot Profile")
        self.plot_btn.clicked.connect(self.update_profile)
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.plot_btn)
        btn_row.addWidget(self.export_btn)

        # ── Matplotlib canvas ──
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.fig.tight_layout()

        # ── Assemble layout ──
        controls = QGroupBox("Settings")
        cl = QVBoxLayout()
        cl.addLayout(img_row)
        cl.addLayout(shp_row)
        cl.addLayout(idx_row)
        cl.addWidget(self.invert_check)
        cl.addWidget(self.normalize_check)
        cl.addLayout(btn_row)
        controls.setLayout(cl)

        layout = QVBoxLayout()
        layout.addWidget(controls)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # ── State ──
        self._last_distances = None
        self._last_profile = None

        # ── Populate combos & connect to layer events ──
        self._refresh_layer_combos()
        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)
        self.viewer.layers.events.reordered.connect(self._on_layers_changed)

    # ── helpers ──

    @staticmethod
    def _labelled_row(text, widget):
        row = QHBoxLayout()
        row.addWidget(QLabel(text))
        row.addWidget(widget)
        return row

    def _on_layers_changed(self, event=None):
        self._refresh_layer_combos()

    def _refresh_layer_combos(self):
        """Re-populate both combo boxes from the viewer's current layers."""
        # Remember current selections
        prev_img = self.image_combo.currentText()
        prev_shp = self.shapes_combo.currentText()

        self.image_combo.clear()
        self.shapes_combo.clear()

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self.image_combo.addItem(layer.name)
            elif isinstance(layer, napari.layers.Shapes):
                self.shapes_combo.addItem(layer.name)

        # Restore previous selection if still present
        idx = self.image_combo.findText(prev_img)
        if idx >= 0:
            self.image_combo.setCurrentIndex(idx)
        idx = self.shapes_combo.findText(prev_shp)
        if idx >= 0:
            self.shapes_combo.setCurrentIndex(idx)

    def _get_selected_layers(self):
        """Return (image_layer, shapes_layer) or (None, None) with message."""
        img_name = self.image_combo.currentText()
        shp_name = self.shapes_combo.currentText()

        if not img_name or not shp_name:
            print("Select both an Image layer and a Shapes layer.")
            return None, None

        try:
            image_layer = self.viewer.layers[img_name]
            shapes_layer = self.viewer.layers[shp_name]
        except KeyError:
            print("Selected layer no longer exists. Refreshing…")
            self._refresh_layer_combos()
            return None, None

        return image_layer, shapes_layer

    # ── core actions ──

    def update_profile(self):
        """Compute and plot the intensity profile."""
        image_layer, shapes_layer = self._get_selected_layers()
        if image_layer is None:
            return

        if len(shapes_layer.data) == 0:
            print("No shapes found. Draw a rectangle first.")
            return

        idx = self.shape_index.value()
        if idx >= len(shapes_layer.data):
            print(f"Shape index {idx} out of range "
                  f"(max {len(shapes_layer.data) - 1}).")
            return

        shape_type = shapes_layer.shape_type[idx]
        if shape_type != 'rectangle':
            print(f"Shape {idx} is '{shape_type}', not a rectangle.")
            return

        corners = np.array(shapes_layer.data[idx])
        if corners.shape[1] > 2:
            corners = corners[:, -2:]

        image = image_layer.data
        if hasattr(image, 'compute'):
            image = np.asarray(image)

        distances, profile = sample_rectangle_profile(image, corners)

        if self.invert_check.isChecked():
            profile = profile.max() - profile + profile.min()

        if self.normalize_check.isChecked() and (profile.max() - profile.min()) > 0:
            profile = (profile - profile.min()) / (profile.max() - profile.min())

        # Plot
        self.ax.clear()
        self.ax.plot(distances, profile, color='#2E86AB', linewidth=1.5)
        self.ax.set_xlabel("Distance along cortical depth (px)")
        self.ax.set_ylabel("Mean intensity (a.u.)")
        self.ax.set_title("Cortical Depth Profile")
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.fig.tight_layout()
        self.canvas.draw()

        self._last_distances = distances
        self._last_profile = profile
        print(f"Profile: {len(distances)} samples over "
              f"{distances[-1]:.0f} px, averaged across short axis.")

    def export_csv(self):
        """Export the last computed profile to CSV via a save dialog."""
        if self._last_distances is None or self._last_profile is None:
            print("No profile to export. Click 'Plot Profile' first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Profile CSV", "cortical_profile.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["distance_px", "mean_intensity"])
            for d, v in zip(self._last_distances, self._last_profile):
                writer.writerow([f"{d:.2f}", f"{v:.4f}"])

        print(f"Profile exported to {path}")


# ── Launch ────────────────────────────────────────────────────────────

def run():
    """Launch the widget in the current napari viewer."""
    viewer = napari.current_viewer()
    if viewer is None:
        viewer = napari.Viewer()

    # Ensure at least one shapes layer exists
    if not any(isinstance(l, napari.layers.Shapes) for l in viewer.layers):
        viewer.add_shapes(
            name='ROI', shape_type='rectangle',
            edge_color='yellow', edge_width=3, face_color='transparent',
        )

    widget = ProfileWidget(viewer)
    viewer.window.add_dock_widget(
        widget, name="Cortical Depth Profile", area='right',
    )
    return widget


if __name__ == '__main__':
    try:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError
        widget = run()
    except Exception:
        viewer = napari.Viewer()
        widget = run()
        napari.run()
