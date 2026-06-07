from __future__ import annotations

import sys
import os
import math
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from PySide6.QtCore import QPoint, QProcess, QProcessEnvironment, QRect, QTimer, Qt, Signal
    from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - runtime dependency message
    print("PySide6 не установлен. Установите его командой: pip install PySide6", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    import pyqtgraph.opengl as gl
except ImportError as exc:  # pragma: no cover - runtime dependency message
    print(
        "pyqtgraph/PyOpenGL не установлены. Установите GUI-зависимости командой: "
        "pip install -r requirements-gui.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


class HeightSurfaceView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((24, 30, 42))
        self.view.setCameraPosition(distance=120, elevation=45, azimuth=-60)
        self.surface_item: gl.GLSurfacePlotItem | None = None
        self.grid_item: gl.GLGridItem | None = None
        self.info_label = QLabel("Здесь появится OpenGL 3D карта высот. Мышь: вращение, колесо: zoom.")
        self.info_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.info_label)
        layout.addWidget(self.view, 1)
        self.setMinimumSize(640, 480)
        self.plot_placeholder()

    def plot_placeholder(self) -> None:
        self.view.clear()
        self.surface_item = None
        self.grid_item = None
        self._add_scene_guides(rows=80, cols=80, z_height=25)
        self.info_label.setText(
            "Здесь появится OpenGL 3D карта высот. Цветные оси: X красная, Y зелёная, Z синяя."
        )

    def plot_height_map(
        self,
        height: np.ndarray,
        height_scale: float,
        stride: int,
        max_side_points: int,
        smooth_surface: bool,
    ) -> None:
        height = np.asarray(height, dtype=np.float32)
        stride = max(1, int(stride))
        max_side_points = max(16, int(max_side_points))
        auto_stride = max(1, math.ceil(max(height.shape) / max_side_points))
        effective_stride = max(stride, auto_stride)
        z = height[::effective_stride, ::effective_stride]
        if smooth_surface:
            z = self._smooth_height_map(z)
        z = np.flipud(z)
        z = z - float(np.nanmin(z))
        z_range = float(np.nanmax(z))
        if z_range > 1e-6:
            z = z / z_range

        rows, cols = z.shape
        xy_size = float(max(rows, cols))
        z = z * float(height_scale) * xy_size * 0.25
        x = np.linspace(-cols / 2.0, cols / 2.0, cols, dtype=np.float32)
        y = np.linspace(-rows / 2.0, rows / 2.0, rows, dtype=np.float32)
        z_gl = np.ascontiguousarray(z.T)
        colors = self._terrain_colors(z_gl)

        self.view.clear()
        self.surface_item = None
        self._add_scene_guides(rows=rows, cols=cols, z_height=max(8.0, float(np.nanmax(z))))

        if self.surface_item is None:
            self.surface_item = gl.GLSurfacePlotItem(
                x=x,
                y=y,
                z=z_gl,
                colors=colors,
                shader=None,
                smooth=False,
                computeNormals=False,
            )
            self.surface_item.setGLOptions("opaque")
            self.view.addItem(self.surface_item)
        else:
            self.surface_item.setData(x=x, y=y, z=z_gl, colors=colors)

        self.view.setCameraPosition(distance=max(rows, cols) * 1.8, elevation=45, azimuth=-60)
        self.info_label.setText(
            f"OpenGL 3D карта высот: {cols}x{rows} вершин, stride={effective_stride}. "
            f"Сглаживание: {'вкл' if smooth_surface else 'выкл'}. "
            "Оси: X красная, Y зелёная, Z синяя. Мышь: вращение, колесо: zoom."
        )

    def save_snapshot(self, path: Path) -> None:
        image = self.view.grabFramebuffer()
        image.save(str(path))

    def _add_scene_guides(self, rows: int, cols: int, z_height: float) -> None:
        xy_size = float(max(rows, cols))
        spacing = max(4, int(xy_size // 8))

        self.grid_item = gl.GLGridItem()
        self.grid_item.setSize(x=cols, y=rows)
        self.grid_item.setSpacing(x=spacing, y=spacing)
        self.view.addItem(self.grid_item)

        x0, x1 = -cols / 2.0, cols / 2.0
        y0, y1 = -rows / 2.0, rows / 2.0
        z1 = max(8.0, float(z_height) * 1.15)
        axes = [
            (np.array([[x0, y0, 0.0], [x1, y0, 0.0]], dtype=np.float32), (1.0, 0.15, 0.10, 1.0), "X", (x1 + 4, y0, 0.0)),
            (np.array([[x0, y0, 0.0], [x0, y1, 0.0]], dtype=np.float32), (0.20, 1.0, 0.25, 1.0), "Y", (x0, y1 + 4, 0.0)),
            (np.array([[x0, y0, 0.0], [x0, y0, z1]], dtype=np.float32), (0.25, 0.45, 1.0, 1.0), "Z", (x0, y0, z1 + 4)),
        ]
        for points, color, label, label_pos in axes:
            line = gl.GLLinePlotItem(pos=points, color=color, width=4, antialias=True, mode="lines")
            self.view.addItem(line)
            text = gl.GLTextItem(pos=label_pos, text=label, color=(255, 255, 255, 255))
            self.view.addItem(text)

        origin = gl.GLTextItem(pos=(x0 - 6, y0 - 6, 0.0), text="0", color=(230, 230, 230, 255))
        self.view.addItem(origin)

    @staticmethod
    def _terrain_colors(z: np.ndarray) -> np.ndarray:
        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))
        t = (z - z_min) / max(z_max - z_min, 1e-6)
        colors = np.empty(z.shape + (4,), dtype=np.float32)
        low = t < 0.45
        mid = (t >= 0.45) & (t < 0.8)
        high = t >= 0.8

        colors[..., 0] = 0.25
        colors[..., 1] = 0.55 + 0.40 * t
        colors[..., 2] = 0.18 + 0.30 * (1.0 - t)
        colors[..., 3] = 1.0
        colors[low, 0] = 0.20 + 0.25 * t[low]
        colors[low, 1] = 0.45 + 0.55 * t[low]
        colors[low, 2] = 0.14
        colors[mid, 0] = 0.55 + 0.25 * t[mid]
        colors[mid, 1] = 0.65 + 0.25 * t[mid]
        colors[mid, 2] = 0.28
        colors[high, 0] = 0.88 + 0.12 * t[high]
        colors[high, 1] = 0.86 + 0.14 * t[high]
        colors[high, 2] = 0.80 + 0.20 * t[high]
        return np.ascontiguousarray(colors.reshape(-1, 4))

    @staticmethod
    def _smooth_height_map(z: np.ndarray) -> np.ndarray:
        padded = np.pad(z, 1, mode="edge")
        return (
            padded[:-2, :-2]
            + 2.0 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2]
            + 4.0 * padded[1:-1, 1:-1]
            + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2]
            + 2.0 * padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 16.0


class SquareImagePreview(QLabel):
    selectionChanged = Signal()

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._source_pixmap: QPixmap | None = None
        self._image_path: Path | None = None
        self._crop_rect: QRect | None = None
        self._drag_start: QPoint | None = None
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    @property
    def image_path(self) -> Path | None:
        return self._image_path

    def set_image(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._source_pixmap = None
            self._image_path = None
            self._crop_rect = None
            self.setText(f"Не удалось открыть:\n{path}")
            self.update()
            self.selectionChanged.emit()
            return False

        self._source_pixmap = pixmap
        self._image_path = path
        self.reset_selection()
        return True

    def reset_selection(self) -> bool:
        if self._source_pixmap is None:
            return False
        width = self._source_pixmap.width()
        height = self._source_pixmap.height()
        side = min(width, height)
        self._crop_rect = QRect((width - side) // 2, (height - side) // 2, side, side)
        self.update()
        self.selectionChanged.emit()
        return True

    def selection_description(self) -> str:
        if self._source_pixmap is None or self._crop_rect is None:
            return "Выберите изображение, затем выделите квадрат на превью."
        rect = self._crop_rect
        return f"Квадрат: x={rect.x()}, y={rect.y()}, сторона={rect.width()} px. Перетащите мышью на превью для выбора."

    def save_selected_crop(self, path: Path) -> QRect | None:
        if self._source_pixmap is None or self._crop_rect is None:
            return None
        crop = self._source_pixmap.copy(self._crop_rect)
        if not crop.save(str(path)):
            return None
        return QRect(self._crop_rect)

    def paintEvent(self, event) -> None:  # noqa: ANN001 - Qt callback signature
        if self._source_pixmap is None:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(17, 17, 17))
        target = self._display_rect()
        painter.drawPixmap(target, self._source_pixmap)

        if self._crop_rect is not None:
            overlay = QColor(0, 0, 0, 115)
            selection = self._source_to_display_rect(self._crop_rect)
            painter.fillRect(QRect(target.left(), target.top(), target.width(), selection.top() - target.top()), overlay)
            painter.fillRect(QRect(target.left(), selection.bottom() + 1, target.width(), target.bottom() - selection.bottom()), overlay)
            painter.fillRect(QRect(target.left(), selection.top(), selection.left() - target.left(), selection.height()), overlay)
            painter.fillRect(QRect(selection.right() + 1, selection.top(), target.right() - selection.right(), selection.height()), overlay)
            painter.setPen(QPen(QColor(255, 214, 80), 3))
            painter.drawRect(selection)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton or self._source_pixmap is None:
            return
        point = self._point_to_source(event.position().toPoint())
        if point is None:
            return
        self._drag_start = point
        self._crop_rect = QRect(point.x(), point.y(), 1, 1)
        self.update()
        self.selectionChanged.emit()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is None or self._source_pixmap is None:
            return
        point = self._point_to_source(event.position().toPoint())
        if point is None:
            return
        self._crop_rect = self._square_from_points(self._drag_start, point)
        self.update()
        self.selectionChanged.emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        self._drag_start = None

    def _display_rect(self) -> QRect:
        if self._source_pixmap is None:
            return self.contentsRect()
        area = self.contentsRect()
        width = self._source_pixmap.width()
        height = self._source_pixmap.height()
        scale = min(area.width() / width, area.height() / height)
        display_width = max(1, int(width * scale))
        display_height = max(1, int(height * scale))
        return QRect(
            area.x() + (area.width() - display_width) // 2,
            area.y() + (area.height() - display_height) // 2,
            display_width,
            display_height,
        )

    def _point_to_source(self, point: QPoint) -> QPoint | None:
        if self._source_pixmap is None:
            return None
        display = self._display_rect()
        if not display.contains(point):
            return None
        x = round((point.x() - display.x()) * self._source_pixmap.width() / display.width())
        y = round((point.y() - display.y()) * self._source_pixmap.height() / display.height())
        x = min(max(int(x), 0), self._source_pixmap.width() - 1)
        y = min(max(int(y), 0), self._source_pixmap.height() - 1)
        return QPoint(x, y)

    def _source_to_display_rect(self, rect: QRect) -> QRect:
        display = self._display_rect()
        if self._source_pixmap is None:
            return display
        scale_x = display.width() / self._source_pixmap.width()
        scale_y = display.height() / self._source_pixmap.height()
        return QRect(
            int(display.x() + rect.x() * scale_x),
            int(display.y() + rect.y() * scale_y),
            max(1, int(rect.width() * scale_x)),
            max(1, int(rect.height() * scale_y)),
        )

    def _square_from_points(self, start: QPoint, end: QPoint) -> QRect:
        if self._source_pixmap is None:
            return QRect()
        width = self._source_pixmap.width()
        height = self._source_pixmap.height()
        side = max(abs(end.x() - start.x()), abs(end.y() - start.y()), 1)
        side = min(side, width, height)
        x = start.x() if end.x() >= start.x() else start.x() - side
        y = start.y() if end.y() >= start.y() else start.y() - side
        x = min(max(x, 0), width - side)
        y = min(max(y, 0), height - side)
        return QRect(x, y, side, side)


class TerrainViewerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.repo_root = Path(__file__).resolve().parents[1]
        self.current_height: np.ndarray | None = None
        self.current_run_dir: Path | None = None
        self.process: QProcess | None = None
        self.redraw_timer = QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(250)
        self.redraw_timer.timeout.connect(self._refresh_3d)

        self.setWindowTitle("Terrain Height Estimation Viewer")
        self.resize(1280, 820)
        self._build_ui()

    def _build_ui(self) -> None:
        self.image_edit = QLineEdit()
        self.checkpoint_edit = QLineEdit(str(self.repo_root / "outputs/rgb_student_distill/checkpoints/best.pt"))
        self.config_edit = QLineEdit(str(self.repo_root / "configs/inference/default.yaml"))
        self.output_root_edit = QLineEdit(str(self.repo_root / "outputs/gui_runs"))
        self.crop_info_label = QLabel("Выберите изображение, затем выделите квадрат на превью.")
        self.crop_info_label.setWordWrap(True)
        self.crop_reset_button = QPushButton("Центральный квадрат")
        self.crop_reset_button.clicked.connect(self._reset_crop_selection)

        crop_layout = QHBoxLayout()
        crop_layout.addWidget(self.crop_info_label, 1)
        crop_layout.addWidget(self.crop_reset_button)
        crop_layout.setContentsMargins(0, 0, 0, 0)
        crop_row = QWidget()
        crop_row.setLayout(crop_layout)

        input_box = QGroupBox("Входные данные")
        form = QFormLayout(input_box)
        form.addRow("Изображение:", self._path_row(self.image_edit, self._browse_image))
        form.addRow("Область:", crop_row)
        form.addRow("Чекпоинт:", self._path_row(self.checkpoint_edit, self._browse_checkpoint))
        form.addRow("Конфиг:", self._path_row(self.config_edit, self._browse_config))
        form.addRow("Папка результатов:", self._path_row(self.output_root_edit, self._browse_output_root))

        self.height_scale_spin = QDoubleSpinBox()
        self.height_scale_spin.setRange(0.1, 10.0)
        self.height_scale_spin.setSingleStep(0.1)
        self.height_scale_spin.setValue(1.0)
        self.height_scale_spin.valueChanged.connect(self._schedule_refresh_3d)

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 16)
        self.stride_spin.setValue(4)
        self.stride_spin.valueChanged.connect(self._schedule_refresh_3d)

        self.max_side_points_spin = QSpinBox()
        self.max_side_points_spin.setRange(16, 256)
        self.max_side_points_spin.setSingleStep(16)
        self.max_side_points_spin.setValue(64)
        self.max_side_points_spin.valueChanged.connect(self._schedule_refresh_3d)

        self.smooth_surface_checkbox = QCheckBox("Сгладить 3D-поверхность")
        self.smooth_surface_checkbox.setToolTip("Уменьшает резкие пики и провалы только в 3D-визуализации.")
        self.smooth_surface_checkbox.stateChanged.connect(self._schedule_refresh_3d)

        self.inference_threads_spin = QSpinBox()
        self.inference_threads_spin.setRange(1, 16)
        self.inference_threads_spin.setValue(max(1, min(2, os.cpu_count() or 2)))

        controls_box = QGroupBox("Визуализация")
        controls = QFormLayout(controls_box)
        controls.addRow("Масштаб высоты:", self.height_scale_spin)
        controls.addRow("Шаг сетки:", self.stride_spin)
        controls.addRow("Макс. точек/сторона:", self.max_side_points_spin)
        controls.addRow("Smooth:", self.smooth_surface_checkbox)
        controls.addRow("CPU-потоки инференса:", self.inference_threads_spin)

        self.run_button = QPushButton("Оценить карту высот")
        self.run_button.clicked.connect(self._run_inference)

        self.status_label = QLabel("Выберите изображение и нажмите кнопку запуска.")
        self.status_label.setWordWrap(True)

        left_layout = QVBoxLayout()
        left_layout.addWidget(input_box)
        left_layout.addWidget(controls_box)
        left_layout.addWidget(self.run_button)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)
        left_panel = QWidget()
        left_panel.setLayout(left_layout)

        self.input_preview = SquareImagePreview("Входное изображение")
        self.input_preview.selectionChanged.connect(self._update_crop_info)
        self.height_preview = QLabel("2D карта высот")
        for label in (self.input_preview, self.height_preview):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(320, 320)
            label.setStyleSheet("QLabel { border: 1px solid #888; background: #111; color: #ddd; }")

        preview_row = QHBoxLayout()
        preview_row.addWidget(self.input_preview)
        preview_row.addWidget(self.height_preview)
        preview_tab = QWidget()
        preview_tab.setLayout(preview_row)

        self.surface_canvas = HeightSurfaceView()
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)

        tabs = QTabWidget()
        tabs.addTab(preview_tab, "2D")
        tabs.addTab(self.surface_canvas, "3D")
        tabs.addTab(self.log_edit, "Лог")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 860])

        self.setCentralWidget(splitter)

    def _path_row(self, edit: QLineEdit, browse_slot) -> QWidget:
        button = QPushButton("Обзор")
        button.clicked.connect(browse_slot)
        layout = QHBoxLayout()
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QWidget()
        row.setLayout(layout)
        return row

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            str(self.repo_root / "data"),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if path:
            self.image_edit.setText(path)
            self._set_preview(self.input_preview, Path(path))

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите чекпоинт",
            str(self.repo_root / "outputs"),
            "PyTorch checkpoints (*.pt *.pth);;All files (*.*)",
        )
        if path:
            self.checkpoint_edit.setText(path)

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите YAML-конфиг",
            str(self.repo_root / "configs"),
            "YAML files (*.yaml *.yml);;All files (*.*)",
        )
        if path:
            self.config_edit.setText(path)

    def _browse_output_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выберите папку результатов", str(self.repo_root / "outputs"))
        if path:
            self.output_root_edit.setText(path)

    def _reset_crop_selection(self) -> None:
        self.input_preview.reset_selection()
        self._update_crop_info()

    def _update_crop_info(self) -> None:
        self.crop_info_label.setText(self.input_preview.selection_description())

    def _run_inference(self) -> None:
        image_path = Path(self.image_edit.text().strip())
        checkpoint_path = Path(self.checkpoint_edit.text().strip())
        config_path = Path(self.config_edit.text().strip())
        output_root = Path(self.output_root_edit.text().strip())

        missing = [path for path in (image_path, checkpoint_path, config_path) if not path.exists()]
        if missing:
            self._show_error("Файл не найден", "\n".join(str(path) for path in missing))
            return

        if self.input_preview.image_path != image_path and not self.input_preview.set_image(image_path):
            self._show_error("Ошибка изображения", f"Не удалось открыть изображение:\n{image_path}")
            return

        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.current_run_dir = output_root / run_name
        try:
            self.current_run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._show_error("Ошибка папки результатов", str(exc))
            return

        selected_image_path = self.current_run_dir / "selected_region.png"
        selected_rect = self.input_preview.save_selected_crop(selected_image_path)
        if selected_rect is None:
            self._show_error("Ошибка области", "Не удалось сохранить выбранную квадратную область изображения.")
            return

        self.current_height = None
        self.surface_canvas.plot_placeholder()
        self.input_preview.update()
        self.height_preview.setText("Ожидание результата...")
        self.log_edit.clear()

        script_path = self.repo_root / "scripts" / "infer.py"
        args = [
            str(script_path),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--image",
            str(selected_image_path),
            f"inference.output_dir={self.current_run_dir}",
        ]

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.repo_root))
        self.process.setProgram(sys.executable)
        self.process.setArguments(args)
        self.process.setProcessEnvironment(self._process_environment())
        self.process.started.connect(self._inference_started)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.errorOccurred.connect(self._process_error)
        self.process.finished.connect(self._inference_finished)

        self.run_button.setEnabled(False)
        self.status_label.setText("Нейросеть считает карту высот...")
        self._append_log(
            f"Selected square: x={selected_rect.x()}, y={selected_rect.y()}, "
            f"side={selected_rect.width()} px -> {selected_image_path}\n"
        )
        self._append_log(f"$ {sys.executable} {' '.join(args)}\n")
        self.process.start()

    def _process_environment(self) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment()
        threads = str(self.inference_threads_spin.value())
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.insert(name, threads)
        return env

    def _inference_started(self) -> None:
        self.status_label.setText("Инференс запущен отдельным процессом. Интерфейс должен оставаться отзывчивым.")

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if self.process is None:
            return
        self.run_button.setEnabled(True)
        self.status_label.setText("Не удалось запустить или выполнить инференс. Подробности в логе.")
        self._append_log(f"\nQProcess error {error}: {self.process.errorString()}\n")

    def _read_stdout(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_log(text)

    def _read_stderr(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self._append_log(text)

    def _inference_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.run_button.setEnabled(True)
        if exit_status != QProcess.NormalExit or exit_code != 0:
            self.status_label.setText("Инференс завершился с ошибкой. Подробности в логе.")
            self._show_error("Ошибка инференса", f"Код завершения: {exit_code}")
            return
        self._load_run_outputs()

    def _load_run_outputs(self) -> None:
        if self.current_run_dir is None:
            return
        height_path = self.current_run_dir / "pred_height.npy"
        height_png_path = self.current_run_dir / "pred_height.png"
        if not height_path.exists():
            self._show_error("Нет результата", f"Не найден файл: {height_path}")
            return

        self.current_height = np.load(height_path)
        if height_png_path.exists():
            self._set_preview(self.height_preview, height_png_path)
        else:
            self.height_preview.setText(str(height_path))

        self._refresh_3d()
        self.status_label.setText(f"Готово. Результаты сохранены в: {self.current_run_dir}")

    def _refresh_3d(self) -> None:
        if self.current_height is None:
            return
        self.status_label.setText("Перерисовка 3D-сцены...")
        self.surface_canvas.plot_height_map(
            self.current_height,
            height_scale=self.height_scale_spin.value(),
            stride=self.stride_spin.value(),
            max_side_points=self.max_side_points_spin.value(),
            smooth_surface=self.smooth_surface_checkbox.isChecked(),
        )
        if self.current_run_dir is not None:
            self.status_label.setText(f"Готово. Результаты сохранены в: {self.current_run_dir}")

    def _schedule_refresh_3d(self) -> None:
        if self.current_height is not None:
            self.redraw_timer.start()

    def _set_preview(self, label: QLabel, path: Path) -> None:
        if isinstance(label, SquareImagePreview):
            label.set_image(path)
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            label.setText(f"Не удалось открыть:\n{path}")
            return
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _append_log(self, text: str) -> None:
        self.log_edit.moveCursor(QTextCursor.End)
        self.log_edit.insertPlainText(text)
        self.log_edit.moveCursor(QTextCursor.End)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)


def main() -> None:
    app = QApplication(sys.argv)
    window = TerrainViewerWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
