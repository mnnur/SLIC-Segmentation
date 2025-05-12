import sys
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from skimage import color, graph
from skimage.segmentation import slic, mark_boundaries
from skimage.filters import gaussian
from PIL import Image
import zipfile
from io import BytesIO
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackError
import traceback, sys

class SegmentationWorker(QtCore.QRunnable):
    def __init__(self, method, image_np, image_gaussian, n_segments, compactness, display_color):
        super().__init__()
        self.method = method
        self.image_np = image_np
        self.image_gaussian = image_gaussian
        self.n_segments = n_segments
        self.compactness = compactness
        self.display_color = display_color
        self.signals = WorkerSignals()

    def run(self):
        try:
            if self.method == "SLIC":
                segments = slic(self.image_gaussian, n_segments=self.n_segments, compactness=self.compactness, start_label=1, convert2lab=True)
                colorized = color.label2rgb(segments, self.image_gaussian, kind='avg')
                boundaries = mark_boundaries(self.image_gaussian, segments)
            else:
                labels = slic(self.image_gaussian, compactness=self.compactness, n_segments=self.n_segments, start_label=1, convert2lab=True)
                rag = graph.rag_mean_color(self.image_np, labels, mode='similarity')
                labels2 = graph.cut_normalized(labels, rag)
                colorized = color.label2rgb(labels2, self.image_np, kind='avg')
                boundaries = mark_boundaries(self.image_np, labels2)
                segments = labels2

            result_img = colorized if self.display_color else boundaries
            self.signals.result.emit((result_img, segments))

        except Exception as e:
            self.signals.error.emit((type(e).__name__, str(e)))
        finally:
            self.signals.finished.emit()

class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal()
    error = QtCore.Signal(tuple)
    result = QtCore.Signal(object)

class SegmentationApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentasi Gambar - SLIC & Normalized Cuts")

        self.result_image_np = None 
        self.image_np = None
        self.image_gaussian = None
        self.label_map = None
        self.method = None
        self.original_pixmap = None
        self.segmented_pixmap = None
        self.threadpool = QtCore.QThreadPool()

        # UI Components
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.original_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.segmented_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)

        # Set size policies so they expand equally
        self.original_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.segmented_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.load_button = QtWidgets.QPushButton("Load Image")
        self.segment_button = QtWidgets.QPushButton("Run Segmentation")
        self.segment_button.setEnabled(False)
        self.save_zip_button = QtWidgets.QPushButton("Save ZIP of Segments")
        self.save_zip_button.setEnabled(False)
        self.save_result_button = QtWidgets.QPushButton("Save Result Image")
        self.save_result_button.setEnabled(False)

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["SLIC", "Normalized Cuts"])

        self.segment_spin = QtWidgets.QSpinBox()
        self.segment_spin.setRange(10, 500)
        self.segment_spin.setValue(250)

        self.compact_spin = QtWidgets.QDoubleSpinBox()
        self.compact_spin.setRange(1.0, 50.0)
        self.compact_spin.setValue(10.0)

        # Radio Buttons
        self.display_radio = QtWidgets.QButtonGroup()
        self.radio_color = QtWidgets.QRadioButton("Colorized Segments")
        self.radio_boundary = QtWidgets.QRadioButton("Boundary Overlay")
        self.radio_color.setChecked(True)
        self.display_radio.addButton(self.radio_color)
        self.display_radio.addButton(self.radio_boundary)

        # Horizontal layout with label and radio buttons
        display_type_layout = QtWidgets.QHBoxLayout()
        display_type_layout.addWidget(QtWidgets.QLabel("Display Type:"))
        display_type_layout.addWidget(self.radio_color)
        display_type_layout.addWidget(self.radio_boundary)

        # Controls Layout - More Compact
        form_layout = QtWidgets.QGridLayout()
        form_layout.setHorizontalSpacing(10)
        form_layout.setVerticalSpacing(5)
        form_layout.setContentsMargins(5, 5, 5, 5)

        form_layout.addWidget(QtWidgets.QLabel("Method:"), 0, 0)
        form_layout.addWidget(self.method_combo, 0, 1)
        form_layout.addWidget(QtWidgets.QLabel("Segments:"), 1, 0)
        form_layout.addWidget(self.segment_spin, 1, 1)
        form_layout.addWidget(QtWidgets.QLabel("Compactness:"), 2, 0)
        form_layout.addWidget(self.compact_spin, 2, 1)

        radio_layout = QtWidgets.QHBoxLayout()
        radio_layout.addWidget(self.radio_color)
        radio_layout.addWidget(self.radio_boundary)
        form_layout.addLayout(display_type_layout, 3, 0, 1, 2)

        # Optional: wrap in group box for clean border
        control_group = QtWidgets.QGroupBox("Segmentation Settings")
        control_group.setLayout(form_layout)

        # Final Layout
        layout = QtWidgets.QVBoxLayout()
        image_layout = QtWidgets.QHBoxLayout()
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.segmented_label)

        layout.addLayout(image_layout)
        layout.addWidget(control_group)  # Use compact control group here

        # Button row layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(10)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.segment_button)
        button_layout.addWidget(self.save_result_button)
        button_layout.addWidget(self.save_zip_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Events
        self.load_button.clicked.connect(self.load_image)
        self.segment_button.clicked.connect(self.run_segmentation)
        self.save_zip_button.clicked.connect(self.save_segments_zip)
        self.save_result_button.clicked.connect(self.save_result_image)

    def np_to_qpixmap(self, img_np):
        """Convert NumPy RGB image to QPixmap"""
        img_np = (img_np * 255).astype(np.uint8) if img_np.dtype != np.uint8 else img_np
        h, w, ch = img_np.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(img_np.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def load_image(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            image = Image.open(file_name).convert('RGB')
            self.image_np = np.array(self.resize_image_keep_ratio(image))
            self.image_gaussian = gaussian(self.image_np, sigma=1, channel_axis=2)
            pixmap = self.np_to_qpixmap(self.image_np)
            self.original_label.setPixmap(pixmap.scaled(self.original_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.segmented_label.clear()
            self.save_zip_button.setEnabled(False)
            self.segment_button.setEnabled(True)
            self.original_pixmap = self.np_to_qpixmap(self.image_np)
            self.original_label.setPixmap(self.original_pixmap.scaled(
                self.original_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resize_image_keep_ratio(self, image: Image.Image, max_width=960) -> Image.Image:
        width, height = image.size
        if width > max_width:
            new_height = int((max_width / width) * height)
            return image.resize((max_width, new_height), Image.LANCZOS)
        return image

    def run_segmentation(self):
        if self.image_np is None or self.image_gaussian is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Please load an image first.")
            return

        method = self.method_combo.currentText()
        self.method = method
        n_segments = self.segment_spin.value()
        compactness = self.compact_spin.value()
        display_color = self.radio_color.isChecked()

        self.segment_button.setEnabled(False)
        self.save_zip_button.setEnabled(False)
        self.save_result_button.setEnabled(False)

        worker = SegmentationWorker(method, self.image_np, self.image_gaussian, n_segments, compactness, display_color)
        worker.signals.result.connect(self.on_segmentation_finished)
        worker.signals.error.connect(self.on_worker_error)
        worker.signals.finished.connect(self.on_worker_finished)

        self.threadpool.start(worker)

    def on_segmentation_finished(self, result):
        result_img, segments = result
        self.label_map = segments
        self.result_image_np = (result_img * 255).astype(np.uint8)
        self.segmented_pixmap = self.np_to_qpixmap(result_img)
        self.segmented_label.setPixmap(self.segmented_pixmap.scaled(
            self.segmented_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def on_worker_error(self, error_tuple):
        error_type, error_message = error_tuple
        QtWidgets.QMessageBox.critical(self, "Error", f"{error_type}: {error_message}")

    def on_worker_finished(self):
        self.segment_button.setEnabled(True)
        self.save_zip_button.setEnabled(True)
        self.save_result_button.setEnabled(True)

    def save_segments_zip(self):
        if self.label_map is None or self.image_np is None:
            return

        zip_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ZIP", f"{self.method}_segments.zip", "ZIP files (*.zip)")
        if not zip_path:
            return

        zip_buffer = BytesIO()
        unique_labels = np.unique(self.label_map)
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for idx, label_id in enumerate(unique_labels, start=1):
                mask = self.label_map == label_id
                segment = np.zeros_like(self.image_np)
                segment[mask] = self.image_np[mask]

                segment_pil = Image.fromarray(segment)
                img_bytes = BytesIO()
                segment_pil.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                zip_file.writestr(f"{self.method.lower()}_segment_{idx}.png", img_bytes.read())

        with open(zip_path, "wb") as f:
            f.write(zip_buffer.getvalue())
        QtWidgets.QMessageBox.information(self, "Saved", "Segments saved successfully!")

    def save_result_image(self):
        if self.result_image_np is None:
            return

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Result Image", "segmented_result.png", "PNG Files (*.png)")
        if not save_path:
            return

        img_to_save = Image.fromarray(self.result_image_np)
        img_to_save.save(save_path)
        QtWidgets.QMessageBox.information(self, "Saved", "Result image saved successfully.")

    def resizeEvent(self, event):
        if self.original_pixmap:
            self.original_label.setPixmap(self.original_pixmap.scaled(
                self.original_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        if self.segmented_pixmap:
            self.segmented_label.setPixmap(self.segmented_pixmap.scaled(
                self.segmented_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = SegmentationApp()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
