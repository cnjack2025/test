import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel, QSlider, QHBoxLayout, QScrollArea, QMessageBox, QStatusBar
from PyQt6.QtGui import QImage, QPixmap, QTransform, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QEvent
import pydicom
import pydicom.errors # Import for specific DICOM error
import numpy as np
import math

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Browser")
        self.setGeometry(100, 100, 1000, 700) # Increased window size

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main horizontal layout
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left side for controls and image
        self.left_v_layout = QVBoxLayout()
        self.open_button = QPushButton("Open DICOM File")
        self.open_button.clicked.connect(self.open_dicom_file)
        self.left_v_layout.addWidget(self.open_button)

        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.clicked.connect(self.reset_zoom_and_pan)
        self.left_v_layout.addWidget(self.reset_zoom_button)

        # Transformation buttons
        self.rotate_cw_button = QPushButton("Rotate 90° CW")
        self.rotate_cw_button.clicked.connect(self.rotate_image_cw)
        self.left_v_layout.addWidget(self.rotate_cw_button)

        self.rotate_ccw_button = QPushButton("Rotate 90° CCW")
        self.rotate_ccw_button.clicked.connect(self.rotate_image_ccw)
        self.left_v_layout.addWidget(self.rotate_ccw_button)

        self.flip_horizontal_button = QPushButton("Flip Horizontal")
        self.flip_horizontal_button.clicked.connect(self.flip_image_horizontal)
        self.left_v_layout.addWidget(self.flip_horizontal_button)

        self.flip_vertical_button = QPushButton("Flip Vertical")
        self.flip_vertical_button.clicked.connect(self.flip_image_vertical)
        self.left_v_layout.addWidget(self.flip_vertical_button)

        # Measurement tool
        self.measure_button = QPushButton("Measure Line")
        self.measure_button.clicked.connect(self.toggle_measure_mode)
        self.left_v_layout.addWidget(self.measure_button)

        self.measurement_label = QLabel("Length: N/A")
        self.left_v_layout.addWidget(self.measurement_label)
        
        # Instance variables for windowing & zoom
        self.raw_pixel_array = None
        self.photometric_interpretation = "MONOCHROME2" # Default
        self.window_level = 0
        self.window_width = 0
        self.zoom_scale = 1.0
        self.base_q_image = None # This will store the QImage after windowing/photometric interpretation
        self.post_windowing_numpy_array = None # Store numpy array after windowing for reset
        
        # Measurement state
        self.is_measuring = False
        self.measurement_points = [] # Stores points in base_q_image coordinates
        self.measurement_distance = 0.0

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Events handled by MainWindow/ScrollArea
        self.image_label.installEventFilter(self) # For mouse events for measurement
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)
        
        self.left_v_layout.addWidget(self.scroll_area) # Add scroll_area instead of image_label
        self.main_layout.addLayout(self.left_v_layout, 2) # Give more stretch factor to image area

        # Right side for metadata and windowing controls
        self.metadata_v_layout = QVBoxLayout()
        self.patient_id_label = QLabel("Patient ID: N/A")
        self.patient_name_label = QLabel("Patient Name: N/A")
        self.study_date_label = QLabel("Study Date: N/A")
        self.modality_label = QLabel("Modality: N/A")

        self.metadata_v_layout.addWidget(self.patient_id_label)
        self.metadata_v_layout.addWidget(self.patient_name_label)
        self.metadata_v_layout.addWidget(self.study_date_label)
        self.metadata_v_layout.addWidget(self.modality_label)
        self.metadata_v_layout.addStretch() # Pushes metadata to the top
        
        # Windowing controls
        self.wl_label = QLabel("Window Level: 0")
        self.metadata_v_layout.addWidget(self.wl_label)
        self.wl_slider = QSlider(Qt.Orientation.Horizontal)
        self.wl_slider.setRange(0, 4095) # Default range, will be updated
        self.wl_slider.valueChanged.connect(self.update_windowing_and_redisplay)
        self.metadata_v_layout.addWidget(self.wl_slider)

        self.ww_label = QLabel("Window Width: 0")
        self.metadata_v_layout.addWidget(self.ww_label)
        self.ww_slider = QSlider(Qt.Orientation.Horizontal)
        self.ww_slider.setRange(1, 4095) # Default range, will be updated
        self.ww_slider.valueChanged.connect(self.update_windowing_and_redisplay)
        self.metadata_v_layout.addWidget(self.ww_slider)
        
        self.main_layout.addLayout(self.metadata_v_layout, 1) # Less stretch factor for metadata

        self.update_image_action_buttons_state(False) # Disable buttons initially

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Open a DICOM file to begin.")

    def update_windowing_and_redisplay(self):
        if self.raw_pixel_array is not None:
            self.window_level = self.wl_slider.value()
            self.window_width = self.ww_slider.value()
            self.wl_label.setText(f"Window Level: {self.window_level}")
            self.ww_label.setText(f"Window Width: {self.window_width}")
            self.display_image()

    def display_image(self): # Signature changed
        if self.raw_pixel_array is None:
            self.image_label.setText("No image loaded")
            self.image_label.setPixmap(QPixmap()) # Clear pixmap
            self.base_q_image = None
            self.post_windowing_numpy_array = None
            self.update_image_action_buttons_state(False)
            return

        pixel_array_windowed = self.raw_pixel_array.copy()
        
        # Windowing Logic (operates on pixel_array_windowed)
        level = self.window_level
        width = self.window_width
        ymin = 0
        ymax = 255
        low = level - width / 2.0
        high = level + width / 2.0
        
        pixel_array_windowed[pixel_array_windowed <= low] = ymin
        pixel_array_windowed[pixel_array_windowed > high] = ymax
        
        condition = (pixel_array_windowed > low) & (pixel_array_windowed <= high)
        if width > 0: 
            pixel_array_windowed[condition] = ((pixel_array_windowed[condition] - low) / width) * (ymax - ymin) + ymin
        else: 
            pixel_array_windowed[condition] = ymin

        if self.photometric_interpretation == "MONOCHROME1":
            pixel_array_windowed = ymax - pixel_array_windowed

        image_scaled_8bit_numpy_array = pixel_array_windowed.astype(np.uint8)
        
        # Store the post-windowed numpy array
        self.post_windowing_numpy_array = np.ascontiguousarray(image_scaled_8bit_numpy_array.copy())
        
        # Create base_q_image from this stored numpy array
        h, w = self.post_windowing_numpy_array.shape[:2]
        self.base_q_image = QImage(self.post_windowing_numpy_array.data, w, h, w, QImage.Format.Format_Grayscale8)
        
        self.update_image_action_buttons_state(True) # Enable buttons
        
        # Clear measurements when image content changes (e.g. windowing)
        self.clear_measurement_state(update_display=False) 
        
        self.update_display_with_potential_measurement() # Display the image
        # Status bar already updated in open_dicom_file for initial load,
        # and update_display_with_potential_measurement will update it for display changes.

    # update_zoomed_image is now effectively replaced by update_display_with_potential_measurement
    # def update_zoomed_image(self): ... (can be removed)

    def wheelEvent(self, event):
        if self.scroll_area.underMouse() and self.base_q_image is not None:
            delta = event.angleDelta().y()
            zoom_factor = 1.15 # Slightly more aggressive zoom
            min_scale = 0.1
            max_scale = 10.0

            if delta > 0: # Zoom in
                self.zoom_scale = min(max_scale, self.zoom_scale * zoom_factor)
            elif delta < 0: # Zoom out
                self.zoom_scale = max(min_scale, self.zoom_scale / zoom_factor)
            
            self.update_display_with_potential_measurement() # Changed from update_zoomed_image
            event.accept() 
        else:
            event.ignore()

    def clear_measurement_state(self, update_display=True):
        self.measurement_points = []
        self.measurement_distance = 0.0
        self.measurement_label.setText("Length: N/A")
        if self.is_measuring: # If was actively measuring, turn it off
            self.is_measuring = False
            self.measure_button.setText("Measure Line")
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            if update_display: # Only update status if this was a cancel action
                 self.status_bar.showMessage("Measurement cancelled.")
        if update_display:
            self.update_display_with_potential_measurement()


    def toggle_measure_mode(self):
        if not self.base_q_image: 
            self.is_measuring = False 
            self.measure_button.setText("Measure Line")
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.measurement_label.setText("Length: N/A")
            self.status_bar.showMessage("No image loaded. Open a DICOM file to measure.")
            return

        self.is_measuring = not self.is_measuring
        if self.is_measuring:
            self.measurement_points = []
            self.measurement_distance = 0.0
            self.measure_button.setText("Cancel Measurement")
            self.measurement_label.setText("Click first point...")
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.status_bar.showMessage("Measurement mode: Click first point.")
        else: 
            self.measure_button.setText("Measure Line")
            if self.measurement_distance > 0 and len(self.measurement_points) == 2 :
                 self.measurement_label.setText(f"Length: {self.measurement_distance:.2f} pixels")
                 # Status will be updated by update_display_with_potential_measurement
            else:
                 self.measurement_label.setText("Length: N/A (Cancelled)")
                 self.status_bar.showMessage("Measurement cancelled.") # Explicitly set for cancel
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
        
        self.update_display_with_potential_measurement()

    def eventFilter(self, source, event):
        if source is self.image_label and event.type() == QEvent.Type.MouseButtonPress:
            if self.is_measuring and self.base_q_image is not None:
                label_pos = event.pos()
                
                # Transform label_pos to base_q_image coordinates
                # Adjust for scrollbar position first, then for zoom
                # Note: self.image_label.pos() might be (0,0) if it's the direct widget of scroll_area
                # and scroll_area.widgetResizable(True)
                
                # Calculate the position on the original base_q_image
                # The event.pos() is relative to the (potentially scaled) image_label widget
                # We need to map this back to the base_q_image coordinates
                
                # Get the current size of the displayed pixmap on the label
                current_pixmap_size = self.image_label.pixmap().size()
                if current_pixmap_size.width() == 0 or current_pixmap_size.height() == 0:
                    return super().eventFilter(source, event) # No pixmap to click on

                # Calculate the actual displayed image's top-left corner within the label due to alignment
                label_width = self.image_label.width()
                label_height = self.image_label.height()
                
                pixmap_x_offset = (label_width - current_pixmap_size.width()) / 2
                pixmap_y_offset = (label_height - current_pixmap_size.height()) / 2

                # Point relative to the top-left of the actual displayed pixmap
                x_on_scaled_pixmap = label_pos.x() - pixmap_x_offset
                y_on_scaled_pixmap = label_pos.y() - pixmap_y_offset

                if x_on_scaled_pixmap < 0 or x_on_scaled_pixmap > current_pixmap_size.width() or \
                   y_on_scaled_pixmap < 0 or y_on_scaled_pixmap > current_pixmap_size.height():
                    return super().eventFilter(source, event) # Click was outside the pixmap

                # Convert to coordinates on the base_q_image
                img_x = x_on_scaled_pixmap / self.zoom_scale
                img_y = y_on_scaled_pixmap / self.zoom_scale
                
                # Ensure coordinates are within base_q_image bounds
                img_x = max(0, min(img_x, self.base_q_image.width() - 1))
                img_y = max(0, min(img_y, self.base_q_image.height() - 1))
                
                self.measurement_points.append((img_x, img_y))
                
                if len(self.measurement_points) == 1:
                    self.measurement_label.setText("Click second point...")
                    self.status_bar.showMessage("Measurement mode: Click second point.")
                elif len(self.measurement_points) == 2:
                    p1 = self.measurement_points[0]
                    p2 = self.measurement_points[1]
                    self.measurement_distance = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                    self.measurement_label.setText(f"Length: {self.measurement_distance:.2f} pixels")
                    # Status bar will be updated by the subsequent call to update_display_with_potential_measurement
                    self.is_measuring = False 
                    self.measure_button.setText("Measure Line")
                    self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
                
                self.update_display_with_potential_measurement()
                return True 
        
        return super().eventFilter(source, event) 

    def update_display_with_potential_measurement(self):
        if self.base_q_image is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("No image loaded or error in processing.")
            self.status_bar.showMessage("No image loaded. Open a DICOM file.") # Step 5
            return
        
        # Update status bar with current image info (Step 4)
        dims = self.base_q_image.size() # QImage size after rotations/flips
        status_message = f"Dimensions: {dims.width()}x{dims.height()} | Zoom: {self.zoom_scale*100:.0f}%"
        if self.is_measuring:
            if not self.measurement_points:
                status_message += " | Measurement: Click first point."
            elif len(self.measurement_points) == 1:
                status_message += " | Measurement: Click second point."
        elif len(self.measurement_points) == 2 and self.measurement_distance > 0: # Just finished a measurement
             status_message += f" | Last Measured: {self.measurement_distance:.2f} pixels."

        self.status_bar.showMessage(status_message)

        pixmap_to_display = QPixmap.fromImage(self.base_q_image)
        
        if self.measurement_points:
            painter = QPainter(pixmap_to_display)
            # Pen width should be constant on screen, so scale it inversely to zoom for drawing on base_q_image
            pen_width_on_screen = 2 # Desired pen width on screen in pixels
            pen_width_on_base_image = pen_width_on_screen / self.zoom_scale 
            pen = QPen(QColor("red"), pen_width_on_base_image) 
            painter.setPen(pen)
            
            for p_idx, p in enumerate(self.measurement_points):
                # Draw a small circle/ellipse for the point
                # The size of the circle should also be somewhat consistent on screen
                radius_on_screen = 3
                radius_on_base_image = radius_on_screen / self.zoom_scale
                painter.drawEllipse(int(p[0] - radius_on_base_image), int(p[1] - radius_on_base_image), 
                                    int(2 * radius_on_base_image), int(2 * radius_on_base_image))
                if p_idx == 0 and len(self.measurement_points) == 1 and self.is_measuring:
                    painter.drawText(int(p[0] + 5 / self.zoom_scale), int(p[1] + 5 / self.zoom_scale), "P1")

            if len(self.measurement_points) == 2:
                p1 = self.measurement_points[0]
                p2 = self.measurement_points[1]
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
                
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                # Adjust text offset based on zoom for consistent appearance
                text_offset_on_screen = 5
                text_offset_on_base_image = text_offset_on_screen / self.zoom_scale
                painter.drawText(int(mid_x), int(mid_y) - int(text_offset_on_base_image), f"{self.measurement_distance:.2f} px")
            painter.end()

        current_width = int(pixmap_to_display.width() * self.zoom_scale)
        current_height = int(pixmap_to_display.height() * self.zoom_scale)
        current_width = max(1, current_width)
        current_height = max(1, current_height)
        
        scaled_pixmap = pixmap_to_display.scaled(current_width, current_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        # self.image_label.adjustSize() # This might not be needed if scroll_area.widgetResizable is True and label alignment handles it

    def rotate_image_cw(self):
        if self.base_q_image is None: return
        self.clear_measurement_state(update_display=False)
        transform = QTransform().rotate(90)
        self.base_q_image = self.base_q_image.transformed(transform, Qt.TransformationMode.SmoothTransformation)
        # Also need to transform post_windowing_numpy_array if we want reset to work with rotations
        # For now, reset_zoom_and_pan will revert to original orientation from post_windowing_numpy_array
        self.update_display_with_potential_measurement()

    def rotate_image_ccw(self):
        if self.base_q_image is None: return
        self.clear_measurement_state(update_display=False)
        transform = QTransform().rotate(-90)
        self.base_q_image = self.base_q_image.transformed(transform, Qt.TransformationMode.SmoothTransformation)
        self.update_display_with_potential_measurement()

    def flip_image_horizontal(self):
        if self.base_q_image is None: return
        self.clear_measurement_state(update_display=False)
        self.base_q_image = self.base_q_image.mirrored(True, False)
        self.update_display_with_potential_measurement()

    def flip_image_vertical(self):
        if self.base_q_image is None: return
        self.clear_measurement_state(update_display=False)
        self.base_q_image = self.base_q_image.mirrored(False, True)
        self.update_display_with_potential_measurement()

    def reset_zoom_and_pan(self):
        self.zoom_scale = 1.0
        if self.post_windowing_numpy_array is not None:
            h, w = self.post_windowing_numpy_array.shape[:2]
            contiguous_array = np.ascontiguousarray(self.post_windowing_numpy_array)
            self.base_q_image = QImage(contiguous_array.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            self.base_q_image = None
        
        self.clear_measurement_state(update_display=False) # Clear measurements on reset
        self.update_display_with_potential_measurement()
        # Reset scrollbar positions
        self.scroll_area.horizontalScrollBar().setValue(0)
        self.scroll_area.verticalScrollBar().setValue(0)
    
    def open_dicom_file(self): 
        fname, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "DICOM Files (*.dcm);;All Files (*)")
        if fname:
            try:
                ds = pydicom.dcmread(fname)
                
                # Verify essential attributes for display
                if not hasattr(ds, 'PixelData') or ds.pixel_array is None:
                    raise pydicom.errors.InvalidDicomError("File does not contain pixel data.")

                self.zoom_scale = 1.0 
                self.base_q_image = None 
                self.raw_pixel_array = ds.pixel_array.astype(np.float32)
                self.photometric_interpretation = ds.get("PhotometricInterpretation", "MONOCHROME2")
                self.post_windowing_numpy_array = None 
                
                self.clear_measurement_state(update_display=False) 

                patient_id = str(ds.get("PatientID", "N/A"))
                patient_name = str(ds.get("PatientName", "N/A"))
                study_date = str(ds.get("StudyDate", "N/A"))
                modality = str(ds.get("Modality", "N/A"))

                self.patient_id_label.setText(f"Patient ID: {patient_id}")
                self.patient_name_label.setText(f"Patient Name: {patient_name}")
                self.study_date_label.setText(f"Study Date: {study_date}")
                self.modality_label.setText(f"Modality: {modality}")
                
                # Status bar message after basic properties are read
                self.status_bar.showMessage(f"Loaded: {fname.split('/')[-1]} | Original Dims: {self.raw_pixel_array.shape[1]}x{self.raw_pixel_array.shape[0]}")

                # Get default WL/WW from DICOM tags or calculate
                wc_val = ds.get("WindowCenter")
                ww_val = ds.get("WindowWidth")

                if isinstance(wc_val, pydicom.multival.MultiValue):
                    self.window_level = float(wc_val[0]) if wc_val else self.raw_pixel_array.mean()
                elif wc_val is not None:
                    self.window_level = float(wc_val)
                else: 
                    self.window_level = self.raw_pixel_array.mean()

                if isinstance(ww_val, pydicom.multival.MultiValue):
                    self.window_width = float(ww_val[0]) if ww_val else (self.raw_pixel_array.max() - self.raw_pixel_array.min())
                elif ww_val is not None:
                    self.window_width = float(ww_val)
                else: 
                    self.window_width = self.raw_pixel_array.max() - self.raw_pixel_array.min()
                
                if self.window_width < 1.0: 
                    self.window_width = np.max([1.0, (self.raw_pixel_array.max() - self.raw_pixel_array.min())])

                min_val = np.min(self.raw_pixel_array)
                max_val = np.max(self.raw_pixel_array)
                
                self.wl_slider.blockSignals(True)
                self.ww_slider.blockSignals(True)
                
                self.wl_slider.setRange(int(min_val), int(max_val))
                self.ww_slider.setRange(1, int(max_val - min_val if (max_val - min_val) >=1 else 1))
                
                self.wl_slider.setValue(int(self.window_level))
                self.ww_slider.setValue(int(self.window_width))

                self.wl_slider.blockSignals(False)
                self.ww_slider.blockSignals(False)

                self.wl_label.setText(f"Window Level: {self.window_level}")
                self.ww_label.setText(f"Window Width: {self.window_width}")
                
                self.display_image() 
                self.image_label.setText("") 
                self.update_image_action_buttons_state(True)
                # The status message in open_dicom_file shows original dims, 
                # update_display_with_potential_measurement will show current (possibly transformed) dims.
                
            except pydicom.errors.InvalidDicomError as e:
                QMessageBox.critical(self, "Error Loading DICOM", f"The file could not be read as a valid DICOM file.\nError: {str(e)}")
                self.status_bar.showMessage("Failed to load DICOM file. Invalid format.") # Step 3
                self.image_label.setText("Failed to load: Invalid DICOM")
                self.image_label.setPixmap(QPixmap()) 
                self.patient_id_label.setText("Patient ID: N/A")
                self.patient_name_label.setText("Patient Name: N/A")
                self.study_date_label.setText("Study Date: N/A")
                self.modality_label.setText("Modality: N/A")
                self.raw_pixel_array = None 
                self.base_q_image = None 
                self.post_windowing_numpy_array = None 
                self.update_image_action_buttons_state(False) 
                self.clear_measurement_state(update_display=False) 
                self.wl_label.setText("Window Level: 0")
                self.ww_label.setText("Window Width: 0")
                self.wl_slider.setRange(0,100)
                self.ww_slider.setRange(1,100)
                self.wl_slider.setValue(0)
                self.ww_slider.setValue(1)
            except Exception as e:
                QMessageBox.critical(self, "Application Error", f"An unexpected error occurred while opening the file.\nError: {str(e)}")
                self.status_bar.showMessage("An unexpected error occurred.")
                self.image_label.setText("Failed to load: Application error")
                self.image_label.setPixmap(QPixmap()) 
                self.patient_id_label.setText("Patient ID: N/A")
                self.patient_name_label.setText("Patient Name: N/A")
                self.study_date_label.setText("Study Date: N/A")
                self.modality_label.setText("Modality: N/A")
                self.raw_pixel_array = None 
                self.base_q_image = None 
                self.post_windowing_numpy_array = None 
                self.update_image_action_buttons_state(False) 
                self.clear_measurement_state(update_display=False)
                self.wl_label.setText("Window Level: 0")
                self.ww_label.setText("Window Width: 0")
                self.wl_slider.setRange(0,100)
                self.ww_slider.setRange(1,100)
                self.wl_slider.setValue(0)
                self.ww_slider.setValue(1)

    def update_image_action_buttons_state(self, enabled):
        self.rotate_cw_button.setEnabled(enabled)
        self.rotate_ccw_button.setEnabled(enabled)
        self.flip_horizontal_button.setEnabled(enabled)
        self.flip_vertical_button.setEnabled(enabled)
        self.measure_button.setEnabled(enabled)

    def open_dicom_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "DICOM Files (*.dcm)")
        if file_name:
            try:
                ds = pydicom.dcmread(file_name)
                
                self.zoom_scale = 1.0 # Reset zoom
                self.base_q_image = None # Reset base image
                self.raw_pixel_array = ds.pixel_array.astype(np.float32)
                self.photometric_interpretation = ds.get("PhotometricInterpretation", "MONOCHROME2")

                # Extract and display DICOM metadata
                patient_id = str(ds.get("PatientID", "N/A"))
                patient_name = str(ds.get("PatientName", "N/A"))
                study_date = str(ds.get("StudyDate", "N/A"))
                modality = str(ds.get("Modality", "N/A"))

                self.patient_id_label.setText(f"Patient ID: {patient_id}")
                self.patient_name_label.setText(f"Patient Name: {patient_name}")
                self.study_date_label.setText(f"Study Date: {study_date}")
                self.modality_label.setText(f"Modality: {modality}")

                # Get default WL/WW from DICOM tags or calculate
                # Get default WL/WW from DICOM tags or calculate
                # WindowCenter (0028,1050) and WindowWidth (0028,1051)
                wc_val = ds.get("WindowCenter")
                ww_val = ds.get("WindowWidth")

                if isinstance(wc_val, pydicom.multival.MultiValue):
                    self.window_level = float(wc_val[0]) if wc_val else self.raw_pixel_array.mean()
                elif wc_val is not None:
                    self.window_level = float(wc_val)
                else: # Tag missing or None
                    self.window_level = self.raw_pixel_array.mean()

                if isinstance(ww_val, pydicom.multival.MultiValue):
                    self.window_width = float(ww_val[0]) if ww_val else (self.raw_pixel_array.max() - self.raw_pixel_array.min())
                elif ww_val is not None:
                    self.window_width = float(ww_val)
                else: # Tag missing or None
                    self.window_width = self.raw_pixel_array.max() - self.raw_pixel_array.min()
                
                if self.window_width < 1.0: # Ensure width is at least 1
                    self.window_width = np.max([1.0, (self.raw_pixel_array.max() - self.raw_pixel_array.min())])


                # Update slider ranges and values
                min_val = np.min(self.raw_pixel_array)
                max_val = np.max(self.raw_pixel_array)
                
                # Block signals while updating sliders to prevent premature display_image calls
                self.wl_slider.blockSignals(True)
                self.ww_slider.blockSignals(True)
                
                self.wl_slider.setRange(int(min_val), int(max_val))
                self.ww_slider.setRange(1, int(max_val - min_val if (max_val - min_val) >=1 else 1))
                
                self.wl_slider.setValue(int(self.window_level))
                self.ww_slider.setValue(int(self.window_width))

                self.wl_slider.blockSignals(False)
                self.ww_slider.blockSignals(False)

                self.wl_label.setText(f"Window Level: {self.window_level}")
                self.ww_label.setText(f"Window Width: {self.window_width}")
                
                self.display_image() # Call display_image which now uses instance variables
                self.image_label.setText("") # Clear any previous error message
                self.update_image_action_buttons_state(True) # Enable buttons on successful load
                
            except Exception as e:
                print(f"Error reading DICOM file: {e}")
                self.image_label.setText("Failed to load image or not a DICOM file")
                self.image_label.setPixmap(QPixmap()) # Clear image
                self.patient_id_label.setText("Patient ID: N/A")
                self.patient_name_label.setText("Patient Name: N/A")
                self.study_date_label.setText("Study Date: N/A")
                self.modality_label.setText("Modality: N/A")
                self.raw_pixel_array = None 
                self.base_q_image = None 
                self.post_windowing_numpy_array = None # Clear this on error
                self.update_image_action_buttons_state(False) # Disable buttons on error
                self.wl_label.setText("Window Level: 0")
                self.ww_label.setText("Window Width: 0")
                self.wl_slider.setRange(0,100)
                self.ww_slider.setRange(1,100)
                self.wl_slider.setValue(0)
                self.ww_slider.setValue(1)

# Main entry point for the DICOM browser application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
