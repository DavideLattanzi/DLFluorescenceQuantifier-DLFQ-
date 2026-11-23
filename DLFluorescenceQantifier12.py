import sys
import json
import pickle
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from skimage import io, filters, morphology, segmentation, feature, measure
from scipy import ndimage as ndi
from readlif.reader import LifFile
from sklearn.ensemble import RandomForestClassifier
from openpyxl.styles import Font, PatternFill, Alignment

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QCheckBox, QSpinBox,
    QGroupBox, QMessageBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QSlider, QComboBox, QDialog, QLineEdit,
    QDialogButtonBox, QScrollArea, QShortcut, QTabWidget, QTextBrowser,
    QTextEdit, QFrame
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QKeySequence

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import EllipseSelector, PolygonSelector
from matplotlib.path import Path as MplPath


# =============================================================================
# DEFAULT CREDITS & WELCOME SCREEN CONFIGURATION (EDITABLE ALSO FROM UI)
# =============================================================================
DEFAULT_DEVELOPER_NAME = "Your Name"
DEFAULT_COLLABORATORS = (
    "• Collaborator 1 (Affiliation / Role)\n"
    "• Collaborator 2 (Affiliation / Role)"
)
WELCOME_CONFIG_PATH = Path.home() / ".dlfluoresencequantifier_welcome.json"


# =============================================================================
# WELCOME / PRESENTATION SCREEN DIALOG (FULLSCREEN, REPRESENTATIVE IMAGE ONLY)
# =============================================================================
class WelcomeScreenDialog(QDialog):
    """Initial fullscreen presentation screen with Developer/Collaborators, Key Features, and a single representative image."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DLFluoresenceQuantifier — Welcome & Overview")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)

        self.config = self.load_welcome_config()
        self.overview_pixmap = None

        self.init_ui()
        self.load_initial_overview_image()

    def load_welcome_config(self):
        if WELCOME_CONFIG_PATH.exists():
            try:
                with open(WELCOME_CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "developer": DEFAULT_DEVELOPER_NAME,
            "collaborators": DEFAULT_COLLABORATORS,
            "overview_image_path": "",
        }

    def save_welcome_config(self):
        try:
            self.config["developer"] = self.edit_developer.text().strip() or DEFAULT_DEVELOPER_NAME
            self.config["collaborators"] = self.edit_collaborators.toPlainText().strip()
            with open(WELCOME_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        # Top Banner
        banner = QFrame()
        banner.setStyleSheet("""
            QFrame {
                background-color: #1f4e78;
                border-radius: 6px;
                padding: 8px 14px;
            }
        """)
        banner_layout = QHBoxLayout(banner)
        lbl_title = QLabel("DLFluoresenceQuantifier")
        lbl_title.setStyleSheet("color: white; font-size: 22px; font-weight: bold;")
        lbl_sub = QLabel("Quantitative Microscopy, Morpho-Template Nuclei AI & Colocalization Suite")
        lbl_sub.setStyleSheet("color: #cfd8dc; font-size: 13.5px;")
        banner_layout.addWidget(lbl_title)
        banner_layout.addStretch()
        banner_layout.addWidget(lbl_sub)
        root_layout.addWidget(banner)

        # Main Body
        body_layout = QHBoxLayout()
        body_layout.setSpacing(12)
        root_layout.addLayout(body_layout, stretch=1)

        # ---------------------------------------------------------------------
        # LEFT COLUMN: DEVELOPER, COLLABORATORS & KEY FEATURES
        # ---------------------------------------------------------------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(395)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body_layout.addWidget(left_scroll)

        left_card = QWidget()
        left_card.setStyleSheet("""
            QWidget {
                background-color: #f8fafc;
                font-size: 12px;
            }
            QGroupBox {
                font-size: 12.5px;
                font-weight: bold;
                color: #1f4e78;
                border: 1px solid #cfd8dc;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        left_col = QVBoxLayout(left_card)
        left_col.setContentsMargins(8, 8, 8, 8)
        left_col.setSpacing(10)
        left_scroll.setWidget(left_card)

        grp_dev = QGroupBox("Lead Developer")
        dev_layout = QVBoxLayout()
        self.edit_developer = QLineEdit()
        self.edit_developer.setText(self.config.get("developer", DEFAULT_DEVELOPER_NAME))
        self.edit_developer.setPlaceholderText("Enter Developer Name...")
        self.edit_developer.setStyleSheet(
            "font-size: 13.5px; font-weight: bold; color: #0d47a1; padding: 6px; "
            "border: 1px solid #b0bec5; border-radius: 4px; background-color: #f1f8ff;"
        )
        self.edit_developer.textChanged.connect(self.save_welcome_config)
        dev_layout.addWidget(self.edit_developer)
        grp_dev.setLayout(dev_layout)
        left_col.addWidget(grp_dev)

        grp_collab = QGroupBox("Collaborators & Contributors")
        collab_layout = QVBoxLayout()
        self.edit_collaborators = QTextEdit()
        self.edit_collaborators.setPlainText(self.config.get("collaborators", DEFAULT_COLLABORATORS))
        self.edit_collaborators.setPlaceholderText("Add collaborator names and affiliations...")
        self.edit_collaborators.setFixedHeight(115)
        self.edit_collaborators.setStyleSheet(
            "font-size: 12px; color: #263238; padding: 5px; "
            "border: 1px solid #cfd8dc; border-radius: 4px; background-color: #ffffff;"
        )
        self.edit_collaborators.textChanged.connect(self.save_welcome_config)
        collab_layout.addWidget(self.edit_collaborators)
        grp_collab.setLayout(collab_layout)
        left_col.addWidget(grp_collab)

        grp_feat = QGroupBox("Main Software Features")
        feat_layout = QVBoxLayout()
        feat_browser = QTextBrowser()
        feat_browser.setFrameShape(QFrame.NoFrame)
        feat_browser.setStyleSheet("background-color: #ffffff; font-size: 12px; line-height: 1.45;")
        feat_browser.setHtml("""
        <ul style="margin-left: -16px; margin-top: 2px;">
            <li style="margin-bottom: 6px;"><b>Multi-Format Microscopy Import:</b> Direct extraction of Leica <code>.lif</code> series, multichannel stacks (8/12/16-bit), and separate single-channel merges (Fluorescence + Brightfield).</li>
            <li style="margin-bottom: 6px;"><b>Raw Bit-Depth Quantification:</b> Computes Area %, Mean Intensity, Integrated Density (IntDen), and CTCF on untouched raw detector values with automatic saturation alerts.</li>
            <li style="margin-bottom: 6px;"><b>Spatial ROIs &amp; Artifact Exclusion:</b> Interactive Circle/Ellipse and Freehand tools to target specific regions and exclude debris or folds.</li>
            <li style="margin-bottom: 6px;"><b>Morpho-Template Nuclei AI &amp; Watershed:</b> Learns nuclear chromatin granularity, Hessian blob geometry, and visual templates (Fluorescence &amp; Brightfield) with Area and Circularity filters.</li>
            <li style="margin-bottom: 6px;"><b>Independent Per-Channel Thresholds:</b> Separate locked thresholds for Red, Green, and Blue channels with real-time Magenta overlay preview.</li>
            <li style="margin-bottom: 6px;"><b>Colocalization &amp; Double-Labeling:</b> Computes Double-Positive Area %, Manders M1/M2 coefficients, Pearson's <i>r</i>, and double-positive nuclei counts with Cyan preview.</li>
            <li style="margin-bottom: 2px;"><b>Session Persistence &amp; Excel Reporting:</b> Full project save/reload (<code>.cfq</code>), reusable AI models (<code>.aim</code>), and formatted multi-sheet Excel export (<code>.xlsx</code>).</li>
        </ul>
        """)
        feat_layout.addWidget(feat_browser)
        grp_feat.setLayout(feat_layout)
        left_col.addWidget(grp_feat, stretch=1)

        # ---------------------------------------------------------------------
        # CENTER AREA: REPRESENTATIVE IMAGE ONLY
        # ---------------------------------------------------------------------
        center_box = QGroupBox("Representative Overview Image")
        center_box.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #1f4e78;
                border: 1px solid #cfd8dc;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        center_layout = QVBoxLayout(center_box)
        center_layout.setSpacing(8)

        img_bar = QHBoxLayout()
        self.lbl_img_info = QLabel("No representative image loaded yet.")
        self.lbl_img_info.setStyleSheet("color: #546e7a; font-size: 12px; font-weight: normal;")

        btn_load_img = QPushButton("Load Representative Image...")
        btn_load_img.setStyleSheet("background-color: #00695c; color: white; font-weight: bold; padding: 6px 14px;")
        btn_load_img.clicked.connect(self.browse_overview_image)

        btn_clear_img = QPushButton("Clear Image")
        btn_clear_img.setStyleSheet("padding: 6px 12px;")
        btn_clear_img.clicked.connect(self.clear_overview_image)

        img_bar.addWidget(self.lbl_img_info)
        img_bar.addStretch()
        img_bar.addWidget(btn_load_img)
        img_bar.addWidget(btn_clear_img)
        center_layout.addLayout(img_bar)

        self.lbl_overview_canvas = QLabel("Click 'Load Representative Image...' above to display your figure.")
        self.lbl_overview_canvas.setAlignment(Qt.AlignCenter)
        self.lbl_overview_canvas.setMinimumSize(640, 480)
        self.lbl_overview_canvas.setStyleSheet(
            "background-color: #161b22; color: #90a4ae; font-size: 14px; "
            "border: 1px solid #b0bec5; border-radius: 6px;"
        )
        center_layout.addWidget(self.lbl_overview_canvas, stretch=1)

        body_layout.addWidget(center_box, stretch=1)

        # ---------------------------------------------------------------------
        # BOTTOM ACTION BAR
        # ---------------------------------------------------------------------
        bottom_bar = QHBoxLayout()

        btn_open_guide = QPushButton("Open User Guide && Parameter Manual")
        btn_open_guide.setStyleSheet(
            "background-color: #1565c0; color: white; font-weight: bold; font-size: 12.5px; padding: 8px 16px;"
        )
        btn_open_guide.clicked.connect(self.open_guide)

        btn_start = QPushButton("Launch DLFluoresenceQuantifier  ->")
        btn_start.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; font-size: 14px; padding: 9px 26px; border-radius: 5px;"
        )
        btn_start.clicked.connect(self.accept_and_launch)

        bottom_bar.addWidget(btn_open_guide)
        bottom_bar.addStretch()
        bottom_bar.addWidget(btn_start)
        root_layout.addLayout(bottom_bar)

    def load_initial_overview_image(self):
        saved_img = self.config.get("overview_image_path", "")
        if saved_img and Path(saved_img).exists():
            pix = QPixmap(saved_img)
            if not pix.isNull():
                self.overview_pixmap = pix
                self.lbl_img_info.setText(f"Representative image: {Path(saved_img).name}")
                self.refresh_overview_pixmap()
                return

        self.overview_pixmap = None
        self.lbl_overview_canvas.clear()
        self.lbl_overview_canvas.setText("Click 'Load Representative Image...' above to display your figure.")

    def browse_overview_image(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Select Representative Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        if not fp:
            return
        pix = QPixmap(fp)
        if pix.isNull():
            QMessageBox.warning(self, "Invalid Image", "Could not load the selected image file.")
            return
        self.overview_pixmap = pix
        self.config["overview_image_path"] = fp
        self.save_welcome_config()
        self.lbl_img_info.setText(f"Representative image: {Path(fp).name}")
        self.refresh_overview_pixmap()

    def clear_overview_image(self):
        self.config["overview_image_path"] = ""
        self.save_welcome_config()
        self.overview_pixmap = None
        self.lbl_overview_canvas.clear()
        self.lbl_overview_canvas.setText("Click 'Load Representative Image...' above to display your figure.")
        self.lbl_img_info.setText("No representative image loaded yet.")

    def refresh_overview_pixmap(self):
        if self.overview_pixmap is None or self.overview_pixmap.isNull():
            return
        target_size = self.lbl_overview_canvas.size()
        if target_size.width() < 50 or target_size.height() < 50:
            target_size = QSize(1100, 720)
        scaled = self.overview_pixmap.scaled(
            target_size - QSize(12, 12),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.lbl_overview_canvas.setPixmap(scaled)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_overview_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_overview_pixmap()

    def open_guide(self):
        dlg = UserGuideDialog(self)
        dlg.exec_()

    def accept_and_launch(self):
        self.save_welcome_config()
        self.accept()


# =============================================================================
# USER GUIDE & PARAMETER MANUAL DIALOG (ENGLISH, PLAIN-READABLE FORMULAS)
# =============================================================================
class UserGuideDialog(QDialog):
    """Comprehensive Step-by-Step Workflow, Tools & Parameter Reference Guide."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DLFluoresenceQuantifier — User Guide & Parameter Manual")
        self.resize(1450, 950)

        layout = QVBoxLayout(self)

        header = QLabel("DLFluoresenceQuantifier — Complete User Manual")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #1f4e78; padding: 6px;")
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabBar::tab {
                font-weight: bold;
                padding: 8px 16px;
            }
            QTextBrowser {
                background-color: #ffffff;
                border: 1px solid #cfd8dc;
                padding: 17px;
                font-size: 17px;
                line-height: 1.5;
            }
        """)

        # ---------------------------------------------------------------------
        # TAB 1: PART 1 - STEP-BY-STEP WORKFLOW
        # ---------------------------------------------------------------------
        tab_workflow = QTextBrowser()
        tab_workflow.setHtml("""
        <h2 style="color:#1f4e78;">Part 1: Step-by-Step Analysis Workflow</h2>
        <p>Follow this standardized pipeline to obtain reproducible, unbiased fluorescence measurements across Control and Treated groups:</p>

        <h3 style="color:#2b5b84;">Step 1 — Load Your Microscopy Images</h3>
        <ul>
            <li><b>Option 1A (Multichannel / LIF):</b> Click <i>Load Multichannel / LIF Images</i> to import Leica <code>.lif</code> containers (all series are extracted automatically) or multichannel <code>.tif/.tiff/.png/.jpg</code> stacks.</li>
            <li><b>Option 1B (Separate Channel Files):</b> If your microscope saved each fluorophore as a separate grayscale file (e.g., one file for Tubulin and one for DAPI/Brightfield), click <i>Merge Separate Files</i> and assign them to Red (R), Green (G), and Blue (B).</li>
            <li><b>Check Saturation:</b> Look at the red warning label at the bottom of the toolbar. If a channel shows saturation (pixels hitting maximum detector capacity), note that intensity comparisons on that channel may be clipped.</li>
        </ul>

        <h3 style="color:#2b5b84;">Step 2 — Inspect &amp; Navigate the Field of View</h3>
        <ul>
            <li>Use the <b>Mouse Wheel</b> over the image to zoom in/out smoothly on specific cells or neurites.</li>
            <li>Hold <b>Right-Click and Drag</b> (or Middle-Click) to pan across the zoomed image. Click <i>Reset Zoom</i> at any time to restore the full field.</li>
            <li>Toggle <b>Channel R / G / B</b> checkboxes above the image to inspect individual markers without altering raw pixel data.</li>
        </ul>

        <h3 style="color:#2b5b84;">Step 3 — Exclude Artifacts &amp; Define ROIs (Optional)</h3>
        <ul>
            <li><b>Exclude Debris (Section 3):</b> If bright dust, dead cell clumps, or folding artifacts are present, draw a Red Circle or Freehand Exclusion mask. Excluded pixels are completely removed from all threshold, background, and intensity calculations.</li>
            <li><b>Define Target ROIs (Section 2):</b> Draw Yellow Circle or Freehand ROIs if you want to measure specific regions or sub-structures. If no Yellow ROI is drawn, the software automatically quantifies the entire valid field of view.</li>
        </ul>

        <h3 style="color:#2b5b84;">Step 4 — Count or Segment Nuclei (Fluorescence &amp; Brightfield — Section 4)</h3>
        <ul>
            <li><b>Option A1 — Fluorescence AI + Watershed:</b> Set <i>Nuclei Channel</i>, draw 2–3 tight green circles inside sample nuclei, set <b>Min Nucleus Area (px)</b> and <b>Splitting Diameter (px)</b>, then click <i>Train &amp; Split All Nuclei</i>.</li>
            <li><b>Option A2 — Brightfield Template &amp; Shape AI:</b> Check <i>Brightfield Mode</i>, draw 3–5 green circles inside sample nuclei (without spilling into cytoplasm), draw orange circles on dense cytoplasm/debris, and set <b>Min Circularity %</b> to <code>40–55</code> to discard elongated cytoplasmic shadows. Save/reuse templates across sessions with <i>Save/Load AI Model (.aim)</i>.</li>
            <li><b>Option B — Manual Tracing (Ideal for Few Cells):</b> Click <b>Draw Manual Nucleus (Circle)</b> or <b>Freehand</b> and trace directly around each nucleus (shown in cyan).</li>
            <li><b>Option C — Direct Manual Number Entry:</b> Type the exact number of nuclei visible in the field into <b>Manual Nuclei Count (0 = Auto)</b>.</li>
        </ul>

        <h3 style="color:#2b5b84;">Step 5 — Calibrate Independent Channel Thresholds on a Control Image (Section 5)</h3>
        <ul>
            <li>Select a representative <b>Control image</b> from the right gallery.</li>
            <li>Set an independent threshold for <b>Threshold Ch 1 (Red)</b>, <b>Threshold Ch 2 (Green)</b>, and <b>Threshold Ch 3 (Blue)</b> (either by clicking <i>Otsu All 3 Channels</i> or adjusting each channel individually with <i>Show Selected Channel Preview (Magenta)</i> active).</li>
            <li><b>Important:</b> Keep these three channel thresholds fixed across all images in the session so Control and Treated samples are compared on the exact same scale.</li>
        </ul>

        <h3 style="color:#2b5b84;">Step 6 — Configure Double Labeling / Colocalization (Section 6)</h3>
        <ul>
            <li>Select the two channels to compare (e.g., <i>Ch 1 (Red) + Ch 2 (Green)</i>).</li>
            <li>Set <b>Threshold Channel A</b> and <b>Threshold Channel B</b> on your Control image and check <i>Show Double-Positive Preview (Cyan)</i> to verify true colocalized signal.</li>
        </ul>

        <h3 style="color:#2b5b84;">Step 7 — Save Project &amp; Export Results to Excel (Sections 0 &amp; 7)</h3>
        <ul>
            <li>Click <b>Save Project (.cfq)</b> in Section 0 to store all images, ROIs, exclusions, manual/automatic nuclear masks, AI templates, and per-channel thresholds in a single reloadable session file.</li>
            <li>Click <b>Export ALL Loaded Images to Excel (.xlsx)</b> to generate a formatted workbook containing <i>Quantification</i>, <i>Colocalization</i>, and <i>Parameter Guide</i> sheets.</li>
        </ul>
        """)
        tabs.addTab(tab_workflow, "1. Step-by-Step Workflow")

        # ---------------------------------------------------------------------
        # TAB 2: PART 2A - HOW THE SOFTWARE & TOOLS WORK
        # ---------------------------------------------------------------------
        tab_tools = QTextBrowser()
        tab_tools.setHtml("""
        <h2 style="color:#1f4e78;">Part 2A: How the Software &amp; Tools Work Internally</h2>

        <h3 style="color:#2b5b84;">1. Raw Bit-Depth Preservation vs. Display Normalization</h3>
        <p>When you toggle R/G/B checkboxes or zoom in, the viewer applies percentile contrast stretching (1st to 99.5th percentile) purely for visual clarity on your monitor. <b>All mathematical measurements (Mean Intensity, IntDen, CTCF, Otsu, and Colocalization) are computed strictly on the untouched raw detector pixel values</b> (8-bit: 0–255, 12-bit: 0–4095, or 16-bit: 0–65535).</p>

        <h3 style="color:#2b5b84;">2. Region Hierarchy &amp; Mask Priority Rules</h3>
        <ol>
            <li><b>Artifact Exclusion Always Wins:</b> Any pixel inside a Red Exclusion mask is subtracted first from every calculation.</li>
            <li><b>Single-Nucleus Mode (if checked):</b> If <i>Quantify ONLY inside detected / traced nuclei</i> is active, each separated or manually traced nucleus (<code>Nucleus_1</code>, <code>Nucleus_2</code>, ...) is measured individually.</li>
            <li><b>ROI Mode vs. Whole Field:</b> Otherwise, if one or more Yellow ROIs exist, each ROI is measured separately. If no Yellow ROI is drawn, the entire non-excluded image is measured as <i>Whole Field</i>.</li>
        </ol>

        <h3 style="color:#2b5b84;">3. Morpho-Textural AI, Template Memory &amp; Circularity Filter (Section 4)</h3>
        <ul>
            <li><b>Chromatin Granularity &amp; Blob Geometry:</b> Each pixel is evaluated for multi-scale local variance (granularity), Difference-of-Gaussians (DoG), and Hessian matrix eigenvalues (separating globular nuclei from elongated neurites/cytoplasmic folds).</li>
            <li><b>Stored Nuclear Templates:</b> Your green sample circles are saved as normalized intensity and gradient templates and matched across the field so the AI remembers nuclear texture even in low-contrast brightfield images.</li>
            <li><b>Circularity Filter:</b> Segmented objects are filtered by roundness <code>4 * pi * Area / Perimeter^2</code> (max 100% = perfect circle) using <b>Min Circularity %</b>.</li>
        </ul>

        <h3 style="color:#2b5b84;">4. Per-Channel Session Thresholding &amp; Colocalization Engine (Sections 5 &amp; 6)</h3>
        <ul>
            <li><b>Independent Channel Cutoffs:</b> Each of the 3 channels (Red, Green, Blue) stores its own session threshold. Any channel left at <code>0</code> falls back to per-image Auto Otsu; any channel with a positive value uses that exact locked threshold across all images.</li>
            <li><b>Visual Overlays:</b> Single-channel suprathreshold pixels for the selected <i>Preview Channel</i> are highlighted in <b>Magenta</b>; double-positive colocalized pixels (exceeding both Threshold A and Threshold B simultaneously) are highlighted in <b>Cyan</b>.</li>
        </ul>
        """)
        tabs.addTab(tab_tools, "2. How the Tools Work")

        # ---------------------------------------------------------------------
        # TAB 3: PART 2B - CALCULATED PARAMETERS & PLAIN-READABLE FORMULAS
        # ---------------------------------------------------------------------
        tab_params = QTextBrowser()
        tab_params.setHtml("""
        <h2 style="color:#1f4e78;">Part 2B: Calculated Parameters &amp; Formulas</h2>
        <p>All formulas below use standard arithmetic notation matching the exported Excel workbook.</p>

        <h3 style="color:#2b5b84;">Sheet 1: Single-Channel Quantification Parameters</h3>
        <table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; width:100%; border-color:#cfd8dc;">
            <tr style="background-color:#1f4e78; color:white;">
                <th align="left" width="26%">Excel Column</th>
                <th align="left" width="37%">Biological / Technical Meaning</th>
                <th align="left" width="37%">Exact Formula</th>
            </tr>
            <tr>
                <td><b>Nuclei Count</b></td>
                <td>Number of nuclei in the region (from automatic detection with Min Area &amp; Circularity filters, manual tracing, or Manual Count override).</td>
                <td>Manual Count override (if &gt; 0), else count of unique valid nuclear labels</td>
            </tr>
            <tr>
                <td><b>Total Analyzed Area (px)</b></td>
                <td>Total number of valid pixels measured after subtracting Red exclusions.</td>
                <td>Count of pixels inside target region and outside exclusion mask</td>
            </tr>
            <tr>
                <td><b>Mean Background</b></td>
                <td>Baseline non-specific background intensity of the image channel.</td>
                <td>5th percentile of pixel intensities across all non-excluded pixels</td>
            </tr>
            <tr>
                <td><b>Area % (Raw)</b></td>
                <td>Percentage of analyzed area occupied by positive fluorescence signal above that channel's threshold.</td>
                <td>(Count of pixels &gt; Channel Threshold / Total Analyzed Area) x 100</td>
            </tr>
            <tr>
                <td><b>Area % per Nucleus</b></td>
                <td>Positive thresholded area percentage normalized per cell/nucleus.</td>
                <td>Area % (Raw) / Nuclei Count</td>
            </tr>
            <tr>
                <td><b>Mean Pixel Intensity (Raw)</b></td>
                <td>Average fluorescence brightness per pixel across the analyzed region.</td>
                <td>IntDen (Raw) / Total Analyzed Area (px)</td>
            </tr>
            <tr>
                <td><b>Mean Pixel Intensity per Nucleus</b></td>
                <td>Average fluorescence intensity normalized by cell/nuclei count.</td>
                <td>Mean Pixel Intensity (Raw) / Nuclei Count</td>
            </tr>
            <tr>
                <td><b>IntDen (Raw)</b></td>
                <td>Integrated Density: total cumulative fluorescence signal in the region.</td>
                <td>Sum of all raw pixel intensities inside the analyzed region</td>
            </tr>
            <tr>
                <td><b>IntDen per Nucleus</b></td>
                <td>Total cumulative raw fluorescence signal per single nucleus.</td>
                <td>IntDen (Raw) / Nuclei Count</td>
            </tr>
            <tr>
                <td><b>CTCF (Raw)</b></td>
                <td>Corrected Total Cell Fluorescence: total signal with local background removed.</td>
                <td>IntDen (Raw) - (Total Analyzed Area (px) x Mean Background)</td>
            </tr>
            <tr>
                <td><b>CTCF per Nucleus</b></td>
                <td>Gold-standard background-corrected fluorescence per single cell/nucleus.</td>
                <td>CTCF (Raw) / Nuclei Count</td>
            </tr>
            <tr>
                <td><b>Saturation Detected</b></td>
                <td>Quality-control flag indicating clipped detector pixels (255, 4095, or 65535).</td>
                <td>YES if any valid pixel &gt;= max bit-depth value, otherwise NO</td>
            </tr>
        </table>

        <h3 style="color:#00695c; margin-top:16px;">Sheet 2: Colocalization / Double-Labeling Parameters</h3>
        <table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; width:100%; border-color:#cfd8dc;">
            <tr style="background-color:#00695c; color:white;">
                <th align="left" width="26%">Excel Column</th>
                <th align="left" width="37%">Biological / Technical Meaning</th>
                <th align="left" width="37%">Exact Formula</th>
            </tr>
            <tr>
                <td><b>Ch A Positive Area %</b></td>
                <td>Percentage of analyzed area positive for the first marker (Channel A).</td>
                <td>(Count of pixels with Ch A &gt; Threshold A / Total Area) x 100</td>
            </tr>
            <tr>
                <td><b>Ch B Positive Area %</b></td>
                <td>Percentage of analyzed area positive for the second marker (Channel B).</td>
                <td>(Count of pixels with Ch B &gt; Threshold B / Total Area) x 100</td>
            </tr>
            <tr>
                <td><b>Double-Positive Area % (A &amp; B)</b></td>
                <td>Percentage of analyzed area simultaneously positive for both markers.</td>
                <td>(Count of pixels where Ch A &gt; Thresh A AND Ch B &gt; Thresh B) / Total Area x 100</td>
            </tr>
            <tr>
                <td><b>Manders M1 (Ch A in Ch B)</b></td>
                <td>Fraction of total positive Channel A fluorescence intensity that overlaps with positive Channel B signal (range: 0 to 1).</td>
                <td>Sum(Ch A intensity where Ch A &gt; Thresh A AND Ch B &gt; Thresh B) / Sum(Ch A intensity where Ch A &gt; Thresh A)</td>
            </tr>
            <tr>
                <td><b>Manders M2 (Ch B in Ch A)</b></td>
                <td>Fraction of total positive Channel B fluorescence intensity that overlaps with positive Channel A signal (range: 0 to 1).</td>
                <td>Sum(Ch B intensity where Ch A &gt; Thresh A AND Ch B &gt; Thresh B) / Sum(Ch B intensity where Ch B &gt; Thresh B)</td>
            </tr>
            <tr>
                <td><b>Pearson's r</b></td>
                <td>Linear correlation between Channel A and Channel B pixel intensities across the entire region (range: -1 to +1).</td>
                <td>Covariance(Ch A, Ch B) / (StdDev(Ch A) x StdDev(Ch B))</td>
            </tr>
            <tr>
                <td><b>Double-Positive Nuclei</b></td>
                <td>Number of detected/traced nuclei containing double-positive (Ch A &gt; Thresh A AND Ch B &gt; Thresh B) signal.</td>
                <td>Count of nuclear masks with at least one double-positive pixel</td>
            </tr>
            <tr>
                <td><b>Double-Positive Nuclei %</b></td>
                <td>Percentage of detected/traced nuclei that are double-positive for both markers.</td>
                <td>(Double-Positive Nuclei / Total Nuclei Count) x 100</td>
            </tr>
        </table>
        """)
        tabs.addTab(tab_params, "3. Parameters & Formulas")

        layout.addWidget(tabs)

        btn_close = QPushButton("Close Guide")
        btn_close.setStyleSheet("font-weight: bold; padding: 6px 18px;")
        btn_close.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)


# =============================================================================
# INTERACTIVE NUCLEI LEARNING, TEMPLATE MEMORY & INDIVIDUAL SEPARATION ENGINE
# =============================================================================
def extract_nucleus_templates(norm_img, nuc_example_masks):
    """
    Extracts normalized intensity + gradient patches (templates) from user-drawn
    sample nuclei so the model remembers nuclear appearance and granularity.
    """
    templates = []
    grad_img = filters.sobel(filters.gaussian(norm_img, sigma=1.0)).astype(np.float32)

    for m in nuc_example_masks:
        ys, xs = np.where(m)
        if len(ys) < 16:
            continue
        y0, y1 = max(0, ys.min() - 2), min(norm_img.shape[0], ys.max() + 3)
        x0, x1 = max(0, xs.min() - 2), min(norm_img.shape[1], xs.max() + 3)
        patch_int = norm_img[y0:y1, x0:x1].astype(np.float32)
        patch_grad = grad_img[y0:y1, x0:x1].astype(np.float32)

        if patch_int.shape[0] >= 5 and patch_int.shape[1] >= 5 and patch_int.std() > 1e-4:
            templates.append((patch_int, patch_grad))
    return templates


def compute_template_similarity_maps(norm_img, templates):
    """
    Matches stored nuclear templates (including 90-deg rotations) across the image
    to produce intensity-template and edge-template correlation feature maps.
    """
    h, w = norm_img.shape
    if not templates:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    grad_img = filters.sobel(filters.gaussian(norm_img, sigma=1.0)).astype(np.float32)
    src_int = norm_img.astype(np.float32)

    best_int_corr = np.full((h, w), -1.0, dtype=np.float32)
    best_grad_corr = np.full((h, w), -1.0, dtype=np.float32)

    for patch_int, patch_grad in templates[:8]:
        for k_rot in (0, 1):
            p_i = np.rot90(patch_int, k_rot)
            p_g = np.rot90(patch_grad, k_rot)
            th, tw = p_i.shape
            if th >= h - 2 or tw >= w - 2 or th < 5 or tw < 5:
                continue

            res_i = cv2.matchTemplate(src_int, p_i, cv2.TM_CCOEFF_NORMED)
            pad_top = th // 2
            pad_bottom = h - res_i.shape[0] - pad_top
            pad_left = tw // 2
            pad_right = w - res_i.shape[1] - pad_left
            aligned_i = np.pad(res_i, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
            best_int_corr = np.maximum(best_int_corr, aligned_i)

            if p_g.std() > 1e-4:
                res_g = cv2.matchTemplate(grad_img, p_g, cv2.TM_CCOEFF_NORMED)
                aligned_g = np.pad(res_g, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
                best_grad_corr = np.maximum(best_grad_corr, aligned_g)

    best_int_corr = np.clip(best_int_corr, 0.0, 1.0)
    best_grad_corr = np.clip(best_grad_corr, 0.0, 1.0)
    return best_int_corr, best_grad_corr


def extract_pixel_features(img_raw, templates=None):
    """
    Extracts multi-scale intensity, gradient, chromatin granularity, Difference-of-Gaussians,
    Hessian shape eigenvalues, and stored nuclear template correlation maps per pixel.
    """
    img = img_raw.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    norm = np.clip((img - p1) / (p99 - p1 + 1e-5), 0, 1).astype(np.float32)

    feats = [norm]
    blurs = {}
    for sigma in [1.0, 2.5, 5.0, 10.0]:
        blur = filters.gaussian(norm, sigma=sigma).astype(np.float32)
        edges = filters.sobel(blur).astype(np.float32)
        blurs[sigma] = blur
        feats.append(blur)
        feats.append(edges)

    # Difference of Gaussians (highlights globular/round nuclear bodies)
    feats.append(blurs[1.0] - blurs[5.0])
    feats.append(blurs[2.5] - blurs[10.0])

    # Multi-scale granularity (fine size=3 & medium size=7 local std)
    for win_size in (3, 7):
        mean_sq = ndi.uniform_filter(norm ** 2, size=win_size)
        sq_mean = ndi.uniform_filter(norm, size=win_size) ** 2
        local_std = np.sqrt(np.clip(mean_sq - sq_mean, 0, None)).astype(np.float32)
        feats.append(local_std)

    # Laplacian & Hessian eigenvalues (distinguishes round blobs from elongated fibers)
    lap = ndi.gaussian_laplace(norm, sigma=3.0).astype(np.float32)
    feats.append(lap)

    H_elems = feature.hessian_matrix(blurs[2.5], sigma=2.0, order="rc", use_gaussian_derivatives=True)
    l1, l2 = feature.hessian_matrix_eigvals(H_elems)
    feats.append(l1.astype(np.float32))
    feats.append(l2.astype(np.float32))

    # Stored nuclear template similarity maps
    tmpl_int, tmpl_grad = compute_template_similarity_maps(norm, templates)
    feats.append(tmpl_int)
    feats.append(tmpl_grad)

    return np.stack(feats, axis=-1), norm


def train_and_predict_nuclei(
    img_raw,
    nuc_example_masks,
    bg_example_masks,
    trained_clf=None,
    est_diameter=25,
    min_nucleus_area=150,
    min_circularity=0.0,
    prob_threshold=0.65,
    is_brightfield=False
):
    """
    Trains a morpho-textural Random Forest classifier + nuclear template memory on
    user-drawn example ROIs, separates touching nuclei using an intensity- and
    distance-weighted Watershed, and filters out objects below min_nucleus_area (px)
    or min_circularity.
    """
    img_f = img_raw.astype(np.float32)
    p1, p99 = np.percentile(img_f, (1, 99))
    norm_preview = np.clip((img_f - p1) / (p99 - p1 + 1e-5), 0, 1).astype(np.float32)

    if len(nuc_example_masks) > 0:
        templates = extract_nucleus_templates(norm_preview, nuc_example_masks)
        feat_stack, norm_img = extract_pixel_features(img_raw, templates=templates)
        h, w, n_feats = feat_stack.shape

        pos_mask = np.zeros((h, w), dtype=bool)
        for m in nuc_example_masks:
            pos_mask |= m

        neg_mask = np.zeros((h, w), dtype=bool)
        for m in bg_example_masks:
            neg_mask |= m

        if not neg_mask.any():
            dilated_pos = morphology.binary_dilation(pos_mask, morphology.disk(12))
            neg_mask = ~dilated_pos

        X_pos = feat_stack[pos_mask]
        X_neg = feat_stack[neg_mask]

        max_samples = 4500
        if len(X_pos) > max_samples:
            idx = np.random.choice(len(X_pos), max_samples, replace=False)
            X_pos = X_pos[idx]
        if len(X_neg) > max_samples:
            idx = np.random.choice(len(X_neg), max_samples, replace=False)
            X_neg = X_neg[idx]

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(len(X_pos), dtype=np.uint8), np.zeros(len(X_neg), dtype=np.uint8)])

        clf = RandomForestClassifier(
            n_estimators=75,
            max_depth=14,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X, y)
        model_bundle = {"clf": clf, "templates": templates}
    else:
        if trained_clf is None:
            raise ValueError("Please draw at least 2-3 ROIs inside sample nuclei first (or load a saved .aim model)!")
        if isinstance(trained_clf, dict):
            clf = trained_clf["clf"]
            templates = trained_clf.get("templates", [])
        else:
            clf = trained_clf
            templates = []

        feat_stack, norm_img = extract_pixel_features(img_raw, templates=templates)
        h, w, n_feats = feat_stack.shape
        model_bundle = {"clf": clf, "templates": templates}

    flat_feats = feat_stack.reshape(-1, n_feats)
    probs = clf.predict_proba(flat_feats)[:, 1].reshape(h, w)

    binary = probs > prob_threshold

    binary = ndi.binary_fill_holes(binary)
    binary = morphology.remove_small_objects(binary, min_size=max(10, int(min_nucleus_area)))

    if not binary.any():
        return np.zeros((h, w), dtype=np.int32), model_bundle

    signal_map = (1.0 - norm_img) if is_brightfield else norm_img
    signal_smooth = filters.gaussian(signal_map, sigma=max(1.0, est_diameter / 10.0))

    dist_map = ndi.distance_transform_edt(binary)
    dist_norm = dist_map / (dist_map.max() + 1e-5)

    combined_height = (0.65 * signal_smooth + 0.35 * dist_norm) * binary

    min_dist = max(4, int(est_diameter * 0.55))
    coords = feature.peak_local_max(combined_height, min_distance=min_dist, labels=binary)

    mask_peaks = np.zeros((h, w), dtype=bool)
    if len(coords) > 0:
        mask_peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_peaks)

    raw_labels = segmentation.watershed(-combined_height, markers, mask=binary)

    clean_labels = np.zeros_like(raw_labels, dtype=np.int32)
    next_id = 1
    for prop in measure.regionprops(raw_labels):
        if prop.area < min_nucleus_area:
            continue
        if min_circularity > 0.0:
            perim = prop.perimeter if prop.perimeter > 0 else 1.0
            circularity = (4.0 * np.pi * prop.area) / (perim ** 2)
            if circularity < min_circularity:
                continue
        clean_labels[raw_labels == prop.label] = next_id
        next_id += 1

    return clean_labels, model_bundle


def extract_single_channel(filepath):
    """Extracts a single 2D plane from an image file."""
    img = io.imread(filepath)
    if img.ndim == 2:
        return img
    elif img.ndim == 3:
        c_axis = np.argmin(img.shape)
        planes = [np.take(img, c, axis=c_axis) for c in range(img.shape[c_axis])]
        return max(planes, key=lambda p: np.var(p))
    return img


class MergeChannelsDialog(QDialog):
    """Dialog to combine separate single-channel files into R, G, B channels."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Merge Separate Channel Files into R, G, B")
        self.resize(620, 290)

        self.paths = {"R": "", "G": "", "B": ""}

        layout = QVBoxLayout(self)
        info = QLabel(
            "Select the image files corresponding to the same field of view.\n"
            "You can load just 2 files (e.g., Tubulin in R or G and DAPI/Brightfield in B) and leave the third empty."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Field / Sample Name:"))
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("e.g. CTR_Field1 (leave blank for automatic name)")
        row_name.addWidget(self.edit_name)
        layout.addLayout(row_name)

        self.edits = {}
        labels = [
            ("R", "Channel R (Red - e.g. Tubulin):", "#d32f2f"),
            ("G", "Channel G (Green - e.g. Marker 2):", "#388e3c"),
            ("B", "Channel B (Blue / Brightfield - e.g. DAPI):", "#1976d2"),
        ]

        for key, text, color in labels:
            row = QHBoxLayout()
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            lbl.setFixedWidth(285)
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText("No file selected...")
            btn = QPushButton("Browse...")
            btn.clicked.connect(lambda _, k=key: self.browse_file(k))
            row.addWidget(lbl)
            row.addWidget(edit)
            row.addWidget(btn)
            layout.addLayout(row)
            self.edits[key] = edit

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse_file(self, key):
        fp, _ = QFileDialog.getOpenFileName(
            self, f"Select Image for Channel {key}", "", "Images (*.tif *.tiff *.jpg *.jpeg *.png)"
        )
        if fp:
            self.paths[key] = fp
            self.edits[key].setText(Path(fp).name)
            if not self.edit_name.text():
                self.edit_name.setText(Path(fp).stem)

    def validate_and_accept(self):
        if not any(self.paths.values()):
            QMessageBox.warning(self, "Warning", "Please select at least one image file!")
            return
        self.accept()


class ImageRecord:
    """Stores image data, channel arrays, masks, manual count, and undo history for a single field."""
    def __init__(self, filepath, name, channels):
        self.filepath = filepath
        self.name = name
        self.channels = channels
        self.shape = channels[0].shape
        self.roi_masks = []
        self.excl_mask = np.zeros(self.shape, dtype=bool)
        self.nuc_train_masks = []
        self.bg_train_masks = []
        self.nuclei_mask = None
        self.manual_nuclei_count = 0
        self.history = []

    def push_history(self):
        """Saves a deep snapshot of current masks and manual count for Undo support."""
        snapshot = {
            "roi_masks": [(n, m.copy()) for n, m in self.roi_masks],
            "excl_mask": self.excl_mask.copy(),
            "nuc_train_masks": [m.copy() for m in self.nuc_train_masks],
            "bg_train_masks": [m.copy() for m in self.bg_train_masks],
            "nuclei_mask": self.nuclei_mask.copy() if self.nuclei_mask is not None else None,
            "manual_nuclei_count": self.manual_nuclei_count,
        }
        self.history.append(snapshot)
        if len(self.history) > 20:
            self.history.pop(0)

    def pop_history(self):
        """Restores the previous mask state."""
        if not self.history:
            return False
        snapshot = self.history.pop()
        self.roi_masks = snapshot["roi_masks"]
        self.excl_mask = snapshot["excl_mask"]
        self.nuc_train_masks = snapshot["nuc_train_masks"]
        self.bg_train_masks = snapshot["bg_train_masks"]
        self.nuclei_mask = snapshot["nuclei_mask"]
        self.manual_nuclei_count = snapshot.get("manual_nuclei_count", 0)
        return True


class DLFluoresenceQuantifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLFluoresenceQuantifier")
        self.resize(1560, 930)

        self.records = []
        self.current_idx = -1

        self.active_selector = None
        self.draw_mode = None

        self.trained_nuclei_clf = None

        self.xlim = None
        self.ylim = None
        self.pan_start = None

        self.init_ui()

        # Keyboard shortcut Ctrl+Z for Undo
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo_last_action)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # =====================================================================
        # COLUMN 1 (LEFT): BALANCED 11.5px FONT + FULL-FIT TOOLBAR (650px)
        # =====================================================================
        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_left.setFixedWidth(650)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll_left)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 10, 6)
        left_layout.setSpacing(8)
        scroll_left.setWidget(left_panel)

        left_panel.setStyleSheet("""
            QWidget {
                font-size: 11.5px;
            }
            QGroupBox {
                font-size: 11.5px;
                font-weight: bold;
                border: 1px solid #b0bec5;
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPushButton {
                font-size: 11.5px;
                text-align: left;
                padding: 4px 8px;
                min-height: 22px;
            }
            QLabel, QCheckBox, QSpinBox, QComboBox {
                font-size: 11.5px;
            }
        """)

        # 0. Project, Guide & History Controls
        grp_proj = QGroupBox("0. Welcome Screen, User Guide, Project && History")
        proj_layout = QVBoxLayout()
        proj_layout.setSpacing(5)

        row_guides = QHBoxLayout()
        btn_welcome = QPushButton("Welcome / Credits Screen")
        btn_welcome.setStyleSheet("background-color: #0d47a1; color: white; font-weight: bold;")
        btn_welcome.clicked.connect(self.open_welcome_screen)

        btn_user_guide = QPushButton("User Guide && Manual")
        btn_user_guide.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold;")
        btn_user_guide.clicked.connect(self.open_user_guide)
        row_guides.addWidget(btn_welcome)
        row_guides.addWidget(btn_user_guide)
        proj_layout.addLayout(row_guides)

        row_proj_files = QHBoxLayout()
        btn_save_proj = QPushButton("Save Project (.cfq)")
        btn_save_proj.setStyleSheet("background-color: #37474f; color: white; font-weight: bold;")
        btn_save_proj.clicked.connect(self.save_project)

        btn_load_proj = QPushButton("Load Project (.cfq)")
        btn_load_proj.setStyleSheet("background-color: #455a64; color: white; font-weight: bold;")
        btn_load_proj.clicked.connect(self.load_project)

        row_proj_files.addWidget(btn_save_proj)
        row_proj_files.addWidget(btn_load_proj)
        proj_layout.addLayout(row_proj_files)

        row_proj_actions = QHBoxLayout()
        self.btn_undo = QPushButton("Undo Last Action (Ctrl+Z)")
        self.btn_undo.setStyleSheet("background-color: #f57c00; color: white; font-weight: bold;")
        self.btn_undo.clicked.connect(self.undo_last_action)

        btn_close_proj = QPushButton("Close Project")
        btn_close_proj.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        btn_close_proj.clicked.connect(self.close_project)

        row_proj_actions.addWidget(self.btn_undo, stretch=3)
        row_proj_actions.addWidget(btn_close_proj, stretch=2)
        proj_layout.addLayout(row_proj_actions)

        grp_proj.setLayout(proj_layout)
        left_layout.addWidget(grp_proj)

        # 1. Image Loading
        grp_load = QGroupBox("1. Load Images")
        load_layout = QVBoxLayout()
        load_layout.setSpacing(5)

        btn_load = QPushButton("1A. Load Multichannel / LIF Images")
        btn_load.setStyleSheet("font-weight: bold; background-color: #2b5b84; color: white;")
        btn_load.clicked.connect(self.load_images)
        load_layout.addWidget(btn_load)

        btn_merge = QPushButton("1B. Merge Separate Files (Tubulin + DAPI / BF)")
        btn_merge.setStyleSheet("font-weight: bold; background-color: #00695c; color: white;")
        btn_merge.clicked.connect(self.load_and_merge_channels)
        load_layout.addWidget(btn_merge)

        grp_load.setLayout(load_layout)
        left_layout.addWidget(grp_load)

        # 2. Analysis ROIs
        grp_roi = QGroupBox("2. Fluorescence Analysis ROIs (Yellow)")
        roi_layout = QVBoxLayout()
        roi_layout.setSpacing(5)
        btn_roi_circle = QPushButton("Draw Circle / Ellipse ROI")
        btn_roi_circle.clicked.connect(lambda: self.activate_tool("roi_circle"))
        btn_roi_free = QPushButton("Draw Freehand Polygon ROI")
        btn_roi_free.clicked.connect(lambda: self.activate_tool("roi_free"))
        btn_roi_clear = QPushButton("Clear Analysis ROIs")
        btn_roi_clear.clicked.connect(self.clear_rois)
        roi_layout.addWidget(btn_roi_circle)
        roi_layout.addWidget(btn_roi_free)
        roi_layout.addWidget(btn_roi_clear)
        grp_roi.setLayout(roi_layout)
        left_layout.addWidget(grp_roi)

        # 3. Debris Exclusion
        grp_excl = QGroupBox("3. Exclude Artifacts / Debris (Red)")
        excl_layout = QVBoxLayout()
        excl_layout.setSpacing(5)
        btn_excl_circle = QPushButton("Exclude Circle Area")
        btn_excl_circle.clicked.connect(lambda: self.activate_tool("excl_circle"))
        btn_excl_free = QPushButton("Exclude Freehand Area")
        btn_excl_free.clicked.connect(lambda: self.activate_tool("excl_free"))
        btn_excl_clear = QPushButton("Clear Exclusions")
        btn_excl_clear.clicked.connect(self.clear_exclusions)
        excl_layout.addWidget(btn_excl_circle)
        excl_layout.addWidget(btn_excl_free)
        excl_layout.addWidget(btn_excl_clear)
        grp_excl.setLayout(excl_layout)
        left_layout.addWidget(grp_excl)

        # 4. Nuclei Detection, Manual Tracing, Size Limit, Circularity & Template Memory
        grp_nuc = QGroupBox("4. Nuclei Count, Manual Tracing && Morpho-Template AI")
        nuc_layout = QVBoxLayout()
        nuc_layout.setSpacing(5)

        row_manual_cnt = QHBoxLayout()
        lbl_man_cnt = QLabel("Manual Nuclei Count (0 = Auto):")
        lbl_man_cnt.setStyleSheet("font-weight: bold; color: #006064;")
        self.spin_manual_nuc = QSpinBox()
        self.spin_manual_nuc.setRange(0, 9999)
        self.spin_manual_nuc.setValue(0)
        self.spin_manual_nuc.setToolTip("Type the exact number of nuclei for this image (leave 0 to use traced/detected nuclei).")
        self.spin_manual_nuc.valueChanged.connect(self.on_manual_nuclei_count_changed)
        row_manual_cnt.addWidget(lbl_man_cnt)
        row_manual_cnt.addWidget(self.spin_manual_nuc)
        nuc_layout.addLayout(row_manual_cnt)

        row_man_draw = QHBoxLayout()
        btn_man_nuc_circle = QPushButton("Draw Manual Nucleus (Circle)")
        btn_man_nuc_circle.setStyleSheet("background-color: #00838f; color: white; font-weight: bold;")
        btn_man_nuc_circle.clicked.connect(lambda: self.activate_tool("manual_nuc_circle"))

        btn_man_nuc_free = QPushButton("Freehand")
        btn_man_nuc_free.setStyleSheet("background-color: #0097a7; color: white; font-weight: bold;")
        btn_man_nuc_free.clicked.connect(lambda: self.activate_tool("manual_nuc_free"))
        row_man_draw.addWidget(btn_man_nuc_circle, stretch=3)
        row_man_draw.addWidget(btn_man_nuc_free, stretch=1)
        nuc_layout.addLayout(row_man_draw)

        btn_clear_nuclei = QPushButton("Clear All Nuclei Masks && Reset Manual Count")
        btn_clear_nuclei.clicked.connect(self.clear_nuclei_masks)
        nuc_layout.addWidget(btn_clear_nuclei)

        # --- Automatic AI + Watershed + Template Memory Section ---
        row_bf = QHBoxLayout()
        row_bf.addWidget(QLabel("Auto Nuclei Channel (1=R, 2=G, 3=B):"))
        self.spin_bf_ch = QSpinBox()
        self.spin_bf_ch.setRange(1, 10)
        self.spin_bf_ch.setValue(3)
        row_bf.addWidget(self.spin_bf_ch)
        nuc_layout.addLayout(row_bf)

        lbl_instruct = QLabel(
            "Or for Auto AI: draw 2-4 circles inside sample nuclei (memorizes texture & shape templates):"
        )
        lbl_instruct.setWordWrap(True)
        lbl_instruct.setStyleSheet("font-weight: normal; color: #37474f;")
        nuc_layout.addWidget(lbl_instruct)

        row_train_nuc = QHBoxLayout()
        btn_ex_nuc_circle = QPushButton("+ AI Sample Nucleus (Circle)")
        btn_ex_nuc_circle.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        btn_ex_nuc_circle.clicked.connect(lambda: self.activate_tool("train_nuc_circle"))

        btn_ex_nuc_free = QPushButton("+ Freehand")
        btn_ex_nuc_free.setStyleSheet("background-color: #388e3c; color: white;")
        btn_ex_nuc_free.clicked.connect(lambda: self.activate_tool("train_nuc_free"))
        row_train_nuc.addWidget(btn_ex_nuc_circle, stretch=3)
        row_train_nuc.addWidget(btn_ex_nuc_free, stretch=1)
        nuc_layout.addLayout(row_train_nuc)

        btn_ex_bg = QPushButton("+ AI Background / Cytoplasm Sample (Recommended for BF)")
        btn_ex_bg.setStyleSheet("background-color: #ef6c00; color: white;")
        btn_ex_bg.clicked.connect(lambda: self.activate_tool("train_bg_circle"))
        nuc_layout.addWidget(btn_ex_bg)

        btn_clear_train = QPushButton("Clear AI Training Samples")
        btn_clear_train.clicked.connect(self.clear_training_examples)
        nuc_layout.addWidget(btn_clear_train)

        row_prob = QHBoxLayout()
        row_prob.addWidget(QLabel("Tighten Borders (Probability %):"))
        self.spin_nuc_prob = QSpinBox()
        self.spin_nuc_prob.setRange(30, 95)
        self.spin_nuc_prob.setValue(65)
        row_prob.addWidget(self.spin_nuc_prob)
        nuc_layout.addLayout(row_prob)

        row_min_area = QHBoxLayout()
        lbl_min_area = QLabel("Min Nucleus Area (px):")
        lbl_min_area.setStyleSheet("font-weight: bold; color: #1b5e20;")
        self.spin_min_nuc_area = QSpinBox()
        self.spin_min_nuc_area.setRange(10, 50000)
        self.spin_min_nuc_area.setSingleStep(25)
        self.spin_min_nuc_area.setValue(150)
        row_min_area.addWidget(lbl_min_area)
        row_min_area.addWidget(self.spin_min_nuc_area)
        nuc_layout.addLayout(row_min_area)

        row_circ = QHBoxLayout()
        lbl_circ = QLabel("Min Circularity % (0=Off, BF~45):")
        lbl_circ.setStyleSheet("font-weight: bold; color: #4a148c;")
        self.spin_min_circ = QSpinBox()
        self.spin_min_circ.setRange(0, 95)
        self.spin_min_circ.setSingleStep(5)
        self.spin_min_circ.setValue(0)
        self.spin_min_circ.setToolTip("Filters out elongated cytoplasmic shadows/debris (100% = perfect circle; 40-55% ideal for Brightfield).")
        row_circ.addWidget(lbl_circ)
        row_circ.addWidget(self.spin_min_circ)
        nuc_layout.addLayout(row_circ)

        row_sep = QHBoxLayout()
        row_sep.addWidget(QLabel("Splitting Diameter (px):"))
        self.spin_sep_diam = QSpinBox()
        self.spin_sep_diam.setRange(6, 200)
        self.spin_sep_diam.setValue(25)
        row_sep.addWidget(self.spin_sep_diam)
        nuc_layout.addLayout(row_sep)

        self.chk_is_bf = QCheckBox("Brightfield Mode (dark nuclei on bright bg)")
        self.chk_is_bf.setChecked(False)
        nuc_layout.addWidget(self.chk_is_bf)

        self.btn_detect_nuc = QPushButton("Train && Split All Nuclei (Auto AI)")
        self.btn_detect_nuc.setStyleSheet("font-weight: bold; background-color: #0277bd; color: white; padding: 6px 8px;")
        self.btn_detect_nuc.clicked.connect(self.train_and_detect_nuclei)
        nuc_layout.addWidget(self.btn_detect_nuc)

        self.btn_apply_trained = QPushButton("Apply Trained Model to This Image")
        self.btn_apply_trained.setEnabled(False)
        self.btn_apply_trained.clicked.connect(self.apply_existing_model)
        nuc_layout.addWidget(self.btn_apply_trained)

        row_ai_io = QHBoxLayout()
        self.btn_save_ai = QPushButton("Save AI Model (.aim)")
        self.btn_save_ai.setEnabled(False)
        self.btn_save_ai.clicked.connect(self.save_ai_model)

        btn_load_ai = QPushButton("Load AI Model (.aim)")
        btn_load_ai.clicked.connect(self.load_ai_model)
        row_ai_io.addWidget(self.btn_save_ai)
        row_ai_io.addWidget(btn_load_ai)
        nuc_layout.addLayout(row_ai_io)

        self.chk_quant_inside_nuclei = QCheckBox("Quantify ONLY inside detected / traced nuclei")
        nuc_layout.addWidget(self.chk_quant_inside_nuclei)
        grp_nuc.setLayout(nuc_layout)
        left_layout.addWidget(grp_nuc)

        # 5. Independent Per-Channel Session Thresholds
        grp_quant = QGroupBox("5. Per-Channel Thresholds (0 = Auto Otsu)")
        q_layout = QVBoxLayout()
        q_layout.setSpacing(5)

        self.spin_ch_thresholds = []
        ch_thresh_specs = [
            ("Threshold Ch 1 (Red):", "#d32f2f"),
            ("Threshold Ch 2 (Green):", "#388e3c"),
            ("Threshold Ch 3 (Blue):", "#1976d2"),
        ]
        for idx_c, (lbl_txt, col_hex) in enumerate(ch_thresh_specs):
            row_c = QHBoxLayout()
            lbl_c = QLabel(lbl_txt)
            lbl_c.setStyleSheet(f"color: {col_hex}; font-weight: bold;")
            sp_c = QSpinBox()
            sp_c.setRange(0, 65535)
            sp_c.setValue(0)
            sp_c.valueChanged.connect(lambda val, c=idx_c: self.on_channel_thresh_spin_changed(c, val))
            row_c.addWidget(lbl_c)
            row_c.addWidget(sp_c)
            q_layout.addLayout(row_c)
            self.spin_ch_thresholds.append(sp_c)

        row_prev_ch = QHBoxLayout()
        row_prev_ch.addWidget(QLabel("Slider & Preview Channel:"))
        self.combo_thresh_ch = QComboBox()
        self.combo_thresh_ch.addItems(["Channel 1 (Red)", "Channel 2 (Green)", "Channel 3 (Blue)"])
        self.combo_thresh_ch.currentIndexChanged.connect(self.on_preview_channel_changed)
        row_prev_ch.addWidget(self.combo_thresh_ch)
        q_layout.addLayout(row_prev_ch)

        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 255)
        self.slider_thresh.setValue(0)
        self.slider_thresh.valueChanged.connect(self.on_slider_thresh_changed)
        q_layout.addWidget(self.slider_thresh)

        row_otsu_btns = QHBoxLayout()
        btn_otsu_curr = QPushButton("Otsu Preview Ch")
        btn_otsu_curr.clicked.connect(self.suggest_otsu_single_channel)

        btn_otsu_all = QPushButton("Otsu All 3 Channels")
        btn_otsu_all.setStyleSheet("font-weight: bold;")
        btn_otsu_all.clicked.connect(self.suggest_otsu_all_channels)

        row_otsu_btns.addWidget(btn_otsu_curr)
        row_otsu_btns.addWidget(btn_otsu_all)
        q_layout.addLayout(row_otsu_btns)

        self.chk_preview_thresh = QCheckBox("Show Selected Channel Preview (Magenta)")
        self.chk_preview_thresh.setStyleSheet("color: #c2185b; font-weight: bold;")
        self.chk_preview_thresh.stateChanged.connect(self.update_display)
        q_layout.addWidget(self.chk_preview_thresh)

        grp_quant.setLayout(q_layout)
        left_layout.addWidget(grp_quant)

        # 6. Colocalization / Double Labeling
        grp_coloc = QGroupBox("6. Colocalization / Double Labeling")
        coloc_layout = QVBoxLayout()
        coloc_layout.setSpacing(5)

        self.chk_enable_coloc = QCheckBox("Enable Double-Labeling in Excel")
        self.chk_enable_coloc.setChecked(True)
        coloc_layout.addWidget(self.chk_enable_coloc)

        row_coloc_pair = QHBoxLayout()
        row_coloc_pair.addWidget(QLabel("Channel Pair:"))
        self.combo_coloc_pair = QComboBox()
        self.combo_coloc_pair.addItems([
            "Ch 1 (Red) + Ch 2 (Green)",
            "Ch 1 (Red) + Ch 3 (Blue)",
            "Ch 2 (Green) + Ch 3 (Blue)"
        ])
        self.combo_coloc_pair.currentIndexChanged.connect(self.update_display)
        row_coloc_pair.addWidget(self.combo_coloc_pair)
        coloc_layout.addLayout(row_coloc_pair)

        row_coloc_ta = QHBoxLayout()
        row_coloc_ta.addWidget(QLabel("Threshold Channel A:"))
        self.spin_coloc_ta = QSpinBox()
        self.spin_coloc_ta.setRange(0, 65535)
        self.spin_coloc_ta.setValue(0)
        self.spin_coloc_ta.valueChanged.connect(self.update_display)
        row_coloc_ta.addWidget(self.spin_coloc_ta)
        coloc_layout.addLayout(row_coloc_ta)

        row_coloc_tb = QHBoxLayout()
        row_coloc_tb.addWidget(QLabel("Threshold Channel B:"))
        self.spin_coloc_tb = QSpinBox()
        self.spin_coloc_tb.setRange(0, 65535)
        self.spin_coloc_tb.setValue(0)
        self.spin_coloc_tb.valueChanged.connect(self.update_display)
        row_coloc_tb.addWidget(self.spin_coloc_tb)
        coloc_layout.addLayout(row_coloc_tb)

        btn_coloc_otsu = QPushButton("Suggest Otsu Thresholds (Ch A && Ch B)")
        btn_coloc_otsu.clicked.connect(self.suggest_coloc_otsu)
        coloc_layout.addWidget(btn_coloc_otsu)

        self.chk_preview_coloc = QCheckBox("Show Double-Positive Preview (Cyan)")
        self.chk_preview_coloc.setStyleSheet("color: #00838f; font-weight: bold;")
        self.chk_preview_coloc.stateChanged.connect(self.update_display)
        coloc_layout.addWidget(self.chk_preview_coloc)

        grp_coloc.setLayout(coloc_layout)
        left_layout.addWidget(grp_coloc)

        # 7. Excel Export
        grp_export = QGroupBox("7. Excel Export")
        exp_layout = QVBoxLayout()
        exp_layout.setSpacing(5)

        btn_quant_curr = QPushButton("Export Current Image to Excel (.xlsx)")
        btn_quant_curr.setStyleSheet("font-weight: bold; background-color: #2e7d32; color: white;")
        btn_quant_curr.clicked.connect(lambda: self.quantify(all_images=False))
        exp_layout.addWidget(btn_quant_curr)

        btn_quant_all = QPushButton("Export ALL Loaded Images to Excel (.xlsx)")
        btn_quant_all.setStyleSheet("font-weight: bold; background-color: #1b5e20; color: white;")
        btn_quant_all.clicked.connect(lambda: self.quantify(all_images=True))
        exp_layout.addWidget(btn_quant_all)

        grp_export.setLayout(exp_layout)
        left_layout.addWidget(grp_export)

        self.lbl_saturation = QLabel("")
        self.lbl_saturation.setStyleSheet("color: #d32f2f; font-weight: bold;")
        self.lbl_saturation.setWordWrap(True)
        left_layout.addWidget(self.lbl_saturation)

        self.lbl_status = QLabel("Ready. Wheel = Zoom | Right-Drag = Pan | Ctrl+Z = Undo.")
        self.lbl_status.setStyleSheet("background-color: #eceff1; padding: 6px; border-radius: 4px;")
        self.lbl_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_status)
        left_layout.addStretch()

        # =====================================================================
        # COLUMN 2 (CENTER): BALANCED IMAGE FRAME
        # =====================================================================
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(center_panel, stretch=1)

        rgb_bar = QGroupBox("Channel Display && Zoom Controls")
        rgb_bar.setStyleSheet("QGroupBox { font-size: 11.5px; font-weight: bold; } QCheckBox, QPushButton { font-size: 11.5px; }")
        rgb_layout = QHBoxLayout()
        rgb_layout.setAlignment(Qt.AlignCenter)

        self.chk_r = QCheckBox("Channel R")
        self.chk_r.setStyleSheet("color: #d32f2f; font-weight: bold;")
        self.chk_r.setChecked(True)
        self.chk_r.stateChanged.connect(self.update_display)

        self.chk_g = QCheckBox("Channel G")
        self.chk_g.setStyleSheet("color: #388e3c; font-weight: bold;")
        self.chk_g.setChecked(True)
        self.chk_g.stateChanged.connect(self.update_display)

        self.chk_b = QCheckBox("Channel B")
        self.chk_b.setStyleSheet("color: #1976d2; font-weight: bold;")
        self.chk_b.setChecked(True)
        self.chk_b.stateChanged.connect(self.update_display)

        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.clicked.connect(self.reset_zoom)

        rgb_layout.addWidget(self.chk_r)
        rgb_layout.addSpacing(12)
        rgb_layout.addWidget(self.chk_g)
        rgb_layout.addSpacing(12)
        rgb_layout.addWidget(self.chk_b)
        rgb_layout.addSpacing(15)
        rgb_layout.addWidget(btn_reset_zoom)
        rgb_bar.setLayout(rgb_layout)
        center_layout.addWidget(rgb_bar)

        self.fig = Figure(facecolor="#181818")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        center_layout.addWidget(self.canvas, stretch=1)

        self.canvas.mpl_connect("scroll_event", self.on_scroll_zoom)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

        # =====================================================================
        # COLUMN 3 (RIGHT): THUMBNAIL GALLERY
        # =====================================================================
        right_panel = QWidget()
        right_panel.setFixedWidth(215)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(right_panel)

        lbl_gallery = QLabel("Loaded Images")
        lbl_gallery.setStyleSheet("font-size: 12px; font-weight: bold;")
        lbl_gallery.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(lbl_gallery)

        self.list_thumbnails = QListWidget()
        self.list_thumbnails.setViewMode(QListWidget.IconMode)
        self.list_thumbnails.setIconSize(QSize(170, 170))
        self.list_thumbnails.setResizeMode(QListWidget.Adjust)
        self.list_thumbnails.setMovement(QListWidget.Static)
        self.list_thumbnails.setSpacing(8)
        self.list_thumbnails.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_thumbnails.currentRowChanged.connect(self.on_thumbnail_selected)
        right_layout.addWidget(self.list_thumbnails)

    # -------------------------------------------------------------------------
    # WELCOME SCREEN & USER GUIDE DIALOG LAUNCHERS
    # -------------------------------------------------------------------------
    def open_welcome_screen(self):
        dlg = WelcomeScreenDialog(self)
        dlg.showMaximized()
        dlg.exec_()

    def open_user_guide(self):
        dlg = UserGuideDialog(self)
        dlg.exec_()

    # -------------------------------------------------------------------------
    # HELPER FOR COLOCALIZATION CHANNEL INDICES
    # -------------------------------------------------------------------------
    def get_coloc_channel_indices(self):
        idx = self.combo_coloc_pair.currentIndex()
        if idx == 0:
            return 0, 1  # Red + Green
        elif idx == 1:
            return 0, 2  # Red + Blue
        else:
            return 1, 2  # Green + Blue

    # -------------------------------------------------------------------------
    # SAVE / LOAD PERSISTENT AI MODEL & TEMPLATES (.aim)
    # -------------------------------------------------------------------------
    def save_ai_model(self):
        if self.trained_nuclei_clf is None:
            QMessageBox.information(self, "No Model", "Train an AI nuclei model first before saving.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Trained AI Nuclei Model", "Nuclei_Template_Model.aim", "AI Nuclei Model (*.aim)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".aim"):
            out_path += ".aim"

        ai_payload = {
            "bundle": self.trained_nuclei_clf,
            "params": {
                "nuc_prob": self.spin_nuc_prob.value(),
                "min_nuc_area": self.spin_min_nuc_area.value(),
                "min_circ": self.spin_min_circ.value(),
                "sep_diam": self.spin_sep_diam.value(),
                "is_bf": self.chk_is_bf.isChecked(),
            }
        }
        try:
            with open(out_path, "wb") as f:
                pickle.dump(ai_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            n_tmpl = len(self.trained_nuclei_clf.get("templates", [])) if isinstance(self.trained_nuclei_clf, dict) else 0
            self.lbl_status.setText(f"Saved AI model ({n_tmpl} nuclear templates) to {Path(out_path).name}.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save AI model:\n{e}")

    def load_ai_model(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load Trained AI Nuclei Model", "", "AI Nuclei Model (*.aim)"
        )
        if not fp:
            return
        try:
            with open(fp, "rb") as f:
                ai_payload = pickle.load(f)
            self.trained_nuclei_clf = ai_payload["bundle"]
            params = ai_payload.get("params", {})
            self.spin_nuc_prob.setValue(params.get("nuc_prob", 65))
            self.spin_min_nuc_area.setValue(params.get("min_nuc_area", 150))
            self.spin_min_circ.setValue(params.get("min_circ", 0))
            self.spin_sep_diam.setValue(params.get("sep_diam", 25))
            self.chk_is_bf.setChecked(params.get("is_bf", False))

            self.btn_apply_trained.setEnabled(True)
            self.btn_save_ai.setEnabled(True)
            n_tmpl = len(self.trained_nuclei_clf.get("templates", [])) if isinstance(self.trained_nuclei_clf, dict) else 0
            self.lbl_status.setText(f"Loaded AI model ({n_tmpl} nuclear templates) from {Path(fp).name}.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load AI model:\n{e}")

    # -------------------------------------------------------------------------
    # PROJECT SAVE, LOAD, CLOSE & UNDO HANDLERS
    # -------------------------------------------------------------------------
    def undo_last_action(self):
        if self.current_idx < 0:
            return
        rec = self.records[self.current_idx]
        if rec.pop_history():
            self.spin_manual_nuc.blockSignals(True)
            self.spin_manual_nuc.setValue(rec.manual_nuclei_count)
            self.spin_manual_nuc.blockSignals(False)
            self.update_display()
            self.lbl_status.setText(f"Undid last action on {rec.name}.")
        else:
            self.lbl_status.setText("Nothing to undo on this image.")

    def save_project(self):
        if not self.records:
            QMessageBox.information(self, "Empty Project", "Load at least one image before saving a project.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save DLFluoresenceQuantifier Project", "DLFQ_Session.cfq", "DLFQ Project (*.cfq)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".cfq"):
            out_path += ".cfq"

        project_data = {
            "records": self.records,
            "current_idx": self.current_idx,
            "trained_nuclei_clf": self.trained_nuclei_clf,
            "settings": {
                "nuc_channel": self.spin_bf_ch.value(),
                "nuc_prob": self.spin_nuc_prob.value(),
                "min_nuc_area": self.spin_min_nuc_area.value(),
                "min_circ": self.spin_min_circ.value(),
                "sep_diam": self.spin_sep_diam.value(),
                "is_bf": self.chk_is_bf.isChecked(),
                "quant_inside_nuclei": self.chk_quant_inside_nuclei.isChecked(),
                "thresh_ch": self.combo_thresh_ch.currentIndex(),
                "ch_thresholds": [sp.value() for sp in self.spin_ch_thresholds],
                "preview_thresh": self.chk_preview_thresh.isChecked(),
                "enable_coloc": self.chk_enable_coloc.isChecked(),
                "coloc_pair": self.combo_coloc_pair.currentIndex(),
                "coloc_ta": self.spin_coloc_ta.value(),
                "coloc_tb": self.spin_coloc_tb.value(),
                "preview_coloc": self.chk_preview_coloc.isChecked(),
            }
        }

        try:
            with open(out_path, "wb") as f:
                pickle.dump(project_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.lbl_status.setText(f"Project saved: {Path(out_path).name}")
            QMessageBox.information(self, "Project Saved", f"Project saved successfully to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save project:\n{e}")

    def load_project(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open DLFluoresenceQuantifier Project", "", "DLFQ Project (*.cfq)"
        )
        if not fp:
            return

        try:
            with open(fp, "rb") as f:
                project_data = pickle.load(f)

            self.reset_workspace(ask_confirm=False)

            self.records = project_data.get("records", [])
            self.trained_nuclei_clf = project_data.get("trained_nuclei_clf", None)
            has_model = self.trained_nuclei_clf is not None
            self.btn_apply_trained.setEnabled(has_model)
            self.btn_save_ai.setEnabled(has_model)

            settings = project_data.get("settings", {})
            self.spin_bf_ch.setValue(settings.get("nuc_channel", 3))
            self.spin_nuc_prob.setValue(settings.get("nuc_prob", 65))
            self.spin_min_nuc_area.setValue(settings.get("min_nuc_area", 150))
            self.spin_min_circ.setValue(settings.get("min_circ", 0))
            self.spin_sep_diam.setValue(settings.get("sep_diam", 25))
            self.chk_is_bf.setChecked(settings.get("is_bf", False))
            self.chk_quant_inside_nuclei.setChecked(settings.get("quant_inside_nuclei", False))

            saved_ths = settings.get("ch_thresholds", None)
            if saved_ths and len(saved_ths) == 3:
                for i, val in enumerate(saved_ths):
                    self.spin_ch_thresholds[i].setValue(val)
            else:
                legacy_val = settings.get("thresh_val", 0)
                for sp in self.spin_ch_thresholds:
                    sp.setValue(legacy_val)

            self.combo_thresh_ch.setCurrentIndex(settings.get("thresh_ch", 0))
            self.on_preview_channel_changed(self.combo_thresh_ch.currentIndex())
            self.chk_preview_thresh.setChecked(settings.get("preview_thresh", False))

            self.chk_enable_coloc.setChecked(settings.get("enable_coloc", True))
            self.combo_coloc_pair.setCurrentIndex(settings.get("coloc_pair", 0))
            self.spin_coloc_ta.setValue(settings.get("coloc_ta", 0))
            self.spin_coloc_tb.setValue(settings.get("coloc_tb", 0))
            self.chk_preview_coloc.setChecked(settings.get("preview_coloc", False))

            for rec in self.records:
                if not hasattr(rec, "history"):
                    rec.history = []
                if not hasattr(rec, "manual_nuclei_count"):
                    rec.manual_nuclei_count = 0
                pixmap = self.make_composite_pixmap(rec, thumb_size=170)
                item = QListWidgetItem(QIcon(pixmap), rec.name)
                item.setSizeHint(QSize(190, 200))
                self.list_thumbnails.addItem(item)

            target_idx = project_data.get("current_idx", 0)
            if self.records:
                self.list_thumbnails.setCurrentRow(min(max(0, target_idx), len(self.records) - 1))

            self.lbl_status.setText(f"Loaded project: {Path(fp).name} ({len(self.records)} images)")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load project:\n{e}")

    def close_project(self):
        self.reset_workspace(ask_confirm=True)

    def reset_workspace(self, ask_confirm=True):
        if ask_confirm and len(self.records) > 0:
            reply = QMessageBox.question(
                self,
                "Close Project",
                "Are you sure you want to close the current project?\nAny unsaved ROIs, masks, or settings will be lost.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        if self.active_selector:
            self.active_selector.set_active(False)
            self.active_selector = None

        self.records = []
        self.current_idx = -1
        self.draw_mode = None
        self.trained_nuclei_clf = None
        self.xlim = None
        self.ylim = None
        self.pan_start = None

        self.list_thumbnails.clear()
        self.btn_apply_trained.setEnabled(False)
        self.btn_save_ai.setEnabled(False)
        self.spin_manual_nuc.blockSignals(True)
        self.spin_manual_nuc.setValue(0)
        self.spin_manual_nuc.blockSignals(False)
        for sp in self.spin_ch_thresholds:
            sp.setValue(0)
        self.slider_thresh.setValue(0)
        self.spin_coloc_ta.setValue(0)
        self.spin_coloc_tb.setValue(0)
        self.chk_preview_thresh.setChecked(False)
        self.chk_preview_coloc.setChecked(False)
        self.chk_quant_inside_nuclei.setChecked(False)
        self.lbl_saturation.setText("")

        self.ax.clear()
        self.ax.axis("off")
        self.canvas.draw()
        self.lbl_status.setText("Project closed. Ready for a new session.")

    # -------------------------------------------------------------------------
    # ZOOM & PAN HANDLERS
    # -------------------------------------------------------------------------
    def reset_zoom(self):
        if self.current_idx < 0:
            return
        h, w = self.records[self.current_idx].shape
        self.xlim = (0, w)
        self.ylim = (h, 0)
        self.update_display()

    def on_scroll_zoom(self, event):
        if self.current_idx < 0 or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        base_scale = 1.25
        scale_factor = 1.0 / base_scale if event.button == "up" else base_scale

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[0] - cur_ylim[1]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[0] - ydata) / (cur_ylim[0] - cur_ylim[1])

        h, w = self.records[self.current_idx].shape
        x_min = max(0, xdata - new_width * (1 - relx))
        x_max = min(w, xdata + new_width * relx)
        y_max = min(h, ydata + new_height * rely)
        y_min = max(0, ydata - new_height * (1 - rely))

        if (x_max - x_min) > 10 and (y_max - y_min) > 10:
            self.xlim = (x_min, x_max)
            self.ylim = (y_max, y_min)
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
            self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if self.current_idx < 0 or event.inaxes != self.ax:
            return
        if event.button in (2, 3):
            self.pan_start = (event.xdata, event.ydata)

    def on_mouse_move(self, event):
        if self.current_idx < 0 or event.inaxes != self.ax:
            return
        if self.pan_start is not None and event.xdata is not None and event.ydata is not None:
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            self.xlim = (cur_xlim[0] + dx, cur_xlim[1] + dx)
            self.ylim = (cur_ylim[0] + dy, cur_ylim[1] + dy)
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
            self.canvas.draw_idle()

    def on_mouse_release(self, event):
        if event.button in (2, 3):
            self.pan_start = None

    # -------------------------------------------------------------------------
    # IMAGE LOADING
    # -------------------------------------------------------------------------
    def load_and_merge_channels(self):
        dlg = MergeChannelsDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return

        loaded = {}
        ref_shape = None
        ref_dtype = None

        for key in ["R", "G", "B"]:
            fp = dlg.paths[key]
            if fp:
                arr = extract_single_channel(fp)
                if ref_shape is None:
                    ref_shape = arr.shape
                    ref_dtype = arr.dtype
                elif arr.shape != ref_shape:
                    QMessageBox.critical(
                        self, "Dimension Mismatch",
                        f"Selected images have different resolutions ({arr.shape} vs {ref_shape})."
                    )
                    return
                loaded[key] = arr

        channels_rgb = []
        for key in ["R", "G", "B"]:
            if key in loaded:
                channels_rgb.append(loaded[key])
            else:
                channels_rgb.append(np.zeros(ref_shape, dtype=ref_dtype))

        rec_name = dlg.edit_name.text().strip() or "Merged_Field"
        self.add_record(ImageRecord("Merged_Files", rec_name, channels_rgb))
        self.list_thumbnails.setCurrentRow(len(self.records) - 1)
        self.lbl_status.setText(f"Merged channels into '{rec_name}'.")

    def load_images(self):
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Multichannel or LIF Images",
            "",
            "Microscopy (*.lif *.tif *.tiff *.jpg *.jpeg *.png)"
        )
        if not filepaths:
            return

        for fp in filepaths:
            path = Path(fp)
            ext = path.suffix.lower()

            if ext == ".lif":
                lif = LifFile(fp)
                for s_idx in range(lif.num_images):
                    img_lif = lif.get_image(s_idx)
                    chs = [np.array(img_lif.get_frame(z=0, t=0, c=c)) for c in range(img_lif.channels)]
                    rec_name = f"{path.stem}_Series{s_idx+1}"
                    self.add_record(ImageRecord(fp, rec_name, chs))
            else:
                img = io.imread(fp)
                chs = []
                if img.ndim == 2:
                    chs.append(img)
                elif img.ndim == 3:
                    c_axis = np.argmin(img.shape)
                    for c in range(img.shape[c_axis]):
                        chs.append(np.take(img, c, axis=c_axis))
                self.add_record(ImageRecord(fp, path.stem, chs))

        if self.current_idx == -1 and len(self.records) > 0:
            self.list_thumbnails.setCurrentRow(0)

    def add_record(self, record):
        self.records.append(record)
        pixmap = self.make_composite_pixmap(record, thumb_size=170)
        item = QListWidgetItem(QIcon(pixmap), record.name)
        item.setSizeHint(QSize(190, 200))
        self.list_thumbnails.addItem(item)

    def make_composite_pixmap(self, record, thumb_size=170):
        rgb = self.build_rgb_array(record, use_r=True, use_g=True, use_b=True)
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        h, w, _ = rgb_uint8.shape
        resized = cv2.resize(rgb_uint8, (thumb_size, int(thumb_size * h / w)), interpolation=cv2.INTER_AREA)
        rh, rw, _ = resized.shape
        qimg = QImage(resized.data, rw, rh, 3 * rw, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def on_thumbnail_selected(self, row):
        if row < 0 or row >= len(self.records):
            return
        if self.active_selector:
            self.active_selector.set_active(False)
            self.active_selector = None

        self.current_idx = row
        rec = self.records[self.current_idx]
        h, w = rec.shape
        self.xlim = (0, w)
        self.ylim = (h, 0)

        self.spin_bf_ch.setMaximum(len(rec.channels))
        self.spin_manual_nuc.blockSignals(True)
        self.spin_manual_nuc.setValue(rec.manual_nuclei_count)
        self.spin_manual_nuc.blockSignals(False)

        max_bit = 255 if rec.channels[0].dtype == np.uint8 else (4095 if rec.channels[0].max() <= 4095 else 65535)
        self.slider_thresh.setMaximum(max_bit)
        self.on_preview_channel_changed(self.combo_thresh_ch.currentIndex())

        self.check_saturation(rec)
        self.lbl_status.setText(f"Active image: {rec.name} ({w}x{h} px, {len(rec.channels)} channels)")
        self.update_display()

    def check_saturation(self, rec):
        saturated_channels = []
        for idx, ch in enumerate(rec.channels):
            if ch.max() == 0:
                continue
            max_possible = 255 if ch.dtype == np.uint8 else (4095 if ch.max() <= 4095 else 65535)
            sat_pixels = np.sum(ch >= max_possible)
            if sat_pixels > 0:
                perc = (sat_pixels / ch.size) * 100
                saturated_channels.append(f"Ch{idx+1} ({perc:.2f}%)")

        if saturated_channels:
            self.lbl_saturation.setText("SATURATION WARNING: " + ", ".join(saturated_channels))
        else:
            self.lbl_saturation.setText("")

    # -------------------------------------------------------------------------
    # PER-CHANNEL THRESHOLD CONTROLS (SINGLE CHANNEL & COLOCALIZATION)
    # -------------------------------------------------------------------------
    def on_preview_channel_changed(self, ch_idx):
        if 0 <= ch_idx < len(self.spin_ch_thresholds):
            val = self.spin_ch_thresholds[ch_idx].value()
            self.slider_thresh.blockSignals(True)
            self.slider_thresh.setValue(min(val, self.slider_thresh.maximum()))
            self.slider_thresh.blockSignals(False)
        self.update_display()

    def on_slider_thresh_changed(self, val):
        ch_idx = self.combo_thresh_ch.currentIndex()
        if 0 <= ch_idx < len(self.spin_ch_thresholds):
            sp = self.spin_ch_thresholds[ch_idx]
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        if self.chk_preview_thresh.isChecked():
            self.update_display()

    def on_channel_thresh_spin_changed(self, ch_idx, val):
        if ch_idx == self.combo_thresh_ch.currentIndex():
            self.slider_thresh.blockSignals(True)
            self.slider_thresh.setValue(min(val, self.slider_thresh.maximum()))
            self.slider_thresh.blockSignals(False)
            if self.chk_preview_thresh.isChecked():
                self.update_display()

    def suggest_otsu_single_channel(self):
        if self.current_idx < 0:
            return
        rec = self.records[self.current_idx]
        ch_idx = min(self.combo_thresh_ch.currentIndex(), len(rec.channels) - 1)
        valid_pixels = rec.channels[ch_idx][~rec.excl_mask]
        if valid_pixels.max() == 0:
            QMessageBox.information(self, "Empty Channel", "The selected channel is empty.")
            return
        otsu_val = int(round(filters.threshold_otsu(valid_pixels)))

        self.spin_ch_thresholds[ch_idx].setValue(otsu_val)
        self.chk_preview_thresh.setChecked(True)
        self.lbl_status.setText(f"Suggested Otsu threshold for Ch {ch_idx+1} on {rec.name}: {otsu_val}.")

    def suggest_otsu_all_channels(self):
        if self.current_idx < 0:
            return
        rec = self.records[self.current_idx]
        summary = []
        for c_idx in range(min(3, len(rec.channels))):
            valid_pixels = rec.channels[c_idx][~rec.excl_mask]
            if valid_pixels.max() > 0:
                otsu_val = int(round(filters.threshold_otsu(valid_pixels)))
                self.spin_ch_thresholds[c_idx].setValue(otsu_val)
                summary.append(f"Ch{c_idx+1}={otsu_val}")
            else:
                self.spin_ch_thresholds[c_idx].setValue(0)
        self.chk_preview_thresh.setChecked(True)
        self.lbl_status.setText(f"Suggested Otsu on {rec.name}: " + ", ".join(summary))

    def suggest_coloc_otsu(self):
        if self.current_idx < 0:
            return
        rec = self.records[self.current_idx]
        idx_a, idx_b = self.get_coloc_channel_indices()
        if idx_a >= len(rec.channels) or idx_b >= len(rec.channels):
            QMessageBox.warning(self, "Missing Channel", "One of the selected colocalization channels is not present.")
            return

        valid_a = rec.channels[idx_a][~rec.excl_mask]
        valid_b = rec.channels[idx_b][~rec.excl_mask]
        if valid_a.max() == 0 or valid_b.max() == 0:
            QMessageBox.information(self, "Empty Channel", "One of the selected channels is empty.")
            return

        otsu_a = int(round(filters.threshold_otsu(valid_a)))
        otsu_b = int(round(filters.threshold_otsu(valid_b)))

        self.spin_coloc_ta.setValue(otsu_a)
        self.spin_coloc_tb.setValue(otsu_b)
        self.chk_preview_coloc.setChecked(True)
        self.lbl_status.setText(f"Suggested Colocalization Otsu on {rec.name}: Ch A = {otsu_a}, Ch B = {otsu_b}.")

    # -------------------------------------------------------------------------
    # COMPOSITE DISPLAY & OVERLAYS
    # -------------------------------------------------------------------------
    def build_rgb_array(self, rec, use_r=True, use_g=True, use_b=True):
        h, w = rec.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        def norm_ch(arr):
            if arr.max() == 0:
                return np.zeros_like(arr, dtype=np.float32)
            p_low, p_high = np.percentile(arr, (1, 99.5))
            return np.clip((arr - p_low) / (p_high - p_low + 1e-5), 0, 1)

        if len(rec.channels) == 1:
            g = norm_ch(rec.channels[0])
            if use_r:
                rgb[:, :, 0] = g
            if use_g:
                rgb[:, :, 1] = g
            if use_b:
                rgb[:, :, 2] = g
            return rgb

        if len(rec.channels) >= 1 and use_r:
            rgb[:, :, 0] = norm_ch(rec.channels[0])
        if len(rec.channels) >= 2 and use_g:
            rgb[:, :, 1] = norm_ch(rec.channels[1])
        if len(rec.channels) >= 3 and use_b:
            rgb[:, :, 2] = norm_ch(rec.channels[2])

        return np.clip(rgb, 0, 1)

    def update_display(self):
        if self.current_idx < 0:
            return
        rec = self.records[self.current_idx]
        composite = self.build_rgb_array(
            rec,
            use_r=self.chk_r.isChecked(),
            use_g=self.chk_g.isChecked(),
            use_b=self.chk_b.isChecked()
        )

        # Single-channel threshold mask preview (Magenta) using that channel's own threshold
        if self.chk_preview_thresh.isChecked():
            ch_idx = min(self.combo_thresh_ch.currentIndex(), len(rec.channels) - 1)
            th_val = self.spin_ch_thresholds[ch_idx].value()
            if th_val > 0:
                above_thresh = (rec.channels[ch_idx] > th_val) & (~rec.excl_mask)
                composite[above_thresh, 0] = 1.0
                composite[above_thresh, 1] = 0.0
                composite[above_thresh, 2] = 1.0

        # Double-labeling / Colocalization overlap mask preview (Bright Cyan)
        if self.chk_preview_coloc.isChecked():
            idx_a, idx_b = self.get_coloc_channel_indices()
            if idx_a < len(rec.channels) and idx_b < len(rec.channels):
                ta = self.spin_coloc_ta.value()
                tb = self.spin_coloc_tb.value()
                if ta > 0 or tb > 0:
                    double_pos = (
                        (rec.channels[idx_a] > ta) &
                        (rec.channels[idx_b] > tb) &
                        (~rec.excl_mask)
                    )
                    composite[double_pos, 0] = 0.0
                    composite[double_pos, 1] = 1.0
                    composite[double_pos, 2] = 1.0

        self.ax.clear()
        self.ax.imshow(composite)
        self.ax.axis("off")

        for _, mask in rec.roi_masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt = cnt.squeeze()
                if cnt.ndim == 2:
                    self.ax.plot(cnt[:, 0], cnt[:, 1], color="yellow", linewidth=2)

        if rec.excl_mask.any():
            contours, _ = cv2.findContours(rec.excl_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt = cnt.squeeze()
                if cnt.ndim == 2:
                    self.ax.plot(cnt[:, 0], cnt[:, 1], color="red", linewidth=2, linestyle="--")

        for mask in rec.nuc_train_masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt = cnt.squeeze()
                if cnt.ndim == 2:
                    self.ax.plot(cnt[:, 0], cnt[:, 1], color="#00e676", linewidth=2)

        for mask in rec.bg_train_masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt = cnt.squeeze()
                if cnt.ndim == 2:
                    self.ax.plot(cnt[:, 0], cnt[:, 1], color="#ff9100", linewidth=2)

        if rec.nuclei_mask is not None and rec.nuclei_mask.max() > 0:
            for nid in range(1, rec.nuclei_mask.max() + 1):
                n_bin = (rec.nuclei_mask == nid).astype(np.uint8)
                if n_bin.sum() == 0:
                    continue
                contours, _ = cv2.findContours(n_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt = cnt.squeeze()
                    if cnt.ndim == 2:
                        self.ax.plot(cnt[:, 0], cnt[:, 1], color="#00e5ff", linewidth=1.6)

        if self.xlim is not None and self.ylim is not None:
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)

        self.canvas.draw()

    # -------------------------------------------------------------------------
    # ROI, MANUAL NUCLEUS TRACING & TRAINING SAMPLE DRAWING TOOLS
    # -------------------------------------------------------------------------
    def on_manual_nuclei_count_changed(self, val):
        if self.current_idx < 0:
            return
        rec = self.records[self.current_idx]
        rec.manual_nuclei_count = val
        if val > 0:
            self.lbl_status.setText(f"Manual nuclei count set to {val} for {rec.name}.")
        else:
            self.lbl_status.setText(f"Manual nuclei count reset to Auto for {rec.name}.")

    def activate_tool(self, mode):
        if self.current_idx < 0:
            return
        if self.active_selector:
            self.active_selector.set_active(False)

        self.draw_mode = mode
        if "roi" in mode:
            color = "yellow"
        elif "excl" in mode:
            color = "red"
        elif "manual_nuc" in mode:
            color = "#00e5ff"
        elif "train_nuc" in mode:
            color = "#00e676"
        else:
            color = "#ff9100"

        if "circle" in mode:
            self.lbl_status.setText("Click and drag on the image to draw a circle/ellipse.")
            self.active_selector = EllipseSelector(
                self.ax, self.on_ellipse_select, useblit=True,
                button=[1], interactive=False,
                props=dict(facecolor=color, edgecolor=color, alpha=0.35, fill=True)
            )
        elif "free" in mode:
            self.lbl_status.setText("Click polygon vertices and click the starting point to close.")
            self.active_selector = PolygonSelector(
                self.ax, self.on_polygon_select, useblit=True,
                props=dict(color=color, linewidth=2, alpha=0.8)
            )

    def on_ellipse_select(self, eclick, erelease):
        rec = self.records[self.current_idx]
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx, ry = abs(x2 - x1) / 2, abs(y2 - y1) / 2

        yy, xx = np.ogrid[:rec.shape[0], :rec.shape[1]]
        mask = ((xx - cx) / (rx + 1e-5))**2 + ((yy - cy) / (ry + 1e-5))**2 <= 1
        self.apply_drawn_mask(mask)

    def on_polygon_select(self, verts):
        rec = self.records[self.current_idx]
        h, w = rec.shape
        y, x = np.mgrid[:h, :w]
        coords = np.vstack((x.ravel(), y.ravel())).T
        path = MplPath(verts)
        mask = path.contains_points(coords).reshape((h, w))
        self.apply_drawn_mask(mask)

    def apply_drawn_mask(self, mask):
        rec = self.records[self.current_idx]
        rec.push_history()

        if self.draw_mode.startswith("roi"):
            roi_id = len(rec.roi_masks) + 1
            rec.roi_masks.append((f"ROI_{roi_id}", mask))
            self.lbl_status.setText(f"Added ROI_{roi_id} on {rec.name}.")
        elif self.draw_mode.startswith("excl"):
            rec.excl_mask = rec.excl_mask | mask
            self.lbl_status.setText(f"Added exclusion region on {rec.name}.")
        elif self.draw_mode.startswith("manual_nuc"):
            if rec.nuclei_mask is None:
                rec.nuclei_mask = np.zeros(rec.shape, dtype=np.int32)
            next_id = int(rec.nuclei_mask.max()) + 1
            rec.nuclei_mask[mask] = next_id
            total_traced = len(np.unique(rec.nuclei_mask)) - (1 if 0 in rec.nuclei_mask else 0)
            self.lbl_status.setText(f"Added Manual Nucleus #{next_id} (Total traced nuclei: {total_traced}).")
        elif self.draw_mode.startswith("train_nuc"):
            rec.nuc_train_masks.append(mask)
            self.lbl_status.setText(f"AI Nucleus template sample #{len(rec.nuc_train_masks)} added!")
        elif self.draw_mode.startswith("train_bg"):
            rec.bg_train_masks.append(mask)
            self.lbl_status.setText(f"AI Background/cytoplasm sample #{len(rec.bg_train_masks)} added.")

        if self.active_selector:
            self.active_selector.set_active(False)
            self.active_selector = None
        self.update_display()

    def clear_rois(self):
        if self.current_idx < 0:
            return
        rec = self.records[self.current_idx]
        rec.push_history()
        rec.roi_masks = []
        self.update_display()
        self.lbl_status.setText("Analysis ROIs cleared (Ctrl+Z to undo).")

    def clear_exclusions(self):
        if self.current_idx < 0:
            return
        rec = self.records[self.current_idx]
        rec.push_history()
        rec.excl_mask = np.zeros(rec.shape, dtype=bool)
        self.update_display()
        self.lbl_status.setText("Exclusions cleared (Ctrl+Z to undo).")

    def clear_nuclei_masks(self):
        if self.current_idx < 0:
            return
        rec = self.records[self.current_idx]
        rec.push_history()
        rec.nuclei_mask = None
        rec.manual_nuclei_count = 0
        self.spin_manual_nuc.blockSignals(True)
        self.spin_manual_nuc.setValue(0)
        self.spin_manual_nuc.blockSignals(False)
        self.update_display()
        self.lbl_status.setText("Nuclei masks & manual count cleared (Ctrl+Z to undo).")

    def clear_training_examples(self):
        if self.current_idx < 0:
            return
        rec = self.records[self.current_idx]
        rec.push_history()
        rec.nuc_train_masks = []
        rec.bg_train_masks = []
        rec.nuclei_mask = None
        self.update_display()
        self.lbl_status.setText("AI training samples cleared (Ctrl+Z to undo).")

    # -------------------------------------------------------------------------
    # INTERACTIVE TRAINING & NUCLEI SPLITTING (WITH TEMPLATE MEMORY & CIRCULARITY)
    # -------------------------------------------------------------------------
    def train_and_detect_nuclei(self):
        if self.current_idx < 0:
            return
        rec = self.records[self.current_idx]
        ch_idx = min(self.spin_bf_ch.value() - 1, len(rec.channels) - 1)
        img_nuc = rec.channels[ch_idx]

        if img_nuc.max() == 0:
            QMessageBox.warning(self, "Empty Channel", "The selected nuclei channel is empty!")
            return

        if len(rec.nuc_train_masks) == 0:
            QMessageBox.information(
                self,
                "Draw AI Training Samples First",
                "Click '+ AI Sample Nucleus (Circle)' and draw 2-4 circles "
                "inside sample nuclei (so the AI stores their chromatin & shape templates), "
                "OR use 'Draw Manual Nucleus' above to trace them directly."
            )
            return

        self.lbl_status.setText("Extracting nuclear templates, training model and splitting nuclei...")
        QApplication.processEvents()

        try:
            rec.push_history()
            prob_t = self.spin_nuc_prob.value() / 100.0
            min_area = self.spin_min_nuc_area.value()
            min_circ = self.spin_min_circ.value() / 100.0
            sep_d = self.spin_sep_diam.value()
            is_bf = self.chk_is_bf.isChecked()

            masks, clf = train_and_predict_nuclei(
                img_nuc,
                rec.nuc_train_masks,
                rec.bg_train_masks,
                trained_clf=self.trained_nuclei_clf,
                est_diameter=sep_d,
                min_nucleus_area=min_area,
                min_circularity=min_circ,
                prob_threshold=prob_t,
                is_brightfield=is_bf
            )
            self.trained_nuclei_clf = clf
            self.btn_apply_trained.setEnabled(True)
            self.btn_save_ai.setEnabled(True)

            rec.nuclei_mask = masks
            self.update_display()
            n_tmpl = len(clf.get("templates", [])) if isinstance(clf, dict) else 0
            self.lbl_status.setText(
                f"Detected {masks.max()} valid nuclei (stored {n_tmpl} templates, area >= {min_area} px) in {rec.name}!"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during model training:\n{e}")

    def apply_existing_model(self):
        if self.current_idx < 0 or self.trained_nuclei_clf is None:
            return
        rec = self.records[self.current_idx]
        ch_idx = min(self.spin_bf_ch.value() - 1, len(rec.channels) - 1)
        img_nuc = rec.channels[ch_idx]

        self.lbl_status.setText("Applying stored nuclear templates & AI model...")
        QApplication.processEvents()

        try:
            rec.push_history()
            prob_t = self.spin_nuc_prob.value() / 100.0
            min_area = self.spin_min_nuc_area.value()
            min_circ = self.spin_min_circ.value() / 100.0
            sep_d = self.spin_sep_diam.value()
            is_bf = self.chk_is_bf.isChecked()

            masks, _ = train_and_predict_nuclei(
                img_nuc,
                [],
                [],
                trained_clf=self.trained_nuclei_clf,
                est_diameter=sep_d,
                min_nucleus_area=min_area,
                min_circularity=min_circ,
                prob_threshold=prob_t,
                is_brightfield=is_bf
            )
            rec.nuclei_mask = masks
            self.update_display()
            self.lbl_status.setText(f"Detected {masks.max()} valid nuclei (>= {min_area} px) in {rec.name}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying trained model:\n{e}")

    # -------------------------------------------------------------------------
    # QUANTIFICATION & ENGLISH EXCEL EXPORT (.XLSX) + COLOCALIZATION SHEET
    # -------------------------------------------------------------------------
    def get_effective_nuclei_count(self, rec, region_mask):
        """Returns manual override count if set (>0), otherwise counts labeled nuclei in mask."""
        if getattr(rec, "manual_nuclei_count", 0) > 0:
            return int(rec.manual_nuclei_count)
        return self.count_nuclei_in_mask(rec.nuclei_mask, region_mask)

    def quantify(self, all_images=False):
        if not self.records:
            return

        target_records = self.records if all_images else [self.records[self.current_idx]]
        ch_thresh_values = [sp.value() for sp in self.spin_ch_thresholds]
        quant_inside_nuclei = self.chk_quant_inside_nuclei.isChecked()
        do_coloc = self.chk_enable_coloc.isChecked()
        idx_a, idx_b = self.get_coloc_channel_indices()
        coloc_ta_fixed = self.spin_coloc_ta.value()
        coloc_tb_fixed = self.spin_coloc_tb.value()

        if all(t == 0 for t in ch_thresh_values) and all_images:
            reply = QMessageBox.question(
                self,
                "All Channel Thresholds Set to 0 (Auto Otsu)",
                "All channel thresholds are set to 0: the software will compute a separate Otsu threshold per channel and per image.\n"
                "To compare Control vs. Treated groups fairly, setting fixed session thresholds is recommended.\n\n"
                "Do you want to proceed anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        results = []
        coloc_results = []
        rgb_names = ["Channel 1 (Red)", "Channel 2 (Green)", "Channel 3 (Blue)"]

        for rec in target_records:
            regions = []
            if quant_inside_nuclei and rec.nuclei_mask is not None and rec.nuclei_mask.max() > 0:
                for nuc_id in range(1, rec.nuclei_mask.max() + 1):
                    m = (rec.nuclei_mask == nuc_id) & (~rec.excl_mask)
                    if m.sum() > 0:
                        regions.append((f"Nucleus_{nuc_id}", m, 1))
            elif len(rec.roi_masks) > 0:
                for name, m in rec.roi_masks:
                    clean_m = m & (~rec.excl_mask)
                    if clean_m.sum() > 0:
                        n_nuc = self.get_effective_nuclei_count(rec, clean_m)
                        regions.append((name, clean_m, n_nuc))
            else:
                clean_m = ~rec.excl_mask
                n_nuc = self.get_effective_nuclei_count(rec, clean_m)
                regions.append(("Whole Field", clean_m, n_nuc))

            # 1. Single-Channel Quantification (using each channel's own threshold)
            for c_idx, ch_img in enumerate(rec.channels):
                if ch_img.max() == 0:
                    continue

                ch_label = rgb_names[c_idx] if c_idx < 3 else f"Channel {c_idx+1}"
                valid_pixels = ch_img[~rec.excl_mask]

                fixed_t = ch_thresh_values[c_idx] if c_idx < len(ch_thresh_values) else 0
                thresh = fixed_t if fixed_t > 0 else filters.threshold_otsu(valid_pixels)
                bg_mean = float(np.percentile(valid_pixels, 5))

                max_val_type = 255 if ch_img.dtype == np.uint8 else (4095 if ch_img.max() <= 4095 else 65535)
                has_sat = bool(np.any(valid_pixels >= max_val_type))

                for reg_name, mask, num_nuclei in regions:
                    pixels = ch_img[mask]
                    area_px = int(len(pixels))

                    mean_raw = float(np.mean(pixels))
                    perc_area_raw = float((np.sum(pixels > thresh) / area_px) * 100.0)
                    int_den_raw = float(np.sum(pixels))
                    ctcf_raw = float(int_den_raw - (area_px * bg_mean))

                    if num_nuclei > 0:
                        mean_per_nuc = round(mean_raw / num_nuclei, 4)
                        perc_area_per_nuc = round(perc_area_raw / num_nuclei, 4)
                        int_den_per_nuc = round(int_den_raw / num_nuclei, 2)
                        ctcf_per_nuc = round(ctcf_raw / num_nuclei, 2)
                    else:
                        mean_per_nuc = "N/A"
                        perc_area_per_nuc = "N/A"
                        int_den_per_nuc = "N/A"
                        ctcf_per_nuc = "N/A"

                    results.append({
                        "Image": rec.name,
                        "Channel": ch_label,
                        "Analyzed Region": reg_name,
                        "Nuclei Count": num_nuclei,
                        "Area % (Raw)": round(perc_area_raw, 3),
                        "Area % per Nucleus": perc_area_per_nuc,
                        "Mean Pixel Intensity (Raw)": round(mean_raw, 3),
                        "Mean Pixel Intensity per Nucleus": mean_per_nuc,
                        "IntDen (Raw)": round(int_den_raw, 2),
                        "IntDen per Nucleus": int_den_per_nuc,
                        "CTCF (Raw)": round(ctcf_raw, 2),
                        "CTCF per Nucleus": ctcf_per_nuc,
                        "Mean Background": round(bg_mean, 3),
                        "Total Analyzed Area (px)": area_px,
                        "Applied Threshold": round(float(thresh), 2),
                        "Saturation Detected": "YES" if has_sat else "NO",
                    })

            # 2. Double-Labeling / Colocalization Quantification
            if do_coloc and idx_a < len(rec.channels) and idx_b < len(rec.channels):
                ch_a = rec.channels[idx_a]
                ch_b = rec.channels[idx_b]
                if ch_a.max() > 0 and ch_b.max() > 0:
                    label_a = rgb_names[idx_a] if idx_a < 3 else f"Channel {idx_a+1}"
                    label_b = rgb_names[idx_b] if idx_b < 3 else f"Channel {idx_b+1}"

                    valid_a = ch_a[~rec.excl_mask]
                    valid_b = ch_b[~rec.excl_mask]

                    ta = coloc_ta_fixed if coloc_ta_fixed > 0 else float(filters.threshold_otsu(valid_a))
                    tb = coloc_tb_fixed if coloc_tb_fixed > 0 else float(filters.threshold_otsu(valid_b))

                    for reg_name, mask, num_nuclei in regions:
                        pix_a = ch_a[mask].astype(np.float64)
                        pix_b = ch_b[mask].astype(np.float64)
                        area_px = int(len(pix_a))

                        pos_a = pix_a > ta
                        pos_b = pix_b > tb
                        double_pos = pos_a & pos_b

                        perc_a = float((np.sum(pos_a) / area_px) * 100.0)
                        perc_b = float((np.sum(pos_b) / area_px) * 100.0)
                        perc_double = float((np.sum(double_pos) / area_px) * 100.0)

                        sum_a = np.sum(pix_a[pos_a])
                        sum_b = np.sum(pix_b[pos_b])
                        m1 = float(np.sum(pix_a[double_pos]) / sum_a) if sum_a > 0 else 0.0
                        m2 = float(np.sum(pix_b[double_pos]) / sum_b) if sum_b > 0 else 0.0

                        if np.std(pix_a) > 0 and np.std(pix_b) > 0:
                            pearson_r = float(np.corrcoef(pix_a, pix_b)[0, 1])
                        else:
                            pearson_r = 0.0

                        double_pos_nuclei = "N/A"
                        double_pos_nuclei_perc = "N/A"
                        if rec.nuclei_mask is not None and rec.nuclei_mask.max() > 0:
                            if quant_inside_nuclei:
                                is_dp = 1 if np.any(double_pos) else 0
                                double_pos_nuclei = is_dp
                                double_pos_nuclei_perc = round(is_dp * 100.0, 2)
                            elif num_nuclei > 0:
                                dp_count = 0
                                unique_nucs = np.unique(rec.nuclei_mask[mask])
                                for nid in unique_nucs:
                                    if nid == 0:
                                        continue
                                    n_mask = (rec.nuclei_mask == nid) & mask
                                    if np.any((ch_a[n_mask] > ta) & (ch_b[n_mask] > tb)):
                                        dp_count += 1
                                double_pos_nuclei = dp_count
                                double_pos_nuclei_perc = round((dp_count / num_nuclei) * 100.0, 2)

                        coloc_results.append({
                            "Image": rec.name,
                            "Channel Pair": f"{label_a} + {label_b}",
                            "Analyzed Region": reg_name,
                            "Total Nuclei Count": num_nuclei,
                            "Double-Positive Nuclei": double_pos_nuclei,
                            "Double-Positive Nuclei %": double_pos_nuclei_perc,
                            "Ch A Positive Area %": round(perc_a, 3),
                            "Ch B Positive Area %": round(perc_b, 3),
                            "Double-Positive Area % (A & B)": round(perc_double, 3),
                            "Manders M1 (Ch A in Ch B)": round(m1, 4),
                            "Manders M2 (Ch B in Ch A)": round(m2, 4),
                            "Pearson's r": round(pearson_r, 4),
                            "Threshold Ch A": round(float(ta), 2),
                            "Threshold Ch B": round(float(tb), 2),
                            "Total Analyzed Area (px)": area_px,
                        })

        df = pd.DataFrame(results)
        df_coloc = pd.DataFrame(coloc_results)

        guide_data = [
            {
                "Parameter / Column": "Image",
                "Description": "Name of the analyzed image file, LIF series, or merged field.",
                "Formula / Calculation": "File or custom sample identifier."
            },
            {
                "Parameter / Column": "Channel / Channel Pair",
                "Description": "Single fluorescence channel or pair of channels evaluated for colocalization.",
                "Formula / Calculation": "Extracted directly from the raw bit-depth matrices."
            },
            {
                "Parameter / Column": "Analyzed Region",
                "Description": "Spatial region quantified (Whole Field, user-drawn ROI, or individual Nucleus_ID).",
                "Formula / Calculation": "Target mask minus any red artifact exclusion masks."
            },
            {
                "Parameter / Column": "Nuclei Count",
                "Description": "Number of nuclei in the region (from automatic detection with Min Area & Circularity filters, manual tracing, or Manual Count override).",
                "Formula / Calculation": "Manual Count override (if > 0), else count of unique valid nuclear labels."
            },
            {
                "Parameter / Column": "Area % (Raw) & Area % per Nucleus",
                "Description": "Percentage of analyzed pixels with intensity above that channel's threshold (raw and normalized per nucleus).",
                "Formula / Calculation": "(Count of pixels > Channel Threshold / Total Analyzed Area) x 100; normalized by dividing by Nuclei Count."
            },
            {
                "Parameter / Column": "Mean Pixel Intensity (Raw & per Nucleus)",
                "Description": "Average grayscale fluorescence intensity across all pixels in the analyzed region.",
                "Formula / Calculation": "IntDen (Raw) / Total Analyzed Area (px); normalized by dividing by Nuclei Count."
            },
            {
                "Parameter / Column": "IntDen (Raw & per Nucleus)",
                "Description": "Integrated Density: total cumulative sum of all pixel intensity values within the analyzed region.",
                "Formula / Calculation": "Sum of all raw pixel intensities inside the analyzed region."
            },
            {
                "Parameter / Column": "CTCF (Raw & per Nucleus)",
                "Description": "Corrected Total Cell Fluorescence: total signal after subtracting local background.",
                "Formula / Calculation": "IntDen (Raw) - (Total Analyzed Area (px) x Mean Background)"
            },
            {
                "Parameter / Column": "Mean Background",
                "Description": "Estimated baseline non-specific background intensity of the image channel.",
                "Formula / Calculation": "5th percentile of pixel intensities across all non-excluded pixels."
            },
            {
                "Parameter / Column": "Double-Positive Area % (A & B)",
                "Description": "Percentage of pixels exceeding BOTH Channel A and Channel B thresholds simultaneously.",
                "Formula / Calculation": "(Count of pixels where Ch A > Thresh A AND Ch B > Thresh B) / Total Area x 100"
            },
            {
                "Parameter / Column": "Manders M1 & M2",
                "Description": "Manders' Colocalization Coefficients: fraction of thresholded intensity in one channel overlapping the other (0 to 1).",
                "Formula / Calculation": "M1 = Sum(Ch A where Ch A > Thresh A AND Ch B > Thresh B) / Sum(Ch A where Ch A > Thresh A); M2 = inverse."
            },
            {
                "Parameter / Column": "Pearson's r",
                "Description": "Linear intensity correlation between Channel A and Channel B across all pixels in the region (-1 to +1).",
                "Formula / Calculation": "Covariance(Ch A, Ch B) / (StdDev(Ch A) x StdDev(Ch B))"
            },
            {
                "Parameter / Column": "Double-Positive Nuclei & %",
                "Description": "Number and percentage of detected/traced nuclei containing double-positive (A & B) signal.",
                "Formula / Calculation": "(Double-Positive Nuclei / Total Nuclei Count) x 100"
            },
        ]
        df_guide = pd.DataFrame(guide_data)

        default_name = "DLFQ_Session_Results.xlsx" if all_images else f"{target_records[0].name}_Results.xlsx"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel File", default_name, "Excel (*.xlsx)"
        )
        if out_path:
            if not out_path.lower().endswith(".xlsx"):
                out_path += ".xlsx"

            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Quantification")
                ws_data = writer.sheets["Quantification"]

                header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
                header_font = Font(color="FFFFFF", bold=True, size=11)

                for col_idx, col in enumerate(ws_data.columns, start=1):
                    cell_header = ws_data.cell(row=1, column=col_idx)
                    cell_header.fill = header_fill
                    cell_header.font = header_font
                    cell_header.alignment = Alignment(horizontal="center", vertical="center")

                    max_len = max(len(str(cell.value or "")) for cell in col)
                    ws_data.column_dimensions[cell_header.column_letter].width = max_len + 5

                    for cell in col[1:]:
                        cell.alignment = Alignment(horizontal="center")

                if not df_coloc.empty:
                    df_coloc.to_excel(writer, index=False, sheet_name="Colocalization")
                    ws_coloc = writer.sheets["Colocalization"]

                    coloc_fill = PatternFill(start_color="00695C", end_color="00695C", fill_type="solid")
                    for col_idx, col in enumerate(ws_coloc.columns, start=1):
                        cell_header = ws_coloc.cell(row=1, column=col_idx)
                        cell_header.fill = coloc_fill
                        cell_header.font = header_font
                        cell_header.alignment = Alignment(horizontal="center", vertical="center")

                        max_len = max(len(str(cell.value or "")) for cell in col)
                        ws_coloc.column_dimensions[cell_header.column_letter].width = max_len + 5

                        for cell in col[1:]:
                            cell.alignment = Alignment(horizontal="center")

                df_guide.to_excel(writer, index=False, sheet_name="Parameter Guide")
                ws_guide = writer.sheets["Parameter Guide"]

                guide_fill = PatternFill(start_color="2E7D32", end_color="2E7D32", fill_type="solid")
                for col_idx, col in enumerate(ws_guide.columns, start=1):
                    cell_header = ws_guide.cell(row=1, column=col_idx)
                    cell_header.fill = guide_fill
                    cell_header.font = header_font
                    cell_header.alignment = Alignment(horizontal="left", vertical="center")

                    max_len = max(len(str(cell.value or "")) for cell in col)
                    ws_guide.column_dimensions[cell_header.column_letter].width = min(max_len + 6, 85)

                    for cell in col[1:]:
                        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

            self.lbl_status.setText("Excel file exported with per-channel thresholds!")
            QMessageBox.information(
                self,
                "Export Complete",
                f"Excel file successfully saved to:\n{out_path}\n\n"
                "Includes:\n"
                "1. 'Quantification' (each channel quantified with its own threshold)\n"
                "2. 'Colocalization' (double-labeling Area %, Manders M1/M2 & Pearson's r)\n"
                "3. 'Parameter Guide' (definitions & formulas)"
            )

    def count_nuclei_in_mask(self, nuclei_mask, region_mask):
        if nuclei_mask is None:
            return 0
        nuclei_in_region = nuclei_mask[region_mask]
        unique_ids = np.unique(nuclei_in_region)
        return int(np.sum(unique_ids > 0))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    welcome = WelcomeScreenDialog()
    welcome.showMaximized()
    if welcome.exec_() == QDialog.Accepted:
        window = DLFluoresenceQuantifier()
        app.setQuitOnLastWindowClosed(True)
        window.showMaximized()
        sys.exit(app.exec_())
    else:
        sys.exit(0)