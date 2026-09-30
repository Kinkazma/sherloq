import json

from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QWidget, QMenu

from gui.sherloq_app.core.utility import modify_font
from gui.sherloq_app.ui.icons import themed_icon


class ToolWidget(QWidget):
    info_message = Signal(str)

    def __init__(self, parent=None):
        super(ToolWidget, self).__init__(parent)

    def shutdown(self):
        from gui.sherloq_app.ui.jobs import LatestJob
        for job in self.findChildren(LatestJob):
            job.shutdown()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


class ToolTree(QTreeWidget):
    FAVORITES_SETTING = 'interface/favorite_tools_v1'

    def __init__(self, parent=None, *, settings=None):
        super(ToolTree, self).__init__(parent)
        self._settings = settings if settings is not None else QSettings()
        self._tool_items = {}
        self._favorite_items = {}
        self.setProperty('localizeToolTree', True)
        group_names = []
        tool_names = []
        tool_infos = []
        tool_progress = []  # 0 = to-do, 1 = debug, 2 = working, 3 = complete

        # [0]
        group_names.append(self.tr("[General]"))
        tool_names.append(
            [
                self.tr("Original Image"),
                self.tr("File Digest"),
                self.tr("Hex Editor"),
                self.tr("Similarity Search"),
            ]
        )
        tool_infos.append(
            [
                self.tr("Display the unaltered reference image for visual inspection"),
                self.tr(
                    "Retrieve physical file information, crypto and perceptual hashes"
                ),
                self.tr(
                    "Open an external hexadecimal editor to show and edit raw bytes"
                ),
                self.tr(
                    "Browse online search services to find visually similar images"
                ),
            ]
        )
        tool_progress.extend([3, 3, 2, 2])

        # [1]
        group_names.append(self.tr("[Metadata]"))
        tool_names.append(
            [
                self.tr("Header Structure"),
                self.tr("EXIF Full Dump"),
                self.tr("Thumbnail Analysis"),
                self.tr("Geolocation data"),
                self.tr("C2PA Validation"),
            ]
        )
        tool_infos.append(
            [
                self.tr(
                    "Dump the file header structure and display an interactive view"
                ),
                self.tr(
                    "Scan through file metadata and gather all available information"
                ),
                self.tr(
                    "Extract optional embedded thumbnail and compare with original"
                ),
                self.tr(
                    "Retrieve optional geolocation data and show it on a world map"
                ),
                self.tr("Validate signed provenance, asset integrity and local trust offline"),
            ]
        )
        tool_progress.extend([3, 3, 3, 2, 3])

        # [2]
        group_names.append(self.tr("[Inspection]"))
        tool_names.append(
            [
                self.tr("Enhancing Magnifier"),
                self.tr("Channel Histogram"),
                self.tr("Global Adjustments"),
                self.tr("Reference Comparison"),
            ]
        )
        tool_infos.append(
            [
                self.tr(
                    "Magnifying glass with enhancements for better identifying forgeries"
                ),
                self.tr(
                    "Display single color channels or RGB composite interactive histogram"
                ),
                self.tr(
                    "Apply standard image adjustments (brightness, hue, saturation, ...)"
                ),
                self.tr(
                    "Open a synchronized double view for comparison with another picture"
                ),
            ]
        )
        tool_progress.extend([3, 3, 3, 3])

        # [3]
        group_names.append(self.tr("[Detail]"))
        tool_names.append(
            [
                self.tr("Luminance Gradient"),
                self.tr("Echo Edge Filter"),
                self.tr("Wavelet Threshold"),
                self.tr("Frequency Split"),
            ]
        )
        tool_infos.append(
            [
                self.tr(
                    "Analyze horizontal/vertical brightness variations across the image"
                ),
                self.tr(
                    "Use derivative filters to reveal artificial out-of-focus regions"
                ),
                self.tr(
                    "Reconstruct image with different wavelet coefficient thresholds"
                ),
                self.tr(
                    "Divide image luminance into high and low frequency components"
                ),
            ]
        )
        tool_progress.extend([3, 3, 3, 3])

        # [4]
        group_names.append(self.tr("[Colors]"))
        tool_names.append(
            [
                self.tr("RGB/HSV Plots"),
                self.tr("Space Conversion"),
                self.tr("PCA Projection"),
                self.tr("Pixel Statistics"),
            ]
        )
        tool_infos.append(
            [
                self.tr(
                    "Display interactive 2D and 3D plots of RGB and HSV pixel values"
                ),
                self.tr("Convert RGB channels into HSV/YCbCr/Lab/Luv/CMYK/Gray spaces"),
                self.tr("Use color PCA to project pixel onto most salient components"),
                self.tr("Compute minimum/maximum/average RGB values for every pixel"),
            ]
        )
        tool_progress.extend([3, 3, 3, 3])

        # [5]
        group_names.append(self.tr("[Noise]"))
        tool_names.append(
            [
                self.tr("Signal Separation"),
                self.tr("Min/Max Deviation"),
                self.tr("Bit Plane Values"),
                self.tr("Wavelet Blocking"),
                self.tr("PRNU Identification"),
                self.tr("Noisesniffer"),
            ]
        )
        tool_infos.append(
            [
                self.tr(
                    "Estimate and extract different kind of image noise components"
                ),
                self.tr(
                    "Highlight pixels deviating from block-based min/max statistics"
                ),
                self.tr(
                    "Show individual bit planes to find inconsistent noise patterns"
                ),
                self.tr("Noise estimation based on high pass wavelet coefficients & grid blocking"),
                self.tr("Exploit sensor pattern noise introduced by different cameras"),
                self.tr("Detect statistically unusual low-noise regions"),
            ]
        )
        tool_progress.extend([3, 3, 3, 2, 1, 3])

        # [6]
        group_names.append(self.tr("[JPEG]"))
        tool_names.append(
            [
                self.tr("Quality Estimation"),
                self.tr("Error Level Analysis"),
                self.tr("Multiple Compression"),
                self.tr("JPEG Ghost Maps"),
                self.tr("ZERO JPEG Grids"),
            ]
        )
        tool_infos.append(
            [
                self.tr(
                    "Extract quantization tables and estimate last saved JPEG quality"
                ),
                self.tr(
                    "Show pixel-wise differences against a fixed compression level"
                ),
                self.tr("Explore recompression losses and experimental aligned double-JPEG traces"),
                self.tr(
                    "Highlight traces of different compressions in difference images"
                ),
                self.tr("Detect foreign or missing local JPEG grids"),
            ]
        )
        tool_progress.extend([3, 3, 1, 2, 3])

        # [7]
        group_names.append(self.tr("[Tampering]"))
        tool_names.append(
            [
                self.tr("Contrast Enhancement"),
                self.tr("Copy-Move Forgery"),
                self.tr("Composite Splicing"),
                self.tr("Image Resampling"),
                self.tr("Copy-Move Forgery 2"),
                self.tr("Adaptive CFA"),
                self.tr("AI Clone Detection"),
                self.tr("Automatic Clone Search"),
                self.tr("Complete Automatic Analysis"),
            ]
        )
        tool_infos.append(
            [
                self.tr("Analyze color distributions to detect contrast enhancements"),
                self.tr("Use invariant feature descriptors to detect cloned regions"),
                self.tr("Exploit DCT statistics for automatic splicing zone detection"),
                self.tr(
                    "Estimate 2D pixel interpolation for detecting resampling traces"
                ),
                self.tr("Search local copy/move correspondences within selected zones and explore colored biomes"),
                self.tr("Analyze local CFA mosaic consistency with pretrained networks"),
                self.tr("Segment copy-move regions and compare scientific panels with pretrained detectors"),
                self.tr("Automatically select subimages and overlay Forgeryscope microscopy, PatchMatch Zernike and PatchMatch SIFT"),
                self.tr("Overlay automatic clone search and ELA biomes on the original image"),
            ]
        )
        tool_progress.extend([3, 2, 3, 2, 3, 3, 3, 3, 3])


        # [8]
        group_names.append(self.tr("[AI Solutions]"))
        tool_names.append(
            [
                self.tr("TruFor"),
                self.tr("CAT-Net v2"),
                self.tr("SAFIRE"),
                self.tr("FOCAL"),
                self.tr("AdaIFL"),
            ]
        )
        tool_infos.append(
            [
                self.tr("TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization"),
                self.tr("Analyze RGB and stored JPEG coefficient traces with CAT-Net v2"),
                self.tr("Group probable image sources with SAFIRE prompted segmentation"),
                self.tr("Partition image features with the FOCAL ViT-L and HRNet ensemble"),
                self.tr("Localize learned manipulation traces with AdaIFL"),
                
            ]
        )
        tool_progress.extend([2, 3, 3, 3, 3])


        # [9]
        group_names.append(self.tr("[Various]"))
        tool_names.append(
            [
                self.tr("Median Filtering"),
                self.tr("Illuminant Map"),
                self.tr("Dead/Hot Pixels"),
                self.tr("Stereogram Decoder"),
            ]
        )
        tool_infos.append(
            [
                self.tr("Detect nonlinear processing traces left by median filtering"),
                self.tr(
                    "Estimate local illuminant colour using Gray World, Shades of Gray or White Patch"
                ),
                self.tr(
                    "Inspect isolated hot/dead pixel candidates and preview local median corrections"
                ),
                self.tr(
                    "Decode 3D images concealed inside crossed-eye autostereograms"
                ),
            ]
        )
        tool_progress.extend([2, 2, 2, 3])

        from .extensions import ExtensionDelegate,EXTENSION_ROLE
        self.setItemDelegate(ExtensionDelegate(self))
        # New panels, newly implemented placeholders, and the double-JPEG addition.
        extensions={(1,4),(5,5),(6,2),(6,4),(7,4),(7,5),
                    (8,0),(8,1),(8,2),(8,3),(8,4),(9,1),(9,2)}
        count = 0
        for i, group in enumerate(group_names):
            group_item = QTreeWidgetItem()
            group_item.setText(0, group)
            font = group_item.font(0)
            font.setBold(True)
            group_item.setFont(0, font)
            group_item.setData(0, Qt.UserRole, False)
            group_item.setIcon(0, themed_icon(f"{i}.svg"))
            for j, tool in enumerate(tool_names[i]):
                tool_item = QTreeWidgetItem(group_item)
                tool_item.setText(0, tool)
                tool_item.setData(0, Qt.UserRole, True)
                tool_item.setData(0, Qt.UserRole + 1, i)
                tool_item.setData(0, Qt.UserRole + 2, j)
                self._tool_items[(i, j)] = tool_item
                tool_item.setToolTip(0, tool_infos[i][j])
                tool_item.setData(0, EXTENSION_ROLE, 'blue' if (i,j)==(7,8) else 'green' if (i,j) in ((7,6),(7,7)) else (i,j) in extensions)
                if tool_progress[count] == 0:
                    modify_font(tool_item, italic=True)
                count += 1
            self.addTopLevelItem(group_item)
        # Category placement is independent of dispatch/persisted tool IDs.
        complete_analysis = self.tool_item(7, 8)
        previous_group = complete_analysis.parent()
        previous_group.takeChild(previous_group.indexOfChild(complete_analysis))
        self.tool_item(2, 0).parent().addChild(complete_analysis)
        ai_clones = self.tool_item(7, 6)
        previous_group = ai_clones.parent()
        previous_group.takeChild(previous_group.indexOfChild(ai_clones))
        self.tool_item(8, 0).parent().addChild(ai_clones)
        self.favorites_group = QTreeWidgetItem([self.tr('[Favorites]')])
        modify_font(self.favorites_group, bold=True)
        self.favorites_group.setIcon(0, themed_icon('8.svg'))
        self.favorites_group.setData(0, Qt.UserRole, False)
        self.favorites_group.setToolTip(0, self.tr('Right-click a tool to add or remove a favorite'))
        self.insertTopLevelItem(0, self.favorites_group)
        try:
            saved = json.loads(self._settings.value(self.FAVORITES_SETTING, '[]'))
        except (TypeError, ValueError):
            saved = []
        if isinstance(saved, list):
            for value in saved:
                if not isinstance(value, str):
                    continue
                parts = value.split(':')
                if len(parts) == 2 and all(part.isdecimal() for part in parts):
                    self.set_favorite(tuple(map(int, parts)), True, persist=False)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_favorites_menu)
        self.expandAll()
        self.setColumnCount(1)
        self.header().setVisible(False)
        self.setMaximumWidth(300)
        self.version = f"{sum(tool_progress) / 100:.2f}f"

    def tool_item(self, group, tool):
        """Canonical item, independent of the visible category order."""
        return self._tool_items.get((group, tool))

    def set_favorite(self, key, enabled, *, persist=True):
        original = self._tool_items.get(key)
        if original is None:
            return
        if enabled and key not in self._favorite_items:
            # Copy tool IDs, translation sources and colored extension markers.
            favorite = original.clone()
            self._favorite_items[key] = favorite
            self.favorites_group.addChild(favorite)
            self.favorites_group.setExpanded(True)
        elif not enabled and key in self._favorite_items:
            favorite = self._favorite_items.pop(key)
            if self.currentItem() is favorite:
                self.setCurrentItem(original)
            self.favorites_group.takeChild(self.favorites_group.indexOfChild(favorite))
        else:
            return
        if persist:
            self._settings.setValue(self.FAVORITES_SETTING,
                                    json.dumps([f'{g}:{t}' for g, t in self._favorite_items]))
            self._settings.sync()

    def favorites_menu(self, item):
        if item is None or not item.data(0, Qt.UserRole):
            return None
        from .localization import t
        key = (item.data(0, Qt.UserRole + 1), item.data(0, Qt.UserRole + 2))
        if key not in self._tool_items:
            return None
        enabled = key not in self._favorite_items
        menu = QMenu(self)
        action = menu.addAction(t('Add to favorites' if enabled else 'Remove from favorites'))
        action.triggered.connect(lambda checked=False: self.set_favorite(key, enabled))
        return menu

    def show_favorites_menu(self, position):
        menu = self.favorites_menu(self.itemAt(position))
        if menu is not None:
            menu.aboutToHide.connect(menu.deleteLater)
            menu.popup(self.viewport().mapToGlobal(position))

    def set_bold(self, tool, enabled):
        if isinstance(tool, tuple):
            for item in (self._tool_items.get(tool), self._favorite_items.get(tool)):
                if item is not None:
                    modify_font(item, bold=enabled)
            return
        items = self.findItems(tool, Qt.MatchFixedString | Qt.MatchRecursive)
        for item in items:
            modify_font(item, bold=enabled)
