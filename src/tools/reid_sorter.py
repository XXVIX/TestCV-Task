import sys
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QPushButton, QHBoxLayout, QFileDialog, QSizePolicy,
                             QStatusBar, QMainWindow, QAction)
from PyQt5.QtGui import QPixmap, QKeySequence
from PyQt5.QtCore import Qt
import shutil

STYLESHEET = """
QWidget { background-color: #2e2e2e; color: #e0e0e0; }
QPushButton { 
    background-color: #4a4a4a; border: 1px solid #555; 
    padding: 10px; border-radius: 5px; font-size: 16px;
}
QPushButton:hover { background-color: #5a5a5a; }
QPushButton:pressed { background-color: #6a6a6a; }
QLabel#imageLabel { border: 1px solid #444; }
QLabel#infoLabel { font-size: 16px; padding: 5px; background-color: #3a3a3a; border-radius: 4px;}
QStatusBar { font-size: 12px; }
"""

class ReidSorter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Re-ID Sorter")
        self.setGeometry(200, 200, 400, 500)

        self.input_dir = None
        self.output_dir = None
        self.trash_dir = None
        self.image_files = []
        self.current_index = -1
        
        self.class_counters = {'spoon': 0, 'fork': 0, 'knife': 0}
        self.last_class_path = {}

        # --- ИЗМЕНЕНО НАЗНАЧЕНИЕ КЛАВИШ ---
        self.key_map = {
            Qt.Key_Q: 'spoon',
            Qt.Key_W: 'fork',
            Qt.Key_E: 'knife'
        }
        # ------------------------------------

        self.setup_ui()
        self.load_images_from_default_path()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel("Loading images...")
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        main_layout.addWidget(self.image_label, 1)

        self.info_label = QLabel("Hotkeys: [Q] Spoon | [W] Fork | [E] Knife | [Shift+...] New Instance | [Del] Trash")
        self.info_label.setObjectName("infoLabel")
        self.info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.info_label)
        
        self.setStatusBar(QStatusBar(self))
        self.update_status()

    def load_images_from_default_path(self):
        try:
            project_root = Path(__file__).resolve().parents[2]
            self.input_dir = project_root / "data" / "reid_dataset_raw_new"
            self.output_dir = project_root / "data" / "reid_dataset_fina_new"
            self.trash_dir = self.output_dir / "_trash"
        except IndexError:
            self.image_label.setText("Error: Place script in 'src/tools' directory.")
            return

        self.output_dir.mkdir(exist_ok=True)
        self.trash_dir.mkdir(exist_ok=True)

        if not self.input_dir.exists():
            self.image_label.setText(f"Error: Directory not found\n{self.input_dir}")
            return

        self.image_files = sorted([f for f in self.input_dir.glob("**/*.png")])
        if self.image_files:
            self.current_index = 0
            self.show_current_image()
        else:
            self.image_label.setText("No images found in source directory.")

    def show_current_image(self):
        if self.current_index >= len(self.image_files):
            self.image_label.setText("All images sorted!")
            self.info_label.setText("Done!")
            self.update_status()
            return

        filepath = self.image_files[self.current_index]
        pixmap = QPixmap(str(filepath))
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.update_status()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        
        if key in self.key_map:
            class_name = self.key_map[key]
            is_new_instance = modifiers == Qt.ShiftModifier
            self.move_file(class_name, is_new_instance)
        elif key == Qt.Key_Delete:
            self.move_file_to_trash()

    def move_file(self, class_name, new_instance=False):
        if self.current_index >= len(self.image_files): return

        if new_instance or class_name not in self.last_class_path:
            self.class_counters[class_name] += 1
            folder_name = f"{class_name}_{self.class_counters[class_name]}"
            dest_folder = self.output_dir / folder_name
            dest_folder.mkdir(exist_ok=True)
            self.last_class_path[class_name] = dest_folder
            print(f"Created and activated new identity folder: {folder_name}")
        else:
            dest_folder = self.last_class_path[class_name]
            
        source_path = self.image_files[self.current_index]
        shutil.move(str(source_path), str(dest_folder / source_path.name))
        self.show_next_image()
    
    def move_file_to_trash(self):
        if self.current_index >= len(self.image_files): return
        source_path = self.image_files[self.current_index]
        shutil.move(str(source_path), str(self.trash_dir / source_path.name))
        self.show_next_image()

    def show_next_image(self):
        self.current_index += 1
        self.show_current_image()
        
    def update_status(self):
        total = len(self.image_files)
        if not self.image_files or self.current_index >= total:
            self.statusBar().showMessage(f"Done. Sorted {total} images.")
            return

        current = self.current_index + 1
        filepath = self.image_files[self.current_index]
        self.statusBar().showMessage(f"Image {current}/{total} | Source: {filepath.parent.name}")
        
    def resizeEvent(self, event):
        if self.image_files and self.current_index < len(self.image_files):
            self.show_current_image()
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    sorter = ReidSorter()
    sorter.show()
    sys.exit(app.exec_())