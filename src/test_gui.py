import sys
import os
import csv
import numpy as np
import cv2
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QGroupBox,
                               QFileDialog, QInputDialog, QMessageBox)
from PySide6.QtCore import Qt, QUrl, QThread, Signal
from PySide6.QtGui import QDesktopServices

def cv2_imread_chinese(filepath):
    img_data = np.fromfile(filepath, dtype=np.uint8)
    return cv2.imdecode(img_data, cv2.IMREAD_COLOR)

def cv2_imwrite_chinese(filepath, img):
    ext = os.path.splitext(filepath)[1]
    result, n = cv2.imencode(ext, img)
    if result:
        with open(filepath, mode='wb') as f:
            n.tofile(f)

class RealAnalysisWorker(QThread):
    log_signal = Signal(str)
    analysis_complete_signal = Signal(dict, str, str, str) 

    def __init__(self, video_path, song_name, level):
        super().__init__()
        self.video_path = video_path
        self.song_name = song_name
        self.level = level

    def run(self):
        self.log_signal.emit("[System] Loading video and template resources...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        song_folder_name = f"{self.song_name}_{self.level}"
        song_dir = os.path.join(current_dir, song_folder_name)
        os.makedirs(song_dir, exist_ok=True)
        
        templates_dir = os.path.join(current_dir, "templates")
        csv_path = os.path.join(song_dir, "records.csv")

        existing_runs = []
        if os.path.exists(song_dir):
            for item in os.listdir(song_dir):
                item_path = os.path.join(song_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    existing_runs.append(int(item))
        run_number = max(existing_runs) + 1 if existing_runs else 1
        run_dir = os.path.join(song_dir, str(run_number))
        os.makedirs(run_dir, exist_ok=True)

        self.log_signal.emit(f"[System] Run directory: {song_folder_name} / {run_number}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.log_signal.emit("[Error] Failed to read the video file.")
            return

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return

        display_scale = 0.5
        resized_frame = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)

        self.log_signal.emit("[System] OpenCV window opened. Please select ROI.")
        
        cv2.namedWindow("1. Select JUDGEMENT ROI", cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty("1. Select JUDGEMENT ROI", cv2.WND_PROP_TOPMOST, 1)
        roi_judge_rect = cv2.selectROI("1. Select JUDGEMENT ROI", resized_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("1. Select JUDGEMENT ROI")
        jx, jy, jw, jh = [int(val / display_scale) for val in roi_judge_rect]

        cv2.namedWindow("2. Select COMBO ROI", cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty("2. Select COMBO ROI", cv2.WND_PROP_TOPMOST, 1)
        roi_combo_rect = cv2.selectROI("2. Select COMBO ROI", resized_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("2. Select COMBO ROI")
        cx, cy, cw, ch = [int(val / display_scale) for val in roi_combo_rect]

        judgement_templates = {}
        for j_name in ["great", "good", "bad", "miss", "combo"]:
            path = os.path.join(templates_dir, f"{j_name}.png")
            if os.path.exists(path):
                judgement_templates[j_name.capitalize()] = cv2_imread_chinese(path)

        combo_tpl = judgement_templates.pop("Combo", None)
        if combo_tpl is None:
            self.log_signal.emit("[Error] Template combo.png not found.")
            cap.release()
            return

        thresholds = {"Great": 0.58, "Good": 0.65, "Bad": 0.65, "Miss": 0.60, "Combo": 0.55}
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 60.0

        screenshot_count = 0
        cooldown_frames = 0
        cooldown_max = int(fps * 0.15)
        temp_counts = {"Great": 0, "Good": 0, "Bad": 0, "Miss": 0}
        scales = np.linspace(0.5, 1.5, 11)

        self.log_signal.emit("[System] Analyzing video... Press 'q' to stop.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            roi_judge_frame = frame[jy:jy+jh, jx:jx+jw]
            roi_combo_frame = frame[cy:cy+ch, cx:cx+cw]

            if cooldown_frames > 0:
                cooldown_frames -= 1
            else:
                is_combo_active = False
                for scale in scales:
                    c_w = int(combo_tpl.shape[1] * scale)
                    c_h = int(combo_tpl.shape[0] * scale)
                    if c_h > ch or c_w > cw or c_h == 0 or c_w == 0: continue
                    c_tpl_resized = cv2.resize(combo_tpl, (c_w, c_h))
                    c_result = cv2.matchTemplate(roi_combo_frame, c_tpl_resized, cv2.TM_CCOEFF_NORMED)
                    _, c_max_val, _, _ = cv2.minMaxLoc(c_result)
                    if c_max_val >= thresholds["Combo"]:
                        is_combo_active = True
                        break

                global_best_val = -1
                best_match_loc = None
                detected_judgement = None
                best_tpl_w, best_tpl_h = 0, 0

                targets_to_check = ["Great"] if is_combo_active else ["Great", "Miss", "Bad", "Good"]

                for j_name in targets_to_check:
                    if j_name not in judgement_templates: continue
                    tpl = judgement_templates[j_name]
                    for scale in scales:
                        tpl_w = int(tpl.shape[1] * scale)
                        tpl_h = int(tpl.shape[0] * scale)
                        if tpl_h > jh or tpl_w > jw or tpl_h == 0 or tpl_w == 0: continue

                        resized_tpl = cv2.resize(tpl, (tpl_w, tpl_h))
                        result = cv2.matchTemplate(roi_judge_frame, resized_tpl, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)

                        if max_val > global_best_val:
                            global_best_val = max_val
                            best_match_loc = max_loc
                            detected_judgement = j_name
                            best_tpl_w = tpl_w
                            best_tpl_h = tpl_h

                if detected_judgement and global_best_val >= thresholds[detected_judgement]:
                    temp_counts[detected_judgement] += 1
                    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    self.log_signal.emit(f"[Match] Detected {detected_judgement} (Score: {global_best_val:.2f}) at {current_time:.2f}s")

                    screenshot_count += 1
                    frame_with_box = frame.copy()
                    top_left = (best_match_loc[0] + jx, best_match_loc[1] + jy)
                    bottom_right = (top_left[0] + best_tpl_w, top_left[1] + best_tpl_h)
                    cv2.rectangle(frame_with_box, top_left, bottom_right, (0, 0, 255), 3)
                    filename = f"{detected_judgement}_{screenshot_count:03d}_{current_time:.2f}s.jpg"
                    cv2_imwrite_chinese(os.path.join(run_dir, filename), frame_with_box)
                    cooldown_frames = cooldown_max

            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (jx, jy), (jx+jw, jy+jh), (0, 255, 0), 2)
            cv2.rectangle(debug_frame, (cx, cy), (cx+cw, cy+ch), (255, 0, 0), 2)
            cv2.imshow("Analyzing Video (Green=Judge, Red=Combo)", debug_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.analysis_complete_signal.emit(temp_counts, csv_path, self.song_name, self.level)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project SEKAI Performance Analyzer v4.5")
        self.resize(950, 550)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        log_group = QGroupBox("System Monitor Log")
        log_layout = QVBoxLayout()
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #0f172a; color: #38bdf8; font-family: Consolas; font-size: 13px;")
        self.log_output.setText("[System] Ready. Please load a video file.\n")
        
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)

        control_group = QGroupBox("Control Panel")
        control_layout = QVBoxLayout()

        video_layout_h = QHBoxLayout()
        self.video_input = QLineEdit()
        self.video_input.setPlaceholderText("Select video file path (.mp4)")
        self.video_input.setStyleSheet("padding: 6px; font-size: 13px;")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.setStyleSheet("padding: 6px 12px; font-weight: bold;")
        self.btn_browse.clicked.connect(self.action_browse_video)
        video_layout_h.addWidget(self.video_input)
        video_layout_h.addWidget(self.btn_browse)

        self.song_input = QLineEdit()
        self.song_input.setPlaceholderText("Enter song name")
        self.song_input.setStyleSheet("padding: 6px; font-size: 13px;")
        
        self.level_input = QLineEdit()
        self.level_input.setPlaceholderText("Enter difficulty level")
        self.level_input.setStyleSheet("padding: 6px; font-size: 13px;")

        button_style_base = "font-weight: bold; font-size: 14px; border-radius: 6px; min-height: 45px;"

        self.btn_start = QPushButton("Start Video Analysis")
        self.btn_start.setStyleSheet(f"background-color: #f59e0b; color: white; {button_style_base}")

        self.btn_open_folder = QPushButton("Open Records Folder")
        self.btn_open_folder.setStyleSheet(f"background-color: #0284c7; color: white; {button_style_base}")
        
        self.btn_history = QPushButton("View Progress Chart")
        self.btn_history.setStyleSheet(f"background-color: #10b981; color: white; {button_style_base}")

        self.btn_start.clicked.connect(self.action_start_analysis)
        self.btn_open_folder.clicked.connect(self.action_open_folder)
        self.btn_history.clicked.connect(self.action_show_history)

        control_layout.addWidget(QLabel("Video File:"))
        control_layout.addLayout(video_layout_h)
        control_layout.addSpacing(8)
        control_layout.addWidget(QLabel("Song Name:"))
        control_layout.addWidget(self.song_input)
        control_layout.addSpacing(8)
        control_layout.addWidget(QLabel("Difficulty Level:"))
        control_layout.addWidget(self.level_input)
        
        control_layout.addSpacing(25)
        control_layout.addWidget(self.btn_start)
        control_layout.addSpacing(5)
        control_layout.addWidget(self.btn_open_folder)
        control_layout.addSpacing(5)
        control_layout.addWidget(self.btn_history)
        control_layout.addStretch()
        
        control_group.setLayout(control_layout)

        main_layout.addWidget(log_group, 4)
        main_layout.addWidget(control_group, 6)

    def action_browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.mkv *.avi)")
        if file_path:
            self.video_input.setText(file_path)

    def action_start_analysis(self):
        video = self.video_input.text().strip()
        song = self.song_input.text().strip()
        level = self.level_input.text().strip()
        
        if not video or not song or not level:
            self.log_output.append("[Warning] Fields cannot be empty.")
            return
            
        self.btn_start.setEnabled(False)
        self.worker = RealAnalysisWorker(video, song, level)
        self.worker.log_signal.connect(self.log_output.append)
        self.worker.analysis_complete_signal.connect(self.process_user_correction)
        self.worker.start()

    def process_user_correction(self, temp_counts, csv_path, song_name, level):
        self.log_output.append("\n" + "="*40)
        self.log_output.append("[System] Analysis complete. Starting data correction...")
        self.log_output.append("="*40)

        perf, ok1 = QInputDialog.getInt(self, "Data Verification (1/5)", "Enter actual PERFECT count:", 0, 0, 9999, 1)
        if not ok1: perf = 0

        final_counts = {}
        for key in ["Great", "Good", "Bad", "Miss"]:
            detected_val = temp_counts[key]
            val, ok = QInputDialog.getInt(self, f"Data Verification - {key}", f"{key} (Detected: {detected_val})\nModify if incorrect:", detected_val, 0, 9999, 1)
            final_counts[key] = val if ok else detected_val

        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        file_exists = os.path.exists(csv_path)
        
        try:
            with open(csv_path, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Date", "Song", "Perfect", "Great", "Good", "Bad", "Miss"])
                writer.writerow([current_date, song_name, perf, final_counts["Great"], final_counts["Good"], final_counts["Bad"], final_counts["Miss"]])
            
            self.log_output.append(f"[Success] Data saved to: {csv_path}")
            QMessageBox.information(self, "Saved", f"Data saved successfully.")
        except Exception as e:
            self.log_output.append(f"[Error] Failed to write CSV: {e}")

        self.btn_start.setEnabled(True)

    def action_open_folder(self):
        song = self.song_input.text().strip()
        level = self.level_input.text().strip()
        if not song or not level: return
        folder_name = f"{song}_{level}"
        target_path = os.path.join(os.getcwd(), folder_name)
        if os.path.exists(target_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(target_path))

    def action_show_history(self):
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        plt.rcParams['axes.unicode_minus'] = False

        song = self.song_input.text().strip()
        level = self.level_input.text().strip()
        
        csv_path = os.path.join(f"{song}_{level}", "records.csv")
        if not os.path.exists(csv_path):
            self.log_output.append("[Error] Record file not found.")
            return

        play_indices = []
        accuracies = []
        play_count = 1
        
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            try:
                next(reader)
            except StopIteration:
                return

            for row in reader:
                if not row or len(row) < 7: continue 
                try:
                    perfect = int(float(row[2]))
                    great = int(float(row[3]))
                    good = int(float(row[4]))
                    bad = int(float(row[5]))
                    miss = int(float(row[6]))
                    
                    total = perfect + great + good + bad + miss
                    if total == 0: continue
                    acc = ((perfect * 1.0) + (great * 0.7) + (good * 0.4)) / total * 100
                    play_indices.append(f"No. {play_count}")
                    accuracies.append(acc)
                    play_count += 1
                except Exception:
                    continue

        if not accuracies:
            self.log_output.append("[System] No valid data points found in records.")
            return

        plt.figure(f"{song} Progress Chart", figsize=(10, 6))
        plt.plot(play_indices, accuracies, marker='o', linestyle='-', linewidth=2.5, color='#10b981')
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.1, f"{acc:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.title(f"《{song}》[Lv.{level}] Performance Chart", fontsize=16, fontweight='bold')
        plt.xlabel("Play History")
        plt.ylabel("Accuracy (%)")
        plt.ylim(max(0, min(accuracies) - 1.0), 100.5)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = MainWindow()
    window.show()
    sys.exit(app.exec())