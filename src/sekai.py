import cv2
import os
import numpy as np
import csv
import shutil
from datetime import datetime
import sys

IS_GUI_MODE = False

if len(sys.argv) == 4:
    video_path = sys.argv[1]
    song_name = sys.argv[2]
    song_level = sys.argv[3]
    IS_GUI_MODE = True
else:
    print("="*40)
    print("Project SEKAI Video Analysis System")
    print("="*40)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        video_filename = input("Enter video filename: ").strip()
        if video_filename == "": video_filename = "gameplay.mp4"
            
        video_path = os.path.join(current_dir, video_filename)
        if os.path.exists(video_path):
            print(f"File found: {video_filename}\n")
            break 
        else:
            print(f"Error: File '{video_filename}' not found.\n")

    song_name = input("Enter song name: ").strip()
    if song_name == "": song_name = "Untitled"
    
    song_level = input("Enter difficulty level: ").strip()
    if song_level == "": song_level = "0"

def cv2_imread_chinese(filepath):
    img_data = np.fromfile(filepath, dtype=np.uint8)
    return cv2.imdecode(img_data, cv2.IMREAD_COLOR)

def cv2_imwrite_chinese(filepath, img):
    ext = os.path.splitext(filepath)[1]
    result, n = cv2.imencode(ext, img)
    if result:
        with open(filepath, mode='wb') as f:
            n.tofile(f)

def save_to_records_csv(csv_path, temp_counts, song_name):
    final_counts = {}
    perfect_count = 0
    
    if not IS_GUI_MODE:
        print("="*40)
        print("Analysis finished. Entering manual correction phase.")
        print("="*40)
        while True:
            try:
                perfect_count = int(input("Enter actual PERFECT count: "))
                break
            except ValueError:
                print("Error: Please enter an integer.")

        print("\nReview detected data. Press [Enter] to keep, or type number to modify:")
        for judgement, detected_val in temp_counts.items():
            user_input = input(f" -> {judgement} (Detected: {detected_val}) Modified: ").strip()
            if user_input == "":
                final_counts[judgement] = detected_val 
            else:
                try:
                    final_counts[judgement] = int(user_input) 
                except ValueError:
                    print(f"Warning: Invalid input. Kept default: {detected_val}")
                    final_counts[judgement] = detected_val
    else:
        for judgement, detected_val in temp_counts.items():
            final_counts[judgement] = detected_val

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Song", "Perfect", "Great", "Good", "Bad", "Miss"])
            
        writer.writerow([
            current_date, song_name, perfect_count, 
            final_counts["Great"], final_counts["Good"], final_counts["Bad"], final_counts["Miss"]
        ])
        
    print(f"\nData saved successfully to: {csv_path}")

def template_based_detection(video_path, run_dir, templates_dir, csv_path, song_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return "error"

    ret, frame = cap.read()
    if not ret:
        print("Error: Video is empty.")
        cap.release()
        return "error"

    display_scale = 0.5
    resized_frame = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)

    print("\nSelect JUDGEMENT ROI window, complete with SPACE/ENTER.")
    roi_judge_rect = cv2.selectROI("1. Select JUDGEMENT ROI", resized_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("1. Select JUDGEMENT ROI")
    jx, jy, jw, jh = [int(val / display_scale) for val in roi_judge_rect]

    print("Select COMBO ROI window, complete with SPACE/ENTER.")
    roi_combo_rect = cv2.selectROI("2. Select COMBO ROI", resized_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("2. Select COMBO ROI")
    cx, cy, cw, ch = [int(val / display_scale) for val in roi_combo_rect]
    
    judgement_templates = {}
    required_templates = ["great", "good", "bad", "miss", "combo"]
    
    for j_name in required_templates:
        img_filename = f"{j_name}.png"
        path = os.path.join(templates_dir, img_filename)
        if not os.path.exists(path): continue
        judgement_templates[j_name.capitalize()] = cv2_imread_chinese(path)

    combo_tpl = judgement_templates.pop("Combo", None)
    if combo_tpl is None:
        cap.release()
        return "error"

    thresholds = {"Great": 0.58, "Good": 0.65, "Bad": 0.65, "Miss": 0.60, "Combo": 0.55}
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 60.0 

    screenshot_count = 0
    cooldown_frames = 0
    cooldown_max = int(fps * 0.15) 
    
    temp_counts = {"Great": 0, "Good": 0, "Bad": 0, "Miss": 0}
    scales = np.linspace(0.5, 1.5, 11)

    print("Analyzing video... Press 'q' to exit, press 'r' to reset ROI.")
    action_status = "done" 
    
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
                print(f"[Match] {detected_judgement} ({global_best_val:.2f}) at {current_time:.2f}s")

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
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('r'): 
            action_status = "restart"
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if action_status == "restart":
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir) 
        return "restart" 
    else:
        save_to_records_csv(csv_path, temp_counts, song_name)
        return "done"                                                                                                                                         

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    song_folder_name = f"{song_name}_{song_level}"
    song_dir = os.path.join(current_dir, song_folder_name)
    os.makedirs(song_dir, exist_ok=True)

    templates_dir = os.path.join(current_dir, "templates")
    csv_path = os.path.join(song_dir, "records.csv")

    while True:
        existing_runs = []
        if os.path.exists(song_dir):
            for item in os.listdir(song_dir):
                item_path = os.path.join(song_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    existing_runs.append(int(item))
                
        run_number = max(existing_runs) + 1 if existing_runs else 1
        run_dir = os.path.join(song_dir, str(run_number))
        os.makedirs(run_dir, exist_ok=True)

        if not os.path.exists(templates_dir):
            print(f"Error: Templates directory '{templates_dir}' not found.")
            break
        
        status = template_based_detection(video_path, run_dir, templates_dir, csv_path, song_name)
        
        if status == "done":
            break
        elif status == "restart":
            continue