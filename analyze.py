import cv2
import os
import csv
import matplotlib.pyplot as plt
import sys

IS_GUI_MODE = False

if len(sys.argv) == 3:
    song_name = sys.argv[1]
    song_level = sys.argv[2]
    IS_GUI_MODE = True
else:
    print("="*40)
    print("Project SEKAI Historical Chart Analysis System")
    print("="*40)
    song_name = input("Enter song name to query: ").strip()
    if song_name == "": song_name = "Untitled"
    
    song_level = input("Enter difficulty level: ").strip()
    if song_level == "": song_level = "0"

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  
plt.rcParams['axes.unicode_minus'] = False                

def draw_progress_chart(s_name, s_level):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    song_folder_name = f"{s_name}_{s_level}"
    song_dir = os.path.join(current_dir, song_folder_name)
    csv_path = os.path.join(song_dir, "records.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Directory '{song_folder_name}' or 'records.csv' not found.")
        return

    play_indices = []  
    accuracies = []    
    hover_dates = []   
    play_count = 1
    
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)  
        
        for row in reader:
            if not row or len(row) < 7: continue
            
            date_str = row[0]
            perfect = int(float(row[2]))
            great = int(float(row[3]))
            good = int(float(row[4]))
            bad = int(float(row[5]))
            miss = int(float(row[6]))
            
            total_notes = perfect + great + good + bad + miss
            if total_notes == 0: continue
            
            weighted_score = (perfect * 1.0) + (great * 0.7) + (good * 0.4)
            accuracy = (weighted_score / total_notes) * 100
            
            play_indices.append(f"No. {play_count}")
            accuracies.append(accuracy)
            hover_dates.append(date_str)
            play_count += 1

    if not accuracies:
        print("Warning: No valid records found.")
        return

    print(f"Loaded {len(accuracies)} records successfully. Generating chart...")

    plt.figure(f"《{s_name}》Progress Curve", figsize=(10, 6))
    plt.plot(play_indices, accuracies, marker='o', linestyle='-', linewidth=2.5, markersize=8, color='#10b981', label="Accuracy")
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.1, f"{acc:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.title(f"《{s_name}》[Lv.{s_level}] Performance Progress", fontsize=16, fontweight='bold')
    plt.xlabel("Play History")
    plt.ylabel("Accuracy %")
    plt.ylim(max(0, min(accuracies) - 1.0), 100.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_progress_chart(song_name, song_level)