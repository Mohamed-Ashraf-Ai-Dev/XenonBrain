import matplotlib.pyplot as plt
import json
import os
import numpy as np

def generate_visuals(history_file="HISTORY.json", output_dir="docs/assets"):
    if not os.path.exists(history_file):
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(history_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        return

    dates = [d['date'].split(' ')[0] for d in data][-10:] # آخر 10 أيام
    predictions = [d['prediction'] for d in data][-10:]
    
    plt.figure(figsize=(10, 5))
    plt.style.use('dark_background')
    
    plt.plot(dates, predictions, marker='o', linestyle='-', color='#00ff88', linewidth=2)
    plt.fill_between(dates, predictions, color='#00ff88', alpha=0.1)
    
    plt.title('XenonBrain Logic Patterns (Last 10 Sessions)', color='white', pad=20)
    plt.xlabel('Date', color='white')
    plt.ylabel('Logic State (0: Neg, 1: Pos)', color='white')
    plt.ylim(-0.2, 1.2)
    plt.grid(True, alpha=0.2)
    
    plt.savefig(os.path.join(output_dir, 'pattern_history.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("تم توليد الرسوم البيانية بنجاح.")

if __name__ == "__main__":
    generate_visuals()
