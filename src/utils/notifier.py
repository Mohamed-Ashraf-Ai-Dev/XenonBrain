import os
import json

def check_and_notify():
    history_file = "HISTORY.json"
    if not os.path.exists(history_file):
        return
        
    with open(history_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if len(data) < 2:
        return
        
    last = data[-1]
    prev = data[-2]
    
    # تنبيه في حالة حدوث تغيير مفاجئ في النمط (من إيجابي لسلبي أو العكس)
    if last['prediction'] != prev['prediction']:
        status = "صعودي 📈" if last['prediction'] == 1 else "حذر 📉"
        print(f"تنبيه: تم رصد تغير في النمط المنطقي إلى {status}")
        # سيتم استخدام GitHub CLI في الـ Workflow لفتح الـ Issue
        with open("ALERT.txt", "w", encoding="utf-8") as f:
            f.write(f"XenonBrain Alert: Logic Shift Detected!\nNew Pattern: {status}\nDate: {last['date']}")

if __name__ == "__main__":
    check_and_notify()
