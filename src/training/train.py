import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import yaml

# إضافة مسار src للنظام
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.xenon_model import XenonModel
from data.data_processor import get_dataloader

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def train():
    config = load_config()
    m_cfg = config['model']
    t_cfg = config['training']
    
    os.makedirs("models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"XenonBrain Training on: {device}")

    # جلب البيانات الحقيقية
    dataloader = get_dataloader(batch_size=t_cfg['batch_size'])

    # تهيئة النموذج القوي
    model = XenonModel(
        input_dim=m_cfg['input_dim'],
        hidden_dim=m_cfg['hidden_dim'],
        output_dim=m_cfg['output_dim'],
        num_layers=m_cfg['num_layers'],
        nhead=m_cfg['nhead']
    ).to(device)
    
    # المحسن ودالة الخسارة
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=t_cfg['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("بدء دورة التدريب على البيانات الحقيقية...")
    model.train()
    for epoch in range(t_cfg['epochs']):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # منع انفجار التدرجات (Gradient Clipping) لضمان الاستقرار
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss/len(dataloader)
        print(f"Epoch [{epoch+1}/{t_cfg['epochs']}], Loss: {avg_loss:.6f}")

    # حفظ النموذج النهائي
    torch.save(model.state_dict(), t_cfg['model_save_path'])
    print(f"تم تحديث XenonBrain بنجاح! النسخة الجديدة محفوظة في: {t_cfg['model_save_path']}")

    # توليد التقرير اليومي وتحديث الذاكرة والرسوم البيانية
    generate_daily_report(model, dataloader, device)
    from utils.visualizer import generate_visuals
    generate_visuals()

def generate_daily_report(model, dataloader, device):
    from data.data_processor import RealDataCollector
    collector = RealDataCollector()
    import datetime
    model.eval()
    with torch.no_grad():
        # الحصول على عينة من البيانات لتقييم النمط الحالي
        batch_x, _ = next(iter(dataloader))
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        probs = torch.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)
        prediction = torch.argmax(avg_probs).item()
        confidence = avg_probs[prediction].item()
    
    # تحديث ذاكرة النظام بالنتيجة الجديدة
    collector.update_history("Auto-generated summary from training session", prediction)

    date_str = datetime.datetime.now().strftime("%Y/%m/%d")
    status = "📈 صعودي (Positive)" if prediction == 1 else "📉 حذر/سلبي (Negative)"
    risk = "منخفض 🟢" if confidence > 0.6 else "متوسط 🟡" if confidence > 0.5 else "مرتفع 🔴"
    
    report_content = f"""# 🧠 تقرير XenonBrain للذكاء الاصطناعي (V4.5) | بتاريخ: {date_str}

## 🌍 تحليل المشهد الشامل
> "قام النظام بتحليل بيانات من مصادر متعددة تشمل MIT, TechCrunch, Wired, وFinancial Times، بالإضافة لمؤشرات S&P 500 وNasdaq وBitcoin لضمان فهم أعمق للأنماط."

## 📊 تحليل الأنماط المنطقية (Logic Partitioning)
| المعيار | الحالة | التقييم المنطقي |
| :--- | :--- | :--- |
| **اتجاه السوق العالمي** | {status} | تحليل تقاطع البيانات المالية مع أخبار التقنية |
| **قوة النمط (Confidence)** | {confidence*100:.2f}% | درجة اليقين بناءً على الذاكرة التاريخية |
| **مستوى المخاطرة** | {risk} | تقييم استقرار النمط الحالي المكتشف |

## 💡 رؤية XenonBrain (The Insight)
*بناءً على الأنماط المكتشفة، النظام يرى أن الحالة الحالية تشير إلى {'فرصة نمو محتملة ناتجة عن زخم تقني' if prediction == 1 else 'ضرورة توخي الحذر بسبب تقلبات في البيانات العالمية'}. تم دمج هذه التجربة في الذاكرة طويلة الأمد لتحسين الاستنتاجات القادمة.*

---
**المطور الرئيسي:** [Mohamed Ashraf](https://github.com/Mohamed-Ashraf-Ai-Dev)
**حالة النظام:** متصل وشغال (Active & Evolving)
"""
    with open("DAILY_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("تم توليد التقرير اليومي بنجاح في DAILY_REPORT.md")

if __name__ == "__main__":
    train()
