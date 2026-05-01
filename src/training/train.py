import torch
import torch.nn as nn
import torch.optim as optim
from models.xenon_model import XenonModel
from data.data_processor import get_dataloader, RealDataCollector
import yaml
import os
import datetime
import json

def train():
    # تحميل الإعدادات
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    m_cfg = config['model']
    t_cfg = config['training']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"XenonBrain Training on: {device}")

    # إنشاء المجلدات
    os.makedirs('models', exist_ok=True)
    
    # تحميل البيانات
    dataloader = get_dataloader(batch_size=t_cfg['batch_size'])
    
    # بناء النموذج
    model = XenonModel(
        input_dim=m_cfg['input_dim'],
        hidden_dim=m_cfg['hidden_dim'],
        output_dim=m_cfg['output_dim']
    ).to(device)
    
    # محاولة تحميل آخر نسخة لتطويرها (Incremental Learning)
    model_path = 'models/xenon_brain_latest.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("تم تحميل النسخة السابقة لمواصلة التطور...")
        except: pass

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(t_cfg['learning_rate']))

    # دورة التدريب
    print("بدء دورة التدريب على البيانات الحقيقية والمصححة...")
    model.train()
    for epoch in range(t_cfg['epochs']):
        total_loss = 0
        for batch_data, batch_targets in dataloader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{t_cfg['epochs']}], Loss: {total_loss/len(dataloader):.6f}")

    # حفظ النموذج المطور
    torch.save(model.state_dict(), model_path)
    print(f"تم تحديث XenonBrain بنجاح! النسخة الجديدة محفوظة في: {model_path}")

    # توليد التقرير والذاكرة
    generate_daily_report(model, device)

def generate_daily_report(model, device):
    collector = RealDataCollector()
    X, _, latest_news = collector.prepare_data()
    
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(X[-1:]).to(device)
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.max(output).item()

    # تحديث الذاكرة
    collector.update_history(latest_news, prediction, confidence)
    
    # إنشاء التقرير العربي المطور
    date_str = datetime.datetime.now().strftime("%Y/%m/%d")
    status = "📈 صعودي (Positive)" if prediction == 1 else "📉 حذر/سلبي (Negative)"
    risk = "منخفض 🟢" if confidence > 0.7 else "متوسط 🟡" if confidence > 0.5 else "مرتفع 🔴"
    
    report_content = f"""🧠 تقرير XenonBrain للذكاء الاصطناعي (V4.5+) | بتاريخ: {date_str}

🌍 تحليل المشهد الشامل
> "قام النظام بتحليل بيانات من مصادر متعددة تشمل أخبار التقنية العالمية، مؤشرات S&P 500 وNasdaq، بالإضافة لتحليل العملات الرقمية (Bitcoin & Ethereum) لضمان فهم أعمق للأنماط الحالية."

📊 تحليل الأنماط المنطقية (Advanced Logic)
| المعيار | الحالة | التقييم المنطقي |
| :--- | :--- | :--- |
| **اتجاه السوق العالمي** | {status} | دمج بيانات الأسهم والعملات الرقمية |
| **قوة النمط (Confidence)** | {confidence*100:.2f}% | درجة اليقين بناءً على التحليل الأخير |
| **مستوى المخاطرة** | {risk} | تقييم استقرار النمط المكتشف |

💡 رؤية XenonBrain (The Insight)
*بناءً على الأنماط المكتشفة، النظام يرى أن الحالة الحالية تشير إلى {'تفاؤل تقني قد ينعكس إيجاباً على الأصول الرقمية' if prediction == 1 else 'ضرورة الحذر بسبب تقلبات غير منتظمة في البيانات العالمية'}. تم دمج هذه التجربة في الذاكرة التصحيحية لتحسين الاستنتاجات القادمة.*

---
المطور الرئيسي: [Mohamed Ashraf](https://github.com/Mohamed-Ashraf-Ai-Dev)
حالة النظام: متصل وشغال (Active & Evolving)
"""
    with open("DAILY_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    # تشغيل المصور التلقائي
    try:
        from utils.visualizer import generate_visuals
        generate_visuals()
    except: pass

if __name__ == "__main__":
    train()
