import torch
import torch.nn as nn
import torch.optim as optim
from models.xenon_model import XenonModel
from data.data_processor import get_dataloader, RealDataCollector
import yaml
import os
import datetime
import json
import yfinance as yf

def train():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    m_cfg = config['model']
    t_cfg = config['training']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"XenonBrain V5 Training on: {device}")
    os.makedirs('models', exist_ok=True)
    dataloader = get_dataloader(batch_size=t_cfg['batch_size'])
    
    model = XenonModel(
        input_dim=m_cfg['input_dim'],
        hidden_dim=m_cfg['hidden_dim'],
        output_dim=m_cfg['output_dim']
    ).to(device)
    
    model_path = 'models/xenon_brain_latest.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("تم تحميل النسخة السابقة لمواصلة التطور الذاتي...")
        except: pass

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(t_cfg['learning_rate']))
    
    model.train()
    print("بدء دورة التدريب على البيانات الحقيقية والمصححة...")
    
    for epoch in range(t_cfg['epochs']):
        total_loss = 0
        for batch_data, batch_targets in dataloader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch_data)
            
            # الحل الجذري: منع القيم من الخروج عن نطاق (0, 1)
            outputs = torch.clamp(outputs, min=1e-7, max=1.0 - 1e-7)
            
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            # حماية إضافية ضد انفجار التدرجات
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{t_cfg['epochs']}], Loss: {total_loss/len(dataloader):.6f}")
            
    torch.save(model.state_dict(), model_path)
    print(f"تم تحديث XenonBrain V5 بنجاح!")
    generate_daily_report(model, device)

def generate_daily_report(model, device):
    collector = RealDataCollector()
    X, _, latest_news = collector.prepare_data()
    
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(X[-1:]).to(device)
        output = model(input_tensor)
        output = torch.clamp(output, min=0.0, max=1.0)
        
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.max(output).item()

    actual_outcome = None
    try:
        data = yf.download("^GSPC", period="2d", interval="1d", progress=False)
        if len(data) >= 2:
            actual_outcome = 1 if data['Close'].iloc[-1] > data['Close'].iloc[-2] else 0
    except: pass

    collector.update_history(latest_news, prediction, confidence, actual_outcome)
    
    date_str = datetime.datetime.now().strftime("%Y/%m/%d")
    status = "📈 صعودي (Positive)" if prediction == 1 else "📉 حذر/سلبي (Negative)"
    risk = "منخفض 🟢" if confidence > 0.75 else "متوسط 🟡" if confidence > 0.55 else "مرتفع 🔴"
    
    report_content = f"""🧠 تقرير XenonBrain للذكاء الاصطناعي (V5 Evolution) | بتاريخ: {date_str}

🌍 تحليل المشهد الشامل
> "تم دمج بيانات من مصادر عالمية تشمل أخبار التقنية، مؤشرات S&P 500 وNasdaq، مع تحليل معمق للعملات الرقمية (Bitcoin, Ethereum, Solana) لضمان فهم شامل للأنماط."

📊 تحليل الأنماط المنطقية (V5 Deep Logic)
| المعيار | الحالة | التقييم المنطقي |
| :--- | :--- | :--- |
| **اتجاه السوق العالمي** | {status} | تحليل تقاطع الأسهم والعملات الرقمية |
| **قوة النمط (Confidence)** | {confidence*100:.2f}% | درجة اليقين بناءً على التحليل الأخير |
| **مستوى المخاطرة** | {risk} | تقييم استقرار النمط المكتشف |

💡 رؤية XenonBrain (The Insight)
*بناءً على الأنماط المكتشفة، النظام يرى أن الحالة الحالية تشير إلى {'زخم إيجابي ملحوظ في الأصول الرقمية والتقنية' if prediction == 1 else 'ضرورة توخي الحذر الشديد بسبب تقلبات غير منتظمة في البيانات الحالية'}. تم دمج هذه التجربة في الذاكرة التصحيحية V5 لتحسين الاستنتاجات القادمة.*

---
المطور الرئيسي: [Mohamed Ashraf](https://github.com/Mohamed-Ashraf-Ai-Dev)
حالة النظام: متصل وشغال (Active & Evolving V5)
"""
    with open("DAILY_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    try:
        from utils.email_notifier import send_email_report
        send_email_report(report_content)
    except Exception as e:
        print(f"فشل إرسال البريد الإلكتروني: {e}")

    try:
        from utils.visualizer import generate_visuals
        generate_visuals()
    except: pass

if __name__ == "__main__":
    train()
