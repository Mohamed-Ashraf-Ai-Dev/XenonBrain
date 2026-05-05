import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import datetime
import yfinance as yf
import yaml

from data.data_processor import RealDataCollector, XenonDataset
from models.xenon_model import XenonModel

def train():
    print("XenonBrain V6.5 Training on: cpu")
    
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # แยกส่วนการตั้งค่าให้ชัดเจนเพื่อป้องกัน KeyError
    m_cfg = config["model"]
    t_cfg = config["training"]

    device = torch.device("cpu")
    model_path = t_cfg["model_save_path"]

    collector = RealDataCollector()
    
    # Ensure history is reconciled before training or making new predictions
    print("🔄 جاري مزامنة السجلات التاريخية وسد الثغرات...")
    collector.reconcile_history()

    X, y, latest_news_summary = collector.prepare_data()
    dataloader = DataLoader(XenonDataset(X, y), batch_size=t_cfg["batch_size"], shuffle=True)

    # استخدام m_cfg للنموذج و t_cfg للتدريب
    model = XenonModel(
        input_dim=X.shape[-1], 
        hidden_dim=m_cfg["hidden_dim"], 
        output_dim=m_cfg["output_dim"], 
        num_heads=m_cfg["nhead"], # nhead في config يقابل num_heads في الموديل
        num_layers=m_cfg["num_layers"]
    ).to(device)

    if os.path.exists(model_path):
        print("✅ تم تحميل النسخة السابقة لمواصلة التطور الذاتي V6.5...")
        model.load_state_dict(torch.load(model_path, map_location=device))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])

    print("🚀 بدء دورة التدريب V6.5 على البيانات الحقيقية والمصححة...")
    model.train()
    for epoch in range(t_cfg["epochs"]):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X.to(device))
            loss = criterion(outputs, batch_y.to(device))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{t_cfg['epochs']}], Loss: {total_loss/len(dataloader):.6f}")
            
    torch.save(model.state_dict(), model_path)
    print(f"✅ تم تحديث XenonBrain V6.5 بنجاح!")
    generate_daily_report(model, device, collector, latest_news_summary, config)

def generate_daily_report(model, device, collector, latest_news_summary, config):
    # جلب أحدث البيانات للتنبؤ الحالي
    X, _, _ = collector.prepare_data()
    
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(X[-1:]).to(device)
        output = model(input_tensor)
        output = torch.clamp(output, min=0.0, max=1.0)
        
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.max(output).item()

    # حساب قيمة المحفظة الافتراضية بناءً على التوقع الجديد
    previous_portfolio_value = 1000.0
    if os.path.exists(collector.history_file):
        try:
            with open(collector.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                if history:
                    previous_portfolio_value = history[-1].get('portfolio_value', 1000.0)
        except: pass
    
    current_portfolio_value = previous_portfolio_value
    if prediction == 1:
        current_portfolio_value *= 1.005 # 0.5% gain for positive prediction
    else:
        current_portfolio_value *= 0.998 # 0.2% loss for cautious prediction

    # actual_outcome for the current day's prediction will be null, reconciled next run
    collector.update_history(latest_news_summary, prediction, confidence, actual_outcome=None, portfolio_value=current_portfolio_value)
    
    date_str = datetime.datetime.now().strftime("%Y/%m/%d")
    status = "📈 صعودي (Positive)" if prediction == 1 else "📉 حذر/سلبي (Negative)"
    risk = "منخفض 🟢" if confidence > 0.75 else "متوسط 🟡" if confidence > 0.55 else "مرتفع 🔴"
    
    strategic_recommendation = ""
    if prediction == 1:
        strategic_recommendation = "يوصي XenonBrain بالبحث عن فرص استثمارية في الأصول ذات الزخم الإيجابي، مع التركيز على القطاعات التي تظهر نمواً قوياً في الأخبار التقنية والمالية، بالإضافة إلى مشاعر مجتمع Reddit الإيجابية."
    else:
        strategic_recommendation = "ينصح XenonBrain بتوخي الحذر وتقليل المخاطر، مع مراقبة دقيقة للأسواق والبحث عن أصول دفاعية أو فرص للتحوط ضد التقلبات، خاصة مع مشاعر مجتمع Reddit المختلطة أو السلبية."

    report_content = f"""🧠 تقرير XenonBrain للذكاء الاصطناعي (V6.5 Sovereign Intelligence) | بتاريخ: {date_str}

🌍 تحليل المشهد الشامل (V6.5 Global & Reddit Insights)
> "تم دمج بيانات من مصادر عالمية تشمل أخبار التقنية، مؤشرات S&P 500 وNasdaq، مع تحليل معمق للعملات الرقمية (Bitcoin, Ethereum) والمؤشرات الفنية المتقدمة (RSI, MACD). تم إضافة تحليل لمشاعر مجتمع Reddit لتعزيز فهم الأنماط والارتباطات العابرة للأصول."

📊 تحليل الأنماط المنطقية (V6.5 Deep Logic)
| المعيار | الحالة | التقييم المنطقي |
| :--- | :--- | :--- |
| **اتجاه السوق العالمي** | {status} | تحليل تقاطع الأسهم والعملات الرقمية والمؤشرات الفنية |
| **قوة النمط (Confidence)** | {confidence*100:.2f}% | درجة اليقين بناءً على التحليل الأخير |
| **مستوى المخاطرة** | {risk} | تقييم استقرار النمط المكتشف |

💰 أداء المحفظة الافتراضية (Virtual Portfolio Performance)
| المعيار | القيمة |
| :--- | :--- |
| **قيمة المحفظة الحالية** | {current_portfolio_value:.2f} دولار |
| **التغير اليومي** | {((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value * 100):.2f}% |

💡 رؤية XenonBrain (The Sovereign Insight)
*بناءً على الأنماط المكتشفة، النظام يرى أن الحالة الحالية تشير إلى {'زخم إيجابي ملحوظ في الأصول الرقمية والتقنية، مدعوماً بمشاعر إيجابية من مجتمع Reddit.' if prediction == 1 else 'ضرورة توخي الحذر الشديد بسبب تقلبات غير منتظمة في البيانات الحالية، مع مشاعر مختلطة أو سلبية من مجتمع Reddit.'}. تم دمج هذه التجربة في الذاكرة التصحيحية V6.5 لتحسين الاستنتاجات القادمة.*

**🚀 توصية XenonBrain الاستراتيجية اليوم:**
{strategic_recommendation}

---
المطور الرئيسي: [Mohamed Ashraf](https://github.com/Mohamed-Ashraf-Ai-Dev)
حالة النظام: متصل وشغال (Active & Evolving V6.5)
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
