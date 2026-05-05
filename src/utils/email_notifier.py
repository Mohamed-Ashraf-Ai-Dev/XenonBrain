import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
import datetime
import json
import base64

def send_email_report(report_content=None):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    receiver_email = "mohamedashrafaidev@gmail.com"
    
    if not sender_email or not sender_password:
        print("خطأ: لم يتم العثور على إعدادات البريد الإلكتروني (Secrets).")
        return

    # Load DAILY_REPORT.md content
    if report_content is None:
        if os.path.exists("DAILY_REPORT.md"):
            with open("DAILY_REPORT.md", "r", encoding="utf-8") as f:
                report_content = f.read()
        else:
            print("خطأ: لا يوجد تقرير لإرساله.")
            return

    # Load HISTORY.json for dynamic data
    history_data = []
    if os.path.exists("HISTORY.json"):
        try:
            with open("HISTORY.json", "r", encoding="utf-8") as f:
                history_data = json.load(f)
        except json.JSONDecodeError:
            print("تحذير: فشل في قراءة HISTORY.json، قد يكون فارغاً أو تالفاً.")
            history_data = []

    # Extract latest portfolio value and growth from history
    latest_portfolio_value = 1000.0
    portfolio_growth_percentage = 0.0
    if history_data:
        latest_entry = history_data[-1]
        latest_portfolio_value = latest_entry.get("portfolio_value", 1000.0)
        portfolio_growth_percentage = ((latest_portfolio_value - 1000) / 1000 * 100)

    # Prepare HTML content for email
    html_report_body = report_content.replace("\n", "<br>").replace("###", "<h3>").replace("##", "<h2>").replace("**", "<b>")
    
    # Embed chart image if available
    chart_html = ""
    chart_path = "docs/assets/pattern_history.png"
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as img_file:
            img_data = img_file.read()
            encoded_img = base64.b64encode(img_data).decode("utf-8")
            chart_html = f"""
            <h3 style="color: #60a5fa; text-align: right;">📈 منحنى الأداء التاريخي</h3>
            <img src="data:image/png;base64,{encoded_img}" alt="Historical Performance Chart" style="width: 100%; max-width: 600px; height: auto; display: block; margin: 10px auto;">
            <br>
            """

    # Construct the full HTML email
    html_content = f"""
    <html>
    <body style="font-family: 'Tajawal', Arial, sans-serif; line-height: 1.6; color: #333; direction: rtl; text-align: right; background-color: #f4f7f6;">
        <div style="max-width: 700px; margin: 20px auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
            <div style="background: linear-gradient(135deg, #60a5fa, #a78bfa); color: white; padding: 30px; text-align: center;">
                <h1 style="margin: 0; font-size: 28px;">🧠 XenonBrain V6.5 Intelligence Report</h1>
                <p style="font-size: 16px; opacity: 0.9;">محرك الذكاء الاصطناعي السيادي</p>
            </div>
            <div style="padding: 30px;">
                <h2 style="color: #1a202c; text-align: right; margin-bottom: 20px;">ملخص التقرير اليومي</h2>
                {html_report_body}
                <br>
                {chart_html}
                <h3 style="color: #60a5fa; text-align: right;">💰 أداء المحفظة الافتراضية (نظرة سريعة)</h3>
                <p style="text-align: right; font-size: 16px;">
                    **القيمة الحالية:** <span style="color: #22c55e; font-weight: bold;">{latest_portfolio_value:.2f}$</span><br>
                    **النمو الإجمالي:** <span style="color: {'#22c55e' if portfolio_growth_percentage >= 0 else '#ef4444'}; font-weight: bold;">{portfolio_growth_percentage:.2f}%</span>
                </p>
            </div>
            <div style="background-color: #f0f4f8; padding: 20px; text-align: center; font-size: 12px; color: #6b7280; border-top: 1px solid #e2e8f0;">
                هذا التقرير تم إنشاؤه تلقائياً بواسطة محرك XenonBrain V6.5 Sovereign Intelligence.<br>
                Developed by <a href="https://github.com/Mohamed-Ashraf-Ai-Dev" style="color: #60a5fa; text-decoration: none;">Mohamed Ashraf</a>
            </div>
        </div>
    </body>
    </html>
    """
    
    msg = MIMEMultipart()
    msg["From"] = f"XenonBrain AI <{sender_email}>"
    msg["To"] = receiver_email
    msg["Subject"] = f"🧠 تقرير XenonBrain V6.5 اليومي | {datetime.datetime.now().strftime("%Y-%m-%d")}"
    
    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("✅ تم إرسال تقرير البريد الإلكتروني بنجاح!")
    except Exception as e:
        print(f"❌ فشل إرسال البريد الإلكتروني: {e}")

if __name__ == "__main__":
    send_email_report()
