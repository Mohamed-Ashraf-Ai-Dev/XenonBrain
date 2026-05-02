import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import datetime

def send_email_report(report_content=None):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    receiver_email = "mohamedashrafaidev@gmail.com"
    
    if not sender_email or not sender_password:
        print("خطأ: لم يتم العثور على إعدادات البريد الإلكتروني (Secrets).")
        return

    # إذا لم يتم تمرير محتوى، حاول قراءته من ملف التقرير اليومي
    if report_content is None:
        if os.path.exists("DAILY_REPORT.md"):
            with open("DAILY_REPORT.md", "r", encoding="utf-8") as f:
                report_content = f.read()
        else:
            print("خطأ: لا يوجد تقرير لإرساله.")
            return

    msg = MIMEMultipart()
    msg['From'] = f"XenonBrain AI <{sender_email}>"
    msg['To'] = receiver_email
    msg['Subject'] = f"🧠 تقرير XenonBrain اليومي | {datetime.datetime.now().strftime('%Y-%m-%d')}"

    # تحويل التقرير من Markdown إلى تنسيق HTML بسيط ليظهر بشكل جميل في الإيميل
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; direction: rtl; text-align: right;">
        <div style="background-color: #1a202c; color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0;">
            <h1 style="margin: 0;">XenonBrain Intelligence Report</h1>
        </div>
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 0 0 10px 10px;">
            {report_content.replace('\n', '<br>').replace('###', '<h3>').replace('##', '<h2>').replace('**', '<b>')}
        </div>
        <div style="margin-top: 20px; font-size: 0.8em; color: #718096; text-align: center;">
            هذا التقرير تم إنشاؤه تلقائياً بواسطة محرك XenonBrain V5.
        </div>
    </body>
    </html>
    """
    
    msg.attach(MIMEText(html_content, 'html'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("✅ تم إرسال تقرير البريد الإلكتروني بنجاح!")
    except Exception as e:
        print(f"❌ فشل إرسال البريد الإلكتروني: {e}")

if __name__ == "__main__":
    send_email_report()
