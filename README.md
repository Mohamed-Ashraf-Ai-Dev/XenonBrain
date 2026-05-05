# 🧠 XenonBrain V6.5: Sovereign Intelligence Engine

![XenonBrain Dashboard](https://raw.githubusercontent.com/Mohamed-Ashraf-Ai-Dev/XenonBrain/main/docs/assets/dashboard_v6.5.png)

مرحباً بك في **XenonBrain V6.5**، محرك الذكاء الاصطناعي السيادي الذي يتجاوز مجرد التنبؤات ليقدم رؤى استراتيجية عميقة في أسواق المال والأصول الرقمية. هذا الإصدار يمثل قفزة نوعية في قدرة النظام على التعلم، التكيف، وتقديم توصيات دقيقة بناءً على تحليل شامل للبيانات العالمية ومشاعر المجتمع.

## ✨ الميزات الرئيسية في V6.5:

*   **الذكاء السيادي (Sovereign Intelligence):** نموذج Transformer متقدم مع طبقات انتباه متعددة الرؤوس (Multi-Head Attention) لفهم الارتباطات المعقدة بين مختلف الأصول (الأسهم، العملات الرقمية، الأخبار التقنية).
*   **المزامنة التاريخية الذكية (Historical Reconciliation):** يقوم النظام الآن بمراجعة وتصحيح توقعاته السابقة تلقائياً بناءً على بيانات السوق الحقيقية، مما يضمن دقة عالية في سجل التعلم.
*   **تكامل مشاعر Reddit (Reddit Sentiment Integration):** يحلل النظام مشاعر مجتمعات Reddit الرائدة في الكريبتو والأسهم لدمج رؤى المجتمع في قراراته الاستراتيجية.
*   **المؤشرات الفنية المتقدمة (Advanced Technical Indicators):** يدمج مؤشرات مثل RSI و MACD في عملية التحليل لتقديم فهم أعمق لزخم السوق.
*   **محفظة افتراضية للتعلم (Virtual Portfolio for Learning):** يتتبع النظام أداء محفظة افتراضية لتقييم فعالية استراتيجياته وتعديلها ذاتياً.
*   **تقارير يومية ذكية (Intelligent Daily Reports):** تقارير مفصلة عبر البريد الإلكتروني ولوحة تحكم تفاعلية تعرض أداء المحفظة، التوقعات، المؤشرات الفنية، ومشاعر Reddit.
*   **واجهة مستخدم متطورة (Advanced Dashboard):** لوحة تحكم حديثة وبديهية تعرض حالة النظام، دقة التوقعات، ومؤشرات المزامنة التاريخية.

## 🛠️ هيكلية المشروع:

*   `src/`:
    *   `data/data_processor.py`: مسؤول عن جلب البيانات من مصادر متعددة (أخبار، أسواق، Reddit)، حساب المؤشرات الفنية، وتصحيح السجلات التاريخية.
    *   `models/xenon_model.py`: تعريف معمارية نموذج الذكاء الاصطناعي (Transformer مع Attention).
    *   `training/train.py`: منطق التدريب، التنبؤ، إدارة المحفظة الافتراضية، وتوليد التقارير اليومية.
    *   `utils/email_notifier.py`: لإرسال التقارير اليومية عبر البريد الإلكتروني.
    *   `utils/notifier.py`: لإرسال تنبيهات GitHub Issues في حالة حدوث "تحولات منطقية" كبيرة.
*   `config/config.yaml`: ملف إعدادات النموذج والتدريب.
*   `models/xenon_brain_latest.pth`: أحدث نسخة مدربة من النموذج.
*   `HISTORY.json`: سجل تاريخي لجميع التوقعات، النتائج الفعلية، وأداء المحفظة.
*   `DAILY_REPORT.md`: التقرير اليومي الأخير بصيغة Markdown.
*   `docs/`: يحتوي على ملف `index.html` الخاص بلوحة التحكم التفاعلية.
*   `.github/workflows/main_training.yml`: ملف GitHub Actions لأتمتة عملية التدريب والنشر اليومية.

## 🚀 البدء السريع:

1.  **استنساخ المستودع:**
    ```bash
    git clone https://github.com/Mohamed-Ashraf-Ai-Dev/XenonBrain.git
    cd XenonBrain
    ```
2.  **إعداد البيئة:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **إعداد GitHub Secrets:**
    *   `SENDER_EMAIL`: بريد إلكتروني لإرسال التقارير (يفضل Gmail).
    *   `SENDER_PASSWORD`: كلمة مرور التطبيق (App Password) للبريد الإلكتروني.
    *   `GITHUB_TOKEN`: رمز مميز بصلاحيات `repo` و `workflow`.
4.  **تشغيل الـ Workflow:**
    *   قم بتشغيل الـ Workflow يدوياً من تبويب Actions في GitHub، أو انتظر حتى يعمل تلقائياً حسب الجدول الزمني.

## 🌐 لوحة التحكم التفاعلية:

يمكنك متابعة أداء XenonBrain V6.5 مباشرة عبر لوحة التحكم التفاعلية:
[https://Mohamed-Ashraf-Ai-Dev.github.io/XenonBrain/](https://Mohamed-Ashraf-Ai-Dev.github.io/XenonBrain/)

--- 

**XenonBrain V6.5: حيث يلتقي الذكاء الاصطناعي بالاستقلالية السيادية.**

Developed by [Mohamed Ashraf](https://github.com/Mohamed-Ashraf-Ai-Dev)
