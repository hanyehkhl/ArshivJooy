# NOTICE — منابع و حقوق مؤلف

## الهام‌گیری (Inspiration)

ایده‌ی کلی این پروژه از پروژه‌ی **ImageBrain** الهام گرفته شده است:

- مخزن: https://github.com/ostad-ai/Image-Brain
- مجوز آن مخزن: MIT

**هیچ کدی از آن پروژه کپی یا مشتق نشده است.** آن مخزن یک برنامه‌ی
بسته‌بندی‌شده‌ی ویندوزی (`ImageBrain.exe` + آرشیوهای نصب) منتشر می‌کند و کد
منبع پایتونی در آن در دسترس نیست؛ بنابراین تنها چیزی که از آن گرفته شده
«مفهوم محصول» است: جست‌وجوی محلی و آفلاین روی آرشیو شخصی با پشتیبانی از
فارسی و انگلیسی. تمام کد موجود در `ArshivJooy` به‌صورت مستقل و از صفر نوشته
شده و تحت مجوز MIT (فایل `LICENSE`) منتشر می‌شود.

اگر بعداً بخواهید بخشی از کد آن پروژه را واقعاً استفاده کنید، طبق مجوز MIT
موظف هستید متن کامل مجوز و اعلان حق مؤلف آن‌ها را نیز در همین فایل بیاورید.

## مدل‌ها و وابستگی‌های شخص ثالث

| مؤلفه | منبع | مجوز |
|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face | Apache-2.0 |
| `openai/clip-vit-base-patch32` | Hugging Face | MIT |
| ChromaDB | https://github.com/chroma-core/chroma | Apache-2.0 |
| Tesseract OCR (باینری سیستمی) | https://github.com/tesseract-ocr/tesseract | Apache-2.0 |
| `pytesseract`, `pypdf`, `python-docx`, `Pillow`, `Gradio`, `tqdm` | PyPI | MIT / BSD / Apache-2.0 |

وزن‌های مدل‌ها در مخزن قرار نمی‌گیرند؛ در اولین اجرا دانلود و در
`data/models/` ذخیره می‌شوند و پس از آن همه‌چیز آفلاین کار می‌کند.
