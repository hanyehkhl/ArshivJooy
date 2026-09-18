# آرشیوجو (ArshivJooy)

موتور جست‌وجوی **معنایی، محلی و کاملاً آفلاین** برای آرشیو شخصی — عکس و سند —
با پشتیبانی کامل از زبان فارسی.

بعد از دانلود یک‌باره‌ی مدل‌ها، هیچ درخواستی به اینترنت فرستاده نمی‌شود و هیچ
فایلی از دستگاه شما خارج نمی‌گردد.

---

## ویژگی‌ها

- جست‌وجوی معنایی فارسی/انگلیسی (نه صرفاً تطبیق کلمه)
- ایندکس اسناد `.pdf` `.txt` `.docx` و عکس‌های `.jpg` `.jpeg` `.png`
- OCR فارسی + انگلیسی روی عکس‌ها با Tesseract
- بردار تصویری CLIP برای عکس‌ها → جست‌وجوی «شبیه این عکس» و متن↔تصویر
- **جست‌وجوی ترکیبی**: متن و عکس در یک پرس‌وجو، ادغام رتبه‌ها با RRF
- ایندکس افزایشی: فایل تکراری دوباره پردازش نمی‌شود (مقایسه‌ی SHA-256)
- نوار پیشرفت با `tqdm` در ترمینال و نوار Gradio در مرورگر
- دو رابط: **CLI** و **Gradio** فارسی

---

## معماری

```
ArshivJooy/
├── app.py                 رابط گرافیکی فارسی (Gradio)
├── cli.py                 رابط خط فرمان (بدون Gradio)
├── requirements.txt
├── LICENSE                MIT
├── NOTICE.md              منابع الهام و مجوزهای شخص ثالث
├── src/
│   ├── config.py          تنظیمات مرکزی + متغیرهای محیطی
│   ├── models.py          دانلود/کش محلی مدل‌ها و حالت offline
│   ├── extractors.py      استخراج متن (pdf/docx/txt)، OCR، نرمال‌سازی، تکه‌بندی
│   ├── embedder.py        کلاس Embedder (متن ۳۸۴ بعدی + عکس ۵۱۲ بعدی)
│   ├── store.py           لایه‌ی ChromaDB
│   ├── ingest.py          اسکن پوشه و ساخت ایندکس
│   └── search.py          جست‌وجو و ادغام رتبه‌ها (RRF)
├── tests/test_core.py
└── data/
    ├── models/            وزن مدل‌ها (دانلود خودکار)
    └── chroma/            پایگاه‌داده‌ی برداری
```

### چرا دو کالکشن؟

بردار متنی `MiniLM` ۳۸۴ بعدی و بردار تصویری `CLIP` ۵۱۲ بعدی است و ChromaDB
اجازه‌ی نگه‌داری دو بُعد مختلف را در یک کالکشن نمی‌دهد. بنابراین:

| کالکشن | محتوا | مدل |
|---|---|---|
| `archive` | متن اسناد + متنِ OCR شده‌ی عکس‌ها (تکه‌تکه) | `paraphrase-multilingual-MiniLM-L12-v2` |
| `archive_images` | یک بردار دیداری به ازای هر عکس | `clip-vit-base-patch32` |

`metadata` هر ردیف شامل `path`, `type` (`image`/`document`), `extracted_text`,
`filename` و نیز `content_hash`, `chunk_index`, `size_bytes`, `modified_at` است.

---

## نصب

### ۱) پیش‌نیاز سیستمی: Tesseract + بسته‌ی زبان فارسی

```bash
# Debian / Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-fas

# macOS
brew install tesseract tesseract-lang

# Windows: نصب از UB-Mannheim و افزودن fas به زبان‌ها،
# سپس تعیین مسیر اجرایی:
#   set ARSHIVJOOY_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### ۲) وابستگی‌های پایتون

```bash
python -m venv .venv
source .venv/bin/activate      # ویندوز: .venv\Scripts\activate
pip install -r requirements.txt
```

### ۳) دانلود یک‌باره‌ی مدل‌ها (تنها مرحله‌ای که به اینترنت نیاز دارد)

```bash
python cli.py models
```

مدل‌ها در `data/models/` ذخیره می‌شوند. اگر این مرحله را اجرا نکنید، در اولین
ایندکس/جست‌وجو به‌صورت خودکار دانلود می‌شوند.

---

## استفاده

### رابط گرافیکی

```bash
python app.py
```

- تب **ایندکس کردن پوشه**: مسیر پوشه را وارد و دکمه را بزنید.
- تب **جست‌وجو**: عبارت فارسی بنویسید، اختیاری یک عکس نمونه بارگذاری کنید و
  نتایج را به‌صورت کارت (نام فایل، مسیر، امتیاز، تکه متن، بندانگشتی) ببینید.

### خط فرمان

```bash
python cli.py index ~/Archive              # ایندکس افزایشی
python cli.py index ~/Archive --force      # ایندکس مجدد کامل
python cli.py search "قرارداد اجاره" -n 10
python cli.py search "ساحل غروب" --image ~/sample.jpg   # جست‌وجوی ترکیبی
python cli.py search "فاکتور" --only document --json
python cli.py stats
python cli.py reset --yes
```

### استفاده به‌صورت کتابخانه

```python
from src.ingest import ingest_folder
from src.search import search

ingest_folder("~/Archive")

for item in search("رسید بانکی", n_results=5):
    print(item["score"], item["type"], item["path"])
    print(item["text_snippet"])
```

---

## تنظیمات (متغیرهای محیطی)

| متغیر | پیش‌فرض | توضیح |
|---|---|---|
| `ARSHIVJOOY_DATA_DIR` | `./data` | محل مدل‌ها و پایگاه‌داده |
| `ARSHIVJOOY_OCR_LANGS` | `fas+eng` | زبان‌های Tesseract |
| `ARSHIVJOOY_TESSERACT_CMD` | — | مسیر اجرایی Tesseract (ویندوز) |
| `ARSHIVJOOY_DEVICE` | `auto` | `cpu` / `cuda` / `mps` |
| `ARSHIVJOOY_CHUNK_SIZE` | `480` | طول تکه‌ی متن |
| `ARSHIVJOOY_CHUNK_OVERLAP` | `80` | هم‌پوشانی تکه‌ها |
| `ARSHIVJOOY_BATCH_SIZE` | `32` | اندازه‌ی دسته در embedding |

---

## تست

```bash
python -m unittest discover -s tests -v
```

---

## نکته‌ها و محدودیت‌ها

- انکودر **متنیِ** CLIP عمدتاً انگلیسی است؛ برای پرس‌وجوی فارسی وزن کمتری
  می‌گیرد و بار اصلیِ جست‌وجوی فارسی روی MiniLM و متنِ OCR است. برای بهترین
  نتیجه روی عکس‌ها، Tesseract فارسی را نصب کنید.
- PDFهای اسکن‌شده متنِ قابل استخراج ندارند؛ در نسخه‌ی فعلی OCR فقط روی
  فایل‌های تصویری اجرا می‌شود.
- امتیاز نمایش‌داده‌شده رتبه‌ی نسبی (نرمال‌شده نسبت به بهترین نتیجه) است، نه
  احتمال؛ شباهت خام هر منبع در کلید `similarities` موجود است.

---

## مجوز

MIT — فایل `LICENSE`.
منابع الهام و مجوز کتابخانه‌ها و مدل‌ها در `NOTICE.md` آمده است.
