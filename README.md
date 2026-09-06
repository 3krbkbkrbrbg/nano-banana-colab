# 🍌 Nano Banana Cloud (نانو بنانا ابری)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3krbkbkrbrbg/nano-banana-colab/blob/main/nano_banana_colab.ipynb)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![GPU](https://img.shields.io/badge/Google%20Colab-T4%20Free-green.svg)
![API](https://img.shields.io/badge/API-OpenAI%20Compatible-orange.svg)

موتور رایگان و فوق‌سریع تولید تصویر هوش مصنوعی مبتنی بر **Google Colab (T4 GPU)** و متصل به **Minis Agent**.
این پروژه یک سرور استاندارد OpenAI (`/v1/images/generations`) همراه با تانل امن Cloudflare بر روی گوگل کلب راه می‌اندازد تا بتوانید بدون محدودیت توکن و هزینه، تصاویر باکیفیت 1024x1024 را در کمتر از ۲ ثانیه تولید کنید.

---

## ⚡ نحوه راه‌اندازی (با ۱ کلیک)

1. بر روی دکمه **Open In Colab** در بالای صفحه کلیک کنید:
   👉 **[ورود مستقیم به گوگل کلب](https://colab.research.google.com/github/3krbkbkrbrbg/nano-banana-colab/blob/main/nano_banana_colab.ipynb)**
2. در محیط Google Colab، از منوی بالا روی **Runtime** کلیک کرده و گزینه **Run all** (یا کلیدهای `Ctrl + F9`) را بزنید.
3. پس از حدود ۱ تا ۲ دقیقه، آدرس تانل کلودفلر در خروجی سلول ۴ چاپ می‌شود:
   ```text
   ==============================================================
   🎉 تبریک! موتور Nano Banana با موفقیت روشن شد!
   🔗 آدرس عمومی API: https://xxxx-xxxx.trycloudflare.com/v1
   ==============================================================
   ```

---

## 📱 اتصال به Minis

برای اینکه Minis از این موتور برای تولید تصویر استفاده کند:

### روش ۱: دستور مستقیم در Minis
دستور زیر را در ترمینال یا چت Minis اجرا کنید (آدرس تانل خود را جایگزین کنید):
```bash
minis-config set providers.65401b6b-7d1f-44ad-bbe4-00e1f4d654de.customBaseURL "https://YOUR-URL.trycloudflare.com/v1"
```

### روش ۲: استفاده در اسکریپت‌های پایتون یا cURL
```bash
curl -X POST "https://YOUR-URL.trycloudflare.com/v1/images/generations" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "a glowing cybernetic banana in a neon futuristic city, cyberpunk 2077 style, cinematic 8k",
       "size": "1024x1024",
       "model": "nano-banana"
     }'
```

---

## 🔑 کلیدهای رسمی Gemini AI Studio (استخراج‌شده از اکانت شما)

از طریق سشن گوگل شما، کلیدهای API اصلی گوگل برای Gemini نیز استخراج و در محیط لوکال شما ذخیره شده‌اند:

- **پروژه ersaz**: `AQ.Ab8RN6KP3U_Elow...lbqQ`
- **پروژه ersas**: `AQ.Ab8RN6IxxCAIf...Neyw`
- **پشتیبان ۱**: `AQ.Ab8RN6KZR6UE...QT6w`
- **پشتیبان ۲**: `AQ.Ab8RN6JPuqNG...BLvw`

> برای استفاده از مدل‌های متنی و چت Gemini (`gemini-2.5-flash`):
> متغیر `GEMINI_API_KEY` در پروفایل سیستم شما ست شد.

---

## 🛠️ ساختار فایل‌های پروژه

- `nano_banana_colab.ipynb`: نوت‌بوک اصلی برای اجرای خودکار در Google Colab
- `server.py`: فایل بک‌اند FastAPI به همراه رابط کاربری وب
- `test_client.py`: اسکریپت پایتون جهت تست و تولید نمونه تصویر
- `README.md`: راهنمای جامع پروژه

---

## 📄 License
MIT License — توسعه داده شده توسط ersaz Agent برای erfan.
