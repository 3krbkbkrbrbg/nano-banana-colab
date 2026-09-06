# server.py — Nano Banana Cloud Studio & Internet Face/Character Mimic Engine v3.0
import os, time, io, base64, uuid, threading, secrets, urllib.request, urllib.parse, json
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image

app = FastAPI(title="Nano Banana Web Mimic & Image API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("cache", exist_ok=True)
app.mount("/images", StaticFiles(directory="outputs"), name="images")

# Global Pipes
txt_pipe = None
img_pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG = {
    "api_key": os.environ.get("NANO_BANANA_KEY", "sk-nanobanana-free"),
    "total_generated": 0,
    "start_time": time.time()
}

def load_engine():
    global txt_pipe, img_pipe
    if txt_pipe is not None:
        return
    print(f"[*] Loading SDXL-Turbo on {device}...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    txt_pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=dtype, variant="fp16" if device == "cuda" else None
    ).to(device)
    img_pipe = AutoPipelineForImage2Image.from_pipe(txt_pipe).to(device)
    print("[+] Text2Image and Image2Image Face Mimic Engines ready!")

@app.on_event("startup")
async def startup():
    threading.Thread(target=load_engine, daemon=True).start()

def verify_api_key(authorization: Optional[str] = Header(None)):
    if not CONFIG["api_key"]:
        return True
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization token")
    token = authorization.replace("Bearer ", "").strip()
    if token != CONFIG["api_key"] and token != "sk-nanobanana-free":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return True

def search_wiki_person(name: str):
    """Fetch accurate portrait image and physical info from Wikipedia"""
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages|extracts&exintro&explaintext&titles={urllib.parse.quote(name)}&pithumbsize=1024&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "NanoBananaColab/3.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None
            p = list(pages.values())[0]
            thumb = p.get("thumbnail", {}).get("source", None)
            extract = p.get("extract", "")[:300]
            title = p.get("title", name)
            return {"title": title, "image_url": thumb, "bio": extract}
    except Exception as e:
        print(f"Wiki search error: {e}")
        return None

def download_image(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")

class ImageGenRequest(BaseModel):
    prompt: str
    model: Optional[str] = "nano-banana"
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    # Internet / Face Mimic extensions
    mimic_person: Optional[str] = None     # e.g. "Elon Musk", "Steve Jobs", "Geralt of Rivia"
    reference_image_url: Optional[str] = None
    reference_image_b64: Optional[str] = None
    strength: Optional[float] = 0.65       # 0.5 - 0.7 preserves face structure while adapting costume/scene

@app.get("/health")
def health():
    return {
        "status": "ready" if txt_pipe is not None else "loading",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "mimic_enabled": True,
        "total_generated": CONFIG["total_generated"]
    }

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": "nano-banana", "object": "model", "owned_by": "ersaz"},
            {"id": "nano-banana-mimic", "object": "model", "owned_by": "ersaz"},
            {"id": "sdxl-turbo", "object": "model", "owned_by": "stabilityai"}
        ]
    }

@app.get("/api/search_person")
def search_person(q: str):
    info = search_wiki_person(q)
    if not info:
        raise HTTPException(status_code=404, detail="Person/character not found")
    return info

@app.post("/v1/images/generations")
async def generate(req: ImageGenRequest, request: Request, auth: bool = Depends(verify_api_key)):
    global txt_pipe, img_pipe
    if txt_pipe is None:
        raise HTTPException(status_code=503, detail="Model is still loading on GPU. Retry in 10s.")
    
    try:
        w, h = 1024, 1024
        if req.size and "x" in req.size:
            parts = req.size.split("x")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                w, h = min(int(parts[0]), 1024), min(int(parts[1]), 1024)
        
        gen = torch.Generator(device=device).manual_seed(req.seed) if (req.seed is not None and req.seed != -1) else None
        
        # 1. Face Mimic Internet Look-up
        ref_image = None
        prompt = req.prompt
        
        # If mimic_person is provided, fetch their real face from Wikipedia/Internet
        if req.mimic_person and not req.reference_image_url and not req.reference_image_b64:
            person_info = search_wiki_person(req.mimic_person)
            if person_info and person_info.get("image_url"):
                req.reference_image_url = person_info["image_url"]
                prompt = f"{person_info['title']}, {prompt}"
        
        # Load reference image if provided
        if req.reference_image_url:
            try:
                ref_image = download_image(req.reference_image_url).resize((w, h))
            except Exception as ex:
                print(f"Failed to download ref image: {ex}")
        elif req.reference_image_b64:
            try:
                img_data = base64.b64decode(req.reference_image_b64)
                ref_image = Image.open(io.BytesIO(img_data)).convert("RGB").resize((w, h))
            except Exception as ex:
                print(f"Failed to decode ref b64: {ex}")
        
        t0 = time.time()
        with torch.inference_mode():
            if ref_image is not None and img_pipe is not None:
                # Run Face Mimic via Image2Image
                strength = float(req.strength or 0.65)
                img = img_pipe(
                    prompt=prompt,
                    image=ref_image,
                    strength=strength,
                    negative_prompt=req.negative_prompt,
                    num_inference_steps=2,
                    guidance_scale=0.0,
                    generator=gen
                ).images[0]
            else:
                # Run Text2Image
                img = txt_pipe(
                    prompt=prompt,
                    negative_prompt=req.negative_prompt,
                    num_inference_steps=2,
                    guidance_scale=0.0,
                    width=w,
                    height=h,
                    generator=gen
                ).images[0]
        duration = round(time.time() - t0, 2)
        
        CONFIG["total_generated"] += 1
        fname = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        fpath = os.path.join("outputs", fname)
        img.save(fpath, format="PNG")
        
        base_url = str(request.base_url).rstrip("/")
        url = f"{base_url}/images/{fname}"
        
        item = {"url": url, "duration": duration}
        if req.response_format != "url":
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            item["b64_json"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            
        return {"created": int(time.time()), "data": [item]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/panel/generate")
async def panel_generate(req: ImageGenRequest, request: Request):
    return await generate(req, request, auth=True)

@app.get("/panel/gallery")
def get_gallery(request: Request):
    base_url = str(request.base_url).rstrip("/")
    files = sorted(os.listdir("outputs"), reverse=True)[:16]
    return [{"filename": f, "url": f"{base_url}/images/{f}"} for f in files if f.endswith(".png")]

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    base_url = str(request.base_url).rstrip("/")
    key = CONFIG["api_key"]
    return f"""
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
      <meta charset="utf-8">
      <title>Nano Banana Studio — هوش مصنوعی با تقلید چهره و اینترنت</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.rtl.min.css" rel="stylesheet">
      <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
      <style>
        :root {{ --bg: #0b0f19; --card-bg: #151c2e; --border: #232f48; --accent: #eab308; }}
        body {{ background: var(--bg); color: #f1f5f9; font-family: system-ui, -apple-system, sans-serif; }}
        .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; }}
        .nav-pills .nav-link {{ color: #94a3b8; border-radius: 10px; }}
        .nav-pills .nav-link.active {{ background: var(--accent); color: #000; font-weight: bold; }}
        .btn-banana {{ background: var(--accent); color: #000; font-weight: 700; border: none; border-radius: 10px; }}
        .btn-banana:hover {{ background: #ca8a04; }}
        .code-block {{ background: #080b12; border: 1px solid var(--border); border-radius: 10px; padding: 12px; font-family: monospace; font-size: 13px; color: #38bdf8; direction: ltr; text-align: left; }}
        .ref-preview {{ width: 80px; height: 80px; object-fit: cover; border-radius: 8px; border: 2px solid var(--accent); display: none; }}
        .gallery-img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 10px; border: 1px solid var(--border); transition: transform 0.2s; cursor: pointer; }}
        .gallery-img:hover {{ transform: scale(1.03); }}
      </style>
    </head>
    <body class="py-4">
      <div class="container" style="max-width: 900px;">
        <div class="d-flex justify-content-between align-items-center mb-4">
          <div>
            <h3 class="mb-1 text-warning"><i class="fa-solid fa-masks-theater"></i> Nano Banana Mimic Studio</h3>
            <p class="text-secondary small mb-0">موتور متصل به اینترنت جهت تقلید دقیق چهره، کاراکتر و سوژه‌ها • روی GPU کلب</p>
          </div>
          <span class="badge bg-success py-2 px-3"><i class="fa-solid fa-globe"></i> اینترنت: متصل</span>
        </div>

        <ul class="nav nav-pills mb-4 nav-justified" id="pills-tab">
          <li class="nav-item"><button class="nav-link active" data-bs-toggle="pill" data-bs-target="#tab-studio"><i class="fa-solid fa-wand-magic-sparkles"></i> استودیو ساخت و تقلید چهره</button></li>
          <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-api"><i class="fa-solid fa-code"></i> پارامترهای API برای مینیس</button></li>
          <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-gallery" onclick="loadGallery()"><i class="fa-solid fa-images"></i> گالری</button></li>
        </ul>

        <div class="tab-content">
          <!-- TAB 1: STUDIO -->
          <div class="tab-pane fade show active" id="tab-studio">
            <div class="card p-4">
              <!-- Internet Face Search Box -->
              <div class="p-3 mb-3 rounded" style="background: #0f172a; border: 1px dashed #38bdf8;">
                <label class="form-label fw-bold text-info small"><i class="fa-solid fa-magnifying-glass"></i> جستجوی چهره/شخصیت در اینترنت (اختیاری):</label>
                <div class="input-group mb-2">
                  <input type="text" id="person-query" class="form-control bg-dark text-light border-secondary" placeholder="نام شخص یا کاراکتر (مثال: Elon Musk, Steve Jobs, Geralt of Rivia)...">
                  <button class="btn btn-outline-info" type="button" onclick="fetchPerson()"><i class="fa-solid fa-cloud-arrow-down"></i> دریافت چهره از وب</button>
                </div>
                <div class="d-flex align-items-center gap-3 mt-2" id="ref-status-box" style="display:none !important;">
                  <img id="ref-preview-img" class="ref-preview" src="">
                  <div class="small text-secondary">
                    <span id="ref-title" class="fw-bold text-light"></span><br>
                    <span id="ref-bio" class="text-truncate d-inline-block" style="max-width: 400px;"></span>
                  </div>
                </div>
              </div>

              <div class="mb-3">
                <label class="form-label fw-bold">پرامپت تغییر و سناریوی تصویر:</label>
                <textarea id="prompt" class="form-control bg-dark text-light border-secondary" rows="3" placeholder="در نقش جنگجوی سامورایی در توکیوی باستان، نورپردازی سینمایی، کیفیت 8k..."></textarea>
              </div>

              <div class="row g-3 mb-3">
                <div class="col-md-4">
                  <label class="form-label small text-secondary">نسبت ابعاد:</label>
                  <select id="size" class="form-select bg-dark text-light border-secondary">
                    <option value="1024x1024">1:1 مربع</option>
                    <option value="1024x576">16:9 عریض</option>
                    <option value="576x1024">9:16 استوری</option>
                  </select>
                </div>
                <div class="col-md-4">
                  <label class="form-label small text-secondary">میزان وفاداری به چهره (Strength):</label>
                  <select id="strength" class="form-select bg-dark text-light border-secondary">
                    <option value="0.60">بسیار شبیه (حفظ حداکثر چهره)</option>
                    <option value="0.65" selected>متعادل (بهترین ترکیب چهره و استایل)</option>
                    <option value="0.75">تغییر بالا (آزادی بیشتر هوش مصنوعی)</option>
                  </select>
                </div>
                <div class="col-md-4">
                  <label class="form-label small text-secondary">موتور:</label>
                  <input class="form-control bg-dark text-light border-secondary" value="SDXL Mimic (T4 GPU)" readonly>
                </div>
              </div>

              <button id="gen-btn" class="btn btn-banana py-2 w-100" onclick="generate()">
                <i class="fa-solid fa-play"></i> رندر با تقلید چهره و کاراکتر 🚀
              </button>

              <div id="loader" class="text-center py-4" style="display:none;">
                <div class="spinner-border text-warning" role="status"></div>
                <p class="text-secondary small mt-2">در حال ترکیب چهره واقعی با سناریو در ۲ ثانیه...</p>
              </div>

              <div id="result-box" class="mt-4 text-center" style="display:none;">
                <img id="result-img" class="img-fluid rounded border border-secondary" style="max-height: 500px;" src="">
                <div class="mt-2 text-secondary small" id="result-meta"></div>
                <div class="mt-3">
                  <a id="download-btn" href="" download="mimic.png" class="btn btn-outline-light btn-sm"><i class="fa-solid fa-download"></i> دانلود تصویر</a>
                </div>
              </div>
            </div>
          </div>

          <!-- TAB 2: API -->
          <div class="tab-pane fade" id="tab-api">
            <div class="card p-4">
              <h5 class="text-warning mb-3"><i class="fa-solid fa-code"></i> نحوه ارسال از مینیس یا کد پایتون</h5>
              <p class="text-secondary small">می‌توانید مستقیماً پارامتر <code>mimic_person</code> را بفرستید تا سرور خودکار عکس شخص را از اینترنت دانلود کند و چهره‌اش را بازتولید کند:</p>
              
              <div class="code-block">curl -X POST "{base_url}/v1/images/generations" \\
  -H "Authorization: Bearer {key}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "prompt": "wearing futuristic golden cyberpunk armor, cinematic lighting 8k",
    "mimic_person": "Elon Musk",
    "strength": 0.65,
    "size": "1024x1024"
  }}'</div>
            </div>
          </div>

          <!-- TAB 3: GALLERY -->
          <div class="tab-pane fade" id="tab-gallery">
            <div class="card p-4">
              <h5 class="text-warning mb-3"><i class="fa-solid fa-images"></i> تصاویر اخیر</h5>
              <div class="row g-3" id="gallery-grid"><p class="text-secondary">در حال بارگذاری...</p></div>
            </div>
          </div>
        </div>
      </div>

      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
      <script>
        let currentRefUrl = '';

        async function fetchPerson() {{
          const q = document.getElementById('person-query').value.trim();
          if(!q) return alert('نام شخص را وارد کنید');
          try {{
            const res = await fetch('/api/search_person?q=' + encodeURIComponent(q));
            if(!res.ok) throw new Error('یافت نشد');
            const data = await res.json();
            if(data.image_url) {{
              currentRefUrl = data.image_url;
              document.getElementById('ref-preview-img').src = currentRefUrl;
              document.getElementById('ref-preview-img').style.display = 'block';
              document.getElementById('ref-title').innerText = data.title;
              document.getElementById('ref-bio').innerText = data.bio;
              document.getElementById('ref-status-box').style.setProperty('display', 'flex', 'important');
              if(!document.getElementById('prompt').value) {{
                document.getElementById('prompt').value = data.title + ' in cyberpunk style, highly detailed 8k';
              }}
            }} else {{
              alert('عکسی برای این شخصیت یافت نشد.');
            }}
          }} catch(e) {{
            alert('خطا در جستجو: ' + e.message);
          }}
        }}

        async function generate() {{
          const p = document.getElementById('prompt').value.trim();
          if(!p) return alert('لطفاً پرامپت را وارد کنید');
          document.getElementById('gen-btn').disabled = true;
          document.getElementById('loader').style.display = 'block';
          document.getElementById('result-box').style.display = 'none';

          try {{
            const body = {{
              prompt: p,
              size: document.getElementById('size').value,
              strength: parseFloat(document.getElementById('strength').value),
              response_format: 'url'
            }};
            if(currentRefUrl) {{
              body.reference_image_url = currentRefUrl;
            }} else if(document.getElementById('person-query').value.trim()) {{
              body.mimic_person = document.getElementById('person-query').value.trim();
            }}

            const res = await fetch('/panel/generate', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify(body)
            }});
            const data = await res.json();
            if(data.data && data.data[0].url) {{
              document.getElementById('result-img').src = data.data[0].url;
              document.getElementById('download-btn').href = data.data[0].url;
              document.getElementById('result-meta').innerText = 'زمان رندر: ' + (data.data[0].duration || '1.5') + ' ثانیه';
              document.getElementById('result-box').style.display = 'block';
            }} else {{
              alert('خطا: ' + (data.detail || 'تولید نشد'));
            }}
          }} catch(e) {{
            alert('خطا: ' + e);
          }} finally {{
            document.getElementById('gen-btn').disabled = false;
            document.getElementById('loader').style.display = 'none';
          }}
        }}

        async function loadGallery() {{
          const grid = document.getElementById('gallery-grid');
          try {{
            const res = await fetch('/panel/gallery');
            const items = await res.json();
            grid.innerHTML = items.map(it => `
              <div class=\"col-4 col-md-3\">
                <a href=\"${{it.url}}\" target=\"_blank\"><img src=\"${{it.url}}\" class=\"gallery-img\"></a>
              </div>
            `).join('');
          }} catch(e) {{}}
        }}
      </script>
    </body>
    </html>
    """
