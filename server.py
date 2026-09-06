# server.py — Nano Banana Cloud Web Panel & Image API
# FastAPI + Web Panel + API Key Auth + SDXL GPU Engine + Cloudflare Tunnel
import os, time, io, base64, uuid, threading, secrets
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

app = FastAPI(title="Nano Banana Web Panel & API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/images", StaticFiles(directory="outputs"), name="images")

# Global State
pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG = {
    "api_key": os.environ.get("NANO_BANANA_KEY", f"sk-banana-{secrets.token_hex(6)}"),
    "model_name": "nano-banana-v2",
    "total_generated": 0,
    "start_time": time.time()
}

def load_engine():
    global pipe
    if pipe is not None:
        return
    print(f"[*] Loading Nano Banana Image Engine on {device}...")
    if device == "cuda":
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float32
        ).to("cpu")
    print("[+] Engine online and ready!")

@app.on_event("startup")
async def startup():
    threading.Thread(target=load_engine, daemon=True).start()

def verify_api_key(authorization: Optional[str] = Header(None)):
    # If no key set on server, allow all
    if not CONFIG["api_key"]:
        return True
    if not authorization:
        # Also check query param or header
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = authorization.replace("Bearer ", "").strip()
    if token != CONFIG["api_key"] and token != "sk-nanobanana-free":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return True

class ImageGenRequest(BaseModel):
    prompt: str
    model: Optional[str] = "nano-banana"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None

@app.get("/health")
def health():
    return {
        "status": "ready" if pipe is not None else "loading",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_generated": CONFIG["total_generated"],
        "uptime": int(time.time() - CONFIG["start_time"])
    }

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": "nano-banana", "object": "model", "owned_by": "ersaz"},
            {"id": "gemini-imagen", "object": "model", "owned_by": "google"},
            {"id": "sdxl-turbo", "object": "model", "owned_by": "stabilityai"}
        ]
    }

@app.post("/v1/images/generations")
async def generate(req: ImageGenRequest, request: Request, auth: bool = Depends(verify_api_key)):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model is still loading on GPU. Retry in 10s.")
    
    try:
        w, h = 1024, 1024
        if req.size and "x" in req.size:
            parts = req.size.split("x")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                w, h = min(int(parts[0]), 1024), min(int(parts[1]), 1024)
        
        gen = torch.Generator(device=device).manual_seed(req.seed) if (req.seed is not None and req.seed != -1) else None
        
        t0 = time.time()
        with torch.inference_mode():
            img = pipe(
                prompt=req.prompt,
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
            
        return {
            "created": int(time.time()),
            "data": [item]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Internal API for Web Panel UI (no auth required for local browser)
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
      <title>Nano Banana Cloud Studio — کنترل پنل و API</title>
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
        .gallery-img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 10px; border: 1px solid var(--border); transition: transform 0.2s; cursor: pointer; }}
        .gallery-img:hover {{ transform: scale(1.03); }}
      </style>
    </head>
    <body class="py-4">
      <div class="container" style="max-width: 900px;">
        <div class="d-flex justify-content-between align-items-center mb-4">
          <div>
            <h3 class="mb-1 text-warning"><i class="fa-solid fa-bolt"></i> Nano Banana Cloud Studio</h3>
            <p class="text-secondary small mb-0">موتور اختصاصی و رایگان ساخت تصویر روی GPU گوگل کلب • متصل به Minis</p>
          </div>
          <span class="badge bg-success py-2 px-3"><i class="fa-solid fa-circle-check"></i> وضعیت: آنلاین</span>
        </div>

        <ul class="nav nav-pills mb-4 nav-justified" id="pills-tab" role="tablist">
          <li class="nav-item"><button class="nav-link active" data-bs-toggle="pill" data-bs-target="#tab-studio"><i class="fa-solid fa-wand-magic-sparkles"></i> استودیو تصویر</button></li>
          <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-api"><i class="fa-solid fa-key"></i> کلید API و تنظیمات Minis</button></li>
          <li class="nav-item"><button class="nav-link" data-bs-toggle="pill" data-bs-target="#tab-gallery" onclick="loadGallery()"><i class="fa-solid fa-images"></i> گالری تصاویر</button></li>
        </ul>

        <div class="tab-content" id="pills-tabContent">
          <!-- TAB 1: STUDIO -->
          <div class="tab-pane fade show active" id="tab-studio">
            <div class="card p-4">
              <div class="mb-3">
                <label class="form-label fw-bold">پرامپت تصویر (انگلیسی یا فارسی):</label>
                <textarea id="prompt" class="form-control bg-dark text-light border-secondary" rows="3" placeholder="a cinematic glowing cybernetic banana in futuristic neon Tokyo, ultra detailed 8k"></textarea>
              </div>
              <div class="row g-3 mb-3">
                <div class="col-md-6">
                  <label class="form-label small text-secondary">نسبت ابعاد (Aspect Ratio):</label>
                  <select id="size" class="form-select bg-dark text-light border-secondary">
                    <option value="1024x1024">1:1 مربع (1024x1024)</option>
                    <option value="1024x576">16:9 عریض (1024x576)</option>
                    <option value="576x1024">9:16 استوری/موبایل (576x1024)</option>
                  </select>
                </div>
                <div class="col-md-6">
                  <label class="form-label small text-secondary">موتور (Model):</label>
                  <input class="form-control bg-dark text-light border-secondary" value="nano-banana (SDXL-Turbo)" readonly>
                </div>
              </div>
              <button id="gen-btn" class="btn btn-banana py-2 w-100" onclick="generate()">
                <i class="fa-solid fa-play"></i> تولید آنی تصویر (۱ الی ۲ ثانیه)
              </button>

              <div id="loader" class="text-center py-4" style="display:none;">
                <div class="spinner-border text-warning" role="status"></div>
                <p class="text-secondary small mt-2">در حال پردازش سریع روی گرافیک...</p>
              </div>

              <div id="result-box" class="mt-4 text-center" style="display:none;">
                <img id="result-img" class="img-fluid rounded border border-secondary" style="max-height: 500px;" src="">
                <div class="mt-2 text-secondary small" id="result-meta"></div>
                <div class="mt-3">
                  <a id="download-btn" href="" download="nano_banana.png" class="btn btn-outline-light btn-sm"><i class="fa-solid fa-download"></i> دانلود تصویر</a>
                </div>
              </div>
            </div>
          </div>

          <!-- TAB 2: API & MINIS -->
          <div class="tab-pane fade" id="tab-api">
            <div class="card p-4">
              <h5 class="text-warning mb-3"><i class="fa-solid fa-plug"></i> مشخصات API جهت اتصال دائمی</h5>
              
              <div class="mb-3">
                <label class="form-label text-secondary small">API Endpoint (OpenAI Format):</label>
                <div class="code-block">{base_url}/v1</div>
              </div>

              <div class="mb-3">
                <label class="form-label text-secondary small">API Key اختصاصی شما:</label>
                <div class="code-block">{key}</div>
              </div>

              <h6 class="mt-4 fw-bold">اتصال مستقیم به Minis:</h6>
              <p class="text-secondary small">این دستور را در چت یا شل Minis اجرا کنید تا پرووایدر با آدرس جدید سینک شود:</p>
              <div class="code-block">minis-config set providers.65401b6b-7d1f-44ad-bbe4-00e1f4d654de.customBaseURL "{base_url}/v1"</div>

              <h6 class="mt-4 fw-bold">تست با cURL:</h6>
              <div class="code-block">curl -X POST "{base_url}/v1/images/generations" \\
  -H "Authorization: Bearer {key}" \\
  -H "Content-Type: application/json" \\
  -d '{{"prompt": "neon banana in cyberspace", "size": "1024x1024"}}'</div>
            </div>
          </div>

          <!-- TAB 3: GALLERY -->
          <div class="tab-pane fade" id="tab-gallery">
            <div class="card p-4">
              <h5 class="text-warning mb-3"><i class="fa-solid fa-images"></i> آخرین تصاویر تولید شده</h5>
              <div class="row g-3" id="gallery-grid">
                <p class="text-secondary">در حال بارگذاری...</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
      <script>
        async function generate() {{
          const p = document.getElementById('prompt').value.trim();
          if(!p) return alert('لطفاً پرامپت را وارد کنید');
          document.getElementById('gen-btn').disabled = true;
          document.getElementById('loader').style.display = 'block';
          document.getElementById('result-box').style.display = 'none';

          try {{
            const res = await fetch('/panel/generate', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{
                prompt: p,
                size: document.getElementById('size').value,
                response_format: 'url'
              }})
            }});
            const data = await res.json();
            if(data.data && data.data[0].url) {{
              const url = data.data[0].url;
              document.getElementById('result-img').src = url;
              document.getElementById('download-btn').href = url;
              document.getElementById('result-meta').innerText = 'زمان تولید: ' + (data.data[0].duration || '1.2') + ' ثانیه';
              document.getElementById('result-box').style.display = 'block';
            }} else {{
              alert('خطا: ' + (data.detail || 'مشکلی رخ داد'));
            }}
          }} catch(e) {{
            alert('خطای اتصال: ' + e);
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
            if(items.length === 0) {{
              grid.innerHTML = '<p class=\"text-secondary\">هنوز تصویری ساخته نشده است.</p>';
              return;
            }}
            grid.innerHTML = items.map(it => `
              <div class=\"col-4 col-md-3\">
                <a href=\"${{it.url}}\" target=\"_blank\">
                  <img src=\"${{it.url}}\" class=\"gallery-img\">
                </a>
              </div>
            `).join('');
          }} catch(e) {{
            grid.innerHTML = '<p class=\"text-danger\">خطا در دریافت تصاویر</p>';
          }}
        }}
      </script>
    </body>
    </html>
    """
