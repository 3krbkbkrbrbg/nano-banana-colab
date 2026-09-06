# server.py — Nano Banana Colab Image Generation Server
# OpenAI-compatible API on Google Colab T4 GPU + Cloudflare Tunnel
import os, time, io, base64, uuid, asyncio, threading
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
from PIL import Image

app = FastAPI(title="Nano Banana Colab API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/images", StaticFiles(directory="outputs"), name="images")

# Global pipe
pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_engine():
    global pipe
    if pipe is not None:
        return
    print(f"[*] Loading Nano Banana (SDXL-Lightning) on {device}...")
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    if device == "cuda":
        pipe = AutoPipelineForText2Image.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        pipe.unet.load_state_dict(
            torch.load(f"./checkpoints/{ckpt}", map_location="cuda") 
            if os.path.exists(f"./checkpoints/{ckpt}") 
            else AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
            ).to("cuda").unet.state_dict()
        ) if os.path.exists(f"./checkpoints/{ckpt}") else None
        # fallback directly to sdxl-turbo for instant load if lightning unet not pre-downloaded
    else:
        # CPU fallback / test
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float32
        ).to("cpu")
    
    print("[+] Engine loaded successfully!")

class ImageGenRequest(BaseModel):
    prompt: str
    model: Optional[str] = "nano-banana"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json" # or "url"
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=load_engine, daemon=True).start()

@app.get("/health")
def health():
    return {
        "status": "ready" if pipe is not None else "loading",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "timestamp": int(time.time())
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "nano-banana", "object": "model", "owned_by": "ersaz", "permission": []},
            {"id": "gemini-imagen", "object": "model", "owned_by": "ersaz", "permission": []},
            {"id": "sdxl-turbo", "object": "model", "owned_by": "stabilityai", "permission": []}
        ]
    }

@app.post("/v1/images/generations")
async def generate_image(req: ImageGenRequest, request: Request):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Please wait 15 seconds.")
    
    try:
        # Parse size
        width, height = 1024, 1024
        if req.size:
            parts = req.size.lower().split("x")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                width, height = min(int(parts[0]), 1024), min(int(parts[1]), 1024)
        
        generator = None
        if req.seed is not None and req.seed != -1:
            generator = torch.Generator(device=device).manual_seed(req.seed)
        
        # Run inference (SDXL-Turbo / Lightning requires only 1 to 4 steps!)
        steps = 4 if "lightning" in str(type(pipe)).lower() else 2
        with torch.inference_mode():
            result = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=steps,
                guidance_scale=0.0,
                width=width,
                height=height,
                generator=generator
            )
            image = result.images[0]
        
        # Save output
        filename = f"{uuid.uuid4().hex[:12]}.png"
        filepath = os.path.join("outputs", filename)
        image.save(filepath, format="PNG")
        
        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/images/{filename}"
        
        data_item = {}
        if req.response_format == "url":
            data_item["url"] = image_url
        else:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_item["b64_json"] = b64_str
            data_item["url"] = image_url
        
        return {
            "created": int(time.time()),
            "data": [data_item]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return f"""
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
      <meta charset="utf-8">
      <title>Nano Banana Cloud — Minis</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.rtl.min.css" rel="stylesheet">
      <style>
        body {{ background: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; min-height: 100vh; display: flex; align-items: center; justify-content: center; }}
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        .btn-banana {{ background: #facc15; color: #000; font-weight: bold; }}
        .btn-banana:hover {{ background: #eab308; }}
        #output-img {{ max-width: 100%; border-radius: 12px; margin-top: 15px; display: none; }}
      </style>
    </head>
    <body>
      <div class="container py-4">
        <div class="row justify-content-center">
          <div class="col-12 col-md-8 col-lg-6">
            <div class="card p-4 text-center">
              <h2 class="mb-2">🍌 Nano Banana Engine</h2>
              <p class="text-secondary small mb-4">متصل به Minis • سرعت بالا روی GPU رایگان گوگل کلب</p>
              <div class="text-start mb-3" dir="ltr">
                <label class="form-label text-secondary small">API Endpoint:</label>
                <input class="form-control form-control-sm bg-dark text-light border-secondary" value="{str(request.base_url).rstrip('/')}/v1/images/generations" readonly>
              </div>
              <div class="mb-3">
                <textarea id="prompt" class="form-control bg-dark text-light border-secondary" rows="3" placeholder="توصیف تصویر (مثال: a glowing cybernetic banana in futuristic city)..."></textarea>
              </div>
              <button id="gen-btn" class="btn btn-banana w-100 py-2" onclick="generate()">تولید تصویر 🚀</button>
              <div id="loader" class="spinner-border text-warning my-3" style="display:none;" role="status"></div>
              <img id="output-img" src="" alt="Generated image">
            </div>
          </div>
        </div>
      </div>
      <script>
        async function generate() {{
          const p = document.getElementById('prompt').value.trim();
          if(!p) return alert('لطفاً پرامپت وارد کنید');
          document.getElementById('gen-btn').disabled = true;
          document.getElementById('loader').style.display = 'inline-block';
          document.getElementById('output-img').style.display = 'none';
          try {{
            const res = await fetch('/v1/images/generations', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ prompt: p, size: '1024x1024' }})
            }});
            const data = await res.json();
            if(data.data && data.data[0].url) {{
              const img = document.getElementById('output-img');
              img.src = data.data[0].url;
              img.style.display = 'block';
            }} else {{
              alert('خطا: ' + (data.detail || 'تولید نشد'));
            }}
          }} catch(e) {{
            alert('خطای ارتباط: ' + e);
          }} finally {{
            document.getElementById('gen-btn').disabled = false;
            document.getElementById('loader').style.display = 'none';
          }}
        }}
      </script>
    </body>
    </html>
    """
