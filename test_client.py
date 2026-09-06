#!/usr/bin/env python3
"""
Test client for Nano Banana Colab API
Usage: python3 test_client.py "a majestic lion in cyberpunk style" [api_url]
"""
import sys, json, urllib.request, base64, os

prompt = sys.argv[1] if len(sys.argv) > 1 else "a futuristic glowing neon cybernetic banana, dark moody background, high resolution"
api_url = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NANO_BANANA_API_URL", "http://localhost:8000/v1")

endpoint = f"{api_url.rstrip('/')}/images/generations"
print(f"[*] Sending prompt to {endpoint}...")
print(f"[*] Prompt: {prompt}")

payload = {
    "prompt": prompt,
    "model": "nano-banana",
    "size": "1024x1024",
    "response_format": "b64_json"
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        item = res["data"][0]
        out_path = "/var/minis/attachments/colab_generated.png"
        if "b64_json" in item and item["b64_json"]:
            img_bytes = base64.b64decode(item["b64_json"])
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            print(f"[+] Saved image to {out_path}")
        elif "url" in item:
            print(f"[+] Image URL: {item['url']}")
except Exception as e:
    print(f"[-] Error: {e}")
