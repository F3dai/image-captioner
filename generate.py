import os
import json
import time
from datetime import datetime
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import platform

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy

# Configuration
IMAGE_FOLDER = Path("images/")  # Input image directory

# Generate timestamped results directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = Path("results") / timestamp
THUMBS_DIR = RESULTS_DIR / "thumbnails"
RESULT_JSON = RESULTS_DIR / "result.json"
REPORT_HTML = RESULTS_DIR / "report.html"

# Create results directories
THUMBS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Determine device specs
def get_device_specs():
    if device == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            mem_gb = props.total_memory / (1024**3)
            return f"GPU: {name}, Memory: {mem_gb:.1f} GB"
        except Exception:
            return "GPU (name unknown)"
    else:
        cpu = platform.processor() or platform.machine()
        cores = os.cpu_count()
        return f"CPU: {cpu}, Cores: {cores}"

device_specs = get_device_specs()

# Load BLIP model and processor with fast tokenizer to suppress slow processor warning
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base", use_fast=True
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# Load spaCy for noun phrase extraction
nlp = spacy.load("en_core_web_sm")

# Utility functions
def generate_caption(image_path):
    raw = Image.open(image_path).convert('RGB')
    inputs = processor(raw, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

def extract_tags(caption):
    doc = nlp(caption)
    return list({token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]})

def create_thumbnail(image_path, thumb_path, size=(200, 200)):
    img = Image.open(image_path)
    img.thumbnail(size)
    img.save(thumb_path)

# Main processing
data = []
start_time = time.perf_counter()

for img_file in tqdm(list(IMAGE_FOLDER.iterdir())):
    if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
        continue

    img_start = time.perf_counter()
    try:
        caption = generate_caption(img_file)
        tags = extract_tags(caption)
    except Exception:
        caption = None
        tags = []

    img_end = time.perf_counter()
    duration = img_end - img_start

    # Metadata
    file_size = img_file.stat().st_size
    thumb_path = THUMBS_DIR / img_file.name
    create_thumbnail(img_file, thumb_path)

    entry = {
        "filename": img_file.name,
        "caption": caption,
        "tags": tags,
        "file_size_bytes": file_size,
        "time_taken_sec": duration,
        "thumbnail": str(thumb_path.relative_to(RESULTS_DIR))
    }
    data.append(entry)

end_time = time.perf_counter()
total_time = end_time - start_time

# Save JSON data
with open(RESULT_JSON, 'w', encoding='utf-8') as f:
    json.dump({
        "summary": {
            "total_time_sec": total_time,
            "num_images": len(data),
            "device": device,
            "device_specs": device_specs
        },
        "results": data
    }, f, indent=2)

# Generate HTML report
html = [
    "<!DOCTYPE html>",
    "<html lang='en'>",
    "<head><meta charset='UTF-8'><title>Image Labeling Report</title></head>",
    "<body>",
    f"<h1>Image Labeling Report - {timestamp}</h1>",
    f"<p><strong>Device:</strong> {device} ({device_specs})</p>",
    f"<p><strong>Total images:</strong> {len(data)}</p>",
    f"<p><strong>Total time (s):</strong> {total_time:.2f}</p>",
    f"<p><strong>Average time per image (s):</strong> {(total_time/len(data)) if data else 0:.2f}</p>",
    "<hr/>",
]

for entry in data:
    html.append("<div style='margin-bottom:20px;'>")
    html.append(f"<h2>{entry['filename']}</h2>")
    html.append(f"<img src='thumbnails/{entry['filename']}' alt='thumb' style='max-width:200px;'/> <br/>")
    html.append(f"<strong>Caption:</strong> {entry['caption']}<br/>")
    html.append(f"<strong>Tags:</strong> {', '.join(entry['tags'])}<br/>")
    html.append(f"<strong>File size (bytes):</strong> {entry['file_size_bytes']}<br/>")
    html.append(f"<strong>Time taken (s):</strong> {entry['time_taken_sec']:.3f}<br/>")
    html.append("</div><hr/>")

html.append("</body></html>")

with open(REPORT_HTML, 'w', encoding='utf-8') as f:
    f.write("\n".join(html))

print(f"Results saved in: {RESULTS_DIR}")
