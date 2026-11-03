import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import TEST_IMAGES_DIR, RESULTS_DIR
from app.core.inference import pipeline

# App setup
app = FastAPI(
    title="PneumoAI Diagnostic PACS Service",
    description="Decoupled Lesion Delineation & Consensus Reporting System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"

# Mount static directories
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/test_images", StaticFiles(directory=str(BASE_DIR / "test_images")), name="test_images")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Request Schema
class DemoPredictRequest(BaseModel):
    filename: str

@app.get("/")
async def index(request: Request):
    """Renders the Medical PACS Web Dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    """Healthcheck endpoint reporting model status."""
    return {
        "status": "online",
        "unet_model_loaded": pipeline.unet.model is not None,
        "yolo_model_loaded": pipeline.yolo.model is not None,
        "device": str(pipeline.unet.device)
    }

@app.get("/api/demo-cases")
async def get_demo_cases():
    """Returns available benchmark demo CXR images from test_images/demo_cases_raw."""
    cases_meta = [
        {
            "filename": "01_Severe_Large_Pneumothorax.png",
            "title": "Case 01: Severe Tension Pneumothorax",
            "category": "Emergency | Visceral Pleura Detachment",
            "badge": "emergency"
        },
        {
            "filename": "02_Apical_Classic.png",
            "title": "Case 02: Classic Apical Pneumothorax",
            "category": "Priority | Apex Radiolucency",
            "badge": "priority"
        },
        {
            "filename": "03_Lateral_Streak.png",
            "title": "Case 03: Lateral Pleural Separation",
            "category": "Priority | Lateral Rib Margin",
            "badge": "priority"
        },
        {
            "filename": "04_Subtle_Small_Pneumothorax.png",
            "title": "Case 04: Subtle Small Pneumothorax",
            "category": "Priority | Fine Visceral Line",
            "badge": "priority"
        },
        {
            "filename": "05_Clean_Normal_Lung.png",
            "title": "Case 05: Normal Healthy Chest",
            "category": "Routine | Clean Normal Variant",
            "badge": "routine"
        }
    ]

    output_cases = []
    for item in cases_meta:
        file_path = TEST_IMAGES_DIR / item["filename"]
        if file_path.exists():
            output_cases.append({
                **item,
                "thumbnail_url": f"/test_images/demo_cases_raw/{item['filename']}"
            })

    return {"cases": output_cases}

@app.post("/api/predict/demo")
async def predict_demo(req: DemoPredictRequest):
    """Executes AI diagnosis on a selected demo case."""
    target_path = TEST_IMAGES_DIR / req.filename
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Demo image {req.filename} not found.")

    try:
        report = pipeline.run(target_path)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """Executes AI diagnosis on a user-uploaded CXR radiograph."""
    try:
        contents = await file.read()
        report = pipeline.run(contents)
        report["source_image"] = file.filename
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
