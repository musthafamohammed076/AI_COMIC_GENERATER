"""
AI Comic Generator - FastAPI Backend
Entry point for the REST API that orchestrates Gemini + Stable Diffusion.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gemini_service import GeminiService
from image_generator import ComicImageGenerator

# Load .env from the project root so GEMINI_API_KEY is available during startup.
project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=project_root / ".env")

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Comic Generator API",
    description="Converts story scripts into AI-generated comic strips",
    version="1.0.0",
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory where generated comic images are saved
OUTPUT_DIR = Path("generated_comics")
OUTPUT_DIR.mkdir(exist_ok=True)

# Serve generated images as static files at /images/<filename>
app.mount("/images", StaticFiles(directory=str(OUTPUT_DIR)), name="images")

# ── Service Instances ─────────────────────────────────────────────────────────
gemini  = GeminiService()
img_gen = ComicImageGenerator()

# ── Request / Response Models ─────────────────────────────────────────────────
class StoryRequest(BaseModel):
    story: str                          # Raw story text from the user

class PanelData(BaseModel):
    panel_number:   int
    scene:          str
    characters:     str
    emotion:        str
    background:     str
    sd_prompt:      str
    image_filename: str
    image_url:      str

class ComicResponse(BaseModel):
    comic_id: str
    panels:   List[PanelData]
    message:  str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Quick liveness probe."""
    return {"status": "ok", "service": "AI Comic Generator"}


@app.post("/generate-comic", response_model=ComicResponse)
async def generate_comic(request: StoryRequest):
    """
    Full pipeline:
      1. Validate input
      2. Gemini → panel breakdown (scene / characters / emotion / background)
      3. Build Stable Diffusion prompts
      4. Generate images
      5. Return panel metadata + image URLs
    """

    # ── 1. Input validation ───────────────────────────────────────────────────
    story = request.story.strip()
    if not story:
        raise HTTPException(status_code=400, detail="Story text cannot be empty.")
    if len(story) < 20:
        raise HTTPException(status_code=400, detail="Story is too short. Please provide more detail.")

    # ── 2. Gemini: break story into panels ───────────────────────────────────
    try:
        panels_metadata = gemini.break_into_panels(story)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")

    if not panels_metadata:
        raise HTTPException(status_code=500, detail="Gemini returned no panels.")

    # ── 3. Unique folder per comic so images don't clash ─────────────────────
    comic_id  = str(uuid.uuid4())[:8]
    comic_dir = OUTPUT_DIR / comic_id
    comic_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. Generate one image per panel ──────────────────────────────────────
    result_panels: List[PanelData] = []

    for panel in panels_metadata:
        panel_num = panel["panel_number"]
        sd_prompt = img_gen.build_prompt(panel)

        filename = f"panel_{panel_num}.png"
        save_path = comic_dir / filename

        try:
            img_gen.generate_image(sd_prompt, str(save_path))
        except Exception as e:
            # Non-fatal: generate a placeholder so the strip still renders
            img_gen.create_placeholder(str(save_path), panel_num, str(e))

        result_panels.append(
            PanelData(
                panel_number   = panel_num,
                scene          = panel.get("scene", ""),
                characters     = panel.get("characters", ""),
                emotion        = panel.get("emotion", ""),
                background     = panel.get("background", ""),
                sd_prompt      = sd_prompt,
                image_filename = f"{comic_id}/{filename}",
                image_url      = f"/images/{comic_id}/{filename}",
            )
        )

    return ComicResponse(
        comic_id = comic_id,
        panels   = result_panels,
        message  = f"Successfully generated {len(result_panels)} comic panels!",
    )


@app.delete("/delete-comic/{comic_id}")
def delete_comic(comic_id: str):
    """Clean up a previously generated comic from disk."""
    comic_dir = OUTPUT_DIR / comic_id
    if comic_dir.exists():
        shutil.rmtree(comic_dir)
        return {"message": f"Comic {comic_id} deleted."}
    raise HTTPException(status_code=404, detail="Comic not found.")
