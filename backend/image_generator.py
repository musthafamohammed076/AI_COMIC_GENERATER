"""
image_generator.py
------------------
Generates comic-style images using Stable Diffusion running on CPU.

Key optimisations for CPU inference:
  - Only 20 inference steps (vs default 50)
  - 512×512 output (minimum quality threshold)
  - float32 precision (CPU doesn't support float16)
  - Safety checker disabled to save memory
  - Model loaded once and reused across requests
"""

import os
from pathlib import Path
from typing import Dict, Any

from PIL import Image, ImageDraw, ImageFont


# ── Lazy import so the backend starts quickly even without GPU/diffusers ──────
_pipeline = None   # Loaded on first use


def _get_pipeline():
    """Load the Stable Diffusion pipeline once and cache it."""
    global _pipeline
    if _pipeline is None:
        from diffusers import StableDiffusionPipeline
        import torch

        model_id = os.getenv(
            "SD_MODEL_ID",
            "runwayml/stable-diffusion-v1-5"   # Small, well-tested model
        )

        print(f"[ImageGen] Loading Stable Diffusion model: {model_id}")
        print("[ImageGen] This may take a few minutes on first run …")

        _pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype    = torch.float32,   # CPU requires float32
            safety_checker = None,            # Saves ~600 MB of RAM
            requires_safety_checker = False,
        )
        _pipeline = _pipeline.to("cpu")

        # Reduce memory footprint on CPU
        _pipeline.enable_attention_slicing()

        print("[ImageGen] Model loaded successfully.")
    return _pipeline


class ComicImageGenerator:
    """Handles prompt engineering and image generation for comic panels."""

    # Negative prompt discourages common low-quality artefacts
    NEGATIVE_PROMPT = (
        "blurry, low quality, deformed, ugly, bad anatomy, "
        "watermark, signature, extra limbs, duplicate, cropped"
    )

    # ── Prompt builder ─────────────────────────────────────────────────────────
    def build_prompt(self, panel: Dict[str, Any]) -> str:
        """
        Convert a panel metadata dict into a rich Stable Diffusion prompt.

        Template:
          comic book style, <characters>, <emotion> expression,
          <scene>, <background>, cinematic lighting, high detail,
          vibrant colors, sharp lines, professional illustration
        """
        characters = panel.get("characters", "a person")
        emotion    = panel.get("emotion",    "neutral")
        scene      = panel.get("scene",      "")
        background = panel.get("background", "outdoor environment")

        prompt = (
            f"comic book style, graphic novel art, "
            f"{characters}, {emotion} expression, "
            f"{scene}, {background}, "
            f"cinematic lighting, high detail, vibrant colors, "
            f"sharp ink lines, professional comic illustration, "
            f"4k quality"
        )
        return prompt

    # ── Image generation ───────────────────────────────────────────────────────
    def generate_image(self, prompt: str, save_path: str) -> str:
        """
        Run Stable Diffusion inference and save the result as PNG.

        Args:
            prompt    : SD prompt string
            save_path : Full path where the PNG will be saved

        Returns:
            save_path on success.

        Raises:
            RuntimeError if the pipeline fails.
        """
        pipe = _get_pipeline()

        print(f"[ImageGen] Generating image …")
        print(f"[ImageGen] Prompt: {prompt[:120]} …")

        result = pipe(
            prompt          = prompt,
            negative_prompt = self.NEGATIVE_PROMPT,
            num_inference_steps = int(os.getenv("SD_STEPS", "20")),  # Fast on CPU
            width           = 512,
            height          = 512,
            guidance_scale  = 7.5,    # How strongly to follow the prompt
        )

        image: Image.Image = result.images[0]
        image.save(save_path, format="PNG")
        print(f"[ImageGen] Image saved → {save_path}")
        return save_path

    # ── Fallback placeholder ───────────────────────────────────────────────────
    def create_placeholder(
        self,
        save_path: str,
        panel_number: int,
        error_msg: str = "",
    ) -> str:
        """
        Create a styled placeholder image when SD inference fails.
        Keeps the comic strip intact even on partial failures.
        """
        width, height = 512, 512
        img  = Image.new("RGB", (width, height), color="#1a1a2e")
        draw = ImageDraw.Draw(img)

        # Comic-panel border
        draw.rectangle([8, 8, width - 8, height - 8], outline="#e94560", width=4)

        # Panel number label
        draw.rectangle([8, 8, 120, 52], fill="#e94560")
        draw.text((14, 16), f"PANEL {panel_number}", fill="white")

        # Centre message
        msg_lines = [
            "⚠ Image generation",
            "temporarily unavailable",
            "",
            f"Panel {panel_number}",
        ]
        y = height // 2 - 50
        for line in msg_lines:
            # Estimate centre position (no truetype font needed)
            x = width // 2 - len(line) * 4
            draw.text((x, y), line, fill="#e0e0e0")
            y += 28

        # Subtle error hint (truncated)
        if error_msg:
            hint = error_msg[:60]
            draw.text((16, height - 30), f"err: {hint}", fill="#666688")

        img.save(save_path, format="PNG")
        print(f"[ImageGen] Placeholder saved → {save_path}")
        return save_path
