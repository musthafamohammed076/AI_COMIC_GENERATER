"""
gemini_service.py
-----------------
Calls Google Gemini to decompose a story into structured comic panels.

Each panel contains:
  - panel_number  : int  (1-based)
  - scene         : str  (what is happening)
  - characters    : str  (who is in the panel + appearance)
  - emotion       : str  (mood / facial expression)
  - background    : str  (setting / environment)
"""

import os
import json
import re
import textwrap
from typing import List, Dict, Any

import google.generativeai as genai


class GeminiService:
    """Wraps the Gemini 2.5 Flash model for comic-panel decomposition."""

    MODEL_NAME = "gemini-2.5-flash"   # Free-tier friendly model (1.5 is deprecated)

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.MODEL_NAME)

    # ── Public API ─────────────────────────────────────────────────────────────
    def break_into_panels(self, story: str) -> List[Dict[str, Any]]:
        """
        Send the story to Gemini and return a list of panel dicts.
        Raises on API failure or unparseable response.
        """
        prompt = self._build_prompt(story)

        # Call Gemini
        response = self.model.generate_content(prompt)
        raw_text = response.text.strip()

        # Extract and parse JSON from the response
        panels = self._parse_panels(raw_text)
        return panels

    # ── Private helpers ────────────────────────────────────────────────────────
    def _build_prompt(self, story: str) -> str:
        """
        Construct the structured prompt that tells Gemini exactly what JSON
        to return. Being explicit about the format avoids most parsing errors.
        """
        return textwrap.dedent(f"""
            You are a comic book writer. Analyze the following story and break it
            into 3 to 5 sequential comic panels.

            For EACH panel return a JSON object with these EXACT keys:
              - "panel_number"  : integer starting at 1
              - "scene"         : 1-2 sentences describing what is happening
              - "characters"    : detailed description of every visible character
                                  (name, age, clothing, hair, skin tone)
              - "emotion"       : dominant emotion or facial expression
              - "background"    : detailed description of the setting/environment

            Return ONLY a valid JSON array of panel objects — no markdown fences,
            no extra commentary. Example structure:

            [
              {{
                "panel_number": 1,
                "scene": "Hero stands at the edge of the cliff.",
                "characters": "Alex, 16-year-old boy, messy brown hair, blue jacket",
                "emotion": "determined",
                "background": "Rocky cliff at sunset, orange sky, crashing waves below"
              }}
            ]

            STORY:
            {story}
        """).strip()

    def _parse_panels(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Robustly extract a JSON array from Gemini's response, even if it
        accidentally wraps it in markdown code fences.
        """
        # Strip optional ```json ... ``` fences
        cleaned = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()

        # Find the outermost JSON array
        start = cleaned.find("[")
        end   = cleaned.rfind("]")
        if start == -1 or end == -1:
            raise ValueError(
                f"Could not find a JSON array in Gemini response:\n{raw_text[:500]}"
            )

        json_str = cleaned[start : end + 1]
        panels   = json.loads(json_str)

        # Basic validation
        if not isinstance(panels, list) or len(panels) == 0:
            raise ValueError("Gemini returned an empty panel list.")

        required_keys = {"panel_number", "scene", "characters", "emotion", "background"}
        for i, panel in enumerate(panels):
            missing = required_keys - set(panel.keys())
            if missing:
                raise ValueError(
                    f"Panel {i+1} is missing required keys: {missing}"
                )

        # Enforce 3–5 panel limit
        panels = panels[:5]
        return panels
