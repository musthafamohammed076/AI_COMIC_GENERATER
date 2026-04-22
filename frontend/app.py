"""
app.py  –  Streamlit frontend for the AI Comic Generator
---------------------------------------------------------
Workflow:
  1. User enters a story in the text area
  2. App POSTs to FastAPI /generate-comic
  3. Backend calls Gemini → panel metadata
  4. Backend calls Stable Diffusion → panel images
  5. This page fetches images and renders them as a comic strip
"""

import io
import time
import requests
from PIL import Image

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AI Comic Generator",
    page_icon  = "🎨",
    layout     = "wide",
)

# ── Custom CSS (comic-book aesthetic) ─────────────────────────────────────────
st.markdown("""
<style>
  /* Import Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Comic+Neue:wght@400;700&display=swap');

  /* Global background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
  }

  /* Main title */
  .comic-title {
    font-family: 'Bangers', cursive;
    font-size: 4rem;
    letter-spacing: 4px;
    text-align: center;
    background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
    margin-bottom: 0.2rem;
  }

  .comic-subtitle {
    font-family: 'Comic Neue', cursive;
    text-align: center;
    color: #aaa;
    font-size: 1.1rem;
    margin-bottom: 2rem;
  }

  /* Panel card */
  .panel-card {
    background: #1a1a2e;
    border: 3px solid #e94560;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
    box-shadow: 4px 4px 0px #e94560;
  }

  .panel-number {
    font-family: 'Bangers', cursive;
    font-size: 1.4rem;
    color: #feca57;
    letter-spacing: 2px;
  }

  .panel-meta {
    font-family: 'Comic Neue', cursive;
    font-size: 0.85rem;
    color: #ccc;
    margin-top: 4px;
    line-height: 1.6;
  }

  /* Prompt pill */
  .prompt-pill {
    background: #0f3460;
    border-left: 3px solid #48dbfb;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 0.75rem;
    color: #aef;
    font-family: monospace;
    word-break: break-word;
    margin-top: 8px;
  }

  /* Generate button */
  .stButton > button {
    font-family: 'Bangers', cursive !important;
    font-size: 1.4rem !important;
    letter-spacing: 2px !important;
    background: linear-gradient(90deg, #e94560, #0f3460) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    box-shadow: 3px 3px 0px #feca57 !important;
    transition: transform 0.1s !important;
  }
  .stButton > button:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: 5px 5px 0px #feca57 !important;
  }

  /* Text area */
  .stTextArea textarea {
    background: #0d0d1a !important;
    color: #f0f0f0 !important;
    border: 2px solid #e94560 !important;
    border-radius: 6px !important;
    font-family: 'Comic Neue', cursive !important;
    font-size: 1rem !important;
  }

  /* Divider */
  hr { border-color: #e94560 !important; }

  /* Expander */
  .streamlit-expanderHeader {
    font-family: 'Comic Neue', cursive !important;
    color: #48dbfb !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="comic-title">💥 AI COMIC GENERATOR 💥</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="comic-subtitle">Turn your story into a comic strip powered by Gemini + Stable Diffusion</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Sidebar – settings & tips ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)

    st.markdown("---")
    st.markdown("## 💡 Story Tips")
    st.markdown("""
- **Be specific** about characters (name, age, clothing, hair colour)
- **Describe emotions** and actions clearly
- **Set the scene** — indoor/outdoor, time of day
- Aim for **3–5 plot beats** for best panel breakdown
- Keep stories **50–500 words** for optimal results
    """)

    st.markdown("---")
    st.markdown("## 📌 Example Story")
    example = (
        "Maya, a 10-year-old girl with curly red hair and a yellow raincoat, "
        "discovers a tiny glowing door at the base of an ancient oak tree in the rainy "
        "park. Curious and excited, she opens it and finds a miniature magical kingdom "
        "inside. A small fairy with silver wings greets her with a warm smile. "
        "Together they fly over sparkling rivers and candy-coloured mountains. "
        "Maya returns home at sunset, clutching a tiny glowing acorn as a souvenir."
    )
    if st.button("📋 Load Example"):
        st.session_state["story_input"] = example

# ── Main input area ───────────────────────────────────────────────────────────
story = st.text_area(
    label       = "✍️ Enter Your Story",
    value       = st.session_state.get("story_input", ""),
    height      = 200,
    placeholder = "Once upon a time, a brave astronaut named Zara …",
    key         = "story_input",
)

col_btn, col_clear = st.columns([4, 1])
with col_btn:
    generate_clicked = st.button("🎨 GENERATE MY COMIC!")
with col_clear:
    if st.button("🗑️ Clear"):
        st.session_state["story_input"] = ""
        st.rerun()

# ── Backend health check ──────────────────────────────────────────────────────
def check_backend() -> bool:
    try:
        r = requests.get(f"{backend_url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

# ── Generation flow ───────────────────────────────────────────────────────────
if generate_clicked:
    if not story.strip():
        st.error("⚠️ Please enter a story before generating!")
    elif len(story.strip()) < 20:
        st.error("⚠️ Story is too short. Please add more detail.")
    else:
        # Check backend is alive
        if not check_backend():
            st.error(
                "❌ Cannot reach the backend API.\n\n"
                "Make sure FastAPI is running:\n```bash\ncd backend && uvicorn main:app --reload\n```"
            )
            st.stop()

        # ── Progress display ──────────────────────────────────────────────────
        progress_bar  = st.progress(0)
        status_text   = st.empty()

        steps = [
            (10, "🤖 Sending story to Gemini AI …"),
            (30, "🧠 Breaking story into comic panels …"),
            (50, "🎨 Generating panel images (this takes a few minutes on CPU) …"),
            (80, "🖼️ Assembling your comic strip …"),
            (100, "✅ Done!"),
        ]

        def update_progress(step_idx: int):
            pct, msg = steps[step_idx]
            progress_bar.progress(pct)
            status_text.markdown(f"**{msg}**")

        update_progress(0)
        time.sleep(0.3)
        update_progress(1)

        # ── API call ──────────────────────────────────────────────────────────
        try:
            update_progress(2)
            response = requests.post(
                f"{backend_url}/generate-comic",
                json    = {"story": story},
                timeout = 600,   # SD on CPU can be slow
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Stable Diffusion can be slow on CPU — try again or reduce story length.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            detail = response.json().get("detail", str(e))
            st.error(f"❌ API Error: {detail}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

        update_progress(3)
        time.sleep(0.3)
        update_progress(4)

        # ── Store results in session state so they survive reruns ─────────────
        st.session_state["comic_data"] = data
        status_text.empty()
        progress_bar.empty()

# ── Display comic strip ───────────────────────────────────────────────────────
if "comic_data" in st.session_state:
    data   = st.session_state["comic_data"]
    panels = data.get("panels", [])

    st.markdown("---")
    st.markdown(
        f'<div class="comic-title" style="font-size:2rem;">🗂️ YOUR COMIC STRIP</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**Comic ID:** `{data.get('comic_id', 'N/A')}` &nbsp;|&nbsp; **Panels:** {len(panels)}")
    st.markdown("")

    # Render panels in a 3-column grid
    cols = st.columns(min(len(panels), 3))

    for idx, panel in enumerate(panels):
        col = cols[idx % 3]
        with col:
            # Fetch image from backend static server
            img_url = f"{backend_url}{panel['image_url']}"
            try:
                img_response = requests.get(img_url, timeout=30)
                img_response.raise_for_status()
                img = Image.open(io.BytesIO(img_response.content))
                st.image(img, use_container_width=True)
            except Exception:
                st.warning(f"Could not load image for Panel {panel['panel_number']}")

            # Panel metadata card
            st.markdown(
                f"""
                <div class="panel-card">
                  <div class="panel-number">📖 PANEL {panel['panel_number']}</div>
                  <div class="panel-meta">
                    <b>Scene:</b> {panel['scene']}<br>
                    <b>Characters:</b> {panel['characters']}<br>
                    <b>Emotion:</b> {panel['emotion']}<br>
                    <b>Background:</b> {panel['background']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Expandable SD prompt
            with st.expander("🔍 View SD Prompt"):
                st.markdown(
                    f'<div class="prompt-pill">{panel["sd_prompt"]}</div>',
                    unsafe_allow_html=True,
                )

    # ── Download section ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Download Panels")
    dl_cols = st.columns(len(panels))

    for idx, panel in enumerate(panels):
        with dl_cols[idx]:
            img_url = f"{backend_url}{panel['image_url']}"
            try:
                r   = requests.get(img_url, timeout=30)
                btn = st.download_button(
                    label    = f"Panel {panel['panel_number']}",
                    data     = r.content,
                    file_name= f"panel_{panel['panel_number']}.png",
                    mime     = "image/png",
                )
            except Exception:
                st.write("N/A")

    # ── Regenerate ────────────────────────────────────────────────────────────
    st.markdown("")
    if st.button("🔄 Generate a New Comic"):
        del st.session_state["comic_data"]
        st.rerun()
