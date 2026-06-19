"""
genai_curriculum_video.py
─────────────────────────
Pipeline: Curriculum brief → Claude → 8 weekly slides (PIL + TTF) → MP4

"""

import os, json, textwrap, pathlib, tempfile
from dotenv import load_dotenv

load_dotenv()
import anthropic
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, concatenate_videoclips
import imageio_ffmpeg
import moviepy.config as mpy_config
mpy_config.FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()


# ── Fonts  ─────────────────────────────────────
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_REG  = "C:/Windows/Fonts/arial.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"

W, H        = 1280, 720
TOTAL_WEEKS = 8
BG_COLOR    = "#0d1117"   # GitHub dark — consistent across all slides
ACCENT      = "#58a6ff"   # single accent colour; clean and readable on dark


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLAUDE — generate all 8 weeks as structured JSON
# ─────────────────────────────────────────────────────────────────────────────

CURRICULUM_BRIEF = """
You are designing an 8-week GenAI and Agentic AI corporate training programme.
The audience is mixed: software engineers, IT ops, and business analysts.
Sessions are 1 hour each, delivered live via Google Skill Boost notebooks.

The progression is:
  Week 1 — GenAI Fundamentals (Tokens, LLMs, APIs)
  Week 2 — Prompt Engineering (zero-shot, few-shot, CoT, guardrails)
  Week 3 — How Transformers Work (attention, embeddings, fine-tuning preview)
  Week 4 — RAG — Retrieval-Augmented Generation (vector search, chunking)
  Week 5 — Graph RAG with Neo4j (knowledge graphs, RAGAS evaluation)
  Week 6 — AI Agents & Tool Use (function calling, ReAct pattern)
  Week 7 — Multi-Agent Systems (orchestrator, sub-agents, human-in-the-loop)
  Week 8 — Production & ITSM Integration Demo (ServiceNow, audit logging, SLA gates)
"""

def extract_weeks(brief: str) -> list[dict]:
    """
    Ask Claude to produce exactly 8 week objects.
    Each object:
      week_number : int  1–8
      title       : str  short week title (max 10 words)
      topics      : list[str]  exactly 3 bullet topics (max 7 words each)
      hands_on    : str  one hands-on demo label (max 10 words)
      duration    : int  seconds this slide plays (7 or 8)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Please set it in your terminal, e.g., $env:ANTHROPIC_API_KEY='your-api-key'")
        exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""
{brief}

Return ONLY valid JSON — a list of exactly 8 objects. No markdown, no explanation.
Each object must have ALL of these keys:
  "week_number" : integer 1 to 8
  "title"       : short week title, max 6 words
  "topics"      : list of exactly 3 bullet strings, max 7 words each
  "hands_on"    : one hands-on demo label, max 10 words, start with "Demo:"
  "duration"    : integer, either 7 or 8 seconds
"""

    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    weeks = json.loads(raw.strip())
    assert len(weeks) == 8, f"Expected 8 weeks, got {len(weeks)}"
    return weeks


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIL — render each week as a 1280×720 slide
# ─────────────────────────────────────────────────────────────────────────────

def render_slide(week: dict, tmp_dir: str = None) -> str:
    """
    Renders one week slide and saves it to {tmp_dir}/week_{n}.png.

    Layout (top to bottom):
      44px  — top padding
      Label pill  "WEEK 01"  +  counter "01 / 08"  top-right
      96px  — horizontal accent rule
      gap
      Title      64px bold white     centred
      3 topics   28px regular        accent colour  left-aligned block
      hands_on   24px mono           #606070 dimmed centred
      gap
      10px  — bottom accent bar
    """
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    
    img  = Image.new("RGB", (W, H), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    wn = week["week_number"]

    # ── Label pill: "WEEK 01" ─────────────────────────────────────────────────
    font_label = ImageFont.truetype(FONT_MONO, 20)
    pill_text  = f"WEEK {wn:02d}"
    bbox = draw.textbbox((0, 0), pill_text, font=font_label)
    pw = bbox[2] - bbox[0] + 28
    ph = bbox[3] - bbox[1] + 14
    draw.rounded_rectangle([48, 44, 48 + pw, 44 + ph], radius=5, fill=ACCENT)
    draw.text((48 + 14, 44 + 7), pill_text, font=font_label, fill=BG_COLOR)

    # ── Slide counter top-right ───────────────────────────────────────────────
    font_counter = ImageFont.truetype(FONT_REG, 18)
    draw.text((W - 52, 52), f"{wn:02d} / {TOTAL_WEEKS:02d}",
              font=font_counter, fill=ACCENT, anchor="rm")

    # ── Horizontal rule ───────────────────────────────────────────────────────
    draw.rectangle([48, 96, W - 48, 98], fill=ACCENT)

    # ── Week title ────────────────────────────────────────────────────────────
    font_title = ImageFont.truetype(FONT_BOLD, 58)
    title_lines = textwrap.wrap(week["title"], width=26)
    title_line_h = 72
    title_block_h = len(title_lines) * title_line_h

    # Reserve space: title block + gap + 3 topics + gap + hands_on
    topic_block_h  = 3 * 38
    handson_h      = 36
    total_content_h = title_block_h + 24 + topic_block_h + 20 + handson_h

    y = (H - total_content_h) // 2

    for line in title_lines:
        draw.text((W // 2, y), line, font=font_title,
                  fill="white", anchor="mm")
        y += title_line_h

    # ── 3 bullet topics ───────────────────────────────────────────────────────
    font_topic = ImageFont.truetype(FONT_REG, 28)
    y += 24   # gap after title

    # Centre the bullet block
    bullet_x = W // 2 - 260   # left-align start of bullet text block

    for topic in week["topics"][:3]:
        # Accent bullet dot
        draw.ellipse([bullet_x - 14, y + 8, bullet_x - 6, y + 16], fill=ACCENT)
        draw.text((bullet_x, y), topic, font=font_topic, fill=ACCENT)
        y += 38

    # ── Hands-on label ────────────────────────────────────────────────────────
    font_handson = ImageFont.truetype(FONT_MONO, 22)
    y += 20
    draw.text((W // 2, y), week["hands_on"], font=font_handson,
              fill="#606070", anchor="mm")

    # ── Bottom accent bar ─────────────────────────────────────────────────────
    draw.rectangle([0, H - 10, W, H], fill=ACCENT)

    path = pathlib.Path(tmp_dir) / f"week_{wn:02d}.png"
    img.save(str(path))
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MOVIEPY — assemble 8 clips into one MP4
# ─────────────────────────────────────────────────────────────────────────────

def build_video(weeks: list[dict], slide_paths: list[str],
                output: str = "genai_curriculum.mp4") -> str:
    clips = [
        ImageClip(path).set_duration(week["duration"]).set_fps(24)
        for path, week in zip(slide_paths, weeks)
    ]
    final = concatenate_videoclips(clips, method="compose")

    # ── BGM ──────────────────────────────────────────────────────────────────
    from moviepy.editor import AudioFileClip
    bgm_path = os.path.join(os.getcwd(), "bgm.mp3")
    has_audio = False
    if os.path.exists(bgm_path):
        try:
            bgm = AudioFileClip(bgm_path).volumex(0.32)
            bgm = bgm.audio_loop(duration=final.duration)
            final = final.set_audio(bgm)
            has_audio = True
            print("    🎵  BGM loaded: bgm.mp3")
        except Exception as e:
            print(f"    ⚠️  BGM failed: {e} — continuing without audio")

    final.write_videofile(
        output,
        fps=24,
        codec="libx264",
        audio=has_audio,
        threads=2,
        preset="ultrafast",
        logger=None,
    )
    return output

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    OUTPUT_FILE = "genai_curriculum.mp4"
    TMP_DIR     = tempfile.gettempdir()

    # ── Step 1: Claude ────────────────────────────────────────────────────────
    print("🤖  Asking Claude to plan the 8-week curriculum ...")
    weeks = extract_weeks(CURRICULUM_BRIEF)
    print(f"    ✅  {len(weeks)} weeks received\n")
    for w in weeks:
        print(f"    Week {w['week_number']:02d}  {w['title']}")

    # ── Step 2: PIL slides ────────────────────────────────────────────────────
    print("\n🎨  Rendering slides ...")
    slide_paths = []
    for week in weeks:
        path = render_slide(week, tmp_dir=TMP_DIR)
        slide_paths.append(path)
        print(f"    Week {week['week_number']:02d} → {path}")

    # ── Step 3: Video assembly ────────────────────────────────────────────────
    print("\n🎬  Assembling video ...")
    video_path = build_video(weeks, slide_paths, output=OUTPUT_FILE)
    size_kb = pathlib.Path(video_path).stat().st_size // 1024
    print(f"\n    ✅  Saved: {video_path}  ({size_kb} KB)")
    print(f"    Total duration: {sum(w['duration'] for w in weeks)}s")
    print("\n    Download from the file browser and upload to YouTube manually.")
