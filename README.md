<div align="center">

<!-- ============================================================================
🖼️ IMAGE SLOT 1 — HERO BANNER
   File: docs/images/hero-banner.png  |  Size: 1920 x 640 px (3:1 banner)
   Prompt for your image model:
   "Wide futuristic banner illustration for an AI art toolbox called 'StarNodes'.
   A glowing night sky full of connected golden stars forming a node-graph
   constellation, flowing lines of light linking the stars like a neural network.
   Deep indigo and violet background, warm gold accents, soft bokeh, clean modern
   flat-illustration style, high detail, no text."
============================================================================= -->

# ⭐ ComfyUI StarNodes

**A big, friendly toolbox of 100 custom nodes that make ComfyUI easier, faster and more fun.**

*Starters • Samplers • Image tools • Qwen & Flux helpers • Video • PSD export • Wildcards • and much more*

[![Version](https://img.shields.io/badge/version-2.8.1-blueviolet?style=for-the-badge)](#)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20nodes-orange?style=for-the-badge)](#)
[![License](https://img.shields.io/github/license/Starnodes2024/ComfyUI_StarNodes?style=for-the-badge)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Starnodes2024-black?style=for-the-badge&logo=github)](https://github.com/Starnodes2024/ComfyUI_StarNodes)

</div>

---

## 🌟 What is StarNodes?

StarNodes is a **swiss-army-knife node pack** for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
It fills the gaps you hit every day: starting a workflow with one node instead of five,
saving real layered **PSD files**, browsing **360° panoramas**, batching images into
**grids with captions**, managing **wildcards**, **LoRAs**, **sampler presets** —
plus a full set of **Qwen / Flux / LTX-Video** helpers.

> 💡 **In one sentence:** fewer wires, fewer clicks, better defaults — everything lives under one **⭐StarNodes** menu.

<!-- ============================================================================
🖼️ IMAGE SLOT 2 — FEATURE COLLAGE
   File: docs/images/feature-collage.png  |  Size: 1600 x 900 px (16:9)
   Prompt for your image model:
   "Modern software feature collage, 2x3 grid of six rounded cards floating on a
   dark indigo background: a paintbrush with sparkles, a film reel, a photo stack,
   a layered PSD document icon, a 360-degree panorama sphere, and a magic wand.
   Flat design, golden and violet accent colors, soft shadows, minimal, no text."
============================================================================= -->

---

## ✨ Recently Added & Updated — Spotlight

These are the newest and most powerful nodes in the pack. If you try nothing else, try these:

<table>
<tr>
<td width="50%">

### 🆕 ⭐ Star LTXV 2.5 All-in-One
**The complete LTX-2.5 pipeline in a single node.** Text-to-video, image-to-video, image+audio, first/last-frame and audio-only modes — two-pass upscale rendering, baked sigma or plain-step schedules, model/LoRA/CLIP/VAE caching, and sound decoded from the high-step pass.

### 🆕 ⭐ Star Video Sound Enricher
**Fix the scratchy AI-video soundtrack.** De-harsh bell, deep-bass and warmth boost, high-fizz taming and gentle analog saturation — four tuned presets plus a full Custom mode, always delivered at 44.1 kHz.

### 🆕 ⭐ Star Video Sound Enricher Option
**Same sound magic, zero extra wires.** Outputs the enricher settings as a `sound_settings` bundle — plug it into the LTXV 2.5 All-in-One and the soundtrack comes out cleaned and enriched internally.

### 🆕 ⭐ Star SD Upscale Refiner Advanced
**SD1.5 upscaling + refinement in one node.** Built-in LoRA support, tiled diffusion for low VRAM, and optional ControlNet tile guidance — a complete "make it big and beautiful" pipeline.

### 🆕 ⭐ Star Minimax All In One
**The whole MiniMax H3 reference-to-video pipeline in a single node.** Models, text encoder and both VAEs load internally; feed reference images, videos and audios, then sample and decode video + audio — no sub-graph. Includes the new **2:1 (Panorama)** preset.

### 🆕 ⭐ Star Minimax Latent Upscaler + Option
**Second-pass upscale + refine, no extra wiring.** The Option node plugs into the `options` input of ⭐ Star Minimax All In One: the pass-1 video latent is upscaled with a MiniMax H3 3D latent-upscaler model and refined in a short 3/4/5-step pass with the same conditioning, same seed — optionally with a turbo-LoRA-patched model on the node's `model` input, and a toggle for pass-1 or refined audio. The standalone twin does the same upscale + refine + decode in any workflow (connect latent, model, clip, both VAEs and a prompt).

### 🆕 ⭐ StarSampler (Unified)
**One sampler to rule them all.** Extensive configuration in a single node, with a tiled VAE decoder so even big images finish on modest GPUs. Replaces stacks of sampler plumbing.

### 🆕 ⭐ Star Split Sampler Option
**Two samplers, one run.** Plug into the `options` input of ⭐ StarSampler or ⭐ Star SD Upscale Refiner to run the first N steps with one sampler and the rest with another — e.g. `euler` for 6 steps then `ddim` for 6, for 12 steps total. Mix sampler strengths without rewiring anything.

</td>
<td width="50%">

### 🆕 ⭐ Star Advanced Ratio/Latent
**Aspect ratio + megapixels, done for you.** Pick a ratio and a target size — get perfect dimensions and a ready-to-use empty latent, no calculator needed.

### 🔄 ⭐ Star Output Cleaner
**Your output folder, finally under control.** Browse thumbnails, select and delete old generations without ever leaving ComfyUI.

### 🔄 ⭐ Star Image Compare
**Judge your results like a pro.** Interactive before/after wipe slider right inside ComfyUI — drag to compare two images pixel by pixel. Perfect for checking upscales, refiners and filter tweaks.

### 🔄 ⭐ Star Panorama Tools
**Create, save and explore 360° worlds.** The 360 Parallax Viewer (and Pro) let you look around panoramas interactively with mouse parallax, overlays and depth maps — while Save Panorama JPEG/+ embeds proper XMP metadata, including stereoscopic 3D output.

### 🔄 ⭐ Star Video Compressor
**Shrink videos to any target.** Compress to a quality level *or an exact file size* with H.264, H.265, VP9 or AV1 — no external tools needed.

### 🔄 ⭐ Star Slideshow Maker
**Images in, movie out.** Transitions, motion effects and audio, rendered through FFmpeg straight from your workflow.

### 🔄 ⭐ Star Tiled Upscalers (SeedVR & PiD)
**Huge upscales on small GPUs.** Tile-based upscaling with SeedVR2 (with color correction) or PiD/PixelDiT models — overlapping tiles mean no seams, no VRAM panic.

</td>
</tr>
</table>

> 🆕 = recently added &nbsp;•&nbsp; 🔄 = recently updated

<!-- ============================================================================
🖼️ IMAGE SLOT 2b — SPOTLIGHT SHOWCASE
   File: docs/images/spotlight-showcase.png  |  Size: 1600 x 900 px (16:9)
   Prompt for your image model:
   "Dramatic before/after AI image enhancement showcase, split composition:
   left side a small blurry low-resolution fantasy landscape, right side the
   same landscape upscaled to crisp stunning detail with vibrant HDR colors.
   A glowing vertical divider line in the middle, subtle film grain texture,
   dark background, cinematic, high contrast, no text."
============================================================================= -->

---

## 🚀 Installation

### Option A — ComfyUI Manager (recommended, 1 minute)

1. Open **ComfyUI Manager**
2. Search for **`Starnodes`**
3. Click **Install** → restart ComfyUI ✅

### Option B — Manual install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Starnodes2024/ComfyUI_StarNodes
cd ComfyUI_StarNodes
pip install -r requirements.txt
```

Restart ComfyUI. Done. 🎉

> **Where are the nodes?** Press **double-click** on the canvas and search **`star`** —
> or open the **⭐StarNodes** category in the node browser.

<!-- ============================================================================
📸 IMAGE SLOT 3 — SCREENSHOT (take this one yourself, no prompt needed)
   File: docs/images/node-search.png  |  Size: ~1200 x 700 px
   Content: ComfyUI canvas with the node search open, typing "star", showing the
   ⭐StarNodes category list. Tip: use the dark theme for a consistent look.
============================================================================= -->

---

## 🧭 Quick Start (60 seconds)

| I want to… | Use this node |
|---|---|
| Start a Flux / SDXL / Qwen workflow fast | ⭐ **Star Start(t) Settings** nodes |
| Upscale a latent with a model | ⭐ **Star Model Latent Upscaler** |
| Save my image with all metadata | ⭐ **Star Save Image+** |
| Export a layered PSD file | ⭐ **Star PSD Saver** family |
| Use wildcards in prompts | ⭐ **Star Wildcards Advanced** |
| Stack multiple LoRAs | ⭐ **Star Dynamic LoRA** |
| Debug any connection | ⭐ **Star Show Everything** |
| Make a grid of images with captions | ⭐ **Star Grid Composer** |

---

## 📦 What's Inside

Click a category to expand it. All categories appear in ComfyUI exactly as named below.

<details open>
<summary><b>🚀 Starters</b> — one node to kick off a whole workflow</summary>

| Node | What it does |
|------|--------------|
| ⭐ Star FLUX Start(t) Settings | Load UNET + 2 CLIPs and create an empty latent in one step |
| ⭐ Star SD(XL) Start(t) Settings | Load checkpoint + CLIP + VAE and create an empty latent |
| ⭐ Star Qwen Image Start(t) Settings | Load a Qwen model and create an empty latent |

</details>

<details>
<summary><b>🎛️ Sampler</b> — sampling, LoRAs, inpainting, refinement</summary>

| Node | What it does |
|------|--------------|
| ⭐ StarSampler (Unified) | Advanced all-in-one sampler with tiled VAE decode for low VRAM |
| ⭐ Star Split Sampler Option | Run two different samplers in one pass — e.g. euler 6 steps + ddim 6 steps |
| ⭐ Star Split Sampler Info | Split sampler info into its individual components |
| ⭐ Star 3 LoRAs | Apply up to 3 LoRAs with individual weights |
| ⭐ Star Dynamic LoRA | Flexible multi-LoRA node with dynamic slots |
| ⭐ Star Dynamic LoRA (Model Only) | Same, without touching CLIP |
| ⭐ Star Dynamic LoRA Weight | Normalize and fine-tune LoRA weights |
| ⭐ Star Detail Daemon | Enhance details for Flux & all SD models |
| ⭐ Star FluxFill Inpainter | Inpainting for Flux with optimized conditioning |
| ⭐ Star Flux2/Qwen-Image-Edit Inpainter | All-in-one inpaint: crop-and-stitch, ref image, Differential Diffusion |
| ⭐ Star Distilled Optimizer (QWEN/ZIT) | Two-pass distilled refinement for Z-Image-Turbo / Qwen Turbo |

</details>

<details>
<summary><b>🖼️ Image & Latent</b> — filters, upscalers, grids, panoramas, PSD export</summary>

| Node | What it does |
|------|--------------|
| ⭐ Star Model Latent Upscaler | Complete latent upscale pipeline with model choice |
| ⭐ Star Image2Latent | Convert an image to a latent |
| ⭐ Star Latent Resize | Resize latents with ratio / megapixel selector |
| ⭐ Star Advanced Ratio/Latent | Aspect ratio + latent megapixel helper |
| ⭐ Starnodes Aspect Ratio Advanced | Resolution selection with aspect-ratio math done for you |
| ⭐ Star Adaptive Detail Enhancement | Sharpen / denoise via edge, face & texture analysis |
| ⭐ Star Apply Overlay (Depth) | Blend filtered images using depth or masks |
| ⭐ Star Simple Filters | Sharpen, blur, saturation, contrast, brightness, color matching |
| ⭐ Star HighPass Filters | High-pass sharpening for fine detail |
| ⭐ Star Black & White | Flexible B&W conversion with tonal control |
| ⭐ Star Radial Blur | Focus / zoom motion blur effects |
| ⭐ Star HDR Effects | HDR-style enhancement |
| ⭐ Star Realistic Film Grain | Analog grain with real film-stock profiles |
| ⭐ Star Image Compare | Interactive before/after wipe slider |
| ⭐ Star Tiled PiD Upscaler | Upscale with PiD/PixelDiT models in overlapping tiles |
| ⭐ Star Tiled SeedVR Upscaler | SeedVR2 tiled upscaling with color correction |
| ⭐ Star Lucida RMBG | High-quality background removal (BiRefNet fine-tune) |
| ⭐ Star 360 Parallax Viewer | Interactive 360° panorama with mouse parallax |
| ⭐ Star 360 Parallax Viewer Pro | Panorama + video export, overlays, depth maps |
| ⭐ Star Save Panorama JPEG | JPEG with XMP panorama metadata |
| ⭐ Star Save Panorama JPG+ | Panorama export with stereoscopic 3D output |
| ⭐ Star Grid Composer | Compose images into a grid with captions, fonts, colors |
| ⭐ Star Grid Image Batcher | Batch up to 16 images for the Grid Composer |
| ⭐ Star Grid Captions Batcher | Batch up to 16 captions for grid layouts |
| ⭐ Star PSD Saver (Dynamic) | Save layered PSD files |
| ⭐ Star PSD Saver 2 (Optimized) | Faster PSD export |
| ⭐ Star PSD Saver Adv. Layers | PSD export with advanced layer handling |
| ⭐ Star Watermark | Text / image watermarks with placement control |
| ⭐ Star Meta Injector | Copy PNG metadata from one image to another |
| ⭐ Star Icon Exporter | Multi-size PNG/ICO export with shaping & shadow |
| ⭐ Star Random Image Loader | Random images from a folder, seed-controlled |
| ⭐ Star Image Loader 1by1 | Sequential image loading with state memory |
| ⭐ Star Qwen Image Edit Inputs | Stitch up to 4 images for Qwen editing |

</details>

<!-- ============================================================================
🖼️ IMAGE SLOT 4 — PSD SAVER SHOWCASE
   File: docs/images/psd-saver-demo.png  |  Size: 1400 x 800 px
   Prompt for your image model:
   "Clean software showcase illustration: on the left a ComfyUI-style node graph
   with glowing connected nodes, on the right an Adobe Photoshop layers panel
   showing neatly separated layers (background, subject, lighting, text). An arrow
   of golden light flows from the nodes into the layer stack. Dark indigo
   background, flat modern style, no text."
============================================================================= -->

<details>
<summary><b>🎬 Video</b> — LTX-Video, loops, compression, slideshows</summary>

| Node | What it does |
|------|--------------|
| ⭐ Star LTXV All-in-One (2-Pass) | Half-res pass → 2x upscale → full-res pass in a single node |
| ⭐ Star LTX Video Settings | Video dimension & frame calculator |
| ⭐ Star VAE LTXV Save / Load | VAE encode/decode for LTX video |
| ⭐ Star LTX Image Cut | Smart cropping for LTX video with aspect-ratio preservation |
| ⭐ Star LTXV Get Last Frame | Extract the last frame from video latents |
| ⭐ Star LTXV Load Last Image From Folder | Load the last generated image from a folder |
| ⭐ Star Video Joiner | Join multiple videos into one |
| ⭐ Star Video Loader | Decode video to frames + audio + fps — no extra suite needed |
| ⭐ Star Video Compressor | Compress to a target size/quality (H.264/H.265/VP9/AV1) |
| ⭐ Star Slideshow Maker | Slideshows with transitions, motion effects and audio |
| ⭐ Star Image Loop | Seamless looping frames from panoramic images |
| ⭐ Star Video Loop | Seamless looping frames from video |
| ⭐ Star Frame From Video | Pick first / last / specific frame from a batch |

</details>

<details>
<summary><b>📝 Text & Data</b> — wildcards, prompts, storage</summary>

| Node | What it does |
|------|--------------|
| ⭐ Star Wildcards Advanced | Wildcards with folder paths, nesting and multi-prompt support |
| ⭐ Star Text Inputs (Concatenate) | Merge multiple text inputs |
| ⭐ Star Text Filter | Remove words, whitespace and empty lines |
| ⭐ Star Easy-Text-Storage | Save & reuse text snippets across workflows |
| ⭐ Star Prompt Picker | Pick prompts from a file or folder (random or sequential) |
| ⭐ Star Web Scraper (Headlines) 📰 | Scrape news headlines as prompt inspiration |

</details>

<details>
<summary><b>🎨 Prompts & Conditioning</b> — Qwen, Flux2, regional control</summary>

| Node | What it does |
|------|--------------|
| ⭐ Star Image Edit for Qwen/Kontext | Dynamic prompt builder with customizable templates |
| ⭐ Star Qwen-Rebalance-Prompter | Intelligent prompt rebalancing |
| ⭐ Star Qwen Edit Encoder | CLIP encoder tuned for Qwen image editing |
| ⭐ Star QwenEdit+ Conditioner | Enhanced conditioning for Qwen models |
| ⭐ Star Qwen Regional Prompter | Region-based prompting for precise control |
| ⭐ Star Flux2 Conditioner | Text + up to 5 reference images for Flux2 |
| ⭐ Star Conditioning Saver / Loader | Save and reuse conditioning between workflows |

</details>

<details>
<summary><b>💾 IO & Save</b> — smarter saving and loading</summary>

| Node | What it does |
|------|--------------|
| ⭐ Star Save Image+ | PNG/JPG/WEBP/PSD with format chips, mask embedding, 16-bit PNG, 5 metadata fields |
| ⭐ Star Load Image+ | Load from input/output folder, clipboard paste, metadata fields, invert mask |
| ⭐ Star Metadata Saver Option | Labeled key/value metadata for Star Save Image+ |
| ⭐ Star Image Loader Options | Show metadata from Star Load Image+ with copy buttons |
| ⭐ Star Save Folder String | Flexible path builder with date-based organization |

</details>

<details>
<summary><b>🛠️ Helpers & Tools</b> — the little things that save your day</summary>

| Node | What it does |
|------|--------------|
| ⭐ Star Show Everything | Universal debug node — connect anything, see type/shape/stats/preview |
| ⭐ Star Stop And Go | Interactive pause / preview / continue control |
| ⭐ Star Output Cleaner | Browse & clean your output folder with thumbnails |
| ⭐ Star Ollama Prompt Helper | Local Ollama prompts with 15 presets and vision support |
| ⭐ Star Duplicate Model Finder | SHA256 duplicate model scanner |
| ⭐ Star FP8 Converter | Convert checkpoints to FP8 |
| ⭐ Star Model Packer | Merge split safetensors, convert precision |
| ⭐ Star Divisible Dimension | Keep dimensions divisible by a given value |
| ⭐ Star Size Calculator by Side | Resize longest/shortest side, keep aspect ratio |
| ⭐ Star Denoise Slider | Simple denoise-strength slider |
| ⭐ Star Box Drawer | Draw filled or outlined rectangles |
| ⭐ Star Image Shifter | Shift images with seamless wrapping (panoramas/textures) |
| ⭐ Star Krea2 Unbound | Prompt-adherence enhancer for Krea2 models |
| ⭐ Star Multi Inputs To One | Combine dynamic inputs into a single output |
| ⭐ Star Show Last Frame | Extract the last frame from video latents |
| ⭐ Star Save / Load / Delete Sampler Settings | Reusable sampler presets |
| ⭐ Star SD Upscale Refiner Advanced | SD1.5 upscale + refine with LoRA, tiled diffusion, ControlNet tile |
| ⭐ Star Palette Extractor | Extract the dominant color palette from an image |

</details>

---

## 🃏 Wildcards

On first start, StarNodes copies a ready-made wildcard collection to
`[ComfyUI]/wildcards/`. Add your own `.txt` files there — one option per line.

```text
a photo of __animal__ in __place__, {golden hour|blue hour|midnight}, __style__
```

- `__name__` → random line from `wildcards/name.txt`
- `subfolder\__name__` → use subfolders
- `{a|b|c}` → quick random choice
- Nesting works up to 10 levels deep 🪆

<!-- ============================================================================
🖼️ IMAGE SLOT 5 — WILDCARD MOODBOARD
   File: docs/images/wildcard-moodboard.png  |  Size: 1600 x 900 px
   Prompt for your image model:
   "A 2x2 moodboard of four AI-generated fantasy portraits of the same fox
   character in different styles: watercolor, cyberpunk neon, oil painting,
   pixel art. Each panel clearly distinct in mood and color. Thin white borders
   between panels, vibrant, high quality."
   (Idea: generate it WITH the wildcard node itself — perfect demo!)
============================================================================= -->

---

## 🎨 Theme System

StarNodes ships its own **color themes**. Pick a theme in the ComfyUI settings,
then apply theme presets to any node via the **right-click menu** (multi-select
supported). Perfect for keeping big workflows readable.

---

## 📚 Example Workflows

Ready-to-use workflows live in **`example_workflows/`** — drag any `.json` onto
your canvas. Highlights include:

- 🖼️ **Image upscaling & refinement** (SD upscale, SeedVR/PiD tiled upscalers)
- 🎬 **LTX-Video** pipelines (2-pass all-in-one, loops, slideshows)
- 🌌 **360° panorama** generation & interactive viewing
- 🧩 **Qwen image editing** (regional prompting, multi-image edit)

---

## 📦 Dependencies

Everything is listed in `requirements.txt` and installs automatically.
**Good to know:** if a package is missing, the rest of the pack still loads —
only the affected node is skipped, and ComfyUI tells you exactly what to install.

| Package | Needed for |
|---------|------------|
| `psd-tools` | PSD Saver nodes |
| `opencv-python` | Detail Enhancement, Panorama JPG+, Slideshow |
| `scikit-learn`, `webcolors` | Palette Extractor |
| `color-matcher` | Simple Filters (color matching) |
| `imageio-ffmpeg` | Video Loader / Compressor / Slideshow |
| `requests`, `beautifulsoup4` | News Scraper |
| `ollama` | Ollama Prompt Helper |

---

## 🆘 Troubleshooting

| Problem | Fix |
|---|---|
| A node is missing | Check the ComfyUI console — it prints which package to `pip install` |
| Nodes don't appear at all | Make sure you restarted ComfyUI after installing |
| Wildcards don't resolve | Confirm the files exist in `[ComfyUI]/wildcards/` |
| Ollama node can't connect | Start the Ollama app/server first (`ollama serve`) |

📖 Detailed per-node docs are built right into ComfyUI: select a node and open
its **help panel** — StarNodes ships a doc page for every node (`web/docs/`).

---

<div align="center">

<!-- ============================================================================
🖼️ IMAGE SLOT 6 — FOOTER ART (optional, small)
   File: docs/images/footer-stars.png  |  Size: 1200 x 300 px (4:1 slim banner)
   Prompt for your image model:
   "Minimal slim footer banner, a gentle wave of tiny golden stars fading from
   left to right on a deep indigo background, lots of empty space, elegant,
   flat design, no text."
============================================================================= -->

**Made with ⭐ by [Starnodes2024](https://github.com/Starnodes2024/ComfyUI_StarNodes)**

If StarNodes helps your workflows, consider leaving a ⭐ on GitHub — it helps others find the pack!

</div>
