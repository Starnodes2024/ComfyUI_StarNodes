# ⭐ Star Character Builder

## Overview

The **Star Character Builder** node builds a **character description prompt** from simple dropdown selections — no typing required. Pick traits like sex, ethnicity, body build, hairstyle, outfit and more, and the node assembles them into one clean comma-separated prompt string.

All dropdown options are editable in `json/star_character_builder.json` — add your own entries to any category!

All inputs have helpful tooltips when you hover over them in ComfyUI.

## Special dropdown values

Every category dropdown starts with three special options:

- **none**: the category is left out of the prompt
- **random**: a random entry from the category list is picked
- **exclude**: the category is left out of the prompt **and** stays out even when `random_all` is on

## Inputs

### Segments (all dropdowns)

- **sex**: female / male / transgender woman / transgender man / non-binary
- **type**: character type / subculture (goth, punk, emo, supermodel, hipster, viking, …) — unisex, gender comes from the `sex` category
- **ethnicity**: European, East Asian, African, Latin American, …
- **skin_tone**: porcelain, fair, tan, olive, ebony, freckled, …
- **body_build**: slim, athletic, curvy, muscular, plus-size, …
- **hairstyle**: long wavy hair, pixie cut, dreadlocks, bald head, …
  - Combined with **hair_color** automatically: `long wavy hair` + `blonde` → `long wavy blonde hair`
- **hair_color**: blonde, jet black, auburn, pastel pink, …
- **eye_color**: green, ice blue, heterochromia eyes, … (becomes `X eyes` in the prompt)
- **makeup**: natural makeup, smokey eye makeup, bold red lipstick, gothic makeup, …
- Wear categories (see **Wear rule** below):
  - **clothing**: casual everyday outfits (business suit, hoodie and joggers, kimono, …)
  - **female_outfit**: classic female outfits (little black dress, pin-up swing dress, …)
  - **male_outfit**: classic male outfits (three-piece suit, tuxedo, …)
  - **lingerie**: lace bra and panties, silk babydoll, corset, …
  - **kinky_outfit**: latex catsuit, fishnet bodystocking, shibari harness, …
- **accessories**: glasses, earrings, hats, tattoos, …
- **sextoy**: holding a riding crop, wearing nipple clamps, …
- **expression**: gentle smile, confident smirk, fierce expression, …
- **pose**: looking at viewer, full body shot, dynamic action pose, …
- **custom**: your own extra category — replace the example entries in the json file
- **style**: the render style — photorealistic, amateur photography, cinematic film still, anime, 3d anime, manga/western comic, cartoon, 3d render, oil painting, pixel art, …
  - Applied to **both** the `description` output (appended at the end) and the `character_sheet` output (`Rendered in X style, presented in a clinical and scientific way…`)

### Wear rule

When **random_all** is on, only **one** of the five wear categories is picked per run — the character never wears e.g. a suit *and* lingerie at the same time. When you select wear categories manually, all selected ones are combined, e.g. `wearing blouse and skirt and red lace lingerie set`.

### Other inputs

- **age** (STRING)
  - Free text: a plain number becomes `25 year old`, words like `young adult` are used as-is
  - Type `random` to pick a random age, or use the `random_age` toggle instead
  - Empty = no age in the prompt

- **random_age** (BOOLEAN)
  - Picks a random age of **18–100** each run and ignores the age text field
  - The range can be changed via `age_random_min` / `age_random_max` in the json settings

- **additional** (STRING, multiline)
  - Free text appended at the end of the prompt — perfect for style and quality tags like `photorealistic, cinematic lighting`

- **seed** (INT)
  - Seed for all random picks — the same seed gives the same character
  - `0` = new random picks every run

- **random_all** (BOOLEAN)
  - Every category gets a random pick regardless of its dropdown setting
  - Categories set to **exclude** stay out of the prompt; all others are randomized (dropdown values are ignored)
  - The `exclude` list in the json settings does the same globally

## Outputs

### 1. description (STRING)

- The plain generated **comma-separated description prompt**, e.g.
  `25 year old European goth female, fair skin, athletic build, long wavy blonde hair, green eyes, wearing summer dress, glasses, gentle smile, looking at viewer, photorealistic`

### 2. character_sheet (STRING)

- A ready-to-use, strict **character reference sheet** image prompt for the same character:
  - Fixed layout: multi-panel grid on a solid neutral background with **full-body front**, **full-body back**, **front face close-up** and **profile face close-up** views
  - Gender-correct anatomical close-up panels (strict mapping, each gender explicitly excludes the wrong genitalia):
    - **female** → breasts and vulva/vagina panels, absolutely no penis or male genitalia
    - **male** → penis (with testicles) in flaccid **and** erect state, flat chest, absolutely no breasts/vulva
    - **transgender woman** → breasts **and** penis (flaccid and erect), absolutely no vulva/vagina
    - **transgender man** → flat chest and vulva/vagina, absolutely no penis or male genitalia
    - custom sex entries you add yourself fall back to a neutral chest/genitalia panel description
    - **female** and **male** sheets also describe the face distinctly (*a distinctly feminine female face* / *a distinctly masculine male face*); transgender sheets keep the face neutral
    - always with an explicit *no hermaphroditic or mixed traits* safeguard
  - **Naked by default**: the character is 100% naked in all main body views unless a wear category is selected
  - **Isolation rule**: accessories and sextoys are never worn or held — they get their own separate panels
  - If you use **random_all**, set all five wear categories to **exclude** if you want a naked sheet (otherwise an outfit is always randomized)
  - `pose` is ignored in sheet mode (the views are prescribed); `expression` and `makeup` show in the face close-ups; `additional` is appended at the very end of the prompt, same as in normal mode
  - Always ends with: clinical/scientific style, flat even studio lighting, consistent proportions across panels, and an absolute **no text/labels/symbols/watermarks** prohibition

### 3. system_prompt (STRING)

- The **system prompt** you can paste into your LLM so it generates character sheet prompts itself
- The text lives in `json/star_character_builder.json` as the `system_prompt` key — edit it there to customize it permanently

## Typical workflows

- **Direct character**: connect `description` to your prompt/CLIP text input
- **Direct sheet**: connect `character_sheet` straight to a sampler text encoder — no LLM needed
- **LLM flow**: connect `system_prompt` as the system message of your LLM node and `description` as the user message — the LLM will expand it into a full character sheet prompt following the rules

## Customizing the option lists

Edit `json/star_character_builder.json` in the StarNodes folder:

- Each category is a simple list of strings — add, remove or reword entries as you like
- New entries appear in the dropdowns after a **ComfyUI restart** — but `random` picks use the updated file immediately
- The `settings` block holds `age_random_min`, `age_random_max` and a global `exclude` list of category names

## Tips

- Set unused categories to **none** — only selected traits end up in the prompt
- Use **random** on a few categories and fixed values on the rest for controlled variety (great together with a fixed **seed**)
- Pair with ⭐ Star Prompt Picker or any text input downstream to combine the character prompt with scene descriptions
