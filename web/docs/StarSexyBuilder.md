# ⭐ Star Sexy Builder

## Overview

The **Star Sexy Builder** node builds a **sensual character description prompt** from simple dropdown selections — no typing required. It works exactly like the ⭐ Star Character Builder, but is fully focused on the sexually attractive details of a figure: body build, explicit anatomy for male and female, sexy lingerie and night outfits, seductive poses, expressions and mood.

All dropdown options are editable in `json/star_sexy_builder.json` — add your own entries to any category!

All inputs have helpful tooltips when you hover over them in ComfyUI.

## Special dropdown values

Every category dropdown starts with three special options:

- **none**: the category is left out of the prompt
- **random**: a random entry from the category list is picked
- **exclude**: the category is left out of the prompt **and** stays out even when `random_all` is on

## Inputs

### Body

- **sex**: female / male / transgender woman / transgender man
- **skin_tone**: porcelain, tan, olive, ebony — plus sensual surface variants like glossy oiled skin, wet skin, sweat-glistened skin, sun-kissed with tan lines
- **body_build**: slim, athletic, curvy, hourglass figure, voluptuous, petite, muscular, plus-size, …
- **waist**: slim narrow waist, flat toned stomach, six-pack abs, snatched corseted waist, narrow waist with wide hips, …
- **legs**: long shapely legs, thick strong thighs, thigh gap, muscular toned legs, endless long legs, …

### Anatomy details

- **breasts**: small perky, medium natural, large heavy, huge, teardrop-shaped, augmented, flat chest, … (the female chest category)
- **chest**: defined muscular pecs, chiseled six-pack abs, broad strong chest, hairy masculine chest, athletic v-taper, … (the male counterpart to breasts)
- **nipples**: small pink, erect, puffy, pierced, large dark areolas, …
- **butt**: bubble butt, big round butt, firm athletic butt, heart-shaped, wide hips and big butt, …
- **genitals**: explicit anatomy for both — smooth shaved vulva, large labia, aroused, large erect penis, uncircumcised, heavy hanging, hidden genitals, …
- **pubic_hair**: completely shaved, landing strip, natural full bush, heart-shaped patch, …

### Wear categories (see **Wear rule** below)

- **lingerie**: lace bra and panties, silk babydoll, corset, see-through lace, garter belt and stockings, open cup bra, crotchless bodysuit, …
- **nightwear**: sexy night outfits — sheer silk nightgown, transparent lace negligée, open silk robe, satin slip dress with a high slit, unbuttoned oversized shirt, kimono slipping off the shoulders, wet silk gown, …
- **kinky_outfit**: latex catsuit, fishnet bodystocking, dominatrix outfit, shibari harness, bunny girl costume, …

### Finishing touches

- **accessories**: thigh-high stockings, stiletto heels, velvet choker, silk blindfold, body chain, long satin gloves, …
- **expression**: seductive gaze, bedroom eyes, biting the lower lip, lustful stare, teasing wink, orgasmic bliss, …
- **pose**: lying seductively on a bed, arching the back, bent over looking back, presenting the butt, sprawled on silk sheets, teasing striptease pose, …
- **ambiance**: dim warm bedroom light, flickering candlelight, neon glow through the window, moonlight through blinds, red boudoir lighting, …
- **custom**: your own extra category — replace the example entries in the json file
- **style**: the render style — boudoir photography (default), photorealistic, glamour magazine photography, soft focus, pin-up art, anime, 3d render, …

### Wear rule

When **random_all** is on, only **one** of the three wear categories is picked per run — no lingerie *and* nightgown stacking. When you select wear categories manually, all selected ones are combined, e.g. `wearing red lace lingerie set and open silk robe`.

### Other inputs

- **age** (STRING)
  - Free text: a plain number becomes `25 year old`, words like `young adult` are used as-is
  - Type `random` to pick a random age, or use the `random_age` toggle instead
  - Empty = no age in the prompt

- **random_age** (BOOLEAN)
  - Picks a random age of **18–60** each run and ignores the age text field
  - The range can be changed via `age_random_min` / `age_random_max` in the json settings

- **additional** (STRING, multiline)
  - Free text appended at the end of the prompt — perfect for scene and quality tags like `luxury hotel bedroom, silk sheets, cinematic lighting`

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
  `25 year old female, tan skin, hourglass figure, large natural breasts, erect nipples, slim narrow waist, bubble butt, long shapely legs, wearing sheer silk nightgown, thigh-high stockings, biting the lower lip, arching the back, flickering candlelight, boudoir photography`

### 2. sexy_sheet (STRING)

- A ready-to-use, strict **adult figure reference sheet** image prompt for the same character, matching the `system_prompt` layout spec:
  - **Left side (full height)**: two vertical full-body panels — one **full-body front view** and one **full-body back view**
  - **Right side (standard layout)**: a 6-panel detail grid in a **3x2** layout:
    - Top row (head views): face portraits with the selected expression
    - Bottom row (body sections): **chest area from the front** (showing breasts/chest and nipple details), **pelvic area from the front** (genitals and pubic hair shown when selected), **buttocks from the back**
  - **Expanded layout**: when breasts/chest/butt or accessories are selected, the right side expands to **8 panels in a 4x2 layout** adding **dedicated close-up panels of the selected anatomy** and **separate isolated close-ups** of accessory items on neutral backgrounds — never worn or held
  - **Unclothed by default**: the subject is unclothed in the full-body panels unless a wear category is selected (with **random_all**, set all three wear categories to **exclude** for a naked sheet)
  - `pose` and `ambiance` are ignored in sheet mode (the views are prescribed); `additional` is appended at the very end
  - Always ends with: clinical objective style like a figure study plate, flat even studio lighting, absolute consistency across panels, and an absolute **no text/labels/symbols/watermarks** prohibition

### 3. system_prompt (STRING)

- The **system prompt** you can paste into your LLM so it generates figure sheet prompts itself
- The text lives in `json/star_sexy_builder.json` as the `system_prompt` key — edit it there to customize it permanently

## Typical workflows

- **Direct character**: connect `description` to your prompt/CLIP text input
- **Direct sheet**: connect `sexy_sheet` straight to a sampler text encoder — no LLM needed
- **LLM flow**: connect `system_prompt` as the system message of your LLM node and `description` as the user message

## Customizing the option lists

Edit `json/star_sexy_builder.json` in the StarNodes folder:

- Each category is a simple list of strings — add, remove or reword entries as you like
- New entries appear in the dropdowns after a **ComfyUI restart** — but `random` picks use the updated file immediately
- The `settings` block holds `age_random_min`, `age_random_max` and a global `exclude` list of category names

## Tips

- This builder deliberately skips the everyday details (hair, makeup, ethnicity, everyday outfits) — chain it with the ⭐ Star Character Builder upstream and combine both outputs (e.g. with ⭐ Star Prompt Picker or a text join) for the full character
- Don’t mix the gendered anatomy categories against the sex: pick **breasts** for female figures or **chest** for male figures — with **random_all** you may want to set the unused one to **exclude**
- Use **random** on a few categories and fixed values on the rest for controlled variety (great together with a fixed **seed**)
- The **additional** field is perfect for the scene: `luxury hotel bedroom, silk sheets, city lights at night`
