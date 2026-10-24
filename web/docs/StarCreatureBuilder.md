# ⭐ Star Creature Builder

## Overview

The **Star Creature Builder** node builds a **creature description prompt** from simple dropdown selections — no typing required. It works exactly like the ⭐ Star Character Builder, but for fantasy creatures: mix animals, monsters and human features by picking species, body plan, head, skin covering, wings, horns, claws, tails, genitals, colors and more — the node assembles everything into one clean comma-separated prompt string.

All dropdown options are editable in `json/star_creature_builder.json` — add your own entries to any category!

All inputs have helpful tooltips when you hover over them in ComfyUI.

## Special dropdown values

Every category dropdown starts with three special options:

- **none**: the category is left out of the prompt
- **random**: a random entry from the category list is picked
- **exclude**: the category is left out of the prompt **and** stays out even when `random_all` is on

## Inputs

### Identity

- **sex**: female / male / hermaphrodite / androgynous / sexless
- **creature_type**: the species — dragon, wolf, kitsune, harpy, naga-like serpent, minotaur, demon, slime creature, alien, mimic, … (70+ entries)
- **element**: an elemental affinity placed right before the species in the lead (fire, ice, shadow, celestial, void, …) → `fire dragon`, `shadow wolf`
- **body_plan**: how human/animal the body is — anthropomorphic, humanoid, feral, tauric, naga-bodied, human-bodied (human with creature features), amorphous, ethereal, …
- **size**: tiny, human-sized, large, gigantic, colossal, …

### Head & face

- **head**: full animal head, human face, semi-human face with animal features, skull head, beaked head, twin heads, …
- **mouth**: sharp fangs, tusks, powerful beak, mandibles, forked tongue, lamprey-like mouth, … — combined with **head** automatically: `full animal head` + `sharp fangs` → `full animal head with sharp fangs`
- **ears**: pointed elf ears, cat ears, bat ears, fin ears, no visible ears, …
- **horns**: ram horns, dragon horns, antlers, single unicorn horn, crystal horns, crown of horns, …
- **eye_type** + **eye_color**: combined automatically — `glowing eyes` + `amber` → `glowing amber eyes`; eye types include slit-pupiled, compound, multiple eyes, a third eye, cyclops eye, heterochromatic, …
- **hair_mane** + **hair_color**: combined automatically — `long mane` + `silver` → `long silver mane`; includes crest of feathers, fin crest, fire mane, mushroom cap, …

### Body & colors

- **skin_type**: skin, fur, scales, feathers, slime, chitin exoskeleton, stone skin, metal plating, ethereal wispy body, …
- **body_color**: the main body color (white, emerald, obsidian, iridescent, translucent, …) — combined with skin_type: `emerald` + `scales` → `emerald scales`
- **pattern** + **pattern_color**: markings on the body — `gold` + `stripes` → `emerald scales, gold stripes`; includes glowing runes, bioluminescent spots, lava cracks, …
- **body_features**: extra anatomy — spikes along the spine, dorsal fins, gills, an extra pair of arms, a glowing core, moss growing on the body, …

### Limbs & anatomy

- **arms_hands**: clawed hands, sharp talons, paws, human hands, tentacle fingers, scythe-like arms, no arms, …
- **legs_feet**: digitigrade legs, plantigrade legs, hooved legs, taloned feet, serpentine lower body instead of legs, tentacles instead of legs, multiple legs, …
- **wings**: dragon wings, bat wings, feathered angel wings, fairy wings, moth wings, demon wings, mechanical wings, shadow wings, multiple pairs, no wings, …
- **tail**: dragon tail, fluffy fox tail, nine fluffy tails, spaded demon tail, scorpion tail, prehensile tail, blade-tipped tail, no tail, …
- **genitals**: no genitals, human male/female genitals, hermaphrodite, animal genitals with sheath, cloaca, tentacle genitals, ovipositor, exotic alien genitals, …

### Wear categories (see **Wear rule** below)

- **clothing**: loincloth, tattered cloak, ornate robes, bandages, harem outfit, cultist robes, …
- **armor**: ornate plate armor, bone armor, crystal armor, barbarian furs, saddle and reins, …
- **tribal_garb**: tribal loincloth, feathered headdress, shaman robes, war paint and leather, …

### Finishing touches

- **accessories**: gold jewelry, collar and leash, horn rings, tail jewelry, shackles, floating runes, small creature companion, …
- **expression**: fierce expression, snarling, roaring, curious head tilt, regal expression, … (plus the standard set)
- **pose**: wings spread, pouncing pose, coiled pose, on all fours, rearing up, perched, … (plus the standard set)
- **custom**: your own extra category — replace the example entries in the json file
- **style**: the render style — fantasy illustration (default), bestiary illustration, concept art, photorealistic, anime, 3d render, oil painting, …
  - Applied to **both** the `description` output (appended at the end) and the `creature_sheet` output (`Rendered in X style, presented in a clinical and scientific way like a biological field guide…`)

### Wear rule

When **random_all** is on, only **one** of the three wear categories is picked per run — the creature never wears e.g. armor *and* tribal garb at the same time. When you select wear categories manually, all selected ones are combined, e.g. `wearing tattered cloak and bone armor`.

### Other inputs

- **age** (STRING)
  - Free text: a plain number becomes `25 year old`, words like `adult`, `ancient` or `hatchling` are used as-is
  - Type `random` to pick a random age, or use the `random_age` toggle instead
  - Empty = no age in the prompt

- **random_age** (BOOLEAN)
  - Picks a random age of **18–100** each run and ignores the age text field
  - The range can be changed via `age_random_min` / `age_random_max` in the json settings

- **additional** (STRING, multiline)
  - Free text appended at the end of the prompt — perfect for style and quality tags like `photorealistic, cinematic lighting`

- **seed** (INT)
  - Seed for all random picks — the same seed gives the same creature
  - `0` = new random picks every run

- **random_all** (BOOLEAN)
  - Every category gets a random pick regardless of its dropdown setting
  - Categories set to **exclude** stay out of the prompt; all others are randomized (dropdown values are ignored)
  - The `exclude` list in the json settings does the same globally

## Outputs

### 1. description (STRING)

- The plain generated **comma-separated description prompt**, e.g.
  `large adult anthropomorphic female fire dragon, emerald scales, gold stripes, full animal head with sharp fangs, pointed elf ears, ram horns, long silver mane, glowing amber eyes, spikes along the spine, clawed hands, digitigrade legs, large dragon wings, long dragon tail, cloaca, wearing tribal armor, bone jewelry, fierce expression, wings spread, fantasy illustration`

### 2. creature_sheet (STRING)

- A ready-to-use, strict **creature reference sheet** image prompt for the same creature, matching the `system_prompt` layout spec:
  - **Left side (full height)**: two vertical full-body panels — one **full-body front view** and one **full-body back view**
  - **Right side (standard layout)**: a 6-panel detail grid in a **3x2** layout:
    - Top row (head views): **frontal head portrait**, **side profile portrait**, **45-degree angle portrait** (with the selected expression)
    - Bottom row (body sections): **chest area** from the front, **pelvic area** from the front (genitals shown when selected), **hindquarters** from the back
  - **Expanded layout**: when the creature has notable anatomy (ears, horns, mouth, arms/hands, legs/feet, wings, tail) or accessories, the right side expands to **8 panels in a 4x2 layout** adding **dedicated anatomical close-ups** of the creature's special features and **separate isolated close-ups** of accessory items on neutral backgrounds — never worn or held
  - Anatomy categories set to entries starting with "no …" (`no wings`, `no tail`, `no genitals`) are skipped for the close-up panels
  - **Unclothed by default**: the creature is unclothed in the full-body panels unless a wear category is selected
  - If you use **random_all**, set all three wear categories to **exclude** if you want a naked sheet (otherwise an outfit is always randomized)
  - `pose` is ignored in sheet mode (the views are prescribed); `expression` shows in the head portraits; `additional` is appended at the very end of the prompt, same as in normal mode
  - Always ends with: clinical/scientific style like a biological field guide, flat even studio lighting, absolute consistency of features/proportions across panels, and an absolute **no text/labels/symbols/watermarks** prohibition

### 3. system_prompt (STRING)

- The **system prompt** you can paste into your LLM so it generates creature sheet prompts itself
- The text lives in `json/star_creature_builder.json` as the `system_prompt` key — edit it there to customize it permanently

## Typical workflows

- **Direct creature**: connect `description` to your prompt/CLIP text input
- **Direct sheet**: connect `creature_sheet` straight to a sampler text encoder — no LLM needed
- **LLM flow**: connect `system_prompt` as the system message of your LLM node and `description` as the user message — the LLM will expand it into a full creature sheet prompt following the rules

## Customizing the option lists

Edit `json/star_creature_builder.json` in the StarNodes folder:

- Each category is a simple list of strings — add, remove or reword entries as you like
- New entries appear in the dropdowns after a **ComfyUI restart** — but `random` picks use the updated file immediately
- The `settings` block holds `age_random_min`, `age_random_max` and a global `exclude` list of category names

## Tips

- Set unused categories to **none** — only selected traits end up in the prompt
- Combine `body_plan` with `creature_type` for the species-vs-shape mix: `human-bodied` + `fox` gives a human with fox features, `feral` + `fox` gives a plain animal, `anthropomorphic` + `fox` gives an anthro fox-person
- Use **random** on a few categories and fixed values on the rest for controlled variety (great together with a fixed **seed**)
- Pair with ⭐ Star Character Builder for human characters in the same scene, or with ⭐ Star Prompt Picker downstream to combine the creature prompt with scene descriptions
