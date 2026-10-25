# ⭐ Star Horror Villains Builder

## Overview

The **Star Horror Villains Builder** node builds a **horror villain description prompt** from simple dropdown selections — no typing required. It works exactly like the ⭐ Star Character Builder and ⭐ Star Creature Builder, but is tuned for horror villains: masked slashers, killer dolls, evil clowns, cenobites, dream stalkers, ghost girls and many more archetypes.

Recreate the vibes of your favorite horror icons — pick a killer doll archetype with a porcelain face, yarn hair and a tiny knife for that haunted-doll look, or combine a hockey mask, mechanic's coveralls, machete and hulking build for the classic camp-slasher.

All dropdown options are editable in `json/star_horror_villains_builder.json` — add your own entries to any category!

All inputs have helpful tooltips when you hover over them in ComfyUI.

## Special dropdown values

Every category dropdown starts with three special options:

- **none**: the category is left out of the prompt
- **random**: a random entry from the category list is picked
- **exclude**: the category is left out of the prompt **and** stays out even when `random_all` is on

## Inputs

### Identity

- **sex**: female / male / genderless / androgynous
- **villain_type**: the archetype — killer doll, haunted porcelain doll, ventriloquist dummy, masked slasher, undead slasher, dream stalker, chainsaw butcher, cenobite, demon nun, evil clown, ghost girl, vengeful spirit, possessed child, hook-handed revenant, plague doctor, vampire, mummy, grim reaper, living mannequin, mirror entity, … (45+ entries)
- **size**: doll-sized, child-sized, human-sized, towering, massive, unnaturally elongated, …

### Face & body

- **face**: emotionless white mask, scratched hockey mask, porcelain doll face, clown face paint with red nose, scream mask, mask made of human skin, burlap sack mask, plague doctor mask, pig mask, severely burned face, grid-scarred face with embedded pins, bandaged face, no face, skeletal face, …
- **hair**: orange yarn doll hair, twin doll ringlets, long black hair hanging over the face, matted greasy hair, clown hair tufts, slicked-back black hair, …
- **skin**: porcelain skin, chalk-white skin, burned and blistered skin, rotting greenish skin, mummified leathery skin, stitched-together skin patches, waxen skin, …
- **eyes**: doll glass eyes, black void eyes, milky white eyes, glowing red eyes, button eyes, hollow eye sockets, mismatched doll eyes, …
- **body_build**: child-sized doll body, stiff jointed doll body, towering hulking build, emaciated skeletal build, twisted hunched build, …
- **injuries**: burn scars, stitched gashes, a carved glasgow smile, cracked porcelain leaking black veins, dried blood splatter, bullet holes that never healed, …

### Outfit & props

- **outfit**: striped sweater and fedora, mechanic's coveralls, blood-stained butcher apron, white victorian doll dress, black cenobite leather robe, ruffled vintage clown suit, dark nun habit, straightjacket, sailor doll outfit, striped shirt and denim overalls, …
- **weapon**: a machete, a blood-stained chainsaw, a bladed claw glove, a large kitchen knife, meat hooks on chains, a hooked hand, an ice pick, a nail gun, puppet strings, sewing needles, … — appears as `wielding …` in the prompt
- **accessories**: a red clown balloon, a worn music box, a cursed videotape, a rosary with dark beads, chains draped over the shoulders, a tiny tricycle, a stack of polaroid photos, …
- **aura**: fog curling around the feet, flickering fluorescent light, moving shadows, levitating slightly, grainy VHS distortion, rain pouring indoors, blood dripping upward, …

### Finishing touches

- **expression**: dead-eyed emotionless stare, maniacal grin, frozen polite doll smile, unhinged wide smile, vacant stare with tilted head, …
- **pose**: standing motionless, looming over the viewer, slowly walking forward, peeking from behind a corner, sitting perfectly still in a chair, dragging the weapon behind, …
- **custom**: your own extra category — replace the example entries in the json file
- **style**: the render style — cinematic horror film still (default), 1980s slasher film still, found footage VHS still, retro horror movie poster, gothic horror oil painting, grainy black and white photo, photorealistic, …

### Other inputs

- **age** (STRING)
  - Free text: a plain number becomes `35 year old`, words like `ageless` or `ancient` are used as-is
  - Type `random` to pick a random age, or use the `random_age` toggle instead
  - Empty = no age in the prompt

- **random_age** (BOOLEAN)
  - Picks a random age of **25–90** each run and ignores the age text field
  - The range can be changed via `age_random_min` / `age_random_max` in the json settings

- **additional** (STRING, multiline)
  - Free text appended at the end of the prompt — perfect for scene and quality tags like `dark basement, volumetric lighting`

- **seed** (INT)
  - Seed for all random picks — the same seed gives the same villain
  - `0` = new random picks every run

- **random_all** (BOOLEAN)
  - Every category gets a random pick regardless of its dropdown setting
  - Categories set to **exclude** stay out of the prompt; all others are randomized (dropdown values are ignored)
  - The `exclude` list in the json settings does the same globally

## Outputs

### 1. description (STRING)

- The plain generated **comma-separated description prompt**, e.g.
  `doll-sized female haunted porcelain doll, porcelain skin, porcelain doll face, twin doll ringlets, doll glass eyes, child-sized doll body, cracked porcelain leaking black veins, wearing white victorian doll dress with a bow, wielding sewing needles, a worn music box, candle flames guttering nearby, frozen polite doll smile, sitting perfectly still in a chair, cinematic horror film still`

### 2. villain_sheet (STRING)

- A ready-to-use, strict **horror villain reference sheet** image prompt for the same villain, matching the `system_prompt` layout spec:
  - **Left side (full height)**: two vertical full-body panels — one **full-body front view** and one **full-body back view**
  - **Right side (standard layout)**: a 6-panel detail grid in a **3x2** layout:
    - Top row (head views): **frontal face portrait**, **side profile portrait**, **45-degree angle portrait** (showing the expression and face/mask)
    - Bottom row (body sections): **chest area** from the front, **pelvic area** from the front, **pelvic area/buttocks** from the back
  - **Expanded layout**: when a face/mask or weapon is selected, or accessories are set, the right side expands to **8 panels in a 4x2 layout** adding **dedicated close-up panels of the face and weapon** and **separate isolated close-ups** of accessory items on neutral backgrounds — never worn or held
  - **Unclothed unless dressed**: the villain is unclothed in the full-body panels unless an outfit is selected (if you use **random_all** and want a naked sheet, set **outfit** to **exclude**)
  - `pose` is ignored in sheet mode (the views are prescribed); `expression` and `injuries` show in the portraits; `additional` is appended at the very end
  - Always ends with: clinical/forensic style, flat even studio lighting, absolute consistency of features across panels, and an absolute **no text/labels/symbols/watermarks** prohibition

### 3. system_prompt (STRING)

- The **system prompt** you can paste into your LLM so it generates horror villain sheet prompts itself
- The text lives in `json/star_horror_villains_builder.json` as the `system_prompt` key — edit it there to customize it permanently

## Typical workflows

- **Direct villain**: connect `description` to your prompt/CLIP text input
- **Direct sheet**: connect `villain_sheet` straight to a sampler text encoder — no LLM needed
- **LLM flow**: connect `system_prompt` as the system message of your LLM node and `description` as the user message — the LLM will expand it into a full villain sheet prompt following the rules

## Customizing the option lists

Edit `json/star_horror_villains_builder.json` in the StarNodes folder:

- Each category is a simple list of strings — add, remove or reword entries as you like
- New entries appear in the dropdowns after a **ComfyUI restart** — but `random` picks use the updated file immediately
- The `settings` block holds `age_random_min`, `age_random_max` and a global `exclude` list of category names

## Tips

- Set unused categories to **none** — only selected traits end up in the prompt
- Build famous-villain vibes by combining archetype + face + outfit + weapon: *masked slasher + scratched hockey mask + mechanic's coveralls + machete*, or *dream stalker + severely burned face + striped sweater and fedora + bladed claw glove*, or *cenobite + grid-scarred face with embedded pins + black leather robe + meat hooks on chains*
- Use **random** on a few categories and fixed values on the rest for controlled variety (great together with a fixed **seed**)
- Use the **additional** field for the scene: `in a dark basement, single hanging light bulb, volumetric fog`
- Pair with ⭐ Star Character Builder for the unsuspecting victim in the same scene
