# Release Notes - Version 3.4.3

## Summary
This release adds two new prompt-builder nodes to the builder family (Horror Villains and Sexy), raises the CLIP Prompt Refiner default length, and fixes a UI state bug in Star Prompt Preview And Edit.

## Changes

### New Features

#### ⭐ Star Horror Villains Builder (`StarHorrorVillainsBuilder`)
- **Horror villain prompt builder** — build villain descriptions entirely from dropdowns: masked slashers, killer dolls, evil clowns, cenobites, ghost girls and more
- 17 categories: villain_type (48 archetypes), size, face/mask, hair, skin, eyes, body_build, injuries, outfit, weapon, accessories, aura, expression, pose, custom, style
- Three outputs: `description`, `villain_sheet` (forensic-style multi-panel reference sheet with face/mask and weapon close-up panels) and `system_prompt`
- All lists editable in `json/star_horror_villains_builder.json`
- New documentation: `web/docs/StarHorrorVillainsBuilder.md`

#### ⭐ Star Sexy Builder (`StarSexyBuilder`)
- **Sensual figure prompt builder** — focused on sexually attractive details: body build, explicit anatomy (breasts/chest, nipples, waist, butt, legs, genitals, pubic hair), lingerie, sexy nightwear and kinky outfits
- Wear rule: lingerie/nightwear/kinky_outfit are mutually exclusive under random_all
- Three outputs: `description`, `sexy_sheet` (figure study sheet with chest/pelvic/buttock detail panels) and `system_prompt`
- All lists editable in `json/star_sexy_builder.json`
- New documentation: `web/docs/StarSexyBuilder.md`

### Enhancements

#### ⭐ Star CLIP Prompt Refiner
- **`max_length` default raised from 512 to 2048** — longer refined prompts out of the box without hitting the generation ceiling

#### README
- New **builder family spotlight** in the "Recently Added & Updated" section covering Character, Creature, Horror Villains and Sexy Builder

### Bug Fixes

#### ⭐ Star Prompt Preview And Edit
- **Preview size reset after workflow tab switch fixed** — the text area height is now re-synced to the restored node size when a workflow is loaded, so a manually resized preview keeps its size

### Updated Files

#### Core Files
- `__init__.py` - Version 3.4.3, registered StarHorrorVillainsBuilder and StarSexyBuilder
- `README.md` - Version 3.4.3 badge, builder family spotlight
- `pyproject.toml` - Version 3.4.3

#### Text & Prompts
- `text_io/star_horror_villains_builder.py` - New node
- `text_io/star_sexy_builder.py` - New node
- `json/star_horror_villains_builder.json` - Option lists and system prompt
- `json/star_sexy_builder.json` - Option lists and system prompt

#### Image Tools
- `image_tools/star_clip_prompt_refiner.py` - max_length default 2048

#### Web Assets
- `web/js/star_prompt_preview_and_edit.js` - Preview size persistence fix
- `web/docs/StarHorrorVillainsBuilder.md` - New documentation
- `web/docs/StarSexyBuilder.md` - New documentation
- `web/docs/StarClipPromptRefiner.md` - Updated default value
- `web/docs/StarPromptPreviewAndEdit.md` - v1.5 note

---

## Version 3.4.2 (2026-10-24) Recap
- ⭐ Star Creature Builder (`StarCreatureBuilder`) - fantasy creature prompt builder with 31 categories (75 species, body plans, wings, horns, tails, colors) and creature_sheet output
- ⭐ Star Load Image+ - "Load Workflow" button loads the embedded workflow exactly like drag&drop onto the canvas
- ⭐ Star Prompt Preview And Edit - mode dropdown replaced by bypass/Edit toggle (default Bypass), legacy workflow compatibility, tab-switch toggle fix
