# StarryLinks

A visual-only link renderer for ComfyUI that overlays animated stars and dots along connection splines. It does not affect execution; it only decorates how links are drawn.

## Enable StarryLinks

- Right-click canvas → Links → Render mode → StarryLink
- Or set in Settings: `Comfy.LinkRenderMode` → StarryLink

## Features

- Arc-length accurate placement of stars and dots for better alignment with the spline
- Preset color selection (dropdown) for dots and stars
- Optional twinkle animation for stars
- Optional blinking animation for dots
- Optional cute purple otters that climb along connection splines
- Optional PNG sprite otters (with frame animation) as an alternative to vector otters

## Settings

All settings are under the StarryLinks section in the Settings dialog.

- __Dots Enabled__ (`StarryLinks.DotsEnabled`)
  - Toggle dots overlay.
- __Dot Color__ (`StarryLinks.DotColor`)
  - Preset palette for dot fill: White, Yellow, Gold, Orange, Red, Pink, Purple, Blue, Cyan, Teal, Green.
- __Dot Step (points)__ (`StarryLinks.DotStep`)
  - Controls dot density indirectly. Lower values = more dots.
  - Note: With arc-length placement and higher-resolution sampling, this now produces more uniform spacing than before.
- __Dot Size__ (`StarryLinks.DotSize`)
  - Dot radius.
- __Dots Blinking__ (`StarryLinks.DotBlinkEnabled`)
  - Enables blinking animation for dots.
- __Dot Blink Speed__ (`StarryLinks.DotBlinkSpeed`)
  - Speed of dot blinking (0.1–5.0).
- __Dot Blink Strength__ (`StarryLinks.DotBlinkStrength`)
  - Intensity of dot blinking (0.0–1.0).
- __Twinkling Stars__ (`StarryLinks.TwinkleEnabled`)
  - Enables star twinkle animation.
- __Twinkle Speed__ (`StarryLinks.TwinkleSpeed`)
  - Speed of twinkle (0.1–5.0).
- __Twinkle Strength__ (`StarryLinks.TwinkleStrength`)
  - Intensity of twinkle (0.0–1.0).
- __Stars__ (`StarryLinks.StarsEnabled`)
  - Toggle star overlay.
- __Star Color__ (`StarryLinks.StarColor`)
  - Preset palette for star fill and stroke: White, Yellow, Gold, Orange, Red, Pink, Purple, Blue, Cyan, Teal, Green.
- __Star Count__ (`StarryLinks.StarCount`)
  - Number of stars per link (0–10).
- __Star Size__ (`StarryLinks.StarSize`)
  - Star outer radius.
- __Line Width__ (`StarryLinks.LineWidth`)
  - Passed through to base link renderer.
- __Otters__ (`StarryLinks.OttersEnabled`)
  - Enable cute otters that climb along the rope.
- __Otter Count__ (`StarryLinks.OtterCount`)
  - Number of otters per link (0–5).
- __Otter Speed__ (`StarryLinks.OtterSpeed`)
  - Climbing speed along the rope (0.2–3.0). Interpreted as a pleasant canvas pace.
- __Otter Scale__ (`StarryLinks.OtterScale`)
  - Size multiplier for otters (0.5–2.0).
- __Otter Direction__ (`StarryLinks.OtterDirection`)
  - up, down, or both (ping‑pong between ends).
- __Otter Cuteness FX__ (`StarryLinks.OtterCutenessFX`)
  - Adds subtle bobbing and occasional heart sparkles.
- __Otter Sprites (PNG)__ (`StarryLinks.OtterSpritesEnabled`)
  - Use PNG sprite otters instead of vector-drawn otters.
- __Otter Sprite Frames__ (`StarryLinks.OtterSpriteFrames`)
  - Number of frames. 1 uses `web/img/otters/otter.png`; >1 uses `web/img/otters/otter_0.png` ... `otter_(N-1).png`.
- __Otter Sprite FPS__ (`StarryLinks.OtterSpriteFPS`)
  - Frames per second for sprite animation (0 disables frame cycling).
- __Reset All Settings__
  - Resets StarryLinks settings to defaults listed below.

## Defaults

- Dots Enabled: true
- Dot Color: #ffffff (White)
- Dot Step (points): 10
- Dot Size: 1.0
- Dots Blinking: true
- Dot Blink Speed: 1.0
- Dot Blink Strength: 1.0
- Twinkling Stars: true
- Twinkle Speed: 3.0
- Twinkle Strength: 1.0
- Stars: true
- Star Color: #ffd24a (Yellow)
- Star Count: 3
- Star Size: 10.0
- Line Width: 1.0

- Otters: false
- Otter Count: 1
- Otter Speed: 1.2
- Otter Scale: 1.0
- Otter Direction: up
- Otter Cuteness FX: true
- Otter Sprites (PNG): false
- Otter Sprite Frames: 9
- Otter Sprite FPS: 6

## Sprite Assets

- Place files under: `web/img/otters/`
- Sizes: 48×48 PNG with transparency recommended.
- Orientation: face RIGHT; rendering rotates to the rope orientation.
- Center the character in the image; keep a small transparent margin.
- Naming:
  - Static: `otter.png`
  - Animated: `otter_0.png`, `otter_1.png`, ..., `otter_8.png` (for 9 frames by default)

## Tips

- If you want fewer or more dots, adjust __Dot Step (points)__.
- If settings don’t appear after updating, close and reopen the Settings dialog.
- Animations (twinkle/blink) redraw only when menus aren’t open to avoid UI jank.
- Otters use the same animation refresh; enabling them may increase redraws on very large graphs—reduce count or speed if needed.
- Use direction = both for a playful ping‑pong effect.

## Changelog

- 2025-08-24
  - Switched to preset color dropdowns for dots and stars for better compatibility.
  - Renamed setting: Purple Dot Chain → Dots Enabled.
  - Added dot blinking with speed/strength.
  - Improved alignment using arc-length sampling for both stars and dots.
  - Added optional animated cute otters with speed/scale/direction/cuteness settings.
  - Added PNG sprite otters support with frames/FPS and documented asset placement.
