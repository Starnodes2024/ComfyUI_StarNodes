# ⭐ Star 1 Wildcard

A compact version of **⭐ Star Wildcards Advanced** with a single prompt and a single wildcard dropdown.

- **Category**: `⭐StarNodes/Text And Data`
- **Node name**: `Star1Wildcard`
- **Output**: `STRING`

## Inputs

- **seed** (`INT`)
  - Seed for the wildcard line pick and inline syntax. Change it to re-roll.

- **prompt** (`STRING`, multiline)
  - Free text prompt. Supports `{option1|option2}` random-option syntax and inline `__wildcard__` / `folder\__wildcard__` references, same as the advanced node.

- **wildcard** (dropdown)
  - `None`: no extra wildcard is appended.
  - `Random`: picks a random wildcard file from `ComfyUI/wildcards`.
  - Any listed wildcard file: appends one seeded random line from that file.

## Behavior

The processed prompt and the picked wildcard line are joined with a space and returned as a single `STRING`, ready to feed into a text/conditioning node.
