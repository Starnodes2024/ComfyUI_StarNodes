# ⭐ Star Everything to INT/STR

## Overview

This node converts **any incoming value** into:

- an `INT` output (safe conversion)
- a `STRING` output (safe conversion)

It is useful when a workflow accidentally sends a non-number (for example a `tuple`) into something that expects an integer and you get errors like:

`TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'`

---

## Inputs

### Required

- **value** (`*`)
  - Connect anything (STRING / INT / FLOAT / tuple / list / tensor / etc.).

---

## Outputs

- **int** (`INT`)
  - If conversion is not possible, this outputs `0`.
  - If `value` is a tuple/list with a single item, it will try to convert that single item.

- **str** (`STRING`)
  - Uses `str(value)`.

---

## Notes

- For string inputs:
  - Tries `int("...")`
  - If that fails, tries `float("...")` then converts to `int`
  - If that fails, returns `0`

---

**Category**: `⭐StarNodes/Helpers And Tools`

**Node name**: `StarEverythingToIntStr`

**Display name**: `⭐ Star Everything to INT/STR`
