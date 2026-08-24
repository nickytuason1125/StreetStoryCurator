# Running FrameGrade light — low-RAM machines, no GPU, any OS

FrameGrade adapts to the machine it is on. On a laptop with no graphics card
and little free memory, use **Lite mode** — it trades some analysis depth for
a much smaller memory footprint and faster culls.

## What Lite mode changes

| Setting | Normal | Lite |
|---|---|---|
| Vision encoder | Best that fits in RAM (up to Pro) | Fast tier, pinned (~1 GB) |
| Grade depth | Full pipeline | Scan pass by default (~2–3× faster) |
| Deep Grade toggle | Available | Available, but off by default |
| Thumbnail workers | Scaled to RAM | Minimum (2) |
| Writing engine (critiques/story text) | Loaded on demand | Unchanged — only loads when you open those features |

Measured on a no-GPU machine (100 photos): Lite culls at roughly
**2–3 s/image with ~2–3 GB peak RAM**, versus up to 11–14 s/image if a large
encoder were forced onto the CPU.

## How to start it

- **Windows:** double-click `Start-Lite.bat`
- **macOS:** run `./Start.command --lite` or `bash start-lite.sh`
- **Linux:** run `bash start-lite.sh`

They simply set environment variables before launching the same server —
nothing else differs.

## Manual equivalent (any OS)

```bash
# macOS / Linux
export SIGLIP_TIER=low          # pin the small encoder
export FRAMEGRADE_LITE=1        # lite defaults (scan-first, min workers)
python server.py

# Windows (cmd)
set SIGLIP_TIER=low
set FRAMEGRADE_LITE=1
python server.py
```

An explicit `SIGLIP_TIER` always wins over auto-selection, so Lite machines
never try to load an encoder that does not fit.

## If it still feels heavy

1. Close browser tabs — Chrome alone commonly holds 2–4 GB.
2. Use **Scan** mode for the first pass; re-grade only the keepers.
3. Leave Deep Grade off unless a specific photo deserves the detailed read.
4. Check the RAM chip in the header: if it reads amber/red, grading is being
   refused or degraded on purpose to protect the machine.