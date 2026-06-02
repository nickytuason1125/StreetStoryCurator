import json, hashlib
prompts = [
    "minimalist architectural and interior geometry, graphic lines, vanishing points, empty liminal commercial space, Edward Hopper quiet interior light, stark geometric shadow, empty diner or laundromat",
    "cinematic low-key street photography, dark atmospheric shadows, intense chiaroscuro light pools",
    "layered environmental street portrait, crisp subject focus framed by intentional out-of-focus foreground elements",
    "unintentional messy amateur snapshot, accidental random camera angles, domestic clutter, junk, trash, throwaway frame, zero artistic value",
    "highly detailed maximalist environmental documentary photography, authentic traditional shop interior or street life scene, dense cultural artifacts, rich storytelling composition, intentional maximalism",
]
h = hashlib.md5(json.dumps(prompts).encode()).hexdigest()
print("Current prompt hash:", h)
cached = "449d893326cbc57e36f3f7dba0d5307c"
print("Cached hash:        ", cached)
print("Match:", h == cached)
