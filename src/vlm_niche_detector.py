import os
import json
import asyncio
import threading
from pathlib import Path
from llama_cpp import Llama, LlamaGrammar

NICHE_LIST = [
    "Street/Urban", "Portrait/People", "Wedding/Event", "Architecture", "Real Estate",
    "Food/Culinary", "Product/Commercial", "Landscape/Nature", "Night/Nocturnal",
    "Sports/Action", "Macro/Detail", "Documentary/Travel", "Fine Art/Creative",
]

# Grammar enforces valid JSON shape — passed per inference call, not to the constructor.
# Fix: "." not ". " — the space made the rule match "0. 9" not "0.9".
_GRAMMAR_SRC = r'''
root   ::= "{" ws '"niche"' ws ":" ws string ws "," ws '"confidence"' ws ":" ws float ws "}"
string ::= "\"" ([^"\\] | "\\" .)* "\""
float  ::= [0-9]+ "." [0-9]+
ws     ::= [ \t\r\n]*
'''

_SYSTEM_PROMPT = (
    "You are a senior photo editor. Classify the image into EXACTLY ONE niche from this list:\n"
    + json.dumps(NICHE_LIST, ensure_ascii=False) + "\n\n"
    "Rules:\n"
    "- Return ONLY valid JSON: {\"niche\": \"...\", \"confidence\": <0.0-1.0>}\n"
    "- Confidence ≥0.85 required. If below, still return your best guess — the caller filters.\n"
    "- Focus purely on composition, lighting, subject matter, and photographic intent."
)

# Constructor-only parameters — passed to Llama() once at model load.
_VLM_INIT = {
    "n_ctx":     4096,   # covers ~1.5k vision tokens + prompt + JSON output
    "n_threads": max(1, os.cpu_count() or 4),
    "n_batch":   512,
    "mmap":      True,
    "mlock":     False,
    "verbose":   False,
}

# Inference-time parameters — passed to self.llm() on every call.
_VLM_INFER = {
    "max_tokens":     150,
    "temperature":    0.1,
    "top_p":          0.9,
    "top_k":          40,
    "repeat_penalty": 1.1,
}

# Grading uses lower temperature for determinism and more tokens for arrays.
_VLM_GRADE_INFER = {
    "max_tokens":     250,
    "temperature":    0.05,
    "top_p":          0.9,
    "top_k":          40,
    "repeat_penalty": 1.1,
}

_GRADE_GRAMMAR_SRC = r'''
root        ::= "{" ws '"score"' ws ":" ws float ws "," ws '"critique"' ws ":" ws string ws "," ws '"strengths"' ws ":" ws str-array ws "," ws '"improvements"' ws ":" ws str-array ws "}"
str-array   ::= "[" ws (string (ws "," ws string)*)? ws "]"
string      ::= "\"" ([^"\\] | "\\" .)* "\""
float       ::= [0-9]+ "." [0-9]+
ws          ::= [ \t\r\n]*
'''

_GRADE_SYSTEM_PROMPT = (
    "You are a senior photo editor. Grade this image on a strict 0.00–1.00 scale.\n"
    "Output ONLY valid JSON matching this schema exactly:\n"
    "{\"score\": 0.85, \"critique\": \"1-sentence actionable feedback\", "
    "\"strengths\": [\"example strength\"], \"improvements\": [\"example improvement\"]}\n\n"
    "Rules:\n"
    "- score: 0.00–1.00, multiples of 0.05 only.\n"
    "- critique: one sentence, specific and actionable — not generic.\n"
    "- strengths / improvements: 1–3 items each, concrete visual observations.\n"
    "- Return ONLY JSON. No markdown, no explanation."
)


class VLMNicheDetector:
    def __init__(self, model_path: str = "models/qwen2.5-vl-3b-instruct-q4_k_m.gguf"):
        # These are always set so attribute access never raises outside classify_batch.
        self.cache_path = Path("cache/vlm_niche_cache.json")
        self.cache: dict = (
            json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self.cache_path.exists() else {}
        )
        try:
            self.llm     = Llama(model_path=model_path, **_VLM_INIT)
            self._grammar = LlamaGrammar.from_string(_GRAMMAR_SRC)
        except Exception as e:
            print(f"VLM failed to load: {e}")
            self.llm      = None
            self._grammar = None

    async def classify_batch(self, image_paths: list[str]) -> dict:
        if not self.llm:
            return self.cache

        new_paths = [p for p in image_paths if p not in self.cache]
        if not new_paths:
            return self.cache

        print(f"VLM scanning {len(new_paths)} uncached images...")

        for i, path in enumerate(new_paths):
            try:
                _path = path  # capture loop variable for lambda closure
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: self.llm(
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": [
                                {"type": "image_url",
                                 "image_url": {"url": f"file://{os.path.abspath(_path)}"}},
                                {"type": "text", "text": "Classify this image. Return ONLY JSON."},
                            ]},
                        ],
                        grammar=self._grammar,
                        **_VLM_INFER,
                    )),
                    timeout=60.0,
                )
                parsed = json.loads(result["choices"][0]["message"]["content"].strip())
                conf  = parsed.get("confidence", 0.0)
                niche = parsed.get("niche", "")
                # Only persist confident, valid results — others stay absent so the
                # caller falls through to the metric fallback automatically.
                if conf >= 0.85 and niche in NICHE_LIST:
                    self.cache[path] = {"niche": niche, "confidence": conf}
            except asyncio.TimeoutError:
                print(f"VLM inference timed out for {path}")
                continue
            except Exception as e:
                print(f"VLM inference error for {path}: {e}")
                continue

            if (i + 1) % 10 == 0:
                self._save_cache()
                print(f"  {i + 1}/{len(new_paths)} processed...")

        self._save_cache()
        return self.cache

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# Score thresholds for the editorial-note gate (used by server.py).
DEEP_REVIEW_TOP    = 0.65   # Strong grade boundary
DEEP_REVIEW_LOW    = 0.45   # Borderline band lower bound
DEEP_REVIEW_HIGH   = 0.55   # Borderline band upper bound

# Grammar for rationale output — no numeric score field, so VLM can never
# produce a number that the UI might accidentally treat as a quality signal.
_RATIONALE_GRAMMAR_SRC = r'''
root   ::= "{" ws '"critique"' ws ":" ws string ws "," ws '"strength"' ws ":" ws string ws "," ws '"is_suggestion"' ws ":" ws "true" ws "}"
string ::= "\"" ([^"\\] | "\\" .)* "\""
ws     ::= [ \t\r\n]*
'''

_RATIONALE_SYSTEM_PROMPT = (
    "You are a senior photo editor writing brief editorial notes for a photographer's review.\n"
    "Output ONLY valid JSON: "
    "{\"critique\": \"<1-sentence observation about light, composition, or moment>\", "
    "\"strength\": \"<the single strongest visual element>\", "
    "\"is_suggestion\": true}\n\n"
    "Rules:\n"
    "- critique: one sentence, observational tone — not a score, not a verdict.\n"
    "- strength: a short noun phrase (e.g. 'rim lighting', 'diagonal leading line').\n"
    "- Never emit numbers, percentages, or quality ratings.\n"
    "- Return ONLY JSON."
)


class VLMGrader:
    """
    Generates structured per-image critique for gated photos.
    Reuses the Llama instance already loaded by VLMNicheDetector — no second
    model load.  Results are persisted to cache/vlm_grade_cache.json so
    subsequent runs are instant.
    """

    def __init__(self, llm: Llama) -> None:
        if llm is None:
            raise ValueError("VLMGrader requires a live Llama instance")
        self.llm       = llm
        self._grammar  = LlamaGrammar.from_string(_GRADE_GRAMMAR_SRC)
        self.cache_path = Path("cache/vlm_grade_cache.json")
        self.cache: dict = (
            json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self.cache_path.exists() else {}
        )

    def grade_batch_sync(self, image_paths: list[str]) -> dict:
        """
        Synchronous batch grader — safe to call from a ThreadPoolExecutor thread.
        Only processes paths not already in cache.
        """
        new_paths = [p for p in image_paths if p not in self.cache]
        if not new_paths:
            return self.cache

        print(f"VLM deep-review: {len(new_paths)} images...")

        for i, path in enumerate(new_paths):
            try:
                result = self.llm(
                    messages=[
                        {"role": "system", "content": _GRADE_SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"file://{os.path.abspath(path)}"}},
                            {"type": "text", "text": "Grade this image. Return ONLY JSON."},
                        ]},
                    ],
                    grammar=self._grammar,
                    **_VLM_GRADE_INFER,
                )
                parsed = json.loads(result["choices"][0]["message"]["content"].strip())
                # Validate required keys before caching
                if all(k in parsed for k in ("score", "critique", "strengths", "improvements")):
                    self.cache[path] = parsed
            except Exception:
                pass  # path stays absent; UI shows no VLM critique

            if (i + 1) % 5 == 0:
                self._save_cache()

        self._save_cache()
        return self.cache

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )


class VLMRationaleGenerator:
    """
    Production replacement for VLMGrader.

    Outputs only qualitative editorial notes — no numeric scores, no technical
    claims. _analyze() remains the sole source of truth for score/grade/breakdown.
    Every result carries is_suggestion=True so UI and downstream code can gate on it.
    """

    def __init__(self, llm: Llama) -> None:
        if llm is None:
            raise ValueError("VLMRationaleGenerator requires a live Llama instance")
        self.llm        = llm
        self._grammar   = LlamaGrammar.from_string(_RATIONALE_GRAMMAR_SRC)
        self.cache_path = Path("cache/vlm_rationale_cache.json")
        self.cache: dict = (
            json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self.cache_path.exists() else {}
        )

    def generate_batch_sync(self, image_paths: list[str]) -> dict:
        """
        Synchronous batch generator — safe to call from a ThreadPoolExecutor thread.
        Only processes uncached paths.
        """
        new_paths = [p for p in image_paths if p not in self.cache]
        if not new_paths:
            return self.cache

        print(f"VLM rationale: {len(new_paths)} images...")

        for i, path in enumerate(new_paths):
            try:
                result = self.llm(
                    messages=[
                        {"role": "system", "content": _RATIONALE_SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"file://{os.path.abspath(path)}"}},
                            {"type": "text", "text": "Write an editorial note for this image. Return ONLY JSON."},
                        ]},
                    ],
                    grammar=self._grammar,
                    max_tokens=120,
                    temperature=0.05,
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.1,
                )
                parsed = json.loads(result["choices"][0]["message"]["content"].strip())
                if "critique" in parsed and "strength" in parsed:
                    parsed["is_suggestion"] = True   # enforce regardless of model output
                    self.cache[path] = parsed
            except Exception:
                pass  # stays absent; UI shows nothing rather than bad data

            if (i + 1) % 5 == 0:
                self._vlm_save_cache()

        self._vlm_save_cache()
        return self.cache

    def _vlm_save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )
