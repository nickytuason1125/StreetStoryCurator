"""
Niche Registry — 20 photography niches with per-niche:
  - Qwen2.5-VL scoring axes  (5 dimensions + calibration text)
  - SigLIP-2 CLIP probe sets (positive + negative text probes for pre-filter)
  - Display metadata         (label, category, icon)

Used by:
  qwen_vlm_grader.py  → build_niche_prompt() generates the scoring prompt
  grade_pipeline_v2.py → niche_clip_probes() returns (pos, neg) for pre-filter
  frontend/src/App.tsx → NICHE_CATEGORIES for the niche picker UI
"""

from __future__ import annotations
from typing import NamedTuple

# ── Data types ────────────────────────────────────────────────────────────────

class NicheAxes(NamedTuple):
    key:  str
    desc: str


REGISTRY: dict[str, dict] = {}


def _reg(
    key:         str,
    label:       str,
    category:    str,
    mode_label:  str,
    calibration: str,
    axes:        list[tuple[str, str]],
    pos_probes:  list[str],
    neg_probes:  list[str],
) -> None:
    REGISTRY[key] = {
        "label":       label,
        "category":    category,
        "mode_label":  mode_label,
        "calibration": calibration,
        "axes":        axes,
        "pos_probes":  pos_probes,
        "neg_probes":  neg_probes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STREET & DOCUMENTARY
# ═══════════════════════════════════════════════════════════════════════════════

_reg(
    key="classic_street",
    label="Classic Street",
    category="Street & Documentary",
    mode_label="street photography",
    calibration=(
        "A competently exposed street shot scores 55-65; "
        "a decisive-moment image with strong composition scores 70-80; "
        "exceptional human geometry or peak moment scores 85+. "
        "Reserve below 40 for genuine technical failures."
    ),
    axes=[
        ("composition",  "framing, geometry, visual hierarchy, decisive use of space"),
        ("lighting",     "quality, direction, tonal depth, shadow control, mood"),
        ("moment",       "decisiveness, timing, peak action, human tension or irony"),
        ("human",        "gesture, expression, relationship, cultural energy (0 if no person)"),
        ("technical",    "sharpness, exposure accuracy, noise, lens rendering"),
    ],
    pos_probes=[
        "decisive moment candid street photograph Cartier-Bresson",
        "peak action human gesture urban street photography",
        "layered street scene foreground background depth",
        "strong shadow geometry street photography Vivian Maier",
        "candid human interaction public space documentary",
        "graphic silhouette backlight street figure urban",
        "frame within frame doorway arch street scene",
        "wet street reflection puddle urban atmosphere",
        "ironic juxtaposition advertisement street photography",
        "leading lines perspective human subject street",
    ],
    neg_probes=[
        "blurry out of focus accidental camera shake",
        "poorly exposed snapshot tourist photo random",
        "empty boring street no interest no subject",
        "posed fake unnatural street photo",
        "cluttered messy no composition no intent",
    ],
)

_reg(
    key="documentary",
    label="Documentary",
    category="Street & Documentary",
    mode_label="documentary photography",
    calibration=(
        "A contextually clear documentary image scores 55-65; "
        "one with emotional truth and strong composition scores 70-80; "
        "exceptional narrative impact with authenticity scores 85+. "
        "Reserve below 40 for images with no documentary value."
    ),
    axes=[
        ("narrative_impact",  "story clarity, emotional truth, scene significance"),
        ("composition",       "framing, visual hierarchy, contextual depth"),
        ("authenticity",      "candid realism, unposed naturalness, truthful moment"),
        ("context",           "environmental context, social/cultural information"),
        ("technical",         "sharpness, exposure, grain quality, lens rendering"),
    ],
    pos_probes=[
        "powerful documentary photograph social issue authentic",
        "photojournalism candid human condition real life",
        "environmental documentary portrait subject in context",
        "long-form documentary essay image narrative",
        "authentic unstaged moment human truth documentary",
        "social documentary community life cultural record",
        "intimate access documentary close proximity real",
        "witness photography human rights documentary impact",
    ],
    neg_probes=[
        "staged artificial set up fake documentary",
        "tourist snapshot no documentary value",
        "random snapshot no narrative no meaning",
        "posed unnatural awkward forced",
    ],
)

_reg(
    key="photojournalism",
    label="Photojournalism",
    category="Street & Documentary",
    mode_label="photojournalism",
    calibration=(
        "A news-worthy, contextually clear image scores 55-65; "
        "one with impact, clarity, and editorial urgency scores 70-80; "
        "iconic news image with historical weight scores 85+. "
        "Reserve below 40 for images with no news or documentary value."
    ),
    axes=[
        ("news_impact",        "newsworthiness, visual urgency, historical or social weight"),
        ("compositional_urgency", "framing under pressure, decisive capture, editorial clarity"),
        ("context",            "scene information, who/what/where readable from image"),
        ("authenticity",       "unmanipulated reality, ethical documentation, candid truth"),
        ("technical",          "sharpness on subject, exposure under difficult conditions"),
    ],
    pos_probes=[
        "breaking news photojournalism press photograph",
        "editorial news photography wire service AP Reuters",
        "conflict protest crowd documentary news photography",
        "political event news photojournalism decisive moment",
        "disaster rescue human interest news photography",
        "sports news photojournalism peak action editorial",
        "portrait news subject environmental context editorial",
    ],
    neg_probes=[
        "staged manipulated fake news photo",
        "blurry missed focus unusable press photo",
        "no news value random snapshot",
        "tourist postcard no editorial context",
    ],
)

_reg(
    key="travel_cultural",
    label="Travel & Cultural",
    category="Street & Documentary",
    mode_label="travel and cultural documentary photography",
    calibration=(
        "A competent travel image with sense of place scores 55-65; "
        "one with cultural authenticity and human connection scores 70-80; "
        "exceptional immersive cultural documentary scores 85+. "
        "Reserve below 40 for generic tourist snapshots with no cultural insight."
    ),
    axes=[
        ("sense_of_place",      "geographical specificity, cultural atmosphere, 'you are there'"),
        ("cultural_authenticity", "genuine local life, unstaged tradition, human geography"),
        ("composition",         "framing, depth, visual hierarchy, environmental integration"),
        ("light_atmosphere",    "quality of light, color, time of day, mood"),
        ("technical",           "sharpness, exposure, color rendering"),
    ],
    pos_probes=[
        "travel photography sense of place cultural authenticity",
        "local market street life cultural documentary travel",
        "traditional ceremony ritual cultural heritage photography",
        "environmental portrait local person in cultural context",
        "landscape with cultural human element travel documentary",
        "food market craft shop local life travel photography",
        "golden hour travel destination authentic atmosphere",
    ],
    neg_probes=[
        "generic tourist postcard no authenticity",
        "posed unnatural tourist photo at landmark",
        "empty landmark without human or cultural context",
        "random snapshot no sense of place",
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE & SPACE
# ═══════════════════════════════════════════════════════════════════════════════

_reg(
    key="architectural",
    label="Architectural",
    category="Architecture & Space",
    mode_label="architectural photography",
    calibration=(
        "A well-composed architectural shot scores 55-65; "
        "a spatially compelling image with strong geometry scores 70-80; "
        "exceptional structure, light, and spatial narrative scores 85+. "
        "Reserve below 40 for technically poor or randomly framed building shots."
    ),
    axes=[
        ("geometry",     "structural lines, symmetry, rhythm, architectural clarity, spatial depth"),
        ("lighting",     "quality, direction, tonal depth, shadow control, facade modeling"),
        ("composition",  "framing, perspective, spatial hierarchy, decisive use of space"),
        ("detail",       "material texture, surface quality, structural detail, craft"),
        ("technical",    "sharpness, exposure, perspective correction, lens rendering"),
    ],
    pos_probes=[
        "architectural photography building structure geometric composition",
        "modernist architecture strong lines symmetry photography",
        "interior architectural photography spatial light quality",
        "facade detail texture material architecture close-up",
        "urban architecture perspective converging lines dramatic",
        "brutalist concrete architecture dramatic shadow light",
        "glass steel modern architecture reflection abstract",
        "historical architecture classical column arch detail",
    ],
    neg_probes=[
        "random snapshot of building tourist photo",
        "distorted perspective accidental tilt poor framing",
        "blurry building photo no architectural intent",
        "cluttered obstructed building shot no clarity",
    ],
)

_reg(
    key="liminal",
    label="Liminal Spaces",
    category="Architecture & Space",
    mode_label="liminal space photography",
    calibration=(
        "A quiet transitional space scores 55-65; "
        "one with genuine uncanny atmosphere and spatial tension scores 70-80; "
        "an exceptional void with psychological resonance scores 85+. "
        "Reserve below 40 for technically poor or ordinary-looking empty spaces."
    ),
    axes=[
        ("atmosphere",   "sense of void, uncanny stillness, psychological tension, solitude"),
        ("geometry",     "spatial lines, perspective depth, architectural structure, scale"),
        ("light_quality","quality, direction, color, emptiness, vacancy feeling"),
        ("composition",  "framing, leading lines, spatial hierarchy"),
        ("technical",    "sharpness, exposure, color grading, lens rendering"),
    ],
    pos_probes=[
        "liminal space empty corridor uncanny atmosphere",
        "empty swimming pool parking garage abandoned mall",
        "transitional architecture non-place solitude",
        "empty escalator airport terminal corridor quiet",
        "brutalist empty concrete space psychological",
        "empty shopping mall food court solitude eerie",
        "backrooms aesthetic empty fluorescent light corridor",
    ],
    neg_probes=[
        "crowded busy space with people activity",
        "random empty room no atmosphere no composition",
        "cluttered messy disorganized space",
        "generic interior snapshot no mood",
    ],
)

_reg(
    key="urban_city",
    label="Urban / City",
    category="Architecture & Space",
    mode_label="urban city photography",
    calibration=(
        "A competent urban texture image scores 55-65; "
        "one with strong city energy, composition, and mood scores 70-80; "
        "exceptional metropolitan atmosphere with visual poetry scores 85+. "
        "Reserve below 40 for generic city snapshots with no visual interest."
    ),
    axes=[
        ("urban_energy",  "metropolitan atmosphere, city texture, cultural vitality"),
        ("composition",   "framing, geometric rhythm, visual hierarchy, urban geometry"),
        ("light_mood",    "time of day, artificial light, atmospheric quality"),
        ("city_texture",  "signage, materials, density, urban layering, infrastructure"),
        ("technical",     "sharpness, exposure, color rendering"),
    ],
    pos_probes=[
        "urban cityscape metropolitan photography strong composition",
        "neon signs city night urban texture photography",
        "busy intersection pedestrian urban energy photography",
        "city rooftop skyline dramatic light urban landscape",
        "underground subway metro urban commuter photography",
        "chinatown ethnic neighborhood cultural urban texture",
        "construction infrastructure urban industrial photography",
    ],
    neg_probes=[
        "generic city snapshot tourist postcard no composition",
        "blurry city photo accidental no intent",
        "random urban scene no visual interest",
        "empty featureless urban environment",
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# LIGHT & MOOD
# ═══════════════════════════════════════════════════════════════════════════════

_reg(
    key="night",
    label="Night / Low Light",
    category="Light & Mood",
    mode_label="night and low light photography",
    calibration=(
        "A competently exposed night scene scores 55-65; "
        "one with atmospheric mood and strong light design scores 70-80; "
        "exceptional nocturnal atmosphere with visual poetry scores 85+. "
        "Reserve below 40 for technically poor night shots without artistic intent."
    ),
    axes=[
        ("light_quality",   "quality of artificial or ambient light, pools, halos, neon glow"),
        ("nocturnal_mood",  "atmospheric darkness, mystery, urban night energy, isolation"),
        ("composition",     "framing within light constraints, shadow use, silhouettes"),
        ("color_palette",   "color temperature balance, neon vs warm, contrast quality"),
        ("technical",       "noise control, sharpness on subject, exposure balance"),
    ],
    pos_probes=[
        "night photography city lights atmospheric darkness",
        "neon lights rain street night urban photography",
        "chiaroscuro dramatic light pool darkness night photography",
        "long exposure city night light trails atmospheric",
        "night portrait street available light cinematic",
        "blue hour dusk urban twilight atmospheric photography",
        "wet street neon reflection night photography",
        "film noir inspired night street photography shadow light",
    ],
    neg_probes=[
        "overexposed night photo blown highlights harsh",
        "pure darkness underexposed nothing visible",
        "flat boring daytime photo nothing to do with night",
        "noisy blurry technically failed night shot",
    ],
)

_reg(
    key="long_exposure",
    label="Long Exposure",
    category="Light & Mood",
    mode_label="long exposure photography",
    calibration=(
        "A clean long exposure with clear intent scores 55-65; "
        "one with beautiful light trails and strong composition scores 70-80; "
        "exceptional time-layered image with visual poetry scores 85+. "
        "Reserve below 40 for accidental blur or technically failed exposures."
    ),
    axes=[
        ("motion_quality",    "intentional blur quality, smoothness, directional flow"),
        ("light_painting",    "light trail clarity, color, luminance paths, star trails"),
        ("composition",       "framing around motion, static vs dynamic elements"),
        ("temporal_effect",   "sense of time passing, ethereal quality, motion-freeze contrast"),
        ("technical",         "sharpness on static elements, exposure balance, noise"),
    ],
    pos_probes=[
        "long exposure photography light trails city",
        "slow shutter waterfall silky smooth water",
        "long exposure star trails milky way night sky",
        "steel wool spinning light painting photography",
        "long exposure traffic light trails urban night",
        "slow shutter crowd blur ghost motion photography",
        "seascape long exposure silky water misty atmosphere",
    ],
    neg_probes=[
        "accidental camera shake blur technical failure",
        "unintentional motion blur missed focus",
        "static subject no motion effect boring",
        "overexposed blown out long exposure failure",
    ],
)

_reg(
    key="fine_art",
    label="Fine Art",
    category="Light & Mood",
    mode_label="fine art photography",
    calibration=(
        "A visually coherent fine art image scores 55-65; "
        "one with painterly quality and artistic intent scores 70-80; "
        "exceptional conceptual and aesthetic execution scores 85+. "
        "Reserve below 40 for images with no discernible artistic vision."
    ),
    axes=[
        ("artistic_vision",  "conceptual intent, artistic statement, personal vision"),
        ("mood",             "emotional resonance, tonal atmosphere, psychological feeling"),
        ("composition",      "painterly framing, visual poetry, deliberate spatial arrangement"),
        ("visual_poetry",    "metaphorical reading, layers of meaning, aesthetic beauty"),
        ("technical",        "exposure, tonal range, focus as expressive tool, print quality"),
    ],
    pos_probes=[
        "fine art photography painterly light artistic vision",
        "pictorialist photography soft focus atmospheric beauty",
        "conceptual fine art photograph artistic statement",
        "large format film photography tonal range beauty",
        "Sally Mann Francesca Woodman fine art photography",
        "vintage lens bokeh soft focus fine art portrait",
        "infrared photography artistic dreamlike fine art",
    ],
    neg_probes=[
        "snapshot commercial photograph no artistic intent",
        "harsh flat documentary no fine art quality",
        "random snapshot accidental no vision",
        "technically poor failed exposure no artistic excuse",
    ],
)

_reg(
    key="minimalist",
    label="Minimalist",
    category="Light & Mood",
    mode_label="minimalist photography",
    calibration=(
        "A clean minimal composition scores 55-65; "
        "one with strong negative space and graphic power scores 70-80; "
        "exceptional reduction with visual tension and precision scores 85+. "
        "Reserve below 40 for unintentionally sparse or poorly composed images."
    ),
    axes=[
        ("negative_space",    "deliberate emptiness, breathing room, void as subject"),
        ("graphic_simplicity","reduction to essential elements, clean lines, graphic clarity"),
        ("tonal_balance",     "high key/low key elegance, contrast without clutter"),
        ("composition",       "precise placement, ratio, geometric tension"),
        ("technical",         "sharpness, tonal fidelity, clean exposure, no distractions"),
    ],
    pos_probes=[
        "minimalist photography negative space graphic composition",
        "clean lines simple subject isolated background minimalism",
        "fog mist minimalist landscape single element",
        "white snow ice minimalist landscape photography",
        "single subject vast negative space stark photography",
        "Japanese aesthetic wabi-sabi minimalist photography",
        "graphic design photography minimal geometric clean",
    ],
    neg_probes=[
        "cluttered busy distracting background elements",
        "overcrowded frame too many subjects no focus",
        "random sparse image accidental not minimalist",
        "messy disorganized composition no clarity",
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# SUBJECT-FOCUSED
# ═══════════════════════════════════════════════════════════════════════════════

_reg(
    key="portrait",
    label="Portrait",
    category="Subject-Focused",
    mode_label="portrait photography",
    calibration=(
        "A technically competent portrait scores 55-65; "
        "one with genuine expression and environmental relationship scores 70-80; "
        "exceptional character study with psychological depth scores 85+. "
        "Reserve below 40 for technically poor or expressionless portraits."
    ),
    axes=[
        ("expression",          "emotional authenticity, character revelation, psychological depth"),
        ("composition",         "subject placement, environmental relationship, framing, gaze"),
        ("lighting",            "flattering or characterful light quality, shadow modeling, mood"),
        ("environmental_context","subject relationship to surroundings, narrative depth, context"),
        ("technical",           "sharpness on eyes, exposure, depth of field, color rendering"),
    ],
    pos_probes=[
        "environmental portrait subject in context expressive",
        "character study portrait psychological depth candid",
        "natural light portrait window light intimate",
        "street portrait candid authentic character",
        "dramatic portrait chiaroscuro light shadow character",
        "editorial portrait expressive mood styled",
        "intimate portrait connection eye contact authentic",
    ],
    neg_probes=[
        "bad focus eyes not sharp portrait failure",
        "flat unflattering light passport photo no expression",
        "forced smile awkward posed unnatural",
        "random person snapshot no portrait intent",
    ],
)

_reg(
    key="wildlife",
    label="Wildlife",
    category="Subject-Focused",
    mode_label="wildlife photography",
    calibration=(
        "A competently captured wildlife image scores 55-65; "
        "one with peak behavior and strong habitat context scores 70-80; "
        "exceptional rare behavior with technical mastery scores 85+. "
        "Reserve below 40 for technically poor or zoo/captive animal images with no behavior."
    ),
    axes=[
        ("subject_behavior",   "peak action, rare behavior, animal interaction, moment quality"),
        ("habitat_context",    "natural environment integration, ecosystem storytelling"),
        ("composition",        "subject placement, habitat framing, negative space"),
        ("light_quality",      "golden hour, catchlight in eye, tonal quality, mood"),
        ("technical",          "sharpness on subject eye, exposure, background separation"),
    ],
    pos_probes=[
        "wildlife photography animal behavior natural habitat",
        "bird in flight peak action wildlife photography",
        "predator prey interaction wildlife decisive moment",
        "golden hour wildlife portrait natural light catchlight",
        "underwater marine wildlife photography behavior",
        "macro wildlife insect spider natural habitat photography",
        "wildlife landscape environmental portrait habitat context",
    ],
    neg_probes=[
        "zoo animal cage bars captive no behavior",
        "blurry missed focus wildlife technical failure",
        "distant unidentifiable animal no behavior no interest",
        "pet domestic animal snapshot no wildlife context",
    ],
)

_reg(
    key="fashion_editorial",
    label="Fashion / Editorial",
    category="Subject-Focused",
    mode_label="fashion and editorial photography",
    calibration=(
        "A competent fashion image scores 55-65; "
        "one with strong editorial mood and styling scores 70-80; "
        "exceptional concept, execution, and fashion energy scores 85+. "
        "Reserve below 40 for technically poor or unstyled fashion attempts."
    ),
    axes=[
        ("styling_aesthetic",  "garment presentation, styling coherence, fashion vision"),
        ("model_expression",   "pose, attitude, energy, connection with camera, editorial power"),
        ("editorial_mood",     "concept execution, narrative, mood board alignment"),
        ("composition",        "framing, model placement, environmental interaction"),
        ("technical",          "sharpness, exposure, color rendition, lighting quality"),
    ],
    pos_probes=[
        "fashion photography editorial styled model strong concept",
        "high fashion editorial Vogue Harper's Bazaar photography",
        "street style fashion photography urban editorial",
        "beauty editorial close-up makeup fashion photography",
        "avant-garde fashion concept artistic editorial photography",
        "commercial fashion catalog clean product photography",
        "runway fashion show photography editorial backstage",
    ],
    neg_probes=[
        "unstylized casual snapshot no fashion context",
        "poorly styled unflattering unintentional clothing photo",
        "random person no editorial intent no fashion",
        "blurry out of focus fashion photo technical failure",
    ],
)

_reg(
    key="sports_action",
    label="Sports / Action",
    category="Subject-Focused",
    mode_label="sports and action photography",
    calibration=(
        "A competent action shot scores 55-65; "
        "one at peak action with strong composition scores 70-80; "
        "exceptional peak moment with emotion and technical mastery scores 85+. "
        "Reserve below 40 for missed action, blur, or passively posed sports images."
    ),
    axes=[
        ("peak_action",     "decisive peak moment, timing precision, athletic climax"),
        ("motion_quality",  "motion freeze or intentional blur, speed feeling, dynamism"),
        ("composition",     "sport-specific framing, anticipation, negative space for motion"),
        ("emotion",         "athlete intensity, crowd energy, competitive spirit"),
        ("technical",       "sharpness on subject, exposure in fast conditions, background"),
    ],
    pos_probes=[
        "peak action sports photography decisive moment athletic",
        "basketball football soccer peak action sports photography",
        "track and field athletics peak action photography",
        "extreme sports action photography dynamic motion",
        "victory celebration emotional sports photography",
        "underwater sports swimming photography peak action",
        "motorsport speed blur action photography F1 racing",
    ],
    neg_probes=[
        "passive static sport no action standing still",
        "blurry missed peak action technical failure",
        "warm-up sideline no game action no tension",
        "posed team photo no athletic energy",
    ],
)

_reg(
    key="wedding",
    label="Wedding / Ceremony",
    category="Subject-Focused",
    mode_label="wedding and ceremony photography",
    calibration=(
        "A competent ceremony image scores 55-65; "
        "one with genuine emotion and beautiful light scores 70-80; "
        "exceptional intimate moment with technical excellence scores 85+. "
        "Reserve below 40 for technically poor or emotionally flat ceremony images."
    ),
    axes=[
        ("emotional_moment",  "genuine emotion, tears, laughter, connection, tender intimacy"),
        ("composition",       "subject framing, environmental context, visual storytelling"),
        ("light_quality",     "golden hour, window light, candlelight, quality of ceremony light"),
        ("story_telling",     "narrative arc, before/after energy, cultural tradition visible"),
        ("technical",         "sharpness on faces, exposure in difficult ceremony light"),
    ],
    pos_probes=[
        "wedding photography emotional moment ceremony beautiful",
        "first kiss ceremony wedding photography authentic",
        "bride groom portrait golden hour wedding photography",
        "wedding reception dance celebration candid photography",
        "detail wedding ring flowers dress photography",
        "family portrait wedding emotional candid laughter",
        "cultural wedding ceremony tradition photography authentic",
    ],
    neg_probes=[
        "flat emotionless posed wedding photo no feeling",
        "blurry out of focus wedding technical failure",
        "empty venue no subjects no ceremony context",
        "awkward forced pose wedding photo unnatural",
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# CREATIVE & SPECIALIZED
# ═══════════════════════════════════════════════════════════════════════════════

_reg(
    key="landscape",
    label="Landscape",
    category="Creative & Specialized",
    mode_label="landscape photography",
    calibration=(
        "A competent landscape scores 55-65; "
        "one with exceptional light, depth, and atmosphere scores 70-80; "
        "a transformative landscape with painterly quality scores 85+. "
        "Reserve below 40 for flat, featureless, or technically poor landscapes."
    ),
    axes=[
        ("light_quality",   "golden hour, dramatic sky, tonal richness, shadow depth"),
        ("landscape_comp",  "foreground interest, layers, horizon placement, scale"),
        ("weather_drama",   "clouds, storm, mist, fog, atmospheric conditions"),
        ("depth_scale",     "sense of vastness, scale, spatial recession, infinity"),
        ("technical",       "sharpness throughout, exposure balance, color rendering"),
    ],
    pos_probes=[
        "landscape photography golden hour dramatic light",
        "mountain landscape dramatic clouds atmospheric photography",
        "coastal seascape golden hour wave photography",
        "forest landscape misty fog atmospheric photography",
        "desert landscape minimalist vast photography",
        "Ansel Adams style black white landscape photography",
        "aurora borealis northern lights landscape photography",
    ],
    neg_probes=[
        "flat overcast featureless boring landscape",
        "snapshot landscape no composition no light interest",
        "blown sky or foreground exposure failure landscape",
        "tourist snapshot landmark no landscape artistry",
    ],
)

_reg(
    key="abstract",
    label="Abstract",
    category="Creative & Specialized",
    mode_label="abstract photography",
    calibration=(
        "A competent abstract scores 55-65; "
        "one with strong visual rhythm and graphic power scores 70-80; "
        "exceptional abstraction with color, form, and tension scores 85+. "
        "Reserve below 40 for accidental abstractions with no visual interest."
    ),
    axes=[
        ("visual_abstraction", "departure from representation, non-literal reading, visual puzzle"),
        ("pattern_texture",    "rhythm, repetition, surface quality, tactile visual"),
        ("color_form",         "color relationships, tonal contrast, shape tension"),
        ("graphic_impact",     "boldness, visual punch, striking first impression"),
        ("technical",          "sharpness on intended plane, exposure control, color fidelity"),
    ],
    pos_probes=[
        "abstract photography pattern texture visual rhythm",
        "geometric abstraction architecture color form photography",
        "macro texture abstract close-up surface photography",
        "reflection water abstract impressionist photography",
        "shadow pattern abstract graphic photography",
        "intentional camera movement ICM abstract photography",
        "multiple exposure abstract photography visual experiment",
    ],
    neg_probes=[
        "accidental random blur no artistic abstract intent",
        "clearly representational subject no abstraction",
        "random texture no composition no visual interest",
        "technical failure masquerading as abstract",
    ],
)

_reg(
    key="macro",
    label="Macro / Close-up",
    category="Creative & Specialized",
    mode_label="macro and close-up photography",
    calibration=(
        "A competent macro with sharp subject scores 55-65; "
        "one with beautiful depth of field and subject detail scores 70-80; "
        "exceptional micro world with composition and light mastery scores 85+. "
        "Reserve below 40 for out-of-focus or compositionally empty macro attempts."
    ),
    axes=[
        ("subject_detail",  "fine detail revelation, texture richness, close-up discovery"),
        ("depth_of_field",  "deliberate bokeh quality, focus plane choice, optical rendering"),
        ("composition",     "subject placement within tight frame, negative space, angle"),
        ("light_quality",   "macro lighting, rim light, diffused close-up light, color"),
        ("technical",       "critical sharpness on focus plane, diffraction control, stacking"),
    ],
    pos_probes=[
        "macro photography extreme close-up fine detail revelation",
        "flower petal dew drop macro photography",
        "insect eye compound macro photography extreme close-up",
        "water drop splash macro high speed photography",
        "food macro extreme close-up texture photography",
        "circuit board technology macro abstract photography",
        "fabric textile fiber macro texture photography",
    ],
    neg_probes=[
        "out of focus macro wrong plane blurry subject",
        "random close-up no detail interest no composition",
        "normal lens close-up not true macro",
        "technically failed macro diffraction blur",
    ],
)

_reg(
    key="experimental",
    label="Experimental / Conceptual",
    category="Creative & Specialized",
    mode_label="experimental and conceptual photography",
    calibration=(
        "An experimental image with clear concept scores 55-65; "
        "one with strong visual innovation and execution scores 70-80; "
        "exceptional avant-garde work with transformative vision scores 85+. "
        "Reserve below 40 for accidental failures with no discernible concept."
    ),
    axes=[
        ("conceptual_strength", "clarity of concept, intellectual or emotional statement"),
        ("visual_innovation",   "departure from convention, new ways of seeing, originality"),
        ("execution",           "concept translated into visual reality with skill"),
        ("intent_clarity",      "deliberate vs accidental, purposeful choices, cohesion"),
        ("technical",           "technical choices serve concept, exposure, focus as tools"),
    ],
    pos_probes=[
        "experimental photography avant-garde visual concept",
        "conceptual photography artistic statement new vision",
        "multiple exposure surreal photography experimental",
        "cameraless photogram lumen print alternative process",
        "infrared conceptual photography otherworldly",
        "solarization darkroom experimental photography",
        "constructed narrative staged conceptual photography",
    ],
    neg_probes=[
        "accidental technical failure random mistake",
        "random snapshot no concept no artistic intent",
        "technical error presented as experimental",
        "conventional photo labeled experimental with no reason",
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

# Map legacy preset strings to registry keys for backward compatibility
LEGACY_MAP: dict[str, str] = {
    "classic street": "classic_street",
    "street":         "classic_street",
    "architectural":  "architectural",
    "fine art":       "fine_art",
    "liminal":        "liminal",
    "story":          "documentary",
    "competition":    "photojournalism",
}

# Ordered category list for UI grouping
CATEGORIES: list[str] = [
    "Street & Documentary",
    "Architecture & Space",
    "Light & Mood",
    "Subject-Focused",
    "Creative & Specialized",
]


def get_niche(preset: str) -> dict | None:
    """Return niche dict for a preset string (key or legacy name)."""
    key = preset.lower().strip()
    if key in REGISTRY:
        return REGISTRY[key]
    if key in LEGACY_MAP:
        return REGISTRY[LEGACY_MAP[key]]
    return None


# Niches whose genre intrinsically centres on a human subject. For every other
# niche (architecture, travel/cultural, landscape, minimalist, liminal, abstract,
# macro, night, urban, etc.) a peopleless frame is legitimate and must NOT be
# penalised by human-centric scoring signals.
_HUMAN_CENTRIC_KEYS: set[str] = {
    "classic_street", "documentary", "photojournalism",
    "portrait", "fashion_editorial", "sports_action", "wedding",
}

# UI display label (lowercased) → registry key, for callers passing the label.
_LABEL_TO_KEY: dict[str, str] = {v["label"].lower(): k for k, v in REGISTRY.items()}


def resolve_key(preset: str) -> str | None:
    """Resolve a preset string (key, legacy name, or UI label) to a registry key."""
    key = preset.lower().strip()
    if key in REGISTRY:
        return key
    if key in LEGACY_MAP:
        return LEGACY_MAP[key]
    return _LABEL_TO_KEY.get(key)


def is_human_centric(preset: str) -> bool:
    """True when the niche intrinsically requires a human subject.

    Unknown presets default to True (street-style behaviour) to stay conservative.
    Travel/Cultural is deliberately treated as NOT human-centric — a strong
    architectural or environmental travel frame should not be punished for lacking
    a person.
    """
    k = resolve_key(preset)
    return True if k is None else (k in _HUMAN_CENTRIC_KEYS)


def niche_clip_probes(preset: str) -> tuple[list[str], list[str]]:
    """
    Return (pos_probes, neg_probes) for SigLIP-2 CLIP pre-filter.
    Falls back to classic_street probes if preset is unknown.
    """
    niche = get_niche(preset) or REGISTRY["classic_street"]
    return niche["pos_probes"], niche["neg_probes"]


# Scale anchors are MODEL-SPECIFIC: each VLM generation maps language to
# numbers differently. Qwen3-VL-2B graded the same batch 0.53-0.80 (mean 0.69)
# where Qwen2.5-VL-3B gave 0.21-0.74 (mean 0.52) under identical anchors —
# the 2B needs explicit low-end anchoring and an anti-clustering instruction.
_SCALE_ANCHORS = {
    "default": (
        "Scale anchors: a tack-sharp decisive moment with layered composition "
        "scores around 85; a competent but static, unremarkable frame around 55; "
        "a technically flawed image with no clear subject around 25. "
    ),
    "qwen3_vl": (
        "Scale anchors: most ordinary frames belong between 30 and 60. "
        "A competent but static, unremarkable frame scores around 45; "
        "a technically flawed image with no clear subject around 20; "
        "reserve 70+ for genuinely exceptional work — a tack-sharp decisive "
        "moment with layered composition scores around 85. Use the full 0-100 "
        "range and do not cluster your scores above 60. "
    ),
}


def build_niche_prompt(preset: str, rag_block: str = "", model_family: str = "") -> str:
    """
    Build a VLM scoring prompt for the given niche.
    model_family selects the per-model scale anchors (e.g. "qwen3_vl").
    Returns the full prompt string ready to pass to the VLM.
    """
    import json as _json

    niche = get_niche(preset) or REGISTRY["classic_street"]
    mode_label  = niche["mode_label"]
    calibration = niche["calibration"]
    axes        = niche["axes"]

    # Placeholders, NOT example values: literal zeros here anchored the 3B
    # model's outputs low (observed 2026-06-11: near-constant comp=20/light=25/
    # score=35 across a whole batch with greedy decoding).
    # Field ORDER is deliberate (generation is sequential): the model must
    # write what it actually sees (strongest_element / main_flaw) BEFORE any
    # number, then the per-axis values, then the holistic score LAST —
    # evidence-first scoring for small VLMs.
    # No critique sentence in the scoring schema — decode length is the
    # dominant per-photo cost, and the UI critique is synthesised from the
    # evidence phrases by parse_niche_breakdown (deep critique is on-demand).
    json_keys: dict = {
        "strongest_element": "<short phrase: what genuinely works>",
        "main_flaw":         "<short phrase: the biggest weakness>",
    }
    for key, _ in axes:
        json_keys[key] = "<0-100>"
    json_keys["score"] = "<0-100>"
    json_schema = _json.dumps(json_keys)

    axes_desc = "\n".join(f"  {key:<18} {desc}" for key, desc in axes)

    # NOTE: keep tone words ("blunt", "harsh", "brutal") OUT of this prompt
    # entirely — they make the 3B model deflate every numeric axis to 10-30 to
    # sound tough (observed 2026-06: a whole batch collapsed to Weak). The goal
    # is accuracy: scores anchored to the calibration text and the RAG rubric.
    return (
        f"You are an experienced photo editor evaluating a photograph for a {mode_label} edit.\n\n"
        f"Rate this image accurately. Output ONLY valid JSON — no prose, no markdown fences:\n"
        f"{json_schema}\n\n"
        f"Fill strongest_element and main_flaw FIRST — describe what you actually see "
        f"before writing any number.\n"
        f"All numeric values are integers 0–100 and are calibrated measurements: {calibration} "
        f"{_SCALE_ANCHORS.get(model_family, _SCALE_ANCHORS['default'])}"
        f"Score exactly what you see — do not deflate numbers to seem strict, do not "
        f"inflate them to be kind.\n"
        f"  strongest_element  the single element that genuinely works best in this image — "
        f"specific and concrete, no generic praise\n"
        f"  main_flaw          the single biggest weakness in this image — specific and concrete\n"
        f"{axes_desc}\n"
        f"{rag_block}"
        f"JSON only:"
    )


def _norm_key(k: str) -> str:
    """Collapse a key to a comparison form: lowercase alphanumerics only.

    Lets 'Human/Culture', 'human_culture', 'Human Culture' all match 'human'."""
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


# ── Axis → canonical fusion role ─────────────────────────────────────────────
# Mirror of frontend ASPECT_DIM (App.tsx). The fusion step in grade_pipeline_v2
# needs Composition/Lighting/Human/Narrative signals, but Qwen emits niche-
# specific axis names — without this mapping every non-street niche fused on
# 0.5 placeholder defaults. Keys are the Title-Case display names that
# parse_niche_breakdown produces.
_ASPECT_ROLE: dict[str, str] = {
    # canonical 5
    "Technical": "tech", "Composition": "comp", "Lighting": "light",
    "Narrative": "auth", "Human/Culture": "human",
    # tech
    "Detail": "tech", "Execution": "tech", "Depth Of Field": "tech",
    "Detail Retention": "tech", "Cleanliness": "tech", "News Sharpness": "tech",
    "Sharpness & Detail": "tech",
    # comp (geometry / framing / spatial)
    "Geometry": "comp", "Compositional Urgency": "comp", "City Texture": "comp",
    "Landscape Comp": "comp", "Depth Scale": "comp", "Negative Space": "comp",
    "Graphic Simplicity": "comp", "Visual Abstraction": "comp",
    "Pattern Texture": "comp", "Graphic Impact": "comp", "Framing": "comp",
    "Geometry & Balance": "comp", "Framing Instinct": "comp", "Layered Depth": "comp",
    # light (lighting / mood / colour / atmosphere)
    "Light Atmosphere": "light", "Light Quality": "light", "Light Mood": "light",
    "Nocturnal Mood": "light", "Color Palette": "light", "Light Painting": "light",
    "Color Form": "light", "Tonal Balance": "light", "Atmosphere": "light",
    "Weather Drama": "light", "Mood": "light", "Natural Light": "light",
    "Mood & Tone": "light", "Tonal Purity": "light", "Contrast Purity": "light",
    "Available Light": "light", "Natural Light Quality": "light",
    # human (subject / figure / expression / emotion)
    "Human": "human", "Expression": "human", "Model Expression": "human",
    "Subject Behavior": "human", "Subject Detail": "human", "Emotion": "human",
    "Emotional Moment": "human", "Styling Aesthetic": "human",
    "Sense Of Place": "human", "Subject Isolation": "human", "Human Impact": "human",
    "Character Presence": "human", "Emotional Resonance": "human",
    "Scale Element": "human", "Presence": "human", "Scale & Life": "human",
    # auth (moment / narrative / concept / authenticity)
    "Moment": "auth", "Narrative Impact": "auth", "Authenticity": "auth",
    "Context": "auth", "News Impact": "auth", "Cultural Authenticity": "auth",
    "Urban Energy": "auth", "Motion Quality": "auth", "Temporal Effect": "auth",
    "Artistic Vision": "auth", "Visual Poetry": "auth",
    "Environmental Context": "auth", "Habitat Context": "auth",
    "Editorial Mood": "auth", "Peak Action": "auth", "Story Telling": "auth",
    "Conceptual Strength": "auth", "Visual Innovation": "auth",
    "Intent Clarity": "auth", "Decisive Moment": "auth", "Cultural Depth": "auth",
    "Journalistic Integrity": "auth", "Narrative Suggestion": "auth",
    "Conceptual Weight": "auth", "Reduction": "auth", "Immediacy": "auth",
    "Environmental Truth": "auth",
}
_ASPECT_ROLE_NORM = {_norm_key(k): v for k, v in _ASPECT_ROLE.items()}

_ROLE_KEYWORDS: list[tuple[str, str]] = [
    # ordered — first hit wins (mirrors frontend keyword fallback priorities)
    ("light",  "light"), ("tonal", "light"), ("tone", "light"), ("color", "light"),
    ("mood",   "light"), ("atmosphere", "light"), ("weather", "light"),
    ("comp",   "comp"), ("geometr", "comp"), ("fram", "comp"), ("graphic", "comp"),
    ("pattern", "comp"), ("space", "comp"), ("layer", "comp"),
    ("human",  "human"), ("subject", "human"), ("expression", "human"),
    ("emotion", "human"), ("gesture", "human"), ("people", "human"),
    ("moment", "auth"), ("narrative", "auth"), ("story", "auth"),
    ("authentic", "auth"), ("cultural", "auth"), ("concept", "auth"),
    ("sharp",  "tech"), ("detail", "tech"), ("technical", "tech"),
    ("focus",  "tech"), ("noise", "tech"),
]


def aspect_role(axis_key: str) -> str:
    """Map an axis key onto a fusion role: 'tech'|'comp'|'light'|'human'|'auth'|''.

    Exact (normalised) table lookup first, keyword fallback for unseen axes,
    '' for non-aspect keys (private fields, counters, etc.)."""
    nk = _norm_key(axis_key)
    hit = _ASPECT_ROLE_NORM.get(nk)
    if hit:
        return hit
    for kw, role in _ROLE_KEYWORDS:
        if kw in nk:
            return role
    return ""


def _coerce_score(v) -> int | None:
    """Pull an int 0–100 out of whatever the VLM put in a value slot.

    Handles ints, floats, numeric strings ('75', '75/100', '0.75'), and nested
    dicts ({'score': 70}, {'value': 70}). Returns None when nothing usable."""
    import re as _re
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        # A 0–1 float (some models emit normalised scores) → scale to 0–100.
        if 0.0 < f <= 1.0:
            f *= 100.0
        return max(0, min(100, int(round(f))))
    if isinstance(v, dict):
        for sub in ("score", "value", "rating", "val"):
            if sub in v:
                return _coerce_score(v[sub])
        return None
    if isinstance(v, str):
        m = _re.search(r'-?\d+(?:\.\d+)?', v)
        if m:
            return _coerce_score(float(m.group()))
    return None


def parse_niche_breakdown(data: dict, preset: str) -> tuple[float, dict, str]:
    """
    Parse a VLM JSON response into (score_0_1, breakdown_dict, critique).
    Handles all 20 niches dynamically — no hardcoded axis names.

    Robust to the ways a VLM drifts from the requested schema: key casing,
    spaces vs underscores, separator variants ('Human/Culture'), numeric values
    embedded in strings ('75/100'), nested score objects, and a missing overall
    'score' (derived from the aspect mean rather than silently defaulting to 50).
    """
    if not isinstance(data, dict):
        return 0.5, {}, ""

    # Normalised lookup of the model's actual keys → first occurrence value.
    norm_lookup: dict[str, object] = {}
    for k, v in data.items():
        nk = _norm_key(k)
        if nk and nk not in norm_lookup:
            norm_lookup[nk] = v

    niche = get_niche(preset) or REGISTRY["classic_street"]

    display_breakdown: dict[str, float] = {}
    aspect_vals: list[float] = []
    for key, _ in niche["axes"]:
        v = norm_lookup.get(_norm_key(key))
        s = _coerce_score(v) if v is not None else None
        if s is None:
            continue
        display_key = key.replace("_", " ").title()
        display_breakdown[display_key] = s / 100.0
        aspect_vals.append(s / 100.0)

    # Overall score: blend the model's holistic number with the aspect mean.
    # Small VLMs anchor the single "score" field far harder than the per-axis
    # judgments (observed: score=35 constant while axes varied) — weighting
    # the axis mean higher lets the per-dimension reasoning drive the result.
    overall = _coerce_score(norm_lookup.get("score") or norm_lookup.get("overall"))
    if overall is not None and aspect_vals:
        axis_mean = sum(aspect_vals) / len(aspect_vals)
        score = 0.40 * (overall / 100.0) + 0.60 * axis_mean
    elif overall is not None:
        score = overall / 100.0
    elif aspect_vals:
        score = sum(aspect_vals) / len(aspect_vals)
    else:
        score = 0.5

    critique = ""
    for ckey in ("critique", "comment", "summary", "reason", "feedback"):
        cv = norm_lookup.get(_norm_key(ckey))
        if isinstance(cv, str) and cv.strip():
            critique = cv.strip()[:140]
            break
    if not critique:
        # Evidence-first fields land before the critique in generation order,
        # so they survive truncation — synthesise from them instead of
        # returning a blank.
        _se = norm_lookup.get(_norm_key("strongest_element"))
        _mf = norm_lookup.get(_norm_key("main_flaw"))
        _parts = []
        if isinstance(_se, str) and _se.strip():
            _parts.append(f"Strongest: {_se.strip()}")
        if isinstance(_mf, str) and _mf.strip():
            _parts.append(f"main flaw: {_mf.strip()}")
        critique = " — ".join(_parts)[:140]

    return score, display_breakdown, critique
