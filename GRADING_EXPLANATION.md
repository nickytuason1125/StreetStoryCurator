# How Street Story Curator Grades Pictures

## Grading Process Overview

Street Story Curator uses a multi-dimensional analysis approach to grade photographs, evaluating them across several key criteria to determine their overall quality and suitability for various purposes.

## Core Grading Dimensions

### 1. Technical Quality
- **Sharpness Assessment**: Uses Laplacian variance across multiple image regions to evaluate focus quality
- **Exposure Analysis**: Checks for blown highlights and blocked shadows
- **Noise Detection**: Evaluates noise levels in shadow areas
- **Intentional Blur Detection**: Identifies when blur is artistic rather than accidental

### 2. Composition
- **DINOv2 Analysis**: Uses a pre-trained vision transformer model to evaluate compositional elements
- **Rule-of-Thirds Energy**: Measures visual energy at key compositional points
- **Subject Prominence**: Evaluates how well the main subject stands out
- **Focal Hierarchy**: Assesses the spatial organization of elements

### 3. Lighting
- **Brightness Quality**: Evaluates overall exposure balance
- **Contrast Assessment**: Measures global and local contrast levels
- **Color Mood**: Analyzes color temperature and saturation harmony
- **Chiaroscuro Detection**: Identifies dramatic lighting patterns

### 4. Authenticity/Narrative
- **MobileViT Aesthetic Scoring**: Uses a model trained to predict aesthetic quality
- **Decisive Moment Detection**: Evaluates timing and gesture quality
- **Human Presence**: Detects and evaluates human subjects in the frame

### 5. Human/Cultural Presence
- **Face Detection**: Uses YuNet (when available) or Haar cascades to detect faces
- **Subject Isolation**: Evaluates how well human subjects are isolated
- **Cultural Context**: Assesses environmental storytelling elements

## Preset-Based Weighting

The application uses different competition presets that weight these dimensions differently:

- **Magnum Editor**: Emphasizes decisive moments and layered compositions
- **Travel Editor**: Prioritizes cultural authenticity and environmental storytelling
- **World Press Doc**: Focuses on technical clarity and journalistic integrity
- **Cinematic/Editorial**: Rewards mood, color tone, and narrative suggestion
- **Fine Art/Contemporary**: Values abstraction and artistic intent
- **Minimalist/Urbex**: Penalizes clutter and rewards simplicity

## Final Grade Assignment

Images are assigned one of three grades:
- **Strong (✅)**: High overall score (>0.55 in most presets)
- **Mid (⚠️)**: Moderate score with potential for improvement (0.38-0.55)
- **Weak (❌)**: Low score indicating significant issues (<0.38)

## False Negatives and Positives

### Current Limitations

1. **False Positives** (Good photos marked as weak):
   - May occur with artistic blur that's mistaken for technical issues
   - Can happen with unconventional compositions that don't fit standard rules
   - Lighting evaluations may not account for all creative lighting scenarios

2. **False Negatives** (Weak photos marked as strong):
   - Technical sharpness may compensate for lack of narrative content
   - May overvalue certain aesthetic qualities while missing conceptual issues

### Mitigation Strategies

The system includes several mechanisms to reduce these errors:

1. **Multiple Model Approach**: Uses different models (DINOv2, MobileViT, NIMA) to cross-validate assessments
2. **Contextual Analysis**: Adjusts scoring based on detected photo genres and styles
3. **Human Presence Detection**: YuNet provides more accurate face detection than basic Haar cascades
4. **Aesthetic Scoring**: NIMA adds human-rated aesthetic evaluation to technical metrics
5. **Duplicate Detection**: Identifies similar shots to avoid over-scoring burst sequences

### Continuous Improvement

The system is designed to be improved over time through:
- Integration of additional models and analysis techniques
- User feedback incorporation
- Regular calibration against professional judging standards
- Expansion of training data for specialized photography genres

## Conclusion

While no automated system can perfectly replicate human judgment, Street Story Curator's multi-dimensional approach with optional enhanced models (YuNet for better face detection and NIMA for human-rated aesthetics) provides a robust framework for photo evaluation that minimizes false positives and negatives while offering actionable feedback for improvement.