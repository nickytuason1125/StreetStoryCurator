# Street Story Curator Architecture

## Overview
Street Story Curator is an offline street photography analysis and sequencing tool that grades photos based on multiple aesthetic and technical criteria, then generates narrative sequences suitable for storytelling or social media presentation.

## Core Components

### 1. Image Analysis Engine
The `LightweightStreetScorer` class in `src/lightweight_analyzer.py` is the heart of the analysis system:

#### Technical Analysis
- Sharpness assessment using Laplacian variance across multiple image regions
- Exposure quality evaluation (blown highlights, blocked shadows)
- Noise detection in shadow areas
- Intentional blur detection (for film/vintage effects)

#### Composition Analysis
- DINOv2-small ONNX model for patch-based composition analysis
- Rule-of-thirds energy detection
- Subject prominence scoring
- Focal hierarchy detection

#### Lighting Analysis
- Overall brightness assessment
- Contrast evaluation (global and local)
- Color mood scoring
- Chiaroscuro detection (dramatic lighting)

#### Authenticity/Narrative Analysis
- MobileViT ONNX model for aesthetic quality scoring
- Decisive moment detection (subject timing)
- Human presence detection using YuNet face detection or Haar cascades

#### Human/Cultural Presence
- Face detection and counting
- Subject isolation scoring
- Human interaction detection

### 2. Scoring System
The application uses preset-based scoring with configurable weights:

#### Preset Profiles
- Classic Street (decisive moments, layering)
- Travel Editor (cultural authenticity)
- Photojournalism (journalistic clarity)
- Cinematic/Editorial (mood and tone)
- Fine Art/Contemporary (abstraction)
- Minimalist/Urbex (simplicity, negative space)
- Humanist/Everyday (candid intimacy)
- LSPF (London Street) (balanced light and human presence)
- Snapshot / Point-and-Shoot (raw immediacy)
- Landscape with Elements (layered depth)

Each preset has different weights for the five core dimensions:
1. Technical quality
2. Composition
3. Lighting
4. Narrative/Authenticity
5. Human/Cultural presence

### 3. Sequence Generation
The `sequence_story` method creates narrative sequences using:

#### Slot-Based Storytelling
1. Opening Frame - Establishes context
2. Focal Subject - Main narrative element
3. Supporting Detail - Technical or compositional detail
4. Contrast/Shift - Visual or thematic contrast
5. Closing Mood - Concluding atmosphere

#### Diversity Mechanisms
- Cosine similarity for visual diversity
- Dimension-based distance metrics
- Duplicate detection and filtering
- Role-based fitness functions

### 4. User Interface
The application has two UI implementations:

#### Gradio Web Interface (src/app.py)
- Folder browsing and image loading
- Preset selection
- Gallery display with filtering
- Instagram carousel generation
- PDF scorecard export

#### Tauri Desktop Application (frontend/)
- React-based frontend with TypeScript
- Desktop integration via Tauri
- Native file system access

## Data Flow

```
Input Images
    ↓
Image Preprocessing (pyvips/cv2)
    ↓
Feature Extraction (ONNX Models)
    ↓
Dimensional Scoring (5 core metrics)
    ↓
Preset-Based Weighting
    ↓
Final Grade Assignment (Strong/Mid/Weak)
    ↓
Sequence Generation (Narrative Ordering)
    ↓
Output (Gallery, PDF, Instagram Carousel)
```

## Dependencies

### Python Dependencies
- gradio: Web interface framework
- opencv-python-headless: Image processing
- numpy: Numerical computing
- scikit-learn: Machine learning utilities
- pyvips-binary: Fast image loading
- pillow: Image handling
- fpdf2: PDF generation
- piexif: EXIF metadata handling
- tqdm: Progress bars
- onnxruntime: ONNX model inference
- requests: Model downloading
- pywebview: Desktop windowing (via Tauri)

### ONNX Models
1. DINOv2-small: Composition analysis and embeddings
2. MobileViT-small: Aesthetic quality scoring
3. NIMA (optional): Neural image assessment
4. YuNet (optional): Face detection

## File Structure
```
street-story-curator/
├── src/                    # Core Python modules
├── models/onnx/           # ONNX models
├── frontend/              # React/Tauri frontend
├── cache/                 # Thumbnail and score caches
├── output/                # Generated outputs
├── requirements.txt       # Python dependencies
└── BUILD.md              # Build instructions
```

## Key Features

### Image Analysis
- Offline processing (no internet required after initial setup)
- Multi-dimensional quality scoring
- Batch processing with caching
- Duplicate detection
- EXIF metadata handling

### Narrative Sequencing
- Genre-specific storytelling templates
- Visual diversity optimization
- Role-based slot assignment
- Rationale generation

### Export Capabilities
- Instagram carousel generation
- PDF scorecards
- ZIP archives of selected sequences
- Multiple aspect ratios

## Performance Considerations
- Multi-threaded image processing
- ONNX model quantization for speed
- Thumbnail caching
- Lazy loading of optional models
- Memory-efficient image preprocessing with pyvips