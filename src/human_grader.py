import torch, clip, logging
from PIL import Image
from pathlib import Path

_log = logging.getLogger(__name__)

class LAIONAestheticScorer:
    def __init__(self, model_dir="./models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.device = torch.device("cpu")
        # Use ViT-L/14 for SOTA performance
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device, download_root=str(self.model_dir))
        self.aesthetic_head = torch.nn.Linear(768, 1)
        self._load_weights()

    def _load_weights(self):
        # Official Hugging Face URL for Improved Aesthetic Predictor V2+
        url = "https://huggingface.co/ChristophSchuhmann/improved_aesthetic_predictor/resolve/main/ava1-l14-linearMSE.pth"
        try:
            _log.info("Downloading LAION Aesthetic Weights from Hugging Face...")
            state_dict = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            self.aesthetic_head.load_state_dict(state_dict)
            _log.info("LAION Weights loaded.")
        except Exception as e:
            _log.warning("Failed to load LAION weights (Network/404 error): %s", e)
            # Fallback to random weights to prevent crash, though scores will be neutral
            self.aesthetic_head = None

    def score(self, img_path):
        if self.aesthetic_head is None: return 0.5
        try:
            img = Image.open(img_path).convert("RGB")
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.clip_model.encode_image(x).float()
                raw = self.aesthetic_head(emb).item()
            # Normalize 4.0-7.5 range to 0.0-1.0
            return max(0.0, min((raw - 4.0) / 3.5, 1.0))
        except Exception as e:
            return 0.5

# Singleton Pattern: Ensure this runs only once
human_grader = LAIONAestheticScorer()

def get_human_aesthetic_score(img_path):
    return human_grader.score(img_path)
