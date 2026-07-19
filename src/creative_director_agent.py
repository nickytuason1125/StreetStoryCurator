"""
Creative Director Agent — Frontier 2026
Bridges the 'Style Brief' to the NSGA-III Sequencer.
"""

import json

class CreativeDirectorAgent:
    def __init__(self, model_client):
        self.model = model_client # e.g., DeepSeek-R1-7B

    def generate_constraints(self, user_brief: str) -> dict:
        """
        Translates human vibes into sequencer-ready math.
        """
        system_prompt = (
            "You are a professional Creative Director. Translate the user's brief into "
            "technical constraints for an image sequencer. Output ONLY JSON.\n"
            "Constraints to control: \n"
            "- diversity_weight (0.0 to 1.0)\n"
            "- human_limit (0.0 to 1.0, 0.0 = empty)\n"
            "- focal_theme (angle, contrast, or color)\n"
            "- novelty_bias (1.0 = strictly avoid previous results)"
        )
        
        # This is where the 'Repetition' is killed
        user_prompt = f"Brief: {user_brief}. I am seeing repetitive results, so ensure novelty is high."
        
        raw_output = self.model.generate(system_prompt, user_prompt)
        return json.loads(raw_output)

# --- Integration Example for VS Code ---

def run_creative_direction_ui(user_brief, source_anchor):
    agent = CreativeDi
    
    rectorAgent(vlm_client)
    
    # 1. The LLM 'thinks' about the creative direction
    constraints = agent.generate_constraints(user_brief)
    
    # 2. Pass those dynamic weights to the sequencer
    sequence = run_nsga3_sequence(
        candidates=pool,
        target=7,
        diversity_weight=constraints['diversity_weight'],
        human_penalty=8.0 if constraints['human_limit'] < 0.2 else 0.0,
        novelty_boost=constraints['novelty_bias']
    )
    return sequence