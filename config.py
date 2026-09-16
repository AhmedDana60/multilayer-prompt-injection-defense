import os
from dotenv import load_dotenv

load_dotenv()

# Configuration and thresholds
THRESHOLD = 0.50
DEBERTA_MODEL_PATH = "models/deberta-finetuned-v3"
OPENAI_MODEL = "gpt-4o-mini"       # target/generation model (Layer 2 tests)
CRITIC_MODEL = "gpt-4.1-mini"      # critic/refiner model (Layer 3, Layer 4)
API_KEY = os.getenv("OPENAI_API_KEY")
