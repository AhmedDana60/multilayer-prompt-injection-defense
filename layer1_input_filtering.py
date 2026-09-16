import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import THRESHOLD, DEBERTA_MODEL_PATH

# De-obfuscation: fix look-alike letters and spaced-out words
# (e.g. "i g n o r e" -> "ignore") before regex matching,
# so simple disguises cannot bypass the rules. Same as the notebook.
DEOBFUSCATE_MAP = {
    'а': 'a', 'е': 'e', 'о': 'o', 'і': 'i', 'с': 'c', 'р': 'p', 'х': 'x',
    '€': 'e', '@': 'a',
}

def normalize_text(text):
    for fake, real in DEOBFUSCATE_MAP.items():
        text = text.replace(fake, real)
    text = re.sub(r'\b(?:[a-zA-Z]\s+){2,}[a-zA-Z]\b',
                  lambda m: m.group(0).replace(' ', ''), text)
    return text



class InputFilter:
    """
    Layer 1: Multi-Turn Input Screening.
    Context-aware hybrid prompt injection detector.

    Combines:
      - Regex-based rules for explicit/familiar injection phrases, tightened
        through iterative false-positive testing (see thesis Ch.3/4 for the
        full evaluation: detection rate and false-positive rate on Malignant,
        HackAPrompt, and a 70-example realistic-request stress test).
      - A fine-tuned DeBERTa-small classifier for implicit/semantic intent.
      - A sliding window over the current prompt + recent user prompt
        history, to catch injection attempts spread across multiple turns.
    """

    def __init__(self, model_path=None, tokenizer_path="microsoft/deberta-v3-small",
                 window_size=3, threshold=None, flag_action="block"):
        """
        model_path: path to the trained classifier weights. Defaults to
                    DEBERTA_MODEL_PATH from config.py (the fine-tuned checkpoint).
        tokenizer_path: where to load the tokenizer from. Defaults to the
                    original hub model - fine-tuning never changes tokenization,
                    so this avoids version-format mismatches with the saved copy.
        window_size: number of prior user prompts to include alongside
                     the current prompt (default 3, per proposal).
        threshold: classifier probability cutoff for the "injection" class.
                   Defaults to THRESHOLD from config.py.
        flag_action: default action when flagged - "block" or "rephrase".
                     Can be overridden per call to detect().
        """
        model_path = model_path if model_path is not None else DEBERTA_MODEL_PATH

        self.regex_patterns = [
            # Instruction-override attempts - verb synonyms included so
            # tightening for precision doesn't quietly lose recall
            r"(ignore|disregard|forget)\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?|guidelines?)",
            r"reveal\s+(your\s+)?(the\s+)?(system\s+)?(prompt|instructions?|hidden)",
            r"system_prompt=.*",
            # Named jailbreak techniques
            r"\bdo\s+anything\s+now\b",
            r"developer\s+mode",
            r"stay\s+in\s+character",
            r"without\s+any\s+(restrictions?|filters?|limitations?)",
            r"no\s+(restrictions?|filters?|content\s+policy)",
            r"jailbreak",
            r"unrestricted\s+(ai|model|assistant)",
            # Fake role/tag injection - attempts to spoof message structure,
            # directly complementary to Layer 2's role-separation boundaries
            r"<\s*/?\s*system\s*/?>",
            r"<\s*/?\s*(assistant|developer|tool|function)\s*/?>",
            r"\[\s*(System\s*Message|System|Assistant|Internal)\s*\]",
            r"^\s*System:\s+",
            r"<\|(?:im_start|im_end|eot_id|start_header_id|end_header_id|endoftext)\|>",
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.window_size = window_size
        self.threshold = threshold if threshold is not None else THRESHOLD
        self.flag_action = flag_action

    def regex_detect(self, text):
        text = normalize_text(text)
        if any(re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in self.regex_patterns):
            return True
        # "DAN" is checked case-sensitively (not IGNORECASE), since real jailbreak
        # references are almost always written in full caps ("You are DAN"),
        # while a normal name like "Dan" is not -- this avoids flagging real names.
        if re.search(r"\bDAN\b", text):
            return True
        return False

    def classify(self, text):
        """Run the DeBERTa-small classifier on a single message."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs[0][1].item()

    def build_window(self, current_prompt, history=None):
        history = history or []
        recent_history = history[-self.window_size:]
        return recent_history + [current_prompt]

    def detect(self, current_prompt, history=None, flag_action=None):
        """
        Evaluates the current prompt in the context of recent history.
        Returns a dict with flagged / action / trigger / model_score / window /
        joined_window / background_suspicion / escalation_pattern.
        Stateless - never stores history between calls.
        """
        window = self.build_window(current_prompt, history)
        joined_window = "\n".join(window)

        # Fragmentation experiment (thesis Ch.4, Evaluation D2): scoring each
        # message independently and taking the maximum let 6.5% of split
        # attacks slip through (187/200 caught). Scoring the joined window as
        # a single text caught all 200/200. This is now the validated design.
        regex_flag = self.regex_detect(joined_window)
        model_score = self.classify(joined_window)
        model_flag = model_score > self.threshold

        # Diagnostic only - does not affect the flagged decision above, so
        # the validated joined-window result stays exactly as tested.
        history_scores = [self.classify(p) for p in window[:-1]]
        background_suspicion = max(history_scores) if history_scores else 0.0

        flagged = regex_flag or model_flag

        if regex_flag and model_flag:
            trigger = "regex+model"
        elif regex_flag:
            trigger = "regex"
        elif model_flag:
            trigger = "model"
        else:
            trigger = None

        escalation_pattern = None
        if flagged and trigger in ("model", "regex+model"):
            escalation_pattern = "gradual_buildup" if background_suspicion > 0.3 else "single_message_trigger"

        action = "pass"
        if flagged:
            action = flag_action if flag_action is not None else self.flag_action

        return {
            "flagged": flagged,
            "action": action,
            "trigger": trigger,
            "model_score": model_score,
            "window": window,
            "joined_window": joined_window,
            "background_suspicion": background_suspicion,
            "escalation_pattern": escalation_pattern,
        }
