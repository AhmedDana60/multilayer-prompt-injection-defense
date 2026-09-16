import os
from openai import OpenAI
from dotenv import load_dotenv

from layer1_input_filtering import InputFilter
from layer2_structured_format import format_prompt
from layer3_output_validation import validate_output
from layer4_response_refine import refine_output

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class MultiLayeredDefense:
    """
    Full four-layer defense pipeline against prompt injection.

      Layer 1 - Multi-Turn Input Screening
      Layer 2 - Randomized-Tag Prompt Isolation
      Layer 3 - Multi-Criterion Critic Validation
      Layer 4 - Iterative Response Correction
    """

    def __init__(self, main_llm_model="gpt-4o-mini", window_size=3, flag_action="block"):
        self.input_filter = InputFilter(window_size=window_size, flag_action=flag_action)
        self.main_llm_model = main_llm_model
        self.fallback_message = (
            "I'm sorry, I can't provide a response to that request right now. "
            "Please try rephrasing your question."
        )

    def _call_main_llm(self, messages):
        response = client.chat.completions.create(model=self.main_llm_model, messages=messages)
        return response.choices[0].message.content

    def process(self, system_prompt, user_input, history=None, external_context=None):
        result = {
            "final_response": None,
            "layer1_result": None,
            "blocked_at_layer1": False,
            "layer3_verdict": None,
            "layer4_used": False,
            "layer3_recheck_verdict": None,
            "fallback_used": False,
        }

        layer1_result = self.input_filter.detect(user_input, history=history)
        result["layer1_result"] = layer1_result

        if layer1_result["flagged"]:
            result["blocked_at_layer1"] = True
            result["final_response"] = self.fallback_message
            return result

        messages = format_prompt(system_prompt, user_input, external_context=external_context)
        raw_output = self._call_main_llm(messages)

        verdict_result = validate_output(system_prompt, user_input, raw_output)
        result["layer3_verdict"] = verdict_result["verdict"]

        if verdict_result["verdict"] == "SAFE":
            result["final_response"] = raw_output
            return result

        result["layer4_used"] = True
        refined_output = refine_output(
            system_prompt, user_input, raw_output, verdict_result["raw_response"]
        )

        recheck_result = validate_output(system_prompt, user_input, refined_output)
        result["layer3_recheck_verdict"] = recheck_result["verdict"]

        if recheck_result["verdict"] == "SAFE":
            result["final_response"] = refined_output
        else:
            result["fallback_used"] = True
            result["final_response"] = self.fallback_message

        return result
