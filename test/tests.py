"""
LLM Testing Suite for vLLM (Streaming & OpenAI Client Version)
Tests reasoning and function calling capabilities deterministically.
"""

import json
import re
import pytest
from pytest_html import extras as extras_html
from typing import Dict, Any, List
from openai import OpenAI
from text_to_num import text2num

# ===========================
# CONFIGURATION
# ===========================
# Point to your vLLM instance

MODELS = {
    "Llama 3.2 1B base": {
        "url": "https://akeelaf-2022--llama-3-2-1b-base-serve.modal.run/v1/",
        "name": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    },
    "Llama 3.2 3B base": {
        "url": "https://akeelaf-2022--llama-3-2-3b-base-serve.modal.run/v1/",
        "name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    },
    "Llama 3.2 3B lora": {
        "url": "https://akeelaf-2022--llama-3-2-3b-lora-serve.modal.run/v1/",
        "name": "hellstone1918/Llama-3.2-3B-basic-lora-model",
    },
}


CLIENT = OpenAI(
    base_url=MODELS['Llama 3.2 3B base']['url'],
    api_key="token-not-needed-for-local-vllm" 
)
MODEL_NAME = MODELS['Llama 3.2 3B base']['name'] # The client often auto-detects, but good to specify
IS_STREAMING = False

class TestDataLoader:
    """Loads and manages test data"""
    
    def __init__(self, dataset_path: str = "test_dataset.json"):
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
    
    def get_reasoning_tests(self) -> List[Dict]:
        return self.data.get("reasoning", [])
    
    def get_function_calling_tests(self) -> List[Dict]:
        return self.data.get("function_calling", [])

class ModelInterface:
    """Handles communication with the LLM via OpenAI Client"""
    
    @staticmethod
    def query(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """
        Query the model and handle both streaming and non-streaming responses.
        Returns the fully accumulated string content.
        """
        try:
            response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                stream=IS_STREAMING
            )
            
            if IS_STREAMING:
                # Accumulate the stream chunks
                full_content = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
                return full_content
            else:
                # Handle standard response
                return response.choices[0].message.content

        except Exception as e:
            pytest.fail(f"API Request Error: {str(e)}")

class TextMatcher:
    """Utility for matching text deterministically"""
    
    @staticmethod
    def normalize(text: str, extract_pattern: str = None) -> str:
        """
        Normalize text for comparison (lower, strip punctuation/whitespace).
        Optionally extract answer from a pattern first.
        
        Args:
            text: Input text to normalize
            extract_pattern: Optional regex pattern to extract answer from.
                            Use a capturing group to specify what to extract.
        
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # Extract answer from pattern if provided
        if extract_pattern:
            # Find all matches (in case pattern appears multiple times)
            matches = re.findall(extract_pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                # Use the last match (final answer)
                text = matches[-1] if isinstance(matches[-1], str) else matches[-1][0]
        
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.,!?;:\"\'\`]', '', text)
        return text.strip()
    
    @staticmethod
    def contains(
        haystack: str, 
        needle: str | list[str],
        extract_pattern: str = None,
    ) -> bool:
        """
        Check if haystack contains needle(s) with optional pattern extraction.
        
        Args:
            haystack: The text to search in
            needle: Single string or list of acceptable strings
            extract_pattern: Optional regex to extract final answer first
            normalize: Whether to normalize text before comparison
        
        Returns:
            True if any needle is found, False otherwise
        """
        # Extract from pattern if provided
        if extract_pattern:
            haystack = TextMatcher.normalize(haystack, extract_pattern)
        
        # Normalize to list
        needles = [needle] if isinstance(needle, str) else needle
        
        haystack_norm = TextMatcher.normalize(haystack)
        return any(n.lower() in haystack_norm for n in needles)
    
    @staticmethod
    def normalize_number(value):
        """Convert number words or digit strings to float."""
        if isinstance(value, (int, float)):
            return float(value)
        
        # Try parsing as numeric first
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
        
        # Try parsing as number words
        try:
            # Remove extra spaces and normalize
            normalized = re.sub(r'\s+', ' ', str(value).strip())
            return float(text2num(normalized, "en"))
        except:
            raise ValueError(f"Cannot parse '{value}' as number")

    # Use in test assertions
    def assert_number_equals(expected, actual, tolerance=0.01):
        """Compare numbers allowing word or digit format."""
        expected_num = TextMatcher.normalize_number(expected)
        actual_num = TextMatcher.normalize_number(actual)
        assert abs(expected_num - actual_num) < tolerance
    
    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON from text, handling markdown blocks"""
        # 1. Try finding markdown JSON block
        match = re.search(r'``````', text, re.DOTALL)
        json_str = match.group(1) if match else None
        
        # 2. If no block, try finding the first outer bracket pair
        if not json_str:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            json_str = match.group(0) if match else None
            
        if not json_str:
            raise ValueError("No JSON structure found in response")
            
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON syntax: {e}")
        
    def values_match(expected, actual) -> bool:
        """Flexible value comparison."""
        if expected is None or actual is None:
            return expected == actual
        if type(expected) != type(actual):
            try:
                return str(expected).lower() == str(actual).lower()
            except:
                return False
        if isinstance(expected, str):
            return expected.lower().strip() == actual.lower().strip()
        elif isinstance(expected, list):
            return sorted(str(x) for x in expected) == sorted(str(x) for x in actual)
        else:
            return expected == actual

# Initialize Loader
loader = TestDataLoader()

# ===========================
# TEST FIXTURES
# ===========================

@pytest.fixture(scope="session", autouse=True)
def verify_connection():
    """Check API connection before starting tests"""
    try:
        CLIENT.models.list()
    except Exception as e:
        pytest.exit(f"CRITICAL: Cannot connect to vLLM at {CLIENT.base_url}. Error: {e}")

# ===========================
# REASONING TESTS
# ===========================

@pytest.mark.reasoning
@pytest.mark.parametrize("test_case", loader.get_reasoning_tests(), 
                         ids=lambda x: f"{x['category'][:3]}_{x['expected_answer']}")
def test_reasoning(test_case: Dict[str, Any], extras):
    prompt = test_case["prompt"]
    expected = test_case["expected_answer"]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant attempting to answer various queries that the user may have. These queries may require you to think in multiple steps to arrive at the correct answer. Once you reach a final answer, you are to respond at the very last line on a new line with 'Final Answer: <your answer here>'."},
        {"role": "user", "content": prompt},
    ]
    
    # Get full response (accumulated from stream)
    response = ModelInterface.query(messages)

    # --- SAVE FULL RESPONSE FOR REVIEW ---
    # This adds a 'Raw Response' block to the HTML report for this test case
    extras.append(extras_html.text(response, name="Raw LLM Response"))
    extras.append(extras_html.text(json.dumps(messages, indent=2), name="Input Messages"))
    # -------------------------------------
    
    assert TextMatcher.contains(
        response,
        expected,
        extract_pattern=r'Final Answer:\s*(.+?)(?:\n|$)',
    ), \
        f"\nExpected: '{expected}'\nGot: '{response}'\nPrompt: {prompt}"

# ===========================
# FUNCTION CALLING TESTS
# ===========================

@pytest.mark.function_calling
@pytest.mark.parametrize("test_case", loader.get_function_calling_tests(),
                         ids=lambda x: f"{x['category'][:3]}_{x['expected_function']}")
def test_function_calling(test_case: Dict[str, Any], extras):
    system_prompt = test_case["system_prompt"]
    user_prompt = test_case["prompt"]
    expected_func = test_case["expected_function"]
    required_keys = test_case["required_keys"]
    expected_values = test_case.get("expected_values", {})
    
    messages = [
        {"role": "system", "content": system_prompt + "\n**You MUST ONLY respond in JSON format specifying the function to call that you have access to and its respective parameters and values. You are NEVER to write your OWN program/script to meet the users need.**"},
        {"role": "user", "content": user_prompt}
    ]
    
    # Get full response
    response = ModelInterface.query(messages)

    # --- SAVE FULL RESPONSE FOR REVIEW ---
    extras.append(extras_html.text(response, name="Raw LLM Response"))
    extras.append(extras_html.text(json.dumps(messages, indent=2), name="Input Messages"))
    # -------------------------------------
    
    # Parse JSON
    try:
        result = TextMatcher.extract_json(response)
    except ValueError as e:
        pytest.fail(f"JSON Parsing Failed: {e}\nResponse Content: {response}")
    
    # HARD CHECKS (must pass)
    missing_keys = [k for k in required_keys if k not in result]
    assert not missing_keys, \
        f"\nMissing Keys: {missing_keys}\nExpected: {required_keys}\nGot: {list(result.keys())}"
    
    actual_func = result.get("function")
    assert actual_func == expected_func, \
        f"\nWrong Function Called.\nExpected: {expected_func}\nGot: {actual_func}"
    
    # SOFT CHECKS (partial credit)
    value_errors = []
    for key, expected_val in expected_values.items():
        if key == "function":
            continue
        actual_val = result.get(key)
        if not TextMatcher.values_match(expected_val, actual_val):
            value_errors.append(f"{key}: expected={expected_val}, got={actual_val}")
    
    # Report results
    if value_errors:
        error_text = "\n  ".join(value_errors)
        extras.append(extras_html.text(
            f"⚠ PARTIAL PASS ({len(value_errors)} value errors):\n  {error_text}",
            name="Test Result"
        ))
        # Pass test but log warning
        pytest.skip(f"PARTIAL: {len(value_errors)} value mismatch(es) - see extras")
    else:
        extras.append(extras_html.text("✓ FULL PASS", name="Test Result"))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--html=report.html", "--self-contained-html"])

# Run 
# uv run pytest tests.py -n 16 -v --html=reports/llama3.2-3b-base/report.html
# add -n <number of parallel processes> if you'd like
