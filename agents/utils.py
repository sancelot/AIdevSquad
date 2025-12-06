import json
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.llms import ChatMessage

from llama_index.llms.ollama import Ollama
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.embeddings.ollama import OllamaEmbedding
import time
import os
import re
try:
    from google.genai.errors import ClientError
except ImportError:
    ClientError = None  # In case you run with ollama and don't have the lib installed
# Import the specific exception for Google's API
from google.api_core import exceptions as google_exceptions


def parse_json_from_response(response_str):
    """
    Finds and parses a JSON object from a string, handling nested objects and
    markdown code fences. It scans for a balanced number of braces {}.
    """
    # First, try to find a markdown block ```json ... ```
    match = re.search(r'```json\s*(.*?)```', response_str, re.DOTALL)
    if match:
        search_str = match.group(1)
    else:
        search_str = response_str

    # Find the first opening brace or bracket
    idx_obj = search_str.find('{')
    idx_arr = search_str.find('[')

    if idx_obj == -1 and idx_arr == -1:
        print("❌ Error: No JSON object or array found in the LLM response.")
        return None

    # Determine which comes first
    if idx_arr != -1 and (idx_obj == -1 or idx_arr < idx_obj):
        json_start = idx_arr
        opener = '['
        closer = ']'
    else:
        json_start = idx_obj
        opener = '{'
        closer = '}'

    # Start scanning from the first brace/bracket to find the matching closer
    balance_count = 1
    json_end = -1
    for i in range(json_start + 1, len(search_str)):
        char = search_str[i]
        if char == opener:
            balance_count += 1
        elif char == closer:
            balance_count -= 1

        if balance_count == 0:
            json_end = i + 1
            break

    if json_end == -1:
        print(
            f"❌ Error: Malformed JSON in the LLM response (unbalanced {opener}{closer}).")
        return None

    # Extract the complete, balanced JSON string
    json_str = search_str[json_start:json_end]

    try:
        # Parse directly - no need to "repair" valid JSON!
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        print(
            f"--- Extracted JSON String ---\n{json_str}\n-----------------------------")
        return None


def configure_llm_and_embed(provider="ollama", model=None):
    """
    Configures and returns LLM and embedding models based on the provider.
    Also sets the global embedding model for LlamaIndex.
    """
    print(f"Configuring models for provider: {provider.upper()}")

    if provider == "google":
        # Assurez-vous que la clé API est chargée (déjà fait par load_dotenv dans orchestrateur.py)
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "Environnement variable GOOGLE_API_KEY is required to use Google Genai.")
        llm_class = GoogleGenAI
        embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")
        llm_args = {"model_name": "models/gemini-1.5-pro-latest"}

    elif provider == "ollama":
        llm_class = Ollama
        # embed_model = OllamaEmbedding(model_name="bge-base-en-v1.5")
        embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")
        llm_args = {"model": "qwen3:8b", "request_timeout": 300.0}

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    # Appliquer la configuration globalement via LlamaIndex Settings
    Settings.embed_model = embed_model

    token_counter = TokenCountingHandler()
    Settings.callback_manager = CallbackManager([token_counter])
    return llm_class, llm_args, token_counter


# Define which exceptions should trigger a retry
retryable_exceptions = (
    google_exceptions.ResourceExhausted,  # This is the 429 error
    google_exceptions.ServiceUnavailable,  # This is the 503 error
)

# @retry(
#     wait=wait_exponential(multiplier=1, min=5, max=60), # Wait 5s, then 10s, 20s, up to 60s
#     stop=stop_after_attempt(5), # Stop after 5 attempts
#     retry=retry_if_exception_type(retryable_exceptions),
#     before_sleep=lambda retry_state: print(f"Rate limit hit. Retrying in {retry_state.next_action.sleep:.2f} seconds...")
# )


def call_llm(llm_instance, system_message, user_prompt, logger=None, agent_name="Unknown Agent"):
    """
    Generic function to call the LLM with a robust, self-healing retry mechanism
    for rate limiting errors.
    """
    if logger:
        logger.log_event_in_step(
            "llm_call", {"agent_name": agent_name, "prompt": user_prompt})
    max_retries = 5
    for attempt in range(max_retries):
        if attempt > 0:
            print("Now retrying...")
        try:
            messages = [
                ChatMessage(role="system", content=system_message),
                ChatMessage(role="user", content=user_prompt),
            ]
            response = llm_instance.chat(messages)
            response_content = response.message.content
            if logger:
                logger.log_event_in_step(
                    "llm_response", {"response": response_content})
            return response_content

        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                # --- NEW, ROBUST PARSING LOGIC ---
                retry_delay = 0

                # The ClientError object has a `response_json` attribute that contains the structured error
                error_details = getattr(e, 'details', {}).get(
                    'error', {}).get('details', [])
                for detail in error_details:
                    if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                        delay_str = detail.get('retryDelay', '0s')
                        print("1", delay_str)
                        # Extract the integer part of the delay string (e.g., "45s" -> 45)
                        match = re.search(r'(\d+)', delay_str)
                        if match:
                            retry_delay = int(match.group(1))
                        break  # Found the retry info, no need to look further
                print("retry delai ", retry_delay)
                # Check if we are hitting a hard daily quota, which is not retryable in the short term
                is_daily_quota_error = any(
                    'FreeTier' in violation.get('quotaId', '')
                    for detail in error_details
                    if detail.get('@type') == 'type.googleapis.com/google.rpc.QuotaFailure'
                    for violation in detail.get('violations', [])
                )

                if is_daily_quota_error:
                    print("\n" + "="*60)
                    print(
                        "🚫 Halted: Google API daily free tier quota has been exceeded.")
                    print(
                        " The application cannot continue. Please upgrade to a paid plan or wait 24 hours.")
                    print("="*60)
                    raise e  # Re-raise the exception to stop the program immediately

                # Fallback if no specific retry_delay was found
                if retry_delay <= 0:
                    # Exponential backoff: 5s, 10s, 20s...
                    retry_delay = 5 * (2 ** attempt)
                    print(
                        f"⚠️ Rate limit hit. Could not parse a specific delay. Retrying with generic backoff in {retry_delay} seconds...")
                else:
                    print(
                        f"API requested a delay. Waiting for {retry_delay} seconds before retrying...")

                time.sleep(retry_delay + 1)
                continue  # Continue to the next attempt in the loop

            else:
                print(f"❌ A non-retryable API client error occurred: {e}")
                raise e

        except Exception as e:
            print(f"❌ An unexpected error occurred during the LLM call: {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e

    raise Exception(f"Failed to call LLM after {max_retries} attempts.")
