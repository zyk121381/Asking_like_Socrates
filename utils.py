import base64
import os
import time
from io import BytesIO
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from volcenginesdkarkruntime import Ark

# Load environment variables
load_dotenv()

# Configuration
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Verifier model configuration (configurable via .env)
VERIFIER_MODEL_API_KEY = os.getenv("VERIFIER_MODEL_API_KEY")
VERIFIER_MODEL_BASE_URL = os.getenv("VERIFIER_MODEL_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

# Determine whether to use OpenAI or Ark client for verifier
USE_ARK_FOR_VERIFIER = os.getenv("USE_ARK_FOR_VERIFIER", "true").lower() == "true"

# Model names (configurable via .env)
REASONER_MODEL = os.getenv("REASONER_MODEL", "gpt-5-mini")
PERCEIVER_MODEL = os.getenv("PERCEIVER_MODEL", "gemini-2.5-flash")
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "doubao-seed-1-6-thinking-250715")

MAX_RETRIES = 3


def encode_pil_image(image: Image.Image, format: str) -> str:
    """Encodes a PIL image to a base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def chat_completions(
    model: str,
    query: str,
    images: Optional[List[Tuple[Any, str]]] = None,
    system_prompt: Optional[str] = None
) -> str:
    """
    Calls OpenAI Chat Completion API. Supports text-only and multimodal inputs.

    Args:
        model: Model identifier (e.g., 'gpt-4o').
        query: User query text.
        images: Optional list of (PIL.Image, suffix) tuples.
        system_prompt: Optional system instruction.

    Returns:
        The model's response content.
    """
    client = OpenAI(
        api_key=OPENAI_KEY,
        base_url=OPENAI_BASE_URL,
    )

    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    user_content = []

    # Process images if provided
    if images:
        for image, suffix in images:
            base64_data = encode_pil_image(image, format=suffix.upper())
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{suffix.lower()};base64,{base64_data}"
                }
            })

    user_content.append({"type": "text", "text": query})

    messages.append({
        "role": "user",
        "content": user_content
    })

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )

    return completion.choices[0].message.content


def chat_verifier(
    model: str,
    query: str,
    system_prompt: Optional[str] = None
) -> str:
    """
    Calls Verifier API for text-only interactions.
    Uses configurable VERIFIER_MODEL_BASE_URL and VERIFIER_MODEL_API_KEY.
    Automatically chooses between OpenAI and Ark client based on USE_ARK_FOR_VERIFIER.
    """
    if USE_ARK_FOR_VERIFIER:
        # Use Ark client for Volcengine
        client = Ark(
            base_url=VERIFIER_MODEL_BASE_URL,
            api_key=VERIFIER_MODEL_API_KEY,
        )
    else:
        # Use OpenAI client for OpenAI-compatible APIs
        client = OpenAI(
            api_key=VERIFIER_MODEL_API_KEY,
            base_url=VERIFIER_MODEL_BASE_URL,
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": query})

    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return completion.choices[0].message.content


class APIModel:
    """
    Wrapper class to handle multi-backend LLM calls with retry logic.
    """

    def __init__(self, model_name: str, system_prompt: Optional[str] = None):
        self.model_name = model_name
        self.system_prompt = system_prompt

    def _single_call(
        self,
        query: str,
        images: Optional[List[Tuple[Image.Image, str]]] = None
    ) -> str:
        """Dispatches the call to the appropriate provider based on model name."""
        # Use OpenAI-compatible API for reasoner and perceiver models
        if self.model_name == REASONER_MODEL or self.model_name == PERCEIVER_MODEL:
            return chat_completions(
                model=self.model_name,
                query=query,
                images=images,
                system_prompt=self.system_prompt,
            )
        # Use verifier API for the verifier model
        elif self.model_name == VERIFIER_MODEL:
            # Note: Verifier API ignores images
            return chat_verifier(self.model_name, query, self.system_prompt)
        else:
            raise NotImplementedError(f"Unknown model: {self.model_name}")

    def get_response(
        self, 
        query: str, 
        images: Optional[List[Tuple[Image.Image, str]]] = None
    ) -> str:
        """
        Executes the API call with automatic retries.

        Args:
            query: Input text.
            images: Optional list of (PIL.Image, suffix) tuples.

        Returns:
            The model response string.
        """
        last_error = None
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._single_call(query, images)
            except Exception as e:
                last_error = e
                print(f"[API RETRY] {self.model_name} attempt {attempt}/{MAX_RETRIES} failed: {e}")
                
                if attempt < MAX_RETRIES:
                    time.sleep(0.5)
                else:
                    raise last_error



# =========================
# Test / Example Code
# =========================
# Run this file directly to test API connectivity and model responses:
#   python utils.py
# This code only executes when utils.py is run directly (not when imported).
if __name__ == "__main__":
    model_name = "gemini-2.5-flash"
    # model_name = "gpt-5-mini"
    # model_name = "doubao-seed-1-6-thinking-250715"

    model = APIModel(model_name, "用小朋友的语气回答我")
    response = model.get_response("维斯塔潘获得过几届世界冠军？")
    print(response)

    print("-" * 20)

    # Ensure this path exists or mock it for testing
    image_path = "assets/Fig1.jpg"
    if os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        model = APIModel(model_name, "用小朋友的语气回答我")
        response = model.get_response("这张图描述了什么？", images=[(img, 'jpeg')])
        print(response)
    else:
        print(f"Test image not found at {image_path}, skipping image test.")