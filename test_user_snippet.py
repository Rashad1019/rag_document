from google import genai
import os

# Set API key from environment or hardcoded fallback
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

try:
    client = genai.Client(api_key=api_key)

    # Use the Gemini 3 Flash model
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Analyze this logic puzzle step-by-step."
    )

    print(f"Success! Response:\n{response.text}")

except Exception as e:
    print(f"Error: {e}")
