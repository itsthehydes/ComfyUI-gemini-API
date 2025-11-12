import google.generativeai as genai
from .google_api_utils import GoogleApiConfig, ApiHandler

# Initialize GoogleApiConfig
google_config = GoogleApiConfig()

class LLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "gemini-1.5-flash", # Changed to Google models
                        "gemini-1.5-pro",
                    ],
                    {"default": "gemini-1.5-flash"},
                ),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "GoogleAPI/LLM" # Renamed category

    def generate_text(self, prompt, model, system_prompt):
        # 1. Configure the client
        if not google_config.configure_client():
            return ApiHandler.handle_text_generation_error(model, "Google API Key is not set or invalid.")

        try:
            # 2. Set up the generative model
            model_client = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt if system_prompt else None
            )

            # 3. Call the API
            response = model_client.generate_content(prompt)
            
            # 4. Return the text
            return (response.text,)

        except Exception as e:
            return ApiHandler.handle_text_generation_error(model, str(e))


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LLM_google": LLMNode, # Renamed node
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_google": "LLM (Google)", # Renamed node
}
