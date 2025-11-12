import google.generativeai as genai
from .google_api_utils import GoogleApiConfig, ImageUtils, ApiHandler

# Initialize GoogleApiConfig
google_config = GoogleApiConfig()

class VLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "gemini-1.5-flash", # Changed to Google models
                        "gemini-1.5-pro",
                        "gemini-pro-vision", # Older, but good example
                    ],
                    {"default": "gemini-1.5-flash"},
                ),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "GoogleAPI/VLM" # Renamed category

    def generate_text(self, prompt, model, system_prompt, image):
        # 1. Configure the client (it will only run once)
        if not google_config.configure_client():
            return ApiHandler.handle_text_generation_error(model, "Google API Key is not set or invalid.")

        try:
            # 2. Convert the image tensor to a PIL Image
            pil_image = ImageUtils.tensor_to_pil(image)
            if not pil_image:
                return ApiHandler.handle_text_generation_error(
                    model, "Failed to convert image tensor to PIL"
                )

            # 3. Set up the generative model
            model_client = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt if system_prompt else None
            )

            # 4. Prepare the content for the API
            #    Send the prompt and the image together
            contents = [prompt, pil_image]

            # 5. Call the API
            response = model_client.generate_content(contents)
            
            # 6. Return the text part of the response
            return (response.text,)
        
        except Exception as e:
            return ApiHandler.handle_text_generation_error(model, str(e))


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VLM_google": VLMNode, # Renamed node
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLM_google": "VLM (Google)", # Renamed node
}
