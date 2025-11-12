from .google_api_utils import google_config, ApiHandler, ResultProcessor
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

class ImagenNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 1536, "step": 64}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "GoogleAPI/Image"

    def generate_image(self, prompt, negative_prompt, width, height, num_images, seed):
        
        client, endpoint = google_config.get_vertex_client_and_endpoint()
        if not client:
            return ApiHandler.handle_image_generation_error("Imagen", "Vertex AI Client not configured. Check config.ini and key file.")

        # --- Build the API Payload ---
        # Note: Vertex AI is very particular about its payload structure.
        instance_dict = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
        parameters_dict = {
            "sampleCount": num_images,
            "aspectRatio": f"{width}:{height}", # Note: This is an example. Imagen 3.2 accepts width/height.
            "seed": seed,
            # Model specific parameters. For Imagen 3.2:
            "width": width,
            "height": height,
        }
        
        # Convert dicts to Google's specific Protobuf format
        instance = json_format.ParseDict(instance_dict, Value())
        parameters = json_format.ParseDict(parameters_dict, Value())
        
        instances = [instance]
        
        try:
            # --- Make the API Call ---
            response = client.predict(
                endpoint=endpoint,
                instances=instances,
                parameters=parameters,
            )
            
            # --- Process the Response ---
            return ResultProcessor.process_imagen_result(response)

        except Exception as e:
            return ApiHandler.handle_image_generation_error("Imagen", str(e))


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "Imagen_google": ImagenNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Imagen_google": "Imagen Text-to-Image (Google)",
}