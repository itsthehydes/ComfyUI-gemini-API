import configparser
import os
import google.generativeai as genai
import numpy as np
import torch
from PIL import Image

class GoogleApiConfig:
    """Singleton class to handle Google API configuration and client setup."""

    _instance = None
    _key = None
    _client_configured = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GoogleApiConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration and API key."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        try:
            if os.environ.get("GOOGLE_API_KEY") is not None:
                print("GOOGLE_API_KEY found in environment variables")
                self._key = os.environ["GOOGLE_API_KEY"]
            else:
                print("GOOGLE_API_KEY not found in environment variables")
                self._key = config["API"]["GOOGLE_API_KEY"]
                print("GOOGLE_API_KEY found in config.ini")
                os.environ["GOOGLE_API_KEY"] = self._key
                print("GOOGLE_API_KEY set in environment variables")

            if self._key == "<your_google_api_key_here>":
                print("WARNING: You are using the default Google API key placeholder!")
                print("Please set your actual Google API key in the config.ini file.")
                self._key = None
        
        except KeyError:
            print("Error: GOOGLE_API_KEY not found in config.ini or environment variables")
            self._key = None

    def get_key(self):
        """Get the Google API key."""
        return self._key

    def configure_client(self):
        """Configures the Google AI client if not already configured."""
        if not self._client_configured and self._key:
            try:
                genai.configure(api_key=self._key)
                self._client_configured = True
                print("Google Generative AI client configured successfully.")
            except Exception as e:
                print(f"Error configuring Google Generative AI client: {e}")
                self._client_configured = False
        return self._client_configured

class ImageUtils:
    """Utility functions for image processing."""

    @staticmethod
    def tensor_to_pil(image):
        """Convert image tensor to PIL Image."""
        try:
            # Convert the image tensor to a numpy array
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            # Ensure the image is in the correct format (H, W, C)
            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)  # Remove batch dimension if present
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
            elif image_np.shape[0] == 3:
                image_np = np.transpose(
                    image_np, (1, 2, 0)
                )  # Change from (C, H, W) to (H, W, C)

            # Normalize the image data to 0-255 range
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8)

            # Convert to PIL Image
            return Image.fromarray(image_np)
        except Exception as e:
            print(f"Error converting tensor to PIL: {str(e)}")
            return None

class ApiHandler:
    """Utility functions for API interactions."""
    
    @staticmethod
    def handle_text_generation_error(model_name, error):
        """Handle text generation errors consistently."""
        print(f"Error generating text with {model_name}: {str(error)}")
        return ("Error: Unable to generate text.",)

    @staticmethod
    def handle_image_generation_error(model_name, error):
        """Handle image generation errors consistently."""
        print(f"Error generating image with {model_name}: {str(error)}")
        # Create a blank black image tensor
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)
