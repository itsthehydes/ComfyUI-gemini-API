import configparser
import os
import google.generativeai as genai
from google.oauth2 import service_account
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import numpy as np
import torch
from PIL import Image
import base64
import io

class GoogleApiConfig:
    """Singleton class to handle all Google API configuration and clients."""

    _instance = None
    
    # Gemini Config
    _gemini_key = None
    _gemini_client_configured = False
    
    # Vertex AI Config
    _vertex_project_id = None
    _vertex_location = None
    _vertex_service_account_file = None
    _vertex_creds = None
    _vertex_endpoint_name = None

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

        # --- Gemini (API Key) Setup ---
        try:
            if os.environ.get("GOOGLE_API_KEY") is not None:
                self._gemini_key = os.environ["GOOGLE_API_KEY"]
            else:
                self._gemini_key = config["API"]["GOOGLE_API_KEY"]
            
            if self._gemini_key == "<your_google_api_key_here>":
                print("WARNING: Gemini API Key is not set in config.ini. LLM/VLM nodes will not work.")
                self._gemini_key = None
        except KeyError:
            print("WARNING: GOOGLE_API_KEY not found in config.ini. LLM/VLM nodes will not work.")
            self._gemini_key = None

        # --- Vertex AI (Service Account) Setup ---
        try:
            self._vertex_project_id = config["VERTEX_AI"]["PROJECT_ID"]
            self._vertex_location = config["VERTEX_AI"]["LOCATION"]
            sa_file = config["VERTEX_AI"]["SERVICE_ACCOUNT_FILE"]
            self._vertex_service_account_file = os.path.join(parent_dir, sa_file)

            if self._vertex_project_id == "<your-project-id-here>":
                print("WARNING: Vertex AI PROJECT_ID is not set in config.ini. Imagen/Veo nodes will not work.")
                self._vertex_project_id = None
            elif not os.path.exists(self._vertex_service_account_file):
                print(f"WARNING: Vertex AI Service Account file not found at: {self._vertex_service_account_file}")
                self._vertex_service_account_file = None
            else:
                # Load credentials
                self._vertex_creds = service_account.Credentials.from_service_account_file(self._vertex_service_account_file)
                self._vertex_endpoint_name = f"projects/{self._vertex_project_id}/locations/{self._vertex_location}/publishers/google/models/imagen-3.2-sdxl"
        except KeyError:
            print("WARNING: [VERTEX_AI] section missing or incomplete in config.ini. Imagen/Veo nodes will not work.")
        except Exception as e:
            print(f"Error loading Vertex AI credentials: {e}")
            self._vertex_creds = None

    def configure_gemini_client(self):
        """Configures the Google AI client if not already configured."""
        if not self._gemini_client_configured and self._gemini_key:
            try:
                genai.configure(api_key=self._gemini_key)
                self._gemini_client_configured = True
                print("Google Gemini (AI Studio) client configured successfully.")
            except Exception as e:
                print(f"Error configuring Google Gemini client: {e}")
                self._gemini_client_configured = False
        return self._gemini_client_configured
    
    def get_vertex_client_and_endpoint(self):
        """Gets the Vertex AI client and endpoint string."""
        if not self._vertex_creds or not self._vertex_endpoint_name:
            print("Vertex AI client is not configured. Check config.ini.")
            return None, None
            
        client_options = {"api_endpoint": f"{self._vertex_location}-aiplatform.googleapis.com"}
        client = aiplatform.gapic.PredictionServiceClient(
            credentials=self._vertex_creds,
            client_options=client_options
        )
        return client, self._vertex_endpoint_name


class ImageUtils:
    """Utility functions for image processing."""
    @staticmethod
    def tensor_to_pil(image):
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)

        if image_np.ndim == 4:
            image_np = image_np.squeeze(0)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))

        if image_np.dtype == np.float32 or image_np.dtype == np.float64:
            image_np = (image_np * 255).astype(np.uint8)

        return Image.fromarray(image_np)

    @staticmethod
    def pil_to_tensor(image):
        """Convert PIL Image to image tensor."""
        try:
            img_array = np.array(image).astype(np.float32) / 255.0
            # Add batch dimension
            img_tensor = torch.from_numpy(img_array)[None,]
            return img_tensor
        except Exception as e:
            print(f"Error converting PIL to tensor: {str(e)}")
            return None

class ResultProcessor:
    """Utility functions for processing API results."""

    @staticmethod
    def process_imagen_result(response):
        """Process Imagen (Vertex AI) result and return tensor."""
        try:
            images = []
            for prediction in response.predictions:
                # Imagen returns image data as a base64 encoded string
                b64_string = prediction.get("bytesBase64Encoded")
                if b64_string:
                    img_bytes = base64.b64decode(b64_string)
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    # Convert PIL image to tensor
                    img_array = np.array(img).astype(np.float32) / 255.0
                    images.append(img_array)

            if not images:
                raise Exception("No images found in API response.")

            stacked_images = np.stack(images, axis=0)
            img_tensor = torch.from_numpy(stacked_images)
            return (img_tensor,)
        except Exception as e:
            print(f"Error processing Imagen result: {str(e)}")
            return ApiHandler.handle_image_generation_error("Imagen", e)


class ApiHandler:
    """Utility functions for API interactions."""
    @staticmethod
    def handle_text_generation_error(model_name, error):
        print(f"Error generating text with {model_name}: {str(error)}")
        return ("Error: Unable to generate text.",)
    
    @staticmethod
    def handle_image_generation_error(model_name, error):
        print(f"Error generating image with {model_name}: {str(error)}")
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)

# Initialize the config singleton when the module is loaded
google_config = GoogleApiConfig()