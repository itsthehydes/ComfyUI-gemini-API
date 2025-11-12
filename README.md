# ComfyUI-Google-API

Custom nodes for ComfyUI to interact directly with Google's generative AI models, including **Gemini (LLM/VLM)** and **Vertex AI (Imagen)**.

This repository provides nodes for:
* **LLM (Google)**: Text generation (Gemini).
* **VLM (Google)**: Image-to-text / Visual Q&A (Gemini).
* **Imagen (Google)**: Text-to-image generation (Vertex AI).

---

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory.
    ```bash
    cd ComfyUI/custom_nodes
    ```

2.  Clone this repository:
    ```bash
    git clone [https://github.com/itsthehydes/ComfyUI-gemini-API.git](https://github.com/itsthehydes/ComfyUI-gemini-API.git)
    ```

3.  Install the required dependencies:
    ```bash
    cd ComfyUI-gemini-API
    pip install -r requirements.txt
    ```

4.  Restart ComfyUI.

---

## Configuration

This repository uses two different Google services, which require **two different authentication methods**.

### 1. Gemini API (for LLM & VLM Nodes)

This uses a simple API Key.

1.  Get your API key from **[Google AI Studio](https://aistudio.google.com/app/apikey)**.
2.  Open the `config.ini` file located in `ComfyUI/custom_nodes/ComfyUI-gemini-API`.
3.  Find the `[API]` section and paste your key:

    ```ini
    [API]
    GOOGLE_API_KEY = your_gemini_api_key_goes_here
    ```

### 2. Vertex AI (for Imagen Text-to-Image Node)

This uses a Google Cloud Project and a Service Account.

1.  **Sign up for Google Cloud** and create a new project.
2.  **Enable Billing** for your project.
3.  **Enable the "Vertex AI API"** in your project's "APIs & Services" dashboard.
4.  **Create a Service Account:**
    * Go to "IAM & Admin" > "Service Accounts".
    * Click "+ Create Service Account".
    * Give it a name (e.g., `vertex-ai-user`).
    * Grant it the "**Vertex AI User**" role.
5.  **Download Your JSON Key:**
    * After creating the service account, click its name.
    * Go to the "**KEYS**" tab.
    * Click "**ADD KEY**" -> "**Create new key**".
    * Select "**JSON**" and click "Create". A `.json` file will download.
6.  **Configure Your Project:**
    * **Move** the downloaded `.json` file into the root of your `ComfyUI-gemini-API` folder (the same place as `config.ini`).
    * **Rename** the `.json` file to `vertex-ai-key.json`.
    * **Open `config.ini`** again.
    * Fill out the `[VERTEX_AI]` section with your Project ID (from your cloud dashboard) and the key file name.

    ```ini
    [VERTEX_AI]
    PROJECT_ID = your-gcloud-project-id-here
    LOCATION = us-central1
    SERVICE_ACCOUNT_FILE = vertex-ai-key.json
    ```

---

## Usage

After installation and configuration, restart ComfyUI. The new nodes will be available in the node browser under the **"GoogleAPI"** category:

* **GoogleAPI/LLM/LLM (Google)**: For text-only prompts.
* **GoogleAPI/VLM/VLM (Google)**: For prompts that include an image.
* **GoogleAPI/Image/Imagen Text-to-Image (Google)**: For generating images from text.