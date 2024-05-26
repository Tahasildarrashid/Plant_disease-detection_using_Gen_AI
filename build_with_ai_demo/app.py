# Required Libraries
import google.generativeai as genai
from pathlib import Path
import gradio as gr

# Setting Up the Model Configuration
generation_config = {
  "temperature": 0,
  "top_p": 1,
  "top_k": 32,
  "max_output_tokens": 4096,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  }
]

# Configure the API Key
genai.configure(api_key="YOUR_SECRET_KEY_HERE")

# Initialize the Model
model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Function to Set Up Image Input
def input_image_setup(file_loc):
    if not (img := Path(file_loc)).exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": Path(file_loc).read_bytes()
        }
    ]
    return image_parts

# Function to Generate Response
def generate_gemini_response(input_prompt, text_input, image_loc):
    image_prompt = input_image_setup(image_loc)
    prompt_parts = [input_prompt + text_input, image_prompt[0]]
    response = model.generate_content(prompt_parts)
    return response.text

# Input Prompt
input_prompt = """This image shows a plant with a possible disease or pest infestation.
Please analyze the image and identify the problem.
Based on your analysis, suggest appropriate remedies and preventive measures, and provide useful website links (Indian links only) which are valid and existing and which can give more information regarding plant disease management.
Also, provide contact information for agricultural helplines in India.
The prompted message is """

# Function to Upload File and Generate Response
def upload_file(files, text_input):
    file_paths = [file.name for file in files]
    if file_paths:
        response = generate_gemini_response(input_prompt, text_input, file_paths[0])
    return file_paths[0], response

# Gradio Interface
with gr.Blocks() as demo:
    header = gr.Label("Please upload an image of the plant and describe any symptoms you have observed:")
    text_input = gr.Textbox(label="Describe the symptoms observed on the plant")
    image_output = gr.Image()
    upload_button = gr.UploadButton("Click to upload an image of the plant",
                                    file_types=["image"],
                                    file_count="multiple")
    file_output = gr.Textbox(label="Diagnosis and Remedies")
    combined_output = [image_output, file_output]

    upload_button.upload(upload_file, [upload_button, text_input], combined_output)

demo.launch(debug=True, share=True)
