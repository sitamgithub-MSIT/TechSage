# Necessary Imports
import os
import time
import base64
import PIL.Image
import gradio as gr
import google.generativeai as genai

from dotenv import load_dotenv

# Load the Environment Variables from .env file
load_dotenv()

# Set the Gemini API Key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Set up the model configuration for content generation
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 1400,
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in [
        "HARASSMENT",
        "HATE_SPEECH",
        "SEXUALLY_EXPLICIT",
        "DANGEROUS_CONTENT",
    ]
]

# Create the Gemini Models for Text and Vision respectively
txt_model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)
vis_model = genai.GenerativeModel(
    model_name="gemini-1.0-pro-vision-latest",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# System Prompt
system_prompt = """
Model: "As a tech blog chatbot, your role is crucial in providing accurate information and guidance to users seeking assistance in various areas of technology, including programming, software development, hardware troubleshooting, and technology trends. Your focus will be on addressing queries related to coding problems, software tools, programming languages, and emerging technologies, offering insights and recommendations to support users in their tech-related endeavors.

**Analysis Guidelines:**

Data Evaluation: Assess data related to coding trends, software development methodologies, programming languages usage, and technological advancements to understand the current landscape and identify areas for exploration and improvement.
Problem Identification: Identify common coding challenges, software bugs, hardware issues, and technology gaps faced by users, considering factors such as complexity, compatibility, and user skill levels.
Solution Discussion: Discuss potential solutions, workarounds, and best practices to resolve coding issues, debug software problems, troubleshoot hardware malfunctions, and stay updated with the latest technological innovations.
Community Engagement: Explore opportunities for community engagement and knowledge-sharing to foster collaboration, learning, and skill development among users, including participation in forums, coding communities, and tech events.
Monitoring and Evaluation: Propose methods for monitoring trends, evaluating software performance, measuring user satisfaction, and tracking technological advancements to ensure users receive relevant and up-to-date information and support.
Collaboration: Emphasize the importance of collaboration with fellow tech enthusiasts, developers, industry experts, and technology companies to facilitate knowledge exchange, promote innovation, and address common challenges in the tech community.

**Refusal Policy:**
If the user provides information not related to technology, programming, software development, hardware troubleshooting, or technology trends, kindly inform them that this chatbot is designed to address queries specific to these areas. Encourage them to seek assistance from appropriate sources for other inquiries.

Your role as a tech blog chatbot is to provide valuable insights and recommendations to support users in navigating the complexities of technology, coding, software development, and hardware troubleshooting. Proceed to assist users with their queries, ensuring clarity, empathy, and accuracy in your responses."

"""


# Image to Base 64 Converter Function
def image_to_base64(image_path):
    """
    Convert an image file to a base64 encoded string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded string representation of the image.
    """
    # Open Image and Encode it to Base64
    with open(image_path, "rb") as img:
        encoded_string = base64.b64encode(img.read())

    # Return the Encoded String
    return encoded_string.decode("utf-8")


# Function that takes User Inputs and displays it on ChatUI
def query_message(history, txt, img):
    """
    Adds a query message to the chat history.

    Parameters:
    history (list): The chat history.
    txt (str): The text message.
    img (str): The image file path.

    Returns:
    list: The updated chat history.
    """
    if not img:
        history += [(txt, None)]
        return history

    # Convert Image to Base64
    base64 = image_to_base64(img)

    # Display Image on Chat UI and return the history
    data_url = f"data:image/jpeg;base64,{base64}"
    history += [(f"{txt} ![]({data_url})", None)]
    return history


# Function that takes User Inputs, generates Response and displays on Chat UI
def llm_response(history, text, img):
    """
    Generate a response based on the input.

    Parameters:
    history (list): A list of previous chat history.
    text (str): The input text.
    img (str): The path to an image file (optional).

    Returns:
    list: The updated chat history.
    """

    # Generate Response based on the Input
    if not img:
        response = txt_model.generate_content(f"{system_prompt}User: {text}")
    else:
        # Open Image and Generate Response
        img = PIL.Image.open(img)
        response = vis_model.generate_content([f"{system_prompt}User: {text}", img])

    # Display Response on Chat UI and return the history
    history += [(None, response.text)]
    return history


# Interface Code using Gradio
with gr.Blocks(theme=gr.themes.Soft()) as app:

    with gr.Row():
        # Image UI
        image_box = gr.Image(type="filepath")

        # Chat UI
        chatbot = gr.Chatbot(scale=2, height=450)
    text_box = gr.Textbox(
        placeholder="Enter text and press enter, or upload an image",
        container=False,
    )

    # Button to Submit the Input and Generate Response
    btn = gr.Button("Submit")
    clicked = btn.click(query_message, [chatbot, text_box, image_box], chatbot).then(
        llm_response, [chatbot, text_box, image_box], chatbot
    )

# Launch the Interface
app.queue()
app.launch(debug=False)
