# Necessary Imports
import gradio as gr

# Import the necessary functions from the src folder
from src.chat import query_message
from src.llm_response import llm_response


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
