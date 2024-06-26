﻿# TechSage

This project aims to provide to help this [blog project](https://github.com/sitamgithub-MSIT/TechwithTea) by providing a chat assistant that can help users with technical queries while reading the blog.

## Project Structure

The project is structured as follows:

- `assets`: This directory contains the output responses screenshots.

- `src`: This directory contains the source code for the project.

  - `chat.py`: This file contains the code for the chat updating the conversation history.
  - `config.py`: This file contains the model configuration settings for the project.
  - `llm_response.py`: This file contains the code for the language model response.
  - `prompt.py`: This file contains the system prompt for the project.
  - `utils.py`: This file contains utility functions for the project.

- `app.py`: This file contains the code for the Gradio application.
- `.env.example`: This file contains the environment variables required for the project.
- `requirements.txt`: This file contains the required dependencies for the project.
- `README.md`: This file contains the project documentation.
- `LICENSE`: This file contains the project license.

## Tech Stack

- Python: Python is used as the primary programming language for building the application.
- Gemini API: These APIs provide advanced natural language processing and computer vision capabilities.
- Gradio: Gradio is used for building interactive UIs for the chat interface.
- Hugging Face spaces: Hugging Face spaces is used for collaborative development and deployment of the gradio application.

## Getting Started

To get started with this project, follow the steps below:

1. Clone the repository: `git clone https://github.com/sitamgithub-MSIT/TechSage.git`
2. Create a virtual environment: `python -m venv tutorial-env`
3. Activate the virtual environment: `tutorial-env\Scripts\activate`
4. Install the required dependencies: `pip install -r requirements.txt`
5. Run the Gradio application: `python app.py`

Now, open up your local host and you should see the web application running. For more information, refer to the Gradio documentation [here](https://www.gradio.app/docs/interface). Also, a live version of the application can be found [here](https://huggingface.co/spaces/sitammeur/TechSage).

## Usage

Once the application is up and running, you can interact with the conversational entity through the provided UI. It can answer queries related to the blog content and provide additional information on the topics discussed in the blog.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please raise an issue to discuss the changes you would like to make. Once the changes are approved, you can create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions regarding the project, feel free to reach out to me on my [GitHub profile](https://github.com/sitamgithub-MSIT).

Happy coding! 🚀
