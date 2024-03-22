# Importing required documents
import os
from operator import itemgetter

import gradio as gr
from huggingface_hub import hf_hub_download
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat

# Initializing model hyperparametes for generation
MAX_MAX_NEW_TOKENS = 1024
DEFAULT_MAX_NEW_TOKENS = 250
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
temperature = 0.3
top_p = 0.9
top_k = 50
repetition_penalty = 1.2

DESCRIPTION = """
# MedPal Chat
A simple demonstration of personal medical chatbot that uses capabilities of BioMistral GGUF.
Advisory Notice! Although BioMistral is intended to encapsulate medical knowledge sourced from high-quality evidence, it hasn't been tailored to effectively, safely, or suitably convey this knowledge within professional parameters for action. We advise refraining from utilizing BioMistral in medical contexts unless it undergoes thorough alignment with specific use cases and undergoes further testing, notably including randomized controlled trials in real-world medical environments. BioMistral 7B may possess inherent risks and biases that have not yet been thoroughly assessed. Additionally, the model's performance has not been evaluated in real-world clinical settings. Consequently, we recommend using BioMistral 7B strictly as a research tool and advise against deploying it in production environments for natural language generation or any professional health and medical purposes.
"""

# Defining model path
model_type = "BioMistral/BioMistral-7B-GGUF"
model_file = "ggml-model-Q5_K_M.gguf"
local_path = "./models/" + model_type

# Downloading model
local_llm = hf_hub_download(
    repo_id = model_type,
    filename = model_file,
    local_dir=local_path
)

# Loading model in the memory with required hyperparameters
llm = LlamaCpp(
    model_path=local_llm,
    max_token=DEFAULT_MAX_NEW_TOKENS,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    repeat_penalty=repetition_penalty
    
)

# Augmenting chat interface wrapper on the loaded model
model = Llama2Chat(llm=llm)

# Defining system prompt and 
system_prompt = "You are a healthcare assistant chatbot that utilizes natural language processing to interact with user. The user has provided their Medical history which contains a summary of either past or current symptoms, medication and lifestyle factors."
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ]
)

# Defining chat memory and langchain 
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)


# defining chat retrieval function
def generate(
    message: str,
    chat_history: list[tuple[str, str]],
):
    # run the chain to retrieve the response
    response = chain.predict(human_input=message)
    return response

# defining chat intereface
chat_interface = gr.ChatInterface(
    fn=generate,
    stop_btn=None,
    examples=[
        ["What should I to prevent cough?"],
    ],
)

# rendering the chat UI
with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()

if __name__ == "__main__":
    # running the chat server
    demo.queue(max_size=20).launch(show_api=False)
