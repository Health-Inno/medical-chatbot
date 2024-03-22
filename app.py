import os
import copy
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


MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

DESCRIPTION = """
# BioMistral-7b GGUF Chat
A simple demonstration of personal medical chatbot.
Advisory Notice! Although BioMistral is intended to encapsulate medical knowledge sourced from high-quality evidence, it hasn't been tailored to effectively, safely, or suitably convey this knowledge within professional parameters for action. We advise refraining from utilizing BioMistral in medical contexts unless it undergoes thorough alignment with specific use cases and undergoes further testing, notably including randomized controlled trials in real-world medical environments. BioMistral 7B may possess inherent risks and biases that have not yet been thoroughly assessed. Additionally, the model's performance has not been evaluated in real-world clinical settings. Consequently, we recommend using BioMistral 7B strictly as a research tool and advise against deploying it in production environments for natural language generation or any professional health and medical purposes.
"""

model_type = "BioMistral/BioMistral-7B-GGUF"
model_file = "ggml-model-Q5_K_M.gguf"
local_path = "./models/" + model_type

local_llm = hf_hub_download(
    repo_id = model_type,
    filename = model_file,
    local_dir=local_path
)

llm = LlamaCpp(
    model_path=local_llm
)

model = Llama2Chat(llm=llm)

system_prompt = "You are a healthcare assistant chatbot that utilizes natural language processing to interact with user. The user has provided their Medical history which contains a summary of either past or current symptoms, medication and lifestyle factors."

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)

# chain = (
#     RunnablePassthrough.assign(
#         chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
#     )
#     | prompt_template
#     | llm
# )


def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    
    # conversation = []
    # if system_prompt:
    #     conversation.append({"role": "system", "content": system_prompt})
    # for user, assistant in chat_history:
    #     conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    # conversation.append({"role": "user", "content": message})

    # print(conversation)

    generate_kwargs = dict(
        max_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repeat_penalty=repetition_penalty,
    )
    
    # output = model.create_chat_completion(**generate_kwargs)
    
    # temp = ""
    # for out in output:
    #     print(out)
    #     stream = copy.deepcopy(out)
    #     temp += stream["choices"][0]["delta"]["content"] if "content" in stream["choices"][0]["delta"] else ""
    #     yield temp


    # for event in chain.astream_events(message, version="v1", **generate_kwargs):
    #     yield event['data']['chunk'] if "chunk" in event['data'] else ""

    # partial_message = ""
    # for response in chain.stream({"human_input":message}):
    #         # partial_message += response.content
    #         print(response)
    #         partial_message += response
    #         yield  partial_message
    response = chain.predict(human_input=message)
    # partial_message = ""
    # for chunk in chain.stream(message):
    #       print(chunk)
    #     #   partial_message += chunk.content
    #       yield chunk
    return response


chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
        ),
    ],
    stop_btn=None,
    examples=[
        ["What should I to prevent cough?"],
    ],
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()

if __name__ == "__main__":
    demo.queue(max_size=20).launch(show_api=False)
