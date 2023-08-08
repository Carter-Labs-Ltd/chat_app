
import gradio as gr  # type: ignore[import]
import json  # type: ignore[import]
from sagemaker.predictor import Predictor
import os
from dotenv import load_dotenv
load_dotenv()



def complete(input, temp):
    stop = ['<|im_end|>']

    endpoint_name = os.environ["MPT_7B_ENDPOINT"]
    data = {
        "inputs": input,
        "parameters": {
            "do_sample": False,
            "temperature": temp,
            "max_new_tokens": 128,
            "stop": stop,
            "return_full_text": False,
            "details": True,
        },
    }

    predictor = Predictor(endpoint_name)
    inference_response = predictor.predict(
        data=json.dumps(data), initial_args={"ContentType": "application/json"}
    )

    response = json.loads(inference_response)[0]

    result = response["generated_text"]
    

    if stop is not None:
        for stop_token in stop + ['<|im_start|>assistant']:
            result = result.replace(stop_token, '')

    return result


def build_input_str(prompt, history):
    history_str = ""
    history_str += '<|im_start|>system\n'
    history_str += prompt + '<|im_end|>\n'
    
    for i, item in enumerate(history):
        if i % 2 == 0:
            history_str += f"<|im_start|>user\n{item}<|im_end|>\n"
        else:
            history_str += f"<|im_start|>assistant\n{item}<|im_end|>\n"
        history_str += f"<|im_start|>assistant\n"
    return history_str


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    temperature = gr.Slider(0.1, 2.0, value=1., label="Temperature")
    clear = gr.ClearButton([msg, chatbot])
    prompt = gr.Textbox(
        label="Prompt",
        placeholder="Enter master prompt.",
        max_lines=10,
    )

    async def respond(
        message, chat_history, prompt, temperature
    ):
        history = [item for sublist in chat_history for item in sublist] + [message]

        input_str = build_input_str(prompt, history)
        print(input_str)
        response = complete(input_str, temp=temperature)
        print(response)
        chat_history.append((message, response))

        return "", chat_history

    msg.submit(respond, [msg, chatbot, prompt, temperature], [msg, chatbot])
print(os.environ)
demo.launch()