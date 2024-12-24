import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import pipeline
from peft import PeftModel, PeftConfig
import gradio as gr

from transformers.utils import logging
logging.set_verbosity(logging.ERROR)

import configparser

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

CONTEXT_MESSAGE_COUNT = CONFIG['Settings'].getint('CONTEXT_MESSAGE_COUNT')
print(f'> CONTEXT_MESSAGE_COUNT = {CONTEXT_MESSAGE_COUNT} <')

DEVICE = None
if torch.cuda.is_available():
  DEVICE = torch.device("cuda") 
  print(f"> GPU is available, using {torch.cuda.get_device_name(0)} <")
else:
  DEVICE = torch.device("cpu")
  print("> GPU is not available, using CPU <")

ORIGINAL_MODEL = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-instruct", torch_dtype=torch.bfloat16)
TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-instruct")
MODEL = PeftModel.from_pretrained(ORIGINAL_MODEL, 'EduBot', torch_dtype=torch.bfloat16, is_trainable=False)
SYSTEM = {"role": "system", "content": "You are EduBot, an educational chatbot, and you will not share personal opinions. Additionally, you will avoid topics that are too far from academic questions."}
PIPE = pipeline(
  "text-generation",
  model=MODEL,
  tokenizer=TOKENIZER,
  torch_dtype=torch.bfloat16,
  device=DEVICE
)

def chat(message, history):
  request = {"role": "user", "content": message}
  if CONTEXT_MESSAGE_COUNT > 1:
    context = history[-CONTEXT_MESSAGE_COUNT + 1:]
    request = [SYSTEM] + context + [request]
  else:
    request = [SYSTEM] + [request]
  # print(request)
  outputs = PIPE(
    request,
    max_new_tokens=128,
    min_new_tokens=4,
    do_sample=True,
    top_k=8,
    max_time=32.0,
  )
  response = outputs[0]["generated_text"][-1]['content']
  if response[-1] != '.' and response[-1] != '?' and response[-1] != '!':
    response += ' ...'
  # print(response)
  return response

gr.ChatInterface(chat, type="messages").launch()
