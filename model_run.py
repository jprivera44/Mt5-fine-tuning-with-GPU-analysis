#File to now run the 

#checking on the version of pytorch
import torch
print(torch.__version__)

from transformers import GPTJForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device usage",device)

#Importing in the GPT-J model from Hugging Face, using low memory usage

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", low_cpu_mem_usage=True)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

#Running Inference on the model
prompt = "The Belgian national football team "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=200)
generated_text = tokenizer.decode(generated_ids[0])
print(generated_text)