
#Import statements
import torch
from transformers import AutoTokenizer, FlaxGPTNeoModel
#from transformers import GPTNeoForCausalLM, GPT2Tokenizer



# Ensure the code runs on the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#New tokenizer for Flax
#Now using the auto tokenizer instead
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
#tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")


#Adding in new Fl
model = FlaxGPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
#model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = model.to(device) # Move model to GPU if available




#new


#inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
#outputs = model(**inputs)

#last_hidden_states = outputs.last_hidden_state


#new



prompts = [
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English.",

    "As the sun set, the sky turned a brilliant orange. It was a sight to behold, "
    "as if the heavens themselves were ablaze.",
    
    "Creating neural networks, is the best past time on a Saturday morning "
    "in order to drive innovation in every corner of the land.",
    
    "Running around San Francisco is the best past time, when taking a break from coding "
    "but it's important to always be rocking nike shoes",
    
    "The only way to make a pizza pie, is to use the most refined flour "
    "otherwise people will not enjoy the pizza to the fullest potential.",

    # Add more prompts as needed
]


#TO push this to increase GPU usage change below to 2K for max length. Changing to lower value for continuation of app.

for prompt in prompts:
    #changing this to now handle JAX
    inputs = tokenizer(prompt,  return_tensors="jax")
    
    input_ids = inputs.input_ids.to(device) # Move input to the same device as the model

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=200, # Increased max_length to push the GPU further
    )

    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)
