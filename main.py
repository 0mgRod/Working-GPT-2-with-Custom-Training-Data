import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define a function to generate a response from the model
def generate_response(prompt):
    # Encode the prompt and generate output
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Start the chatbot loop
while True:
    # Get user input
    user_input = input("You: ")

    # Generate a response from the model
    response = generate_response(user_input)

    # Print the response
    print("AI: " + response)
