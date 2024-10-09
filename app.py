from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_response(prompt):
    response = generator(prompt, max_length=100, num_return_sequences=1)
    return response[0]["generated_text"]

if __name__ == "__main__":
    prompt = input("Enter a prompt: ")  # Get user input
    response = generate_response(prompt)
    print("Generated Response: ", response)

generate_response("Its my birthday today. I used to get sad on my birthday for the past 10 years, but i started feeling grateful and happy. what can you say about this")