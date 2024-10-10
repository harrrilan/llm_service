from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_response(prompt):
    response = generator(prompt, max_length=100, num_return_sequences=1)
    return response[0]["generated_text"]

if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            print("Thank you for using the text generator. Goodbye!")
            break
        response = generate_response(prompt)
        print("Generated Response:", response)
        print()  # Add a blank line for readability