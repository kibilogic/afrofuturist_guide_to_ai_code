import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import math

class SimpleCalculatorAI:

    # Loads the pretrained GPT-2 model and tokenizer
    def __init__(self):
        print("Loading GPT-2...")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  
        print("Model ready!")
    
    # Function that handles the math question
    def ask(self, question):
        if self.is_math(question):
            explanation = self.generate_response(question)
            result = self.calculate(question)
            return f"{explanation}\nCalculation: {result}" if result else explanation
        return self.generate_response(question)


    # Check if it's a math question 
    def is_math(self, text):
        keywords = ['+', '-', '*', '/', 'square root', 'calculate']
        return any(op in text.lower() for op in keywords)
    
    # AI explanation, create prompt and generate response  
    def generate_response(self, text):
          # Sets the conversation format as User (Student) and AI (Tutor)  
          prompt = f"Student: {text}\nTutor:"

          inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
          attention_mask = inputs['attention_mask']

          with torch.no_grad():
              outputs = self.model.generate(
                  inputs['input_ids'],
                  attention_mask=attention_mask,
                  max_length=inputs['input_ids'].shape[1] + 50,
                  pad_token_id=self.tokenizer.eos_token_id,
                  do_sample=True,
                  top_k=50,
                  top_p=0.95,
                  temperature=0.7
              )

          decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

          if "Tutor:" in decoded:
              response = decoded.split("Tutor:")[-1].strip()
              return response.split("\n")[0] or "Let's solve it together"

          return decoded.strip() or "Let's solve it together"

    def calculate(self, text):
        text = text.lower()
        text = text.replace("square root of", "sqrt")
        text = text.replace("plus", "+").replace("minus", "-")
        text = text.replace("times", "*").replace("divided by", "/")
        text = text.replace("calculate", "")

        # Handle square root
        match = re.search(r'sqrt\s*(\d+(\.\d+)?)', text)
        if match:
            number = float(match.group(1))
            return f"√{number} = {round(math.sqrt(number), 2)}"

        # Evaluates expression
        try:
            expr = re.sub(r'[^0-9+\-*/(). ]', '', text)
            result = eval(expr)
            result = round(result, 2) if isinstance(result, float) else result
            return f"{expr} = {result}"
        except:
            return None

def main():
    ai = SimpleCalculatorAI()
    print("\nType a math question or 'quit'")
    while True:
        q = input("You: ")
        if q.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        if q.strip():
            print("AI:", ai.ask(q))

if __name__ == "__main__":
    main()



