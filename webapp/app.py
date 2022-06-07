from fastapi import FastAPI

app = FastAPI()

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/gpt2-base-bne")
model = AutoModelForCausalLM.from_pretrained("PlanTL-GOB-ES/gpt2-base-bne")

def text_generation(input_text, seed):
  input_ids = tokenizer(input_text, return_tensors="pt").input_ids
  #torch.manual_seed(seed) # Max value: 18446744073709551615
  outputs = model.generate(input_ids, do_sample=True, min_length=35, max_length=100)
  generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
  return generated_text


@app.get("/")
def root():
    result = text_generation("esto es un ejemplo de", 0)
    return {"text": str(result)}


if __name__ == "__main__":
    app.run(debug=True, port=80, host="0.0.0.0")
