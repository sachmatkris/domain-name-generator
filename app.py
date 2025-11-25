import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.output_parsers import PydanticOutputParser
import uvicorn

PROMPT = """
You are a domain-name generation assistant. Your job is to read the user-provided 
*business_description* and propose ONE realistic, human-feeling domain name that fits the business.

Follow these rules carefully:

1. If the business description mentions a specific city or country, use the correct ccTLD 
   for that location (e.g., .uk, .de, .fr, .nl, .es, .pt, .it, .lt, .ca, .us, .au, .jp, etc...).

2. If no location is mentioned, choose from REAL modern TLDs ONLY:
   .com, .io, .co, .app, .ai, .net, .org, etc...

3. The domain MUST:
   - sound natural and brandable
   - accurately reflect the type of business
   - use ONLY real TLDs (no invented extensions)

4. Make sure the domain is in the exact format: `name.tld`

5. Return ONLY one domain name.

6. The final output MUST be valid JSON that matches the schema:

{format_instructions}

Do NOT include explanations, reasoning, or extra text. Only return valid JSON.
"""

class Response(BaseModel):
    answer: str

class BD(BaseModel):
    business_description: str


parser = PydanticOutputParser(pydantic_object=Response)
format_instructions = parser.get_format_instructions()


def parse(output_text: str) -> str:
    try:
        parsed: Response = parser.parse(output_text)
        return parsed.answer
    except Exception as e:
        raise e(f"Could not parse from: {output_text!r}")


def propose_domain_name(description, prompt, model, tokenizer):
    prompt_specific = prompt.format(format_instructions=format_instructions)
    messages = [
        {"role" : "system", "content" : prompt_specific},
        {"role" : "user", "content" : f"**business_description**:\n{description}"}
    ]
    tokenized_text_processed = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokenized_text_processed = tokenizer([tokenized_text_processed], return_tensors="pt")
    tokenized_text_processed = {k: v.to(model.device) for k, v in tokenized_text_processed.items()}
    with torch.no_grad():
        generated_ids = model.generate(**tokenized_text_processed, max_new_tokens=1024, do_sample=False)
    output_ids = generated_ids[0][len(tokenized_text_processed["input_ids"][0]):]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(output_text)
    output = parse(output_text)
    return output.lower()


MODEL_NAME = "ft_model/checkpoint-81/"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

app = FastAPI(title="Domain Suggestion API")

@app.post("/suggest", response_model=Response)
def suggest_domain(bd: BD):
    suggestion = propose_domain_name(bd.business_description, PROMPT, model, tokenizer)
    return Response(answer=suggestion)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)