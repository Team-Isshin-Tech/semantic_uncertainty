"""
Inspect the structure of the 'responses' field
"""
import pickle
from pathlib import Path

generations_path = Path('../results/validation_generations.pkl')

with open(generations_path, 'rb') as f:
    generations = pickle.load(f)

qid = list(generations.keys())[0]
gen_data = generations[qid]

print(f"Question: {gen_data['question']}")
print(f"\nNumber of responses: {len(gen_data['responses'])}")
print("\n" + "="*80)

# Inspect first response structure
first_response = gen_data['responses'][0]
print(f"First response type: {type(first_response)}")
print(f"Tuple length: {len(first_response)}")

print("\nTuple elements:")
for i, element in enumerate(first_response):
    print(f"  Index {i}: {type(element)}")
    if isinstance(element, dict):
        print(f"    Keys: {list(element.keys())}")
        for k, v in element.items():
            print(f"      {k}: {type(v)} = {str(v)[:100]}")
    elif isinstance(element, str):
        print(f"    Value: {element[:200]}")
    elif isinstance(element, list):
        print(f"    Length: {len(element)}")

print("\n" + "="*80)
print("All 10 responses:")
for idx, response in enumerate(gen_data['responses'], 1):
    print(f"\n{idx}. Type: {type(response)}")
    if len(response) >= 1:
        # First element is usually the answer text
        answer = response[0]
        if isinstance(answer, dict) and 'response' in answer:
            print(f"   Answer: {answer['response']}")
        else:
            print(f"   Answer: {str(answer)[:200]}")
