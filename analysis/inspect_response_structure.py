"""
Inspect the detailed structure of responses to extract probabilities and log-likelihoods
"""
import pickle
from pathlib import Path

generations_path = Path('../results/validation_generations.pkl')

with open(generations_path, 'rb') as f:
    generations = pickle.load(f)

# Get first question
qid = list(generations.keys())[0]
gen_data = generations[qid]

print(f"Question ID: {qid}")
print(f"Question: {gen_data['question']}\n")

responses = gen_data.get('responses', [])
print(f"Number of responses: {len(responses)}\n")

if responses:
    for idx, response in enumerate(responses, 1):
        print(f"\n{'='*80}")
        print(f"Response {idx}")
        print(f"{'='*80}")
        print(f"Tuple length: {len(response)}")
        print(f"Tuple structure:")
        
        for i, element in enumerate(response):
            print(f"\n  Index {i}: {type(element).__name__}")
            
            if isinstance(element, str):
                print(f"    Value: {element[:100]}")
            elif isinstance(element, list):
                print(f"    Length: {len(element)}")
                if len(element) > 0:
                    print(f"    First element type: {type(element[0])}")
                    if isinstance(element[0], (int, float)):
                        print(f"    First 5 values: {element[:5]}")
                    else:
                        print(f"    First element: {element[0]}")
            elif hasattr(element, 'shape'):  # Tensor
                print(f"    Shape: {element.shape}")
                print(f"    Dtype: {element.dtype}")
            elif isinstance(element, (int, float)):
                print(f"    Value: {element}")
            else:
                print(f"    Value: {str(element)[:200]}")

print("\n" + "="*80)
print("INTERPRETATION GUIDE:")
print("="*80)
print("""
Typical structure:
  [0] = Answer text (string)
  [1] = Log-likelihoods per token (list of floats)
  [2] = Embedding or logits (torch.Tensor)
  [3] = Token IDs or indices (list)
  [4] = Token probabilities (list) - may need to exponentiate from log-space
  [5] = Overall log-likelihood (float)
""")
