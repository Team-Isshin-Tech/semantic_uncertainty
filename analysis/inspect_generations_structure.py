"""
Inspect the structure of validation_generations.pkl to understand how to extract 10 answers
"""
import pickle
from pathlib import Path

generations_path = Path('../results/validation_generations.pkl')

if generations_path.exists():
    print("Loading validation_generations.pkl...")
    with open(generations_path, 'rb') as f:
        generations = pickle.load(f)
    
    # Get first question ID
    qid = list(generations.keys())[0]
    
    print(f"\nFirst question ID: {qid}")
    print(f"Data type: {type(generations[qid])}")
    
    gen_data = generations[qid]
    
    if isinstance(gen_data, dict):
        print(f"\nDict keys: {list(gen_data.keys())}")
        print("\nDetailed structure:")
        for key, value in gen_data.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, list) and len(value) > 0:
                print(f"    - Length: {len(value)}")
                print(f"    - First element type: {type(value[0])}")
                if isinstance(value[0], dict):
                    print(f"    - First element keys: {list(value[0].keys())}")
    
    elif isinstance(gen_data, (list, tuple)):
        print(f"\nList/Tuple length: {len(gen_data)}")
        for i, item in enumerate(gen_data):
            print(f"  Index {i}: {type(item)}")
            if isinstance(item, list) and len(item) > 0:
                print(f"    - Length: {len(item)}")
                print(f"    - Sample: {str(item[0])[:200]}")
    
    # Try different keys
    print("\n" + "="*80)
    print("Searching for generations/answers...")
    if isinstance(gen_data, dict):
        for possible_key in ['generations', 'answers', 'responses', 'generated_texts', 'samples', 'model_answers']:
            if possible_key in gen_data:
                print(f"  Found key: '{possible_key}'")
                print(f"    Type: {type(gen_data[possible_key])}")
                print(f"    Length: {len(gen_data[possible_key]) if hasattr(gen_data[possible_key], '__len__') else 'N/A'}")
    
    print("\nFull sample (first 1000 chars):")
    print(str(gen_data)[:1000])
else:
    print(f"File not found: {generations_path}")
