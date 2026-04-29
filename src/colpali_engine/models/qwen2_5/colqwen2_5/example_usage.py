"""
Example usage of ColQwen2_5 model as both embedding model and VL chat model.
"""

from typing import List
import torch
from PIL import Image   
from transformers import BatchFeature
import time
# Import your ColQwen2_5 model and processor
from modeling_colqwen2_5 import ColQwen2_5
from processing_colqwen2_5 import ColQwen2_5_Processor
from transformers import Qwen2_5_VLConfig

MODEL_PATH = "vidore/colqwen2.5-base"
DEVICE = "cuda"
TAG_SUMMARY_KEYWORD_PROMPT = """
# Role Description
- You are a multimodal vision-language model specialized in generating structured multi-label annotations for images based on a provided comprehensive field schema.

# Input Context
- You will receive an image along with a complete field definition table (including field names, descriptions, and example tags).
- Your task is to analyze the image content and assign appropriate tags to each field according to its definition.

# Output Requirements
- The output must be a valid JSON dictionary with keys as field names (matching exactly the provided field table) and values as lists of strings representing the tags assigned to that field.
- If the information for a field cannot be determined or is not applicable from the image, return an empty list `[]`.
- Each field may contain only one tag; the tag must be accurate, truthful, highly relevant to the image content, and comply with the semantic constraints and format requirements of the field.
- The output should contain no additional explanation or formatting beyond the JSON structure.

# Annotation Principles
- All tags must be directly or clearly inferred from the image content; do not fabricate, guess, or supplement information that is not visually present.
- The language of the tags should be consistent with the image’s linguistic context (e.g., use Chinese tags for a Chinese poster) unless otherwise specified by the field definitions.
- All fields must be output completely without omission, even if some fields have empty lists.
- It is preferred that the assigned tags are among the example tags in the field definition table. Custom tags can only be used when the required tag is not present in the example tags.

# Note
- All tags must be nouns unless the field explicitly requires verbs or other parts of speech. Tags should be clear and unambiguous.
- The value for each field must be an array of strings; even if there is only one tag, brackets cannot be omitted.
- For abstract, blurry, or artistic images, label as much as can be recognized, and return empty arrays for fields that cannot be judged.

# Output Format Example:
```json
{
  "people_tags": ["Elon Musk"],
  "company_tags": ["Tesla"],
  "task_tags": ["product launch", "brand promotion"]
}
```
"""

# Example 1: Using as VL Chat Model (similar to original Qwen2.5-VL)
def vl_chat_example():
    """Example of using ColQwen2_5 as a VL chat model."""
    
    # Load your ColQwen2_5 model and processor
    config = Qwen2_5_VLConfig.from_pretrained(MODEL_PATH)
    model = ColQwen2_5.from_pretrained(MODEL_PATH, config=config)
    processor = ColQwen2_5_Processor.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    
    # Example messages (similar to original Qwen2.5-VL usage)
    
    messages = [
        {
            "role": "system",
            "content": TAG_SUMMARY_KEYWORD_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/workspace/docpc/image.png",  # Replace with actual image path
                },
            ],
        }
    ]
    
    # Process the messages using the VL chat methods
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Load and process images
    images = [Image.open("/workspace/docpc/image.png")]  # Replace with actual image path
    
    inputs = processor.process_vl_chat(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)
    
    # Generate response using VL chat methods
    generated_ids = model.vl_chat_generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("VL Chat Output:", output_text)

# Example 2: Using as Embedding Model (ColQwen functionality)
def embedding_example():
    """Example of using ColQwen2_5 as an embedding model."""
    
    # Load your ColQwen2_5 model and processor
    config = Qwen2_5_VLConfig.from_pretrained(MODEL_PATH)
    model = ColQwen2_5.from_pretrained(MODEL_PATH, config=config)
    processor = ColQwen2_5_Processor.from_pretrained(MODEL_PATH)
    
    # Example documents with images
    documents = ["This is a document about cats.", "This is a document about dogs."]
    images = [Image.open("/workspace/docpc/example_image.png"), Image.open("/workspace/docpc/example_image.png")]  # Replace with actual paths
    
    # Process documents using ColQwen methods
    batch_doc = processor.process_images(images, context_prompts=documents)
    
    # Get document embeddings
    with torch.no_grad():
        doc_embeddings = model(**batch_doc)  # Shape: (batch_size, seq_len, dim)
    
    # Example queries
    queries = ["Find documents about cats", "Find documents about dogs"]
    batch_query = processor.process_queries(queries)
    
    # Get query embeddings
    with torch.no_grad():
        query_embeddings = model(**batch_query)  # Shape: (batch_size, seq_len, dim)
    
    # Compute similarity scores
    scores = processor.score(
        [query_embeddings], [doc_embeddings], device=DEVICE
    )
    print("Similarity Scores:", scores)

# Example 3: Using both modes in the same session
def combined_usage_example():
    """Example of using ColQwen2_5 for both embedding and VL chat in the same session."""
    
    # Load your ColQwen2_5 model and processor
    model = ColQwen2_5.from_pretrained(MODEL_PATH, device_map=DEVICE)
    processor = ColQwen2_5_Processor.from_pretrained(MODEL_PATH)
    start_time = time.time()
    # First, use as embedding model
    print("=== Using as Embedding Model ===")
    images = [Image.open("/workspace/docpc/example_image.png")]  # Replace with actual path
    
    image_batch = processor.process_images([images[0]]).to(DEVICE)
    with torch.no_grad():
        image_embedding = model(**image_batch)
    print("Image embedding shape:", image_embedding.shape)
    end_time = time.time()
    print(f"Time taken for image embedding: {end_time - start_time} seconds")
    # Then, use as VL chat model
    print("\n=== Using as VL Chat Model ===")
    start_time = time.time()
    messages = [
        {
            "role": "system",
            "content": TAG_SUMMARY_KEYWORD_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/workspace/docpc/example_image.png",  # Replace with actual image path
                }
            ],
        }
    ]
    
    # Process the messages using the VL chat methods
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Load and process images
    images = [Image.open("/workspace/docpc/example_image.png")]  # Replace with actual image path
    
    inputs = processor.process_vl_chat(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)
    
    # Generate response using VL chat methods
    generated_ids = model.vl_chat_generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("VL Chat Output:", output_text)
    end_time = time.time()
    print(f"Time taken for VL chat: {end_time - start_time} seconds")

if __name__ == "__main__":
    vl_chat_example()