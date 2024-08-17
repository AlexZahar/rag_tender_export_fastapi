from transformers import AutoTokenizer, AutoModel
from rag_tender_export_fastapi.config.settings import load_config

config=load_config()
# Load model and tokenizer
model_name = config["embedding_model"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Print the embedding size from the model's configuration
print("Embedding size:", model.config.hidden_size)

# Check if the model has a sparse embedding layer
has_sparse_embedding = hasattr(model, 'sparse_embedding')
print("Has sparse embedding layer:", has_sparse_embedding)

# Print model architecture
print("\nModel Architecture:")
print(model)

# Print model config
print("\nModel Config:")
print(model.config)