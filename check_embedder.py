from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/deepset-mxbai-embed-de-large-v1")
model = AutoModel.from_pretrained("mixedbread-ai/deepset-mxbai-embed-de-large-v1")

# Print the embedding size from the model’s configuration
print("Embedding size:", model.config.hidden_size)
