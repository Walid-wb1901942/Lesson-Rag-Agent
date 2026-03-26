from app.services.ollama_client import embed_texts

texts = [
    "Photosynthesis is how plants make energy",
    "A food chain describes how organisms eat each other"
]

vectors = embed_texts(texts)

print("Texts embedded:", len(vectors))
print("Vector length:", len(vectors[0]))
print("First vector sample:", vectors[0][:10])
