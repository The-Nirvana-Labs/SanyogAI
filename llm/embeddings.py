from sentence_transformers import SentenceTransformer


def mpnet_embeddings(text):
        model = SentenceTransformer('all-mpnet-base-v2')

        embedding = model.encode(text,show_progress_bar=True)

        return embedding



