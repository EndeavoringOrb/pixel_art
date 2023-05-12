from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer
from torch import dot, norm, tensor, no_grad
from numpy import argmax
from transformers import AutoTokenizer, CLIPTextModel

maxlen = 256

clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", max_length=maxlen)
model = SentenceTransformer("C:/Users/aaron/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")
model_name = 'bert-base-uncased'
bert_model = BertModel.from_pretrained(model_name)
bert_tokenizer = BertTokenizer.from_pretrained(model_name)


def get_clip_embeddings(sentences):
    '''
    currently truncates input len to 77
    '''

    if type(sentences) == str:
        sentences = [sentences]
        
    inputs = clip_tokenizer(sentences, padding=True, return_tensors="pt")

    while inputs.data['input_ids'].shape[-1] > 77:
        biggest_len = max([len(i) for i in sentences])
        for i in range(len(sentences)):
            if len(sentences[i]) == biggest_len:
                sentences[i] = sentences[i][1:]
        inputs = clip_tokenizer(sentences, padding=True, return_tensors="pt")

    with no_grad():
        clip_model.eval()
        outputs = clip_model(**inputs)
    pooled_output = outputs.pooler_output  # pooled (EOS token) states
    
    return pooled_output

def get_bert_embedding(sentence):

    # Tokenize the input sentence
    tokens = bert_tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tensor([input_ids])

    # Generate the text embeddings
    with no_grad():
        model.eval()
        outputs = bert_model(input_ids)
        embeddings = outputs[0][0]  # Extract the embeddings from the last layer

    # Calculate the mean of the embeddings
    mean_embedding = embeddings.mean(dim=0)

    return mean_embedding


def embedding_cosine_similarity(embedding1, embedding2):
    """
    Computes the cosine similarity between two PyTorch tensors representing sentence embeddings.
    """
    embedding1 = tensor(embedding1)
    embedding2 = tensor(embedding2)
    dot_product = dot(embedding1, embedding2)
    norm1 = norm(embedding1)
    norm2 = norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def get_embeddings(sentences):
    if type(sentences) == str:
        sentences = [sentences]
    embeddings = model.encode(sentences)
    return embeddings

def get_bert_sentence_similarity(sentence1, sentence2):
    embedding1, time1 = get_bert_embedding(sentence1)
    embedding2, time2 = get_bert_embedding(sentence2)
    print(time1)
    print(time2)
    return embedding_cosine_similarity(embedding1, embedding2)

def get_sentence_similarity(sentence1, sentence2):
    embeddings = model.encode([sentence1, sentence2])
    return embedding_cosine_similarity(embeddings[0],embeddings[1])

def get_most_similar_sentence(sentence, sentences):
    """
    returns index of most similar sentence, most similar sentence
    """
    index_of_sentences = argmax([get_sentence_similarity(sentence, sentences[i]) for i in range(len(sentences))])
    return index_of_sentences, sentences[index_of_sentences]

print("\n\n\n")

'''iters = 100
print(f"Testing with {iters} iterations")
start = perf_counter()
for i in range(iters):
    get_sentence_similarity("This is a sentence", "This is a sentence")
end = perf_counter()
print(f"Average time: {(end-start)/iters} seconds")'''