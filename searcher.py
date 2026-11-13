from dataclasses import dataclass
import numpy as np
import spacy
import torch
from transformers import BatchEncoding
from yasbd import register_spacy_component

"""
Utility class for storing information about a chunk.

Attributes:
    text (str): The text captured in the chunk.
    char_bounds (tuple): The beginning and ending char indices of the text, end exclusive.
    token_bounds (tuple): The beginning and ending token indices of the text, end exclusive.
    tokens (BatchEncoding): The tokenized chunk text.
    embedding (np.ndarray): The text's sentence embedding.
"""
@dataclass
class Chunk:
    text: str
    char_bounds: tuple[int, int]
    token_bounds: tuple[int, int]
    tokens: BatchEncoding | None = None
    embedding: np.ndarray | None = None

"""
Utility class for the results of a document search.

Attributes:
    chunk (Chunk): Semantic and positional information about a textual chunk.
    score (float): The ranked similarity of the chunk to a query. Exact scales may vary.
"""
@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    
class Searcher:
    """
    A class for ingesting and executing look-ahead+look-behind semantic searches on an input text.
    Set window_offset=0 to disable look-ahead/behind, or -1 to use late chunking.
    
    Args:
        input_text (str): The entire document text.
        nlp (spacy.language.Language | None): a spaCy pipeline for splitting the document sentences.
        tokenizer (AutoTokenizer): The tokenizer used to pre-process the text and map characters to tokens.
        model (SentenceTransformer): The model used to embed the document text.
        max_chunk_size (int): The maximum chunk size allowed without truncation.
        reranker (AutoModelForSequenceClassification): The reranker applied to a subset of the search results.
        window_offset (int): The number of preceding and successive chunks to factor into each current chunk.

    Attributes:
        input_text (str): See above.
        tokenizer (AutoTokenizer): See above.
        model (SentenceTransformer): See above.
        max_chunk_size (int): See above.
        reranker (AutoModelForSequenceClassification): See above.
        window_offset (int): See above.
        input_tokens (BatchEncoding): The entire tokenized input_text.
        tokens_char_bounds (torch.Tensor): A list of beginning and ending char indices for each token, end exclusive.
        chunks: (List[Chunk]): A list of chunks derived from the input_text.
    """

    def __init__(self, input_text, nlp, tokenizer, model, max_chunk_size, reranker, window_offset=0):
        self.input_text = input_text.strip()
        self.tokenizer = tokenizer
        self.model = model
        self.max_chunk_size = max_chunk_size
        self.reranker = reranker
        self.window_offset = window_offset
        self.chunks = []

        # NLProcessor will be used to split sentences
        if nlp is None:
            register_spacy_component()
            self.nlp = spacy.blank('xx')
            self.nlp.add_pipe('yasbd', first=True, config={'lang': 'auto'})
        else: self.nlp = nlp

        self.input_tokens = self.tokenizer(
            self.input_text,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        # Store the mappings of tokens to character positions
        self.tokens_char_bounds = self.input_tokens.pop('offset_mapping')[0]

        self.__chunk_by_sentences()

        if self.window_offset == 0:
            self.__embed_blind()
        elif self.window_offset >= 1:
            self.__embed_sliding()
        else:
            self.__embed_late()

    """
    Splits self.input_text into sentences and tags their token positions using self.tokenizer.
    Initializes self.chunks with each sentence and the sentence's char and token bounds.
    """
    def __chunk_by_sentences(self):
        try:
            nlp_output = self.nlp(self.input_text)
        except:
            # If input_text language cannot be determined, default to Spanish
            self.nlp.remove_pipe('yasbd')
            self.nlp.add_pipe('yasbd', first=True, config={'lang': 'es'})

            nlp_output = self.nlp(self.input_text)

            # Switch back to auto-detecting the language
            self.nlp.remove_pipe('yasbd')
            self.nlp.add_pipe('yasbd', first=True, config={'lang': 'auto'})

        sentences = nlp_output.sents

        for sent in sentences:
            sent_start = sent.start_char
            sent_end = sent.end_char

            # Ex) tokens_char_bounds[0] = (char_start, char_end + 1) for token 0
            for token_pos, (token_char_start, token_char_end) in enumerate(self.tokens_char_bounds):
                # If starting sent char is within current token's chars:
                if token_char_start <= sent_start <= token_char_end and token_char_end > token_char_start:
                    starting_token_pos = token_pos

                # If ending sent char is within current token's chars:
                if token_char_start <= sent_end <= token_char_end and token_char_end > token_char_start:
                    ending_token_pos = token_pos
                    self.chunks.append(Chunk(
                        text=sent.text,
                        char_bounds=(sent_start, sent_end),
                        token_bounds=(starting_token_pos, ending_token_pos + 1)
                    ))

    """
    Naively embeds every chunk independently of any other chunk.
    """
    def __embed_blind(self):
        self.model.max_seq_length = self.max_chunk_size
        embeddings = self.model.encode([chunk.text for chunk in self.chunks], max_length=self.max_chunk_size)

        # Store each chunk's associated tokens
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i]

            # Store non-chunk [CLS] starting token if first chunk, else normal
            token_idx_start = 0 if i == 0 else chunk.token_bounds[0]

            token_idx_end = min(
                # Store non-chunk [SEP] ending token if last chunk, else normal
                chunk.token_bounds[1] + (i == len(self.chunks) - 1),
                # Clamp right_token_idx to maximum allowed size
                token_idx_start + self.max_chunk_size
            )

            batch = {
                k: v[:, token_idx_start:token_idx_end]
                for k, v in self.input_tokens.items()
            }
            chunk.tokens = batch['input_ids']


    """
    Embeds each chunk on a "sliding" basis; each chunk receives context from n previous and next sentences.
    n is determined by window_chunk_offset.
    """
    def __embed_sliding(self):
        for chunk_index, chunk in enumerate(self.chunks):
            # Determine leftmost token position of current window
            if chunk_index == 0:
                # Grab the starting token [CLS], which is not part of a sentence
                window_start = 0
            else:
                # Leftmost token position of starting chunk
                window_start = self.chunks[
                    max(chunk_index - self.window_offset, 0)
                ].token_bounds[0]

            # Determine rightmost token position of current window
            if chunk_index == len(self.chunks) - 1:
                # Grab the ending token [SEP], which is not part of a sentence
                window_end = chunk.token_bounds[1] + 1
            else:
                # Rightmost token position of ending chunk, +1
                window_end = self.chunks[
                    min(chunk_index + self.window_offset, len(self.chunks) - 1)
                ].token_bounds[1]

            # If window exceeds maximum size, truncate to maximum size without sliding
            if window_end - window_start > self.max_chunk_size:
                if chunk_index != 0:
                    window_start = chunk.token_bounds[0]
                window_end = min(chunk.token_bounds[1], window_start + self.max_chunk_size)

            # input_tokens is sliced in accordance with window_start and window_end above
            batch_inputs = {k: v[:, window_start:window_end] for k, v in self.input_tokens.items()}
            chunk.tokens = batch_inputs['input_ids']
            
            with torch.no_grad():
                model_output = self.model(**batch_inputs)

            chunk_start = chunk.token_bounds[0]
            chunk_end = chunk.token_bounds[1] + 1 # +1 for slicing

            relative_chunk_start = chunk_start - window_start
            relative_chunk_end = chunk_end - window_start

            # Append only the (averaged) embeddings that belong to current chunk
            chunk.embedding = (
                (model_output[0][:, relative_chunk_start:relative_chunk_end][0].sum(dim=0) / (relative_chunk_end - relative_chunk_start)).detach().cpu().numpy()
            )

    """
    Embeds the entire document to contextually inform each chunk embedding. Adapted from:
    https://colab.research.google.com/drive/15vNZb6AsU7byjYoaEtXuNu567JWNzXOz?usp=sharing

    Not for use on large input texts with token counts beyond the model limits.
    """
    def __embed_late(self):
        with torch.no_grad():
            model_output = self.model(**self.input_tokens)

        token_embeddings = model_output[0]
        span_annotation = [[chunk.token_bounds for chunk in self.chunks]]

        for embeddings, annotations in zip(token_embeddings, span_annotation):
            if (
                self.max_chunk_size is not None
            ):  # Remove annotations which go beyond the max-length of the model
                annotations = [
                    (start, min(end, self.max_chunk_size - 1))
                    for (start, end) in annotations
                    if start < (self.max_chunk_size - 1)
                ]
            pooled_embeddings = [
                embeddings[start:end].sum(dim=0) / max((end - start), 1)
                for start, end in annotations
            ]

            for i, chunk in enumerate(self.chunks):
                chunk.embedding = pooled_embeddings[i].detach().cpu().numpy()
                token_idx_start = 0 if i == 0 else chunk.token_bounds[0]
                token_idx_end = chunk.token_bounds[1] + (i == len(self.chunks) - 1)
                batch = {
                    k: v[:, token_idx_start:token_idx_end]
                    for k, v in self.input_tokens.items()
                }
                chunk.tokens = batch['input_ids']

    """
    Utility method for calculating cosine similarity of two strings.
    """
    @staticmethod
    def __cosine_similarity(term_1, term_2):
        return np.dot(term_1, term_2) / (np.linalg.norm(term_1) * np.linalg.norm(term_2))

    """
    Returns a sorted list of chunks in order of decreasing similarity to a search term.

    Args:
        search_term (str): The searched text.
        rerank_top_k (int): Optional. The number of top-k results to rerank and return.
        query_prefix (str): Optional. A prefix to attach to the search_term for reranker prompt engineering.

    Returns:
        (List[SearchResult]) ranked list of search results. Only returns top k if rerank_top_k > 0.
    """
    def get_ranked_similarities(self, search_term, rerank_top_k=0, query_prefix = ''):
        search_results = []
        search_embedding = self.model.encode(search_term)

        for chunk in self.chunks:
            cos_sim = self.__cosine_similarity(search_embedding, chunk.embedding)
            search_results.append(SearchResult(chunk=chunk, score=cos_sim))

        search_results = sorted(search_results, key=lambda r: r.score, reverse=True)

        if rerank_top_k > 0:
            # Format search lists in expected form [[query, chunk], [query, chunk]...]
            sentence_pairs = [
                [query_prefix + search_term, result.chunk.text]
                for result in search_results[:rerank_top_k]
            ]

            rerank_scores = self.reranker.compute_score(sentence_pairs, max_length=1024)

            # Rebuild search results with new reranked scores
            search_results =  [
                SearchResult(chunk=search_results[i].chunk, score=score)
                for i, score in enumerate(rerank_scores)
            ]

            # Rerank search results by new scores
            return sorted(search_results, key=lambda r: r.score, reverse=True)

        return search_results