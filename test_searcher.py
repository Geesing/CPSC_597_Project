from contextlib import suppress
import unittest

import spacy
import torch
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from yasbd import register_spacy_component

from searcher import Searcher

tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-es', trust_remote_code=True, local_files_only=True)

model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-es', trust_remote_code=True, local_files_only=True)
model.to('cpu')
model.eval()

reranker = AutoModelForSequenceClassification.from_pretrained(
    'jinaai/jina-reranker-v2-base-multilingual',
    dtype="auto",
    trust_remote_code=True,
)
reranker.to('cpu')
reranker.eval()

register_spacy_component()
spacy_model = spacy.blank('xx')
spacy_model.add_pipe('yasbd', first=True, config={'lang': 'auto'})

class TestSearcher(unittest.TestCase):

    # ==========================================
    # Tests for expected embedding shape
    # ==========================================
    def test_embedding_shape_window_offset_0(self):
        input_text = 'Sit on the perch and tell the others what to do. Slide the box into that empty space. A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire. The plush chair leaned against the wall. The case was puzzling to the old and wise. The cloud moved in a stately way and was gone. Madam, this is the best brand of corn. Oats are a food eaten by horse and man. A cruise in warm waters in a sleek yacht is fun.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=0
        )
        base_embedding = model.encode('')

        for chunk in searcher.chunks:
            self.assertEqual(chunk.embedding.shape, base_embedding.shape)

    def test_embedding_shape_window_offset_1(self):
        input_text = 'Sit on the perch and tell the others what to do. Slide the box into that empty space. A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire. The plush chair leaned against the wall. The case was puzzling to the old and wise. The cloud moved in a stately way and was gone. Madam, this is the best brand of corn. Oats are a food eaten by horse and man. A cruise in warm waters in a sleek yacht is fun.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=1
        )
        base_embedding = model.encode('')

        for chunk in searcher.chunks:
            self.assertEqual(chunk.embedding.shape, base_embedding.shape)

    def test_embedding_shape_window_offset_2(self):
        input_text = 'Sit on the perch and tell the others what to do. Slide the box into that empty space. A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire. The plush chair leaned against the wall. The case was puzzling to the old and wise. The cloud moved in a stately way and was gone. Madam, this is the best brand of corn. Oats are a food eaten by horse and man. A cruise in warm waters in a sleek yacht is fun.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=2
        )
        base_embedding = model.encode('')

        for chunk in searcher.chunks:
            self.assertEqual(chunk.embedding.shape, base_embedding.shape)

    def test_embedding_shape_window_offset_neg_1(self):
        input_text = 'Sit on the perch and tell the others what to do. Slide the box into that empty space. A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire. The plush chair leaned against the wall. The case was puzzling to the old and wise. The cloud moved in a stately way and was gone. Madam, this is the best brand of corn. Oats are a food eaten by horse and man. A cruise in warm waters in a sleek yacht is fun.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=-1
        )
        base_embedding = model.encode('')

        for chunk in searcher.chunks:
            self.assertEqual(chunk.embedding.shape, base_embedding.shape)



    # ==========================================
    # Tests for trivial exact matches
    # ==========================================
    def test_short_exact_similarity_window_offset_0(self):
        input_text = 'Hi'

        search_term = 'Hi'
    
        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=0
        )
        search_results = searcher.get_ranked_similarities(search_term)

        self.assertEqual(len(search_results), 1)

        self.assertAlmostEqual(
            float(search_results[0].score), 1.0, places=5
        )

    def test_long_exact_similarity_window_offset_0(self):
        input_text = 'The quick brown fox jumps over the lazy dog.'

        search_term = 'The quick brown fox jumps over the lazy dog.'
    
        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=0
        )
        search_results = searcher.get_ranked_similarities(search_term)

        self.assertEqual(len(search_results), 1)

        self.assertAlmostEqual(
            float(search_results[0].score), 1.0, places=5
        )



    # ==========================================
    # Tests for sentence splitting
    # ==========================================
    def test_sentence_splits_es(self):
        input_text = 'Las regulaciones describen una variedad de circunstancias. Por ejemplo, la presencia de material prohibido en la bolsa resultará en una multa de $125. ¡Qué duro! ¿Has visto algo así?'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=0
        )

        chunks = searcher.chunks

        self.assertEqual(len(searcher.chunks), 4)
        self.assertEqual(chunks[0].text, 'Las regulaciones describen una variedad de circunstancias.')
        self.assertEqual(chunks[1].text, 'Por ejemplo, la presencia de material prohibido en la bolsa resultará en una multa de $125.')
        self.assertEqual(chunks[2].text, '¡Qué duro!')
        self.assertEqual(chunks[3].text, '¿Has visto algo así?')

    def test_sentence_splits_en(self):
        input_text = 'The regulations describe a variety of circumstances. For example, the presence of prohibited material in your purse will result in a fine of $125. How harsh! Have you seen anything like that?'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=0
        )

        chunks = searcher.chunks

        self.assertEqual(len(searcher.chunks), 4)
        self.assertEqual(chunks[0].text, 'The regulations describe a variety of circumstances.')
        self.assertEqual(chunks[1].text, 'For example, the presence of prohibited material in your purse will result in a fine of $125.')
        self.assertEqual(chunks[2].text, 'How harsh!')
        self.assertEqual(chunks[3].text, 'Have you seen anything like that?')



    # ==========================================
    # Tests for 1-sentence tokenization
    # ==========================================
    def test_1_sent_tokenization_window_offset_0(self):
        input_text = 'The quick brown fox jumps over the lazy dog.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=0
        )

        self.assertEqual(len(searcher.chunks), 1)

        expected_tokens = tokenizer(text=input_text, return_tensors='pt')['input_ids'][0].tolist()
        custom_tokens = searcher.chunks[0].tokens[0].tolist()

        self.assertEqual(expected_tokens, custom_tokens,
                         'Mismatch in tokens between window_offset=0 and raw tokenizer')

    def test_1_sent_tokenization_window_offset_1(self):
        input_text = 'The quick brown fox jumps over the lazy dog.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=1
        )

        self.assertEqual(len(searcher.chunks), 1)

        expected_tokens = tokenizer(text=input_text, return_tensors='pt')['input_ids'][0].tolist()
        custom_tokens = searcher.chunks[0].tokens[0].tolist()

        self.assertEqual(expected_tokens, custom_tokens,
                         'Mismatch in tokens between window_offset=1 and raw tokenizer')

    def test_1_sent_tokenization_window_offset_2(self):
        input_text = 'The quick brown fox jumps over the lazy dog.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=2
        )

        self.assertEqual(len(searcher.chunks), 1)

        expected_tokens = tokenizer(text=input_text, return_tensors='pt')['input_ids'][0].tolist()
        custom_tokens = searcher.chunks[0].tokens[0].tolist()

        self.assertEqual(expected_tokens, custom_tokens,
                         'Mismatch in tokens between window_offset=2 and raw tokenizer')

    def test_1_sent_tokenization_window_offset_neg_1(self):
        input_text = 'The quick brown fox jumps over the lazy dog.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=-1
        )

        self.assertEqual(len(searcher.chunks), 1)

        expected_tokens = tokenizer(text=input_text, return_tensors='pt')['input_ids'][0].tolist()
        custom_tokens = searcher.chunks[0].tokens[0].tolist()

        self.assertEqual(expected_tokens, custom_tokens,
                         'Mismatch in tokens between window_offset=-1 and raw tokenizer')



    # ==========================================
    # Tests for max_chunk_size
    # ==========================================
    def test_max_chunk_size_5_window_offset_0(self):
        input_text = 'The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.'
        sent_1 = 'The quick brown fox jumps over the lazy dog.'
        sent_2 = ' Pack my box with five dozen liquor jugs.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=5,
            reranker=reranker,
            window_offset=0
        )

        # expected_tokens are sliced to first 5 only to match max_chunk_size
        expected_tokens_sent_1 = tokenizer(text=sent_1, return_tensors='pt')['input_ids'][0].tolist()[:5]
        custom_tokens_sent_1 = searcher.chunks[0].tokens[0].tolist()

        expected_tokens_sent_2 = tokenizer(text=sent_2, return_tensors='pt', add_special_tokens=False)['input_ids'][0].tolist()[:5]
        custom_tokens_sent_2 = searcher.chunks[1].tokens[0].tolist()

        # Trim filler tokens from first sentence tokens
        with suppress(ValueError):
            expected_tokens_sent_1.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            custom_tokens_sent_1.remove(tokenizer.cls_token_id)

        self.assertEqual(expected_tokens_sent_1, custom_tokens_sent_1,
                         'Mismatch in tokens between window_offset=0 and raw tokenizer')
        self.assertEqual(expected_tokens_sent_2, custom_tokens_sent_2,
                         'Mismatch in tokens between window_offset=0 and raw tokenizer')  

    def test_max_chunk_size_5_window_offset_1(self):
        input_text = 'The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.'
        sent_1 = 'The quick brown fox jumps over the lazy dog.'
        sent_2 = ' Pack my box with five dozen liquor jugs.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=5,
            reranker=reranker,
            window_offset=1
        )

        # expected_tokens are sliced to first 5 only to match max_chunk_size
        expected_tokens_sent_1 = tokenizer(text=sent_1, return_tensors='pt')['input_ids'][0].tolist()[:5]
        custom_tokens_sent_1 = searcher.chunks[0].tokens[0].tolist()

        expected_tokens_sent_2 = tokenizer(text=sent_2, return_tensors='pt', add_special_tokens=False)['input_ids'][0].tolist()[:5]
        custom_tokens_sent_2 = searcher.chunks[1].tokens[0].tolist()

        # Trim filler tokens from first sentence tokens
        with suppress(ValueError):
            expected_tokens_sent_1.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            custom_tokens_sent_1.remove(tokenizer.cls_token_id)

        self.assertEqual(expected_tokens_sent_1, custom_tokens_sent_1,
                         'Mismatch in tokens between window_offset=1 and raw tokenizer')
        self.assertEqual(expected_tokens_sent_2, custom_tokens_sent_2,
                         'Mismatch in tokens between window_offset=1 and raw tokenizer')  

    def test_max_chunk_size_5_window_offset_2(self):
        input_text = 'The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.'
        sent_1 = 'The quick brown fox jumps over the lazy dog.'
        sent_2 = ' Pack my box with five dozen liquor jugs.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=5,
            reranker=reranker,
            window_offset=2
        )

        # expected_tokens are sliced to first 5 only to match max_chunk_size
        expected_tokens_sent_1 = tokenizer(text=sent_1, return_tensors='pt')['input_ids'][0].tolist()[:5]
        custom_tokens_sent_1 = searcher.chunks[0].tokens[0].tolist()

        expected_tokens_sent_2 = tokenizer(text=sent_2, return_tensors='pt', add_special_tokens=False)['input_ids'][0].tolist()[:5]
        custom_tokens_sent_2 = searcher.chunks[1].tokens[0].tolist()

        # Trim filler tokens from first sentence tokens
        with suppress(ValueError):
            expected_tokens_sent_1.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            custom_tokens_sent_1.remove(tokenizer.cls_token_id)
            
        self.assertEqual(expected_tokens_sent_1, custom_tokens_sent_1,
                         'Mismatch in tokens between window_offset=2 and raw tokenizer')
        self.assertEqual(expected_tokens_sent_2, custom_tokens_sent_2,
                         'Mismatch in tokens between window_offset=2 and raw tokenizer')  



    # ==========================================
    # Tests for truncating edge chunks
    # ==========================================
    def test_edge_chunks_window_offset_1(self):
        input_text = 'Sit on the perch and tell the others what to do. Slide the box into that empty space. A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire. The plush chair leaned against the wall.'
        first_chunk = 'Sit on the perch and tell the others what to do. Slide the box into that empty space.'
        last_chunk = ' Corn cobs can be used to kindle a fire. The plush chair leaned against the wall.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=1
        )

        expected_tokens_first = tokenizer(text=first_chunk, return_tensors='pt')['input_ids'][0].tolist()
        custom_tokens_first = searcher.chunks[0].tokens[0].tolist()

        expected_tokens_last = tokenizer(text=last_chunk, return_tensors='pt', add_special_tokens=False)['input_ids'][0].tolist()
        custom_tokens_last = searcher.chunks[4].tokens[0].tolist()

        # Trim filler tokens
        with suppress(ValueError):
            expected_tokens_first.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            expected_tokens_first.remove(tokenizer.sep_token_id)
        with suppress(ValueError):
            expected_tokens_last.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            expected_tokens_last.remove(tokenizer.sep_token_id)
        with suppress(ValueError):
            custom_tokens_first.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            custom_tokens_first.remove(tokenizer.sep_token_id)
        with suppress(ValueError):
            custom_tokens_last.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            custom_tokens_last.remove(tokenizer.sep_token_id)

        self.assertEqual(expected_tokens_first, custom_tokens_first,
                         'Mismatch in tokens between window_offset=1 and raw tokenizer')
        self.assertEqual(expected_tokens_last, custom_tokens_last,
                         'Mismatch in tokens between window_offset=1 and raw tokenizer')  

    def test_edge_chunks_window_offset_2(self):
        input_text = 'Sit on the perch and tell the others what to do. Slide the box into that empty space. A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire. The plush chair leaned against the wall.'
        second_chunk = 'Sit on the perch and tell the others what to do. Slide the box into that empty space. A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire.'
        penult_chunk = ' Slide the box into that empty space. A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire. The plush chair leaned against the wall.'

        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=2
        )

        expected_tokens_second= tokenizer(text=second_chunk, return_tensors='pt')['input_ids'][0].tolist()
        custom_tokens_second = searcher.chunks[1].tokens[0].tolist()

        expected_tokens_penult = tokenizer(text=penult_chunk, return_tensors='pt', add_special_tokens=False)['input_ids'][0].tolist()
        custom_tokens_penult = searcher.chunks[3].tokens[0].tolist()

        # Trim filler tokens
        with suppress(ValueError):
            expected_tokens_second.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            expected_tokens_second.remove(tokenizer.sep_token_id)
        with suppress(ValueError):
            expected_tokens_penult.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            expected_tokens_penult.remove(tokenizer.sep_token_id)
        with suppress(ValueError):
            custom_tokens_second.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            custom_tokens_second.remove(tokenizer.sep_token_id)
        with suppress(ValueError):
            custom_tokens_penult.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            custom_tokens_penult.remove(tokenizer.sep_token_id)
            
        self.assertEqual(expected_tokens_second, custom_tokens_second,
                         'Mismatch in tokens between window_offset=2 and raw tokenizer')
        self.assertEqual(expected_tokens_penult, custom_tokens_penult,
                         'Mismatch in tokens between window_offset=2 and raw tokenizer')  



    # ==========================================
    # Tests for chunk splits
    # ==========================================
    def test_chunk_split_window_offset_0(self):
        input_text = 'Sit on the perch and tell the others what to do. Slide the box into that empty space. A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire. The plush chair leaned against the wall. The case was puzzling to the old and wise. The cloud moved in a stately way and was gone. Madam, this is the best brand of corn. Oats are a food eaten by horse and man. A cruise in warm waters in a sleek yacht is fun.'

        expected_tokens = tokenizer(text=input_text, return_tensors='pt')['input_ids'][0].tolist()
        
        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=0
        )

        custom_tokens = torch.empty((1, 0), dtype=torch.int)

        for chunk in searcher.chunks:
            custom_tokens = torch.cat((custom_tokens, chunk.tokens), dim=1)

        custom_tokens = custom_tokens[0].tolist()

        self.assertEqual(expected_tokens, custom_tokens,
                         'Mismatch in tokens between window_offset=0 and raw tokenizer')

    def test_chunk_split_window_offset_1(self):
        input_text = 'Sit on the perch and tell the others what to do. Slide the box into that empty space. A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire. The plush chair leaned against the wall. The case was puzzling to the old and wise. The cloud moved in a stately way and was gone. Madam, this is the best brand of corn. Oats are a food eaten by horse and man. A cruise in warm waters in a sleek yacht is fun.'

        # Sentences 2 to 4 of input_text (starting from 0)
        input_excerpt = ' A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire. The plush chair leaned against the wall.'

        expected_tokens = tokenizer(text=input_excerpt, return_tensors='pt')['input_ids'][0].tolist()
        
        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=1
        )

        # Tokens for sentences 2 to 4 (chunk 3's tokens)
        custom_tokens = searcher.chunks[3].tokens[0].tolist()

        # Trim filler tokens
        with suppress(ValueError):
            expected_tokens.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            expected_tokens.remove(tokenizer.sep_token_id)
        with suppress(ValueError):
            custom_tokens.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            custom_tokens.remove(tokenizer.sep_token_id)

        self.assertEqual(expected_tokens, custom_tokens,
                         'Mismatch in tokens between window_offset=1 and raw tokenizer')

    def test_chunk_split_window_offset_2(self):
        input_text = 'Sit on the perch and tell the others what to do. Slide the box into that empty space. A brown leather bag hung from its strap. Corn cobs can be used to kindle a fire. The plush chair leaned against the wall. The case was puzzling to the old and wise. The cloud moved in a stately way and was gone. Madam, this is the best brand of corn. Oats are a food eaten by horse and man. A cruise in warm waters in a sleek yacht is fun.'

       # Sentences 4 to 8 of input_text (starting from 0)
        input_excerpt = ' The plush chair leaned against the wall. The case was puzzling to the old and wise. The cloud moved in a stately way and was gone. Madam, this is the best brand of corn. Oats are a food eaten by horse and man.'

        expected_tokens = tokenizer(text=input_excerpt, return_tensors='pt')['input_ids'][0].tolist()
        
        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=2
        )

        # Tokens for sentences 4 to 8 (chunk 6's tokens)
        custom_tokens = searcher.chunks[6].tokens[0].tolist()

        # Trim filler tokens
        with suppress(ValueError):
            expected_tokens.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            expected_tokens.remove(tokenizer.sep_token_id)
        with suppress(ValueError):
            custom_tokens.remove(tokenizer.cls_token_id)
        with suppress(ValueError):
            custom_tokens.remove(tokenizer.sep_token_id)

        self.assertEqual(expected_tokens, custom_tokens,
                            'Mismatch in tokens between window_offset=1 and raw tokenizer')



    # ==========================================
    # Tests for bad input_text
    # ==========================================
    def test_spaces_input_text(self):
        searcher = Searcher(
            input_text='    ',
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=0
        )

        empty_token_char_bounds = torch.tensor([[0, 0], [0, 0]])

        # Blank string should be tokenized into start marker tokenizer.cls_token_id and end marker tokenizer.sep_token_id
        self.assertEqual(
            searcher.input_tokens['input_ids'][0].tolist(),
            [tokenizer.cls_token_id, tokenizer.sep_token_id]
        )

        # Character boundaries for empty text's start and end markers should be (0, 0)
        empty_token_char_bounds = torch.tensor([[0, 0], [0, 0]])
        self.assertTrue(torch.equal(searcher.tokens_char_bounds, empty_token_char_bounds))

        self.assertEqual(searcher.chunks, [])

    def test_empty_input_text(self):
        searcher = Searcher(
            input_text='    ',
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=0
        )

        # Blank string should be tokenized into start marker tokenizer.cls_token_id and end marker tokenizer.sep_token_id
        self.assertEqual(
            searcher.input_tokens['input_ids'][0].tolist(),
            [tokenizer.cls_token_id, tokenizer.sep_token_id]
        )

        # Character boundaries for empty text's start and end markers should be (0, 0)
        empty_token_char_bounds = torch.tensor([[0, 0], [0, 0]])
        self.assertTrue(torch.equal(searcher.tokens_char_bounds, empty_token_char_bounds))

        self.assertEqual(searcher.chunks, [])

    def test_gibberish_input_text(self):
        input_text = 'Adnfjs. Adsflkjewklasd! Ssjfiowejoe? SDjfiosdjfods.'
        searcher = Searcher(
            input_text=input_text,
            nlp=spacy_model,
            tokenizer=tokenizer,
            model=model,
            max_chunk_size=8192,
            reranker=reranker,
            window_offset=0
        )

        chunks = searcher.chunks

        self.assertEqual(chunks[0].text, 'Adnfjs.')
        self.assertEqual(chunks[1].text, 'Adsflkjewklasd!')
        self.assertEqual(chunks[2].text, 'Ssjfiowejoe?')
        self.assertEqual(chunks[3].text, 'SDjfiosdjfods.')


if __name__ == '__main__':
    unittest.main()