import numpy as np
from typing import List, Tuple, Union
from spacy.tokens import Doc, Span
from collections import defaultdict


def Rake(doc: Union[Doc, Span]) -> List[Tuple[Span, float]]:
    """
    RAKE (Rapid Automatic Keyphrase Extraction) implemention extension for Spacy.

    Tweaks to original algorithm
    - Exclude number like key phrases
    - Assign word scores based on its lemma (POS tag required)
    - Calculate score based on mean word score vs sum of word scores

    Args:
        doc: spaCy Doc or Span object
    Returns:
        List of (Span, score) tuples
    """
    phrases = []
    assert isinstance(doc, Doc) or isinstance(
        doc, Span
    ), "Input must be a spaCy Doc or Span object"
    assert (
        doc.is_sentenced
    ), "Input must be sentenced, try using the default spaCy pipelines to process your text"
    assert (
        doc.is_tagged
    ), "Input must be tagged, try using the default spaCy pipelines to process your text"
    for line in doc.sents:
        last_index = 0
        for index, token in enumerate(line):
            if not token.is_stop and token.text not in [",", ".", "!", "?", ":", ";"]:
                continue
            else:
                if index != last_index:
                    phrase_candidate = Span(
                        doc, line.start + last_index, line.start + index
                    )
                    ##Do not include phrases that are entirely composed of number/punctuation like tokens
                    if not all(
                        [
                            token.is_punct or token.like_num or token.is_space
                            for token in phrase_candidate
                        ]
                    ):
                        phrases.append(phrase_candidate)
                last_index = index + 1

    word_freq = defaultdict(int)
    word_degree = defaultdict(int)
    word_score = defaultdict(float)

    for phrase in phrases:
        for word in phrase:
            ##Do not assign scores to punctuation-like tokens
            if not word.is_punct:
                word_freq[word.lemma] += 1
                word_degree[word.lemma] += len(phrase)

    for word_lemma, freq in word_freq.items():
        degree = word_degree[word_lemma]
        score = (1.0 * degree) / (1.0 * freq)
        word_score[word_lemma] = score

    def score_phrase(phrase, word_score):
        ##Original algorithm takes sum of word scores, biased to longer phrases?
        ##Here, average is taken, favoring longer phrases only if they comprise of more "important" words
        return sum([word_score[word.lemma] for word in phrase]) / len(phrase)

    phrase_scores = [score_phrase(phrase, word_score) for phrase in phrases]
    sort_index = np.argsort(phrase_scores)[::-1]
    return [(phrases[i], phrase_scores[i]) for i in sort_index]
