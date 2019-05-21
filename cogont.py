#from sklearn import decomposition, metrics, cluster
import string, nltk
import numpy as np
import pandas as pd
import gensim as gs
from sklearn import metrics


def doc_to_stcs(df_doc):
    """
    Converts a dataframe table of documents into a list of strings.
    """
    doc_stcs = []
    for r_i, row in df_doc.iterrows():
        doc_stcs.append(row['Title'])
        # check if the abstract is empty
        if type(row['Abstract']) is str:
            doc_stcs = doc_stcs + nltk.sent_tokenize(row['Abstract'])

    puncs = set(string.punctuation) - {'-', '\''} # don't discard dashes or '
    doc_stcs = [''.join(ch for ch in s if ch not in puncs).lower().split() for s in doc_stcs if s is not '']
    return doc_stcs

def concat_terms(doc_stcs, terms, merge_char='-'):
    """
    Takes a sentence list (prepped for word2vec) and find multi-word phrases from
    a dictionary (terms) and merge them into single terms
    """
    doc_stcs_out = [s for s in doc_stcs]
    terms_out = [t for t in terms]

    # takes multi-word terms and merge them into phrases, and change the doc_stcs_out accordingly
    for t_i, t in enumerate(terms):
        t_split = t.split(sep=' ')
        # if it's a multi-word term
        if len(t_split)>1:
            terms_out[t_i] = merge_char.join(t_split)
            #print(terms_out[t_i])
            for s in doc_stcs_out:
                # go through all the sentences to find and combine that phrase
                for i,w in enumerate(s):
                    if w == t_split[0] and s[i:i+len(t_split)]==t_split:
                        s[i:i+len(t_split)] = [merge_char.join(t_split)]

    return doc_stcs_out, terms_out

def print_similarity(similarity_list):
    """
    Just a helper function to pretty print the similarity list.
    """
    print('----------------\n Similiarty\n---------------')
    for w in similarity_list:
        print('%.4f: '%w[1] + w[0])

def compute_similarity(query_vec, vecs, sim_func=None):
    """
    Computes the similarity between a query vector and a set of vectors, using
    a given similarity metric function. Defaults to cosine.
    """
    if sim_func == None:
        # defaults to cosine similarity
        sim_func = metrics.pairwise.cosine_similarity

    similarity = sim_func(query_vec.astype('float64').reshape(1,-1), vecs.astype('float64')).squeeze()
    return similarity

def sort_similarity(similarity):
    """
    Sort similarity and return the sorted index and similarity
    """
    # sort by descending (most similar first)
    sorted_inds = np.argsort(similarity)[::-1]
    return sorted_inds, similarity[sorted_inds]

def return_subset_inds(terms_all, terms_subset, pad_missing=False):
    """
    Returns the indices of the subset of terms from the full corpus terms.
    """
    if pad_missing:
        subset_inds = [terms_all.index(t) if t in terms_all else np.nan for t in terms_subset]
    else:
        subset_inds = [terms_all.index(t) for t in terms_subset if t in terms_all]
    return subset_inds

def most_similar_subset(terms_all, terms_subset, vecs, positive, negative=[], topn=10):
    """
    Performs similarity query, given positive and negative terms, on
    a subset of all the terms in the corpus.
    """
    query_words = positive+negative

    # do the query vector computation in one go with some fancy matrix multiply
    weights = np.array([1]*len(positive)+[-1]*len(negative)) # normalize
    query_inds = [terms_all.index(w) if w in terms_all else np.nan for w in query_words]
    good_inds, nan_inds = np.where(~np.isnan(query_inds))[0], np.where(np.isnan(query_inds))[0]
    if len(nan_inds):
        print([query_words[i] for i in nan_inds], ' not in vocabulary. Dropped.')
        query_inds = np.array(query_inds)[good_inds].astype(int)
        weights = weights[good_inds]

    query_vec = np.dot(weights,vecs[query_inds,:])

    # compute similarity between query vector and subset of vectors
    # get indices of terms_subset in terms_all
    subset_inds = return_subset_inds(terms_all, terms_subset)
    sim_inds, similarity = sort_similarity(compute_similarity(query_vec,vecs[subset_inds]))
    return [(terms_all[subset_inds[w_ind]], similarity[i]) for i, w_ind in enumerate(sim_inds[:topn])]


#
# def corpus_from_list(doc_list, terms):
#     """ Transforms a list of strings where each string is a document,
#     into gensim friendly corpus form.
#
#     Parameters
#     ----------
#     doc_list : list
#         List of strings, where string is one document.
#     terms: list or array
#         List of terms to use/to be indexed.
#
#     Returns
#     -------
#     corpus_index: list
#         term index and number of times it occurs in each document
#
#     corpus_terms: list
#         terms (in string) that exists in each document
#
#     term_dict: dict
#         term:index pairing, just for gensim accessibility
#     """
#     # initialize corpus index and terms
#     corpus_index, corpus_terms = [None]*len(doc_list), [None]*len(doc_list)
#
#     # create term:index dictionary
#     term_dict = dict(zip(terms, range(len(terms))))
#
#     # iterate through corpus documents
#     for doc_ind, doc in enumerate(doc_list):
#         cur_corp_index = []
#         cur_corp_terms = []
#
#         # loop through all terms
#         for term in terms:
#             # use all lower case to search
#             t_count = doc.lower().count(term)
#             if t_count:
#                 # if a term exists, append it and its occurence to the list
#                 cur_corp_index.append((term_dict[term], t_count))
#                 cur_corp_terms.append(term)
#
#         corpus_index[doc_ind] = cur_corp_index
#         corpus_terms[doc_ind] = cur_corp_terms
#
#     return corpus_index, corpus_terms, term_dict
#
# def df_to_doclist(df):
#     """ Returns a corpus of documents in list format from a dataframe.
#
#     Parameters
#     ----------
#     df : pandas dataframe
#         Documents in tabular format.
#         Must have columns Abstract and Title, even if one is always empty.
#
#     Returns
#     -------
#     doc_list: list
#         Corpus in list format.
#     """
#     # combine title & abstract of each document and put into list
#     doc_list = []
#     for r_i, row in df.iterrows():
#         # if either title or abstract is not a string, only append the other
#         # otherwise, combine into one document.
#         if type(row['Abstract']) is not str:
#             doc_list.append(row['Title'])
#         elif type(row['Title']) is not str:
#             doc_list.append(row['Abstract'])
#         else:
#             doc_list.append(row['Title']+'. ' + row['Abstract'])
#
#     return doc_list
