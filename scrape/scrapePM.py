import numpy as np
from bs4 import BeautifulSoup

from erpsc.core.urls import URLS
from erpsc.core.requester import Requester


def scrape_terms(terms, base_phrase=None, fieldkey='ALL'):
    """
    scrape_terms(terms, base_phrase=None, fieldkey='ALL'):

    Takes in a list of terms, as well as a base phrase (appened to all search terms
    + AND) and searches PubMed for number of entries including the search terms. 

    parameters: 
    	terms: list of search terms
    	base_phrase: extra words to be appended to every term
    	fieldkey: PubMed fields to include in the search, default to 'ALL'
    		or use 'TIAB' for titles and abstracts only, go here:
            https://www.ncbi.nlm.nih.gov/books/NBK3827/
            Search Field Descriptions and Tags

	return: 
		term_counts: a dictionary where the key:value pair represents
			term:count
    """
    req = Requester()
    urls = URLS(db='pubmed', retmax='0', retmode='xml', field=fieldkey)
    urls.build_info(['db'])
    urls.build_search(['db', 'retmax', 'retmode', 'field'])

    # initialize term dictionary
    term_counts = dict()
    for term in terms:

        # Make URL - Exact Term Version, using double quotes
        url = urls.search + '"' + term + '"'

        # if there is a base phrase for cog/neuro specific searches, tack it on
        if base_phrase is not None:
            url = url + base_phrase

        # super hacky way to return just count, should really modify the URL lib
        url = url + "&rettype=count"

        # Pull the page, and parse with Beautiful Soup
        page = req.get_url(url)
        page_soup = BeautifulSoup(page.content, 'lxml')

        # Check whether the search term returned something
        if page_soup.find('quotedphrasenotfound') is None:
            # if the search returned somthing, find the first count
            # which returns how many papers found the term
            count = int(page_soup.find('count').text.strip())
        else:
            # search found nothing
            count = 0

        #print url
        print((term, count))

        # Add the total number of papers for term
        term_counts[term] = count

    req.close()
    return term_counts


def scrape_pairs(terms, base_phrase=None, fieldkey='ALL'):
    """
    scrape_pairs(terms, base_phrase=None, fieldkey='ALL'):

    Takes in a list of terms, as well as a base phrase (appened to all search terms
    + AND) and searches PubMed for counts of co-occurrences for all pairs of terms. 

    parameters: 
    	terms: list of search terms
    	base_phrase: extra words to be appended to every term
    	fieldkey: PubMed fields to include in the search, default to 'ALL'
    		or use 'TIAB' for titles and abstracts only, go here:
            https://www.ncbi.nlm.nih.gov/books/NBK3827/
            Search Field Descriptions and Tags

	return: 
		pair_wise: upper trianglular co-occurrence matrix, ordered by term
			appeareance order
    """
    print_verbose = False
    req = Requester()
    urls = URLS(db='pubmed', retmax='0', retmode='xml', field=fieldkey)
    urls.build_info(['db'])
    urls.build_search(['db', 'retmax', 'retmode', 'field'])

    # --- search pair occurrences ---
    pair_wise = np.zeros((len(terms), len(terms)))
    for ind1, term_1 in enumerate(terms):
        print term_1, (ind1 + 1), '/', len(terms)
        # skip all elements in lower half of matrix
        for ind2, term_2 in enumerate(terms[ind1:]):

            # Make URL - Exact Term Version, using double quotes
            url = urls.search + '"' + term_1 + '"AND"' + term_2 + '"'

            # if there is a base phrase for cog/neuro specific searches, tack it on
            if base_phrase is not None:
                url = url + base_phrase

            # super hacky way to return just count, should really modify the URL lib
            url = url + "&rettype=count"

            # Pull the page, and parse with Beautiful Soup
            #
            # sometimes this piece times out, probably because PM is disconnecting
            # if connection fails, retry (up to 10 times)
            tries = 10
            while tries > 0:
                try:
                    page = req.get_url(url)
                    tries = 0
                except:
                    print 'Connection timed out, retrying...'
                    tries -= 1
                    continue

            page_soup = BeautifulSoup(page.content, 'lxml')

            # Check whether the search term returned something
            if page_soup.find('quotedphrasenotfound') is None:
                # if the search returned somthing, find the first count
                # which returns how many papers found the term
                count = int(page_soup.find('count').text.strip())
            else:
                # search found nothing
                count = 0

            if print_verbose:
                print((term_1, term_2, count))
                print url
                print '--------------'

            # Add the total number of papers for term
            pair_wise[ind1, ind2] = count

    req.close()
    return pair_wise
