"""Classes and functions for Count analysis (key word co-occurences in papers)."""
from __future__ import print_function, division

import datetime
import numpy as np
from bs4 import BeautifulSoup

# Import custom code
from erpsc.base import Base
from erpsc.core.utils import comb_terms, extract
from erpsc.core.urls import URLS

#################################################################################################
#################################### ERPSC - COUNT - CLASSES ####################################
#################################################################################################


class Count(Base):
    """This is a class for counting co-occurence of pre-specified ERPs & terms.

    Attributes
    ----------
    erp_counts : 1d array
        Counts of how many articles found for each ERP word.
    term_counts : 1d array
        Counts of how many articles found for each term word.
    dat_numbers : 2d array
        How many papers found for each ERP / term combination.
    dat_percent : 2d array
        Percent of papers that with co-occuring ERP and term words.
    """

    def __init__(self):
        """Initialize ERP-SCANR Count() object."""

        # Inherit from the ERPSC base class
        Base.__init__(self)

        # Initialize vector of counts of number of papers for each term
        self.erp_counts = np.zeros(0)
        self.term_counts = np.zeros(0)

        # Initialize data output variables
        self.dat_numbers = np.zeros(0)
        self.dat_percent = np.zeros(0)

    def scrape_term(self, db=None, base_phrase=None):
        """Search through pubmed for all abstracts with term occurrence.        
        """

        # Set date of when data was scraped
        self.date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # Get e-utils URLS object. Set retmax as 0, since not using UIDs in this analysis
        urls = URLS(db=db, retmax='0', retmode='xml')
        urls.build_info(['db'])
        urls.build_search(['db', 'retmax', 'retmode'])

        # Get current information about database being used
        self.get_db_info(urls.info)

        # Initialize count variables to the correct length
        self.term_counts = np.zeros([self.n_terms])

        # Loop through each term
        for term_ind, term_ls in enumerate(self.terms):

            # Make URL - Exact Term Version, using double quotes
            url = urls.search + '"' + term_ls[0] + '"'
            # if there is a base phrase, tack it on
            if base_phrase != None:
                url = url + base_phrase

            #url = urls.search + comb_terms(erp_ls, 'or') + 'AND' + comb_terms(term_ls, 'or')

            # Make URL - Non-exact term version
            #url = self.eutils_search + erp + ' erp ' + term

            # Pull the page, and parse with Beautiful Soup
            page = self.req.get_url(url)
            page_soup = BeautifulSoup(page.content, 'lxml')

            # Get all 'count' tags
            #counts = page_soup.find_all('count')

            if page_soup.find('quotedphrasenotfound') is None:
                #counts = extract(page_soup, 'count', 'raw')
                count = int(page_soup.find('count').text.strip())
            else:
                count = 0

            print((term_ls, count))

            # Add the total number of papers for term            
            self.term_counts[term_ind] = count

        # Set Requester object as finished being used
        self.req.close()

    def scrape_pairs(self, db=None, base_phrase=None):
        """Search through pubmed for all abstracts with co-occurence of terms.

        The scraping does an exact word search for two terms (one ERP and one term)
        The HTML page returned by the pubmed search includes a 'count' field.
        This field contains the number of papers with both terms. This is extracted.
        """

        # Set date of when data was scraped
        self.date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # Get e-utils URLS object. Set retmax as 0, since not using UIDs in this analysis
        urls = URLS(db=db, retmax='0', retmode='xml')
        urls.build_info(['db'])
        urls.build_search(['db', 'retmax', 'retmode'])

        # Get current information about database being used
        self.get_db_info(urls.info)

        # Initialize right size matrices to store data
        self.dat_numbers = np.zeros([self.n_terms, self.n_terms])

        #outer loop the terms
        for term_ind, term_ls in enumerate(self.terms):
            print(term_ind, term_ls[0])
            #inner loop the terms
            for term_ind2, term_ls2 in enumerate(self.terms):
                #url = urls.search + comb_terms(erp_ls, 'or') + 'AND' + comb_terms(term_ls, 'or')
                if term_ind > term_ind2:
                    continue

                # Make URL - Exact Term Version, using double quotes
                url = urls.search + '"' + term_ls[0] + '"AND"' + term_ls2[
                    0] + '"'
                if base_phrase != None:
                    url = url + base_phrase

                # Make URL - Non-exact term version
                #url = self.eutils_search + erp + ' erp ' + term

                # Pull the page, and parse with Beautiful Soup
                page = self.req.get_url(url)
                page_soup = BeautifulSoup(page.content, 'lxml')

                # Get all 'count' tags
                #counts = page_soup.find_all('count')

                if page_soup.find('quotedphrasenotfound') is None:
                    #counts = extract(page_soup, 'count', 'raw')
                    count = int(page_soup.find('count').text.strip())
                else:
                    count = 0

                # Add the total number of papers for term            
                self.dat_numbers[term_ind, term_ind2] = count

            # Set Requester object as finished being used
            self.req.close()

    def scrape_data(self, db=None):
        """Search through pubmed for all abstracts with co-occurence of ERP & terms.

        The scraping does an exact word search for two terms (one ERP and one term)
        The HTML page returned by the pubmed search includes a 'count' field.
        This field contains the number of papers with both terms. This is extracted.
        """

        # Set date of when data was scraped
        self.date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # Get e-utils URLS object. Set retmax as 0, since not using UIDs in this analysis
        urls = URLS(db=db, retmax='0', retmode='xml')
        urls.build_info(['db'])
        urls.build_search(['db', 'retmax', 'retmode'])

        # Get current information about database being used
        self.get_db_info(urls.info)

        # Initialize count variables to the correct length
        self.term_counts = np.zeros([self.n_terms])
        self.erp_counts = np.zeros([self.n_erps])

        # Initialize right size matrices to store data
        self.dat_numbers = np.zeros([self.n_erps, self.n_terms])
        self.dat_percent = np.zeros([self.n_erps, self.n_terms])

        # Loop through each ERP term
        for erp_ls in self.erps:
            erp_ind = self.erps.index(erp_ls)
            print(erp_ind)
            # For each ERP, loop through each term term
            for term_ls in self.terms:

                # Get the indices of the current erp & term terms                
                term_ind = self.terms.index(term_ls)

                if erp_ind >= term_ind:
                    continue

                # Make URL - Exact Term Version, using double quotes
                url = urls.search + '"' + erp_ls[0] + '"AND"' + term_ls[
                    0] + '"'
                #print(term_ls)
                #print(url)

                #url = urls.search + comb_terms(erp_ls, 'or') + 'AND' + comb_terms(term_ls, 'or')

                # Make URL - Non-exact term version
                #url = self.eutils_search + erp + ' erp ' + term

                # Pull the page, and parse with Beautiful Soup
                page = self.req.get_url(url)
                page_soup = BeautifulSoup(page.content, 'lxml')

                # Get all 'count' tags
                #counts = page_soup.find_all('count')
                counts = extract(page_soup, 'count', 'all')
                print(counts)

                # Initialize empty temp vector to hold counts
                vec = []

                # Loop through counts, extracting into vec
                # There should be n+1 count fields, where n is the number of search terms
                #   The number of search terms includes all of them, including 'OR's & 'NOT's
                # Example: term=("N400"OR"N4")AND("language")NOT("cancer"OR"histone")
                #   Here there are 5 search terms, and so 6 count tags
                # The 1st count tag is the number of articles meeting the full search term
                #   Each subsequent count tag is each search term, in order.
                for count in counts:
                    vec.append(int(count.text))

                # Add the total number of papers for erp & term
                self.erp_counts[erp_ind] = vec[1]
                self.term_counts[term_ind] = vec[2]

                # Add the number & percent of overlapping papers
                self.dat_numbers[erp_ind, term_ind] = vec[0]
                self.dat_percent[erp_ind, term_ind] = vec[0] / vec[1]

        # Set Requester object as finished being used
        self.req.close()

    def check_cooc_erps(self):
        """"Prints out the terms most associatied with each ERP."""

        # Loop through each erp term, find maximally associated term term and print out
        for erp_ls in self.erps:

            # Find the index of the most common term for current erp
            erp_ind = self.erps.index(erp_ls)
            term_ind = np.argmax(self.dat_percent[erp_ind, :])

            # Print out the results
            print("For the  {:5} the most common association is \t {:10} with \t %{:05.2f}"
                  .format(erp_ls[0], self.terms[term_ind][0], \
                  self.dat_percent[erp_ind, term_ind]*100))

    def check_cooc_terms(self):
        """Prints out the ERP terms most associated with each term."""

        # Loop through each cig term, find maximally associated erp term and print out
        for term_ls in self.terms:

            # Find the index of the most common erp for current term
            term_ind = self.terms.index(term_ls)
            erp_ind = np.argmax(self.dat_percent[:, term_ind])

            # Print out the results
            print("For  {:12} the strongest associated ERP is \t {:5} with \t %{:05.2f}"
                  .format(term_ls[0], self.erps[erp_ind][0], \
                  self.dat_percent[erp_ind, term_ind]*100))

    def check_top(self):
        """Check the terms with the most papers."""

        # Find and print the erp term for which the most papers were found
        print("The most studied ERP is  {:6}  with {:8.0f} papers"
              .format(self.erps[np.argmax(self.erp_counts)][0], \
              self.erp_counts[np.argmax(self.erp_counts)]))

        # Find and print the term term for which the most papers were found
        print("The most studied term is  {:6}  with {:8.0f}  papers"
              .format(self.terms[np.argmax(self.term_counts)][0], \
              self.term_counts[np.argmax(self.term_counts)]))

    def check_counts(self, dat):
        """Check how many papers found for each term.

        Parameters
        ----------
        dat : {'erp', 'term'}
            Which data type to print out.
        """

        # Check counts for all ERP terms
        if dat is 'erp':
            for erp_ls in self.erps:
                erp_ind = self.erps.index(erp_ls)
                print('{:5} - {:8.0f}'.format(erp_ls[0], self.erp_counts[
                    erp_ind]))

        # Check counts for all term terms
        elif dat is 'term':
            for term_ls in self.terms:
                term_ind = self.terms.index(term_ls)
                print('{:18} - {:10.0f}'.format(term_ls[0], self.term_counts[
                    term_ind]))
