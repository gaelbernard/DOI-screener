import pickle
import pandas as pd
from tqdm import tqdm
import requests
from more_itertools import chunked
from source.Classification import LocalCategorizationProcessor, OverlapCategorizationProcessor, GlobalCategorizationProcessor
import matplotlib.pyplot as plt

class Repository:
    '''
    This class is an abstract class that defines the basic structure of a Repository.
    It is meant to be subclassed by other classes such as LocalRepository and OpenAlexRepository.
    '''
    def __init__(self, year_min=None, year_max=None, ror_id=None):
        '''
        Initialize the Repository object with the year range and ROR ID.
        :param year_min: Minimum issued year of publication
        :param year_max: Maximum issued year of publication
        :param ror_id: ROR ID of the institution (e.g., 'https://ror.org/02s376052')
        '''
        self.publications = []
        self.yr_min = year_min
        self.yr_max = year_max
        self.ror_id = self.clean_ror_id(ror_id)

        self.inverted_index = None # Mapping DOI => Publication used for faster search (lazy loading)

    def clean_ror_id(self, ror_id):
        '''
        Basic data cleaning for ROR ID
        :param ror_id:
        :return:
        '''
        if not ror_id:
            return None
        ror_id = ror_id.lower()
        ror_id = ror_id.replace('https://ror.org/', '')
        return ror_id

    def load_from_dois(self, dois, year_range):
        raise NotImplementedError("This method should be implemented in the child class")

    def load_from_ror(self):
        raise NotImplementedError("This method should be implemented in the child class")

    def initialize_inverted_index(self):
        '''
        Initialize the inverted index for faster search. DOI => publication object
        :return:
        '''
        self.inverted_index = {doi: pub for pub in self.publications for doi in pub.DOIs}

    def search_by_doi(self, DOIs):
        '''
        A publication can have multiple DOIs.
        This function takes a set of DOIs and returns a publication object where one of the DOIs matches.
        This function will return the publication object if one of the DOIs matches.
        :param DOIs:
        :return:
        '''
        assert isinstance(DOIs, set), 'DOI must be a set'

        if self.inverted_index is None:
            self.initialize_inverted_index()

        for doi in DOIs:
            if doi in self.inverted_index:
                return self.inverted_index[doi]

        return None

    def save_to_disk(self, path):
        '''
        Save the repository object to disk using pickle
        :param path: path of the file
        :return:
        '''
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(path):
        '''
        Load the repository object from disk using pickle
        :param path: path of the file
        :return:
        '''
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __str__(self):
        return f'{(type(self)).__name__} containing {len(self.publications)} publications'

class LocalRepository(Repository):
    '''
    This class represents a local repository.
    It is a repository with minimal information (only DOIs) and is used to compare with other Global repositories.
    '''
    def __init__(self, year_min=None, year_max=None, ror_id=None):
        assert year_min is not None and year_max is not None, 'Year min and year max must be provided for the Local Repository'
        assert ror_id is not None, 'ROR ID must be provided for the Local Repository'
        super().__init__(year_min, year_max, ror_id)

    def load_from_dois(self, list_dois):
        '''
        Load the repository from a list of DOIs
        :param list_dois: Since a publication can have multiple DOIs, this should be a list of lists of DOIs
        :return:
        '''
        for dois in list_dois:
            self.publications.append(Publication(dois))

class OpenAlexRepository(Repository):
    '''
    This class represents a repository that is loaded from the OpenAlex API.
    '''
    def __init__(self, yr_min=None, yr_max=None, ror_id=None):
        '''
        Initialize the OpenAlexRepository object with the year range and ROR ID.
        :param yr_min:
        :param yr_max:
        :param ror_id:
        '''
        super().__init__(yr_min, yr_max, ror_id)

    def load_from_ror(self):

        assert self.yr_min is not None and self.yr_max is not None, 'Year min and year max must be provided when retrieving DOI from ROR'
        assert self.ror_id is not None, 'ROR ID must be provided when retrieving DOI from ROR'

        cursor = '*'
        with tqdm(total=0, desc="Retrieving publications using the ror id") as pbar:
            while True:

                url = f"https://api.openalex.org/works?page=1&filter=authorships.institutions.ror:https://ror.org/{self.ror_id},publication_year:{self.yr_min}-{self.yr_max},has_doi:true&select=doi,publication_year,title,type,locations,authorships&per_page=100&cursor={cursor}"
                records = requests.get(url).json()
                # Retrieve total number of records (to set up tqdm bar initially)
                if pbar.total == 0:
                    pbar.total = records['meta']['count']

                for record in records.get('results', []):

                    dois = set([x['landing_page_url'] for x in record['locations'] if x['landing_page_url'] and x['landing_page_url'].startswith('https://doi.org/')] + [record['doi']])
                    authors_object = []
                    for authors in record['authorships']:
                        orcid = authors.get('author', None).get('orcid', None)
                        rorid = {x['ror'] for x in authors['institutions']}
                        author = Author(orcid, rorid)
                        authors_object.append(author)

                    pub = Publication(dois, authors_object, year_issued=record.get('publication_year'), type=record.get('type'))
                    self.publications.append(pub)

                    pbar.update(1)  # Update progress bar

                cursor = records.get('meta', {}).get('next_cursor', None)
                if not cursor:
                    break

    def load_from_dois(self, dois):

        # Calling OpenAlex API in batches of 100 DOIs
        pbar = tqdm(total=len(dois), desc="Retrieving publications using from DOIs")
        for batch_dois in chunked(dois, 100):
            url = f"https://api.openalex.org/works?filter=doi:{'|'.join(batch_dois)}&select=doi,publication_year,title,type,locations,authorships&per_page=100"

            for record in requests.get(url).json().get('results',[]):
                record_dois = set([x['landing_page_url'] for x in record['locations'] if x['landing_page_url'] and x['landing_page_url'].startswith('https://doi.org/')] + [record['doi']])

                authors_object = []
                for authors in record['authorships']:
                    orcid = authors.get('author', None).get('orcid', None)
                    rorid = {x['ror'] for x in authors['institutions']}
                    author = Author(orcid, rorid)
                    authors_object.append(author)

                pub = Publication(record_dois, authors_object, year_issued=record.get('publication_year'), type=record.get('type'))
                self.publications.append(pub)

            pbar.update(len(batch_dois))  # Update progress bar


class Publication():
    '''
    Represents a scientific publication object.
    The only mandatory field is the DOI.
    '''
    def __init__(self, DOIs, authors=None, year_issued=None, type=None):
        '''
        Initialize the Publication object with the DOIs, authors, year issued and type.
        :param DOIs: List of DOIs
        :param authors: List of Author objects
        :param year_issued: Year of publication
        :param type: Type of publication (e.g., 'journal')
        '''
        self.DOIs = self.clean_doi(DOIs)
        self.year_issued = year_issued
        self.type = type
        self.category = None        # This will be set by the CategorizationProcessor
        self.category_details = {}  # Holder for detailed category information (like which prefix was used to categorize)

        if authors is None:
            authors = []
        self.authors = authors

    def clean_doi(self, doi):
        '''
        Clean the DOI by lower casing and removing 'https://doi.org/'
        :param doi:
        :return:
        '''

        # Make sure the DOI is a set
        if isinstance(doi, str):
            doi = [doi]
        if isinstance(doi, list):
            doi = set(doi)

        # Lower case and remove https://doi.org/
        doi = {x.lower().replace('https://doi.org/', '') for x in doi}

        # Transform to set again to remove duplicates (due to lower case)
        doi = set(doi)

        return doi

    def __str__(self):
        return f'Publication with {len(self.DOIs)} DOI and {len(self.authors)} authors'

    def to_dict(self):
        obj = {
            'DOIs': self.DOIs,
            'year_issued': self.year_issued,
            'type': self.type,
            #'authors': [author.to_dict() for author in self.authors], # Not sure if we need this, can be potentially large
            'category_details': self.category_details
        }
        if self.category:
            obj.update(self.category.to_dict())

        return obj

class Author():
    '''
    Author object with ORCID and ROR IDs
    '''
    def __init__(self, orcid, ror_ids):
        '''
        An author object with ORCID and ROR IDs (since an author can be affiliated with multiple institutions)
        :param orcid:
        :param ror_ids:
        '''
        self.orcid = orcid
        self.ror_ids = self.clean_ror_ids(ror_ids)

    def clean_ror_ids(self, ror_ids):
        '''
        Clean the ROR
        :param ror_ids:
        :return:
        '''
        if not ror_ids:
            return None
        cleaned_ror_ids = []
        for ror_id in ror_ids:
            ror_id = ror_id.lower()
            ror_id = ror_id.replace('https://ror.org/', '')
            cleaned_ror_ids.append(ror_id)
        return cleaned_ror_ids

    def __str__(self):
        return f'Author: {self.orcid}, {self.ror_ids}'

    def to_dict(self):
        return {
            'orcid': self.orcid,
            'ror_ids': self.ror_ids
        }







