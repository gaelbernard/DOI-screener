import pandas as pd

class CategorizationProcessor:
    '''
    This class is responsible for processing the categorization of the publications.
    We used the chain of responsibility pattern to process the categorization.
    It makes it easy to add new categories and to change the order of the categories.
    (instead of having a big if-else block)

    The order in which the categories are tested depends on the 'order' attribute of the category.
    '''
    def __init__(self):
        '''
        The 'categories' are the all the categories that will be used to process the categorization.
        '''
        self.categories = []

    def process(self, publication):
        '''
        This method will process the categorization of the publication
        (i.e., test and assign a category to the publication)
        It is sorted by the 'order' attribute of the category.
        :param publication:
        :return:
        '''
        for cat in sorted(self.categories, key=lambda x: x.order):
            if cat.apply(publication):
                publication.category = cat
                return

class LocalCategorizationProcessor(CategorizationProcessor):
    '''
    This class tries to explain the local only publications.
    The last category is the default category.
    '''
    def __init__(self, *args):
        super().__init__()
        self.categories = set([L1(*args), L2(*args), L3(*args), LocalDefault(*args)])

class OverlapCategorizationProcessor(CategorizationProcessor):
    '''
    This class tries to explain the publications that are found in both repositories.
    For now, we don't have any categories for this case.
    '''
    def __init__(self, *args):
        super().__init__()
        self.categories = set([Overlap(*args)])

class GlobalCategorizationProcessor(CategorizationProcessor):
    '''
    This class tries to explain the global only publications.
    '''
    def __init__(self, *args):
        super().__init__()
        self.categories = set([G1(*args), G2(*args), G3(*args), GlobalDefault(*args)])


class AbstractCategory():
    '''
    This is the abstract class for the categories.
    We already know the super-categories (local only, overlap, global only).
    A category is trying to divide the super-categories into sub-categories to provide more insights.
    If process() returns True, it means that the publication belongs to this category.
    '''
    code = ''
    description = ''
    color = ''
    order = 0

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        self.local_only = local_only
        self.overlap = overlap
        self.global_only = global_only
        self.global_repo_from_missing_doi = global_repo_from_missing_doi
        self.precompute()

    def precompute(self):
        '''
        This method is used to precompute some data that will be used to process the categorization.
        It is called once before the categorization starts.
        :return:
        '''
        pass

    def apply(self, publication):
        '''
        This method is used to process the categorization of the publication.
        :param publication:
        :return:
        '''
        pass

    def to_dict(self):
        return {
            'code': self.code,
            'description': self.description,
            'color': self.color,
            'order': self.order
        }


'''
################
LOCAL CATEGORIES
################
'''

class L1(AbstractCategory):
    code = 'L1'
    description = 'Found DOI in public repo but not affiliated with the institution'
    color = '#f58c87'
    order = 1


    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        super().__init__(local_only, overlap, global_only, global_repo_from_missing_doi)

    def apply(self, publication):
        '''
        Check if the publication is found in the global repository but not affiliated with the institution
        '''
        found_by_doi = self.global_repo_from_missing_doi.search_by_doi(publication.DOIs)
        if not found_by_doi:
            return False

        # if we are here, it means that the publication was found in the global repository
        # Now we need to check if the authors are affiliated with the institution
        # Returns true
        ror_ids = {ror_id for x in found_by_doi.authors if x.ror_ids for ror_id in x.ror_ids}

        pub_foundable_but_not_affiliated = self.local_only.ror_id not in ror_ids

        # Record details
        if pub_foundable_but_not_affiliated:
            publication.category_details['doi_found_global'] = publication.DOIs.intersection(found_by_doi.DOIs)
            publication.category_details['authors_found_global'] = {x.orcid for x in publication.authors}

        return pub_foundable_but_not_affiliated

class L2(AbstractCategory):
    code = 'L2'
    description = 'Found DOI in public repo but not within time range'
    color = '#f58c87'
    order = 2

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        super().__init__(local_only, overlap, global_only, global_repo_from_missing_doi)

    def apply(self, publication):
        found_by_doi = self.global_repo_from_missing_doi.search_by_doi(publication.DOIs)
        if not found_by_doi:
            return False

        pub_found_outside_time_range = found_by_doi.year_issued < self.local_only.yr_min or found_by_doi.year_issued > self.local_only.yr_max

        # Record details
        if pub_found_outside_time_range:
            publication.category_details['doi_found_global'] = publication.DOIs.intersection(found_by_doi.DOIs)
            publication.category_details['year_issued_global'] = found_by_doi.year_issued

        return pub_found_outside_time_range

class L3(AbstractCategory):
    code = 'L3'
    description = 'DOI not found in global, might be explained by the DOI\'s prefix'
    color = '#f58c87'
    order = 3

    PARAM_MIN_RECORDS = 10
    PARAM_ODDS_RATIO_TRESHOLD = 5

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        super().__init__(local_only, overlap, global_only, global_repo_from_missing_doi)

    def precompute(self):

        local_only_dois = set(doi for publications in self.local_only.publications for doi in publications.DOIs)
        overlap_dois = set(doi for publications in self.overlap.publications for doi in publications.DOIs)

        prefix_local_only = [doi.split('/')[0] for doi in local_only_dois]
        prefix_overlap = [doi.split('/')[0] for doi in overlap_dois]
        prefix_analysis = pd.DataFrame([{'prefix': x, 'is_overlap': True} for x in prefix_overlap] + [{'prefix': x, 'is_overlap': False} for x in prefix_local_only])
        prefix_analysis = prefix_analysis.pivot_table(index='prefix', columns='is_overlap', aggfunc='size', fill_value=0)
        prefix_analysis = prefix_analysis[prefix_analysis.sum(axis=1) >= self.PARAM_MIN_RECORDS]

        success_rate = prefix_analysis[True] / prefix_analysis.sum(axis=1)

        expected_treshold = len(self.local_only.publications) / len(self.overlap.publications)

        # We want to keep records where the success rate is significantly lower than the expected success rate
        self.problematic_prefixes = success_rate[success_rate < expected_treshold / self.PARAM_ODDS_RATIO_TRESHOLD].index


    def apply(self, publication):
        #print ('expected success rate', self.expected_success_rate)
        for doi in publication.DOIs:
            prefix = doi.split('/')[0]

            one_prefix_is_problematic = prefix in self.problematic_prefixes

            # Record details
            if one_prefix_is_problematic:
                publication.category_details['problematic_prefix'] = prefix
                return True

        return False



class LocalDefault(AbstractCategory):
    code = 'L?'
    description = 'Unexplained local only publication'
    color = '#ee4037'
    order = 10000

    def apply(self, publication):
        return True


'''
################
OVERLAP CATEGORIES
################
'''

class Overlap(AbstractCategory):
    code = 'Ov'
    description = 'At least one DOI found in both repositories'
    color = '#272261'
    order = 0

    def apply(self, publication):
        return True

'''
################
GLOBAL CATEGORIES
################
'''

class G1(AbstractCategory):
    code = 'G1'
    description = 'DOI not found in local, might be explained by the DOI\'s prefix'
    color = '#fccf8d'
    order = 1

    PARAM_MIN_RECORDS = 10
    PARAM_ODDS_RATIO_TRESHOLD = 5

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        super().__init__(local_only, overlap, global_only, global_repo_from_missing_doi)

    def precompute(self):

        global_only_dois = set(doi for publications in self.global_only.publications for doi in publications.DOIs)
        overlap_dois = set(doi for publications in self.overlap.publications for doi in publications.DOIs)

        prefix_global_only = [doi.split('/')[0] for doi in global_only_dois]
        prefix_overlap = [doi.split('/')[0] for doi in overlap_dois]
        prefix_analysis = pd.DataFrame([{'prefix': x, 'is_overlap': True} for x in prefix_overlap] + [{'prefix': x, 'is_overlap': False} for x in prefix_global_only])
        prefix_analysis = prefix_analysis.pivot_table(index='prefix', columns='is_overlap', aggfunc='size', fill_value=0)
        prefix_analysis = prefix_analysis[prefix_analysis.sum(axis=1) >= self.PARAM_MIN_RECORDS]

        success_rate = prefix_analysis[True] / prefix_analysis.sum(axis=1)

        expected_treshold = len(self.global_only.publications) / len(self.overlap.publications)

        # We want to keep records where the success rate is significantly lower than the expected success rate
        self.problematic_prefixes = success_rate[success_rate < expected_treshold / self.PARAM_ODDS_RATIO_TRESHOLD].index

    def apply(self, publication):
        for doi in publication.DOIs:
            prefix = doi.split('/')[0]
            if prefix in self.problematic_prefixes:

                # Record details
                publication.category_details['problematic_prefix'] = prefix
                return True

        return False

class G2(AbstractCategory):
    code = 'G2'
    description = 'DOI not found in local, might be explained by the type of publication'
    color = '#fccf8d'
    order = 2

    PARAM_MIN_RECORDS = 10
    PARAM_ODDS_RATIO_TRESHOLD = 5

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        super().__init__(local_only, overlap, global_only, global_repo_from_missing_doi)

    def precompute(self):

        #type_with_success = set(doi for publications in self.global_only.publications for doi in publications.DOIs)
        type_with_success = [pub.type for pub in self.overlap.publications]
        type_without_success = [pub.type for pub in self.global_only.publications]

        type_analysis = pd.DataFrame([{'type': x, 'is_overlap': True} for x in type_with_success] + [{'type': x, 'is_overlap': False} for x in type_without_success])

        type_analysis = type_analysis.pivot_table(index='type', columns='is_overlap', aggfunc='size', fill_value=0)
        type_analysis = type_analysis[type_analysis.sum(axis=1) >= self.PARAM_MIN_RECORDS]

        success_rate = type_analysis[True] / type_analysis.sum(axis=1)
        expected_treshold = len(self.global_only.publications) / len(self.overlap.publications)


        # We want to keep records where the success rate is significantly lower than the expected success rate
        self.problematic_prefixes = success_rate[success_rate < expected_treshold / self.PARAM_ODDS_RATIO_TRESHOLD].index

    def apply(self, publication):
        if publication.type in self.problematic_prefixes:

            # Record details
            publication.category_details['problematic_type'] = publication.type
            return True

        return False

class G3(AbstractCategory):
    code = 'G3'
    description = 'DOI not found in local, but written by authors known to the institution'
    color = '#fccf8d'
    order = 3

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        super().__init__(local_only, overlap, global_only, global_repo_from_missing_doi)

    def precompute(self):

        known_authors = {}

        # retrieve the authors known to the institution
        for publication in self.overlap.publications:
            for author in publication.authors:
                if not author.ror_ids:
                    continue
                if self.overlap.ror_id in author.ror_ids:
                    year = publication.year_issued
                    if year not in known_authors:
                        known_authors[year] = set()
                    known_authors[year].add(author.orcid)

        self.known_authors = known_authors

    def apply(self, publication):
        authors_known_to_inst = set()
        year = publication.year_issued
        if year in self.known_authors:
            known_authors = self.known_authors[year]
            for author in publication.authors:
                if author.orcid in known_authors:
                    authors_known_to_inst.add(author.orcid)

        if len(authors_known_to_inst) > 0:
            publication.category_details['authors_known_to_inst'] = authors_known_to_inst

        return len(authors_known_to_inst) > 0

class GlobalDefault(AbstractCategory):
    code = 'G?'
    description = 'Unexplained global only publication'
    color = '#faaf41'
    order = 10000

    def apply(self, publication):
        return True