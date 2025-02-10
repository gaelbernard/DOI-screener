import pandas as pd
import numpy as np
from source.RulesProcessor import RuleProcessor, AbstractRule

'''
################
PROCESSOR
################
We define here the rules that will be applied in each sets (local-only, overlap, global-only)
'''

class LocalRuleProcessor(RuleProcessor):
    '''
    Rules applied to the local-only publications (publications that are not found in the global repository)
    '''
    def __init__(self, *args):
        super().__init__()
        self.name = 'Local-only'
        self.rules = {Linst(*args), Ltime(*args), LPrefix(*args), Lother(*args)} # The later is the default rule (default rule)

class OverlapRuleProcessor(RuleProcessor):
    '''
    Rules applied to the overlap publications (publications that are found in both the local and global repositories
    '''
    def __init__(self, *args):
        super().__init__()
        self.name = 'Matched'
        self.rules = {Matched(*args)}

class GlobalRuleProcessor(RuleProcessor):
    '''
    Rules applied to the global-only publications (publications that are found in the global repository but not in the local repository
    '''
    def __init__(self, *args):
        super().__init__()
        self.name = 'Global-only'
        self.rules = {Gprefix(*args), Gtype(*args), Gauthors(*args), Gother(*args)} # The later is the default rule (default rule)



'''
################
LOCAL RULES
################
Here we define the rules that will be applied to the local-only publications
'''

class Linst(AbstractRule):
    code = 'L-inst'
    description = 'DOI is not affiliated with institution in global repo'

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        super().__init__(local_only, overlap, global_only, global_repo_from_missing_doi)

    def apply(self, publication):
        details = {}
        '''
        Check if the publication is found in the global repository but not affiliated with the institution
        '''
        found_by_doi = self.global_repo_from_missing_doi.search_by_doi(publication.DOIs)
        if not found_by_doi:
            return False, details

        # if we are here, it means that the publication was found in the global repository
        # Now we need to check if the authors are affiliated with the institution
        # Returns true
        ror_ids = {ror_id for x in found_by_doi.authors if x.ror_ids for ror_id in x.ror_ids}
        pub_foundable_but_not_affiliated = self.local_only.ror_id not in ror_ids

        # Record details
        if pub_foundable_but_not_affiliated:
            details['doi_found_global'] = publication.DOIs.intersection(found_by_doi.DOIs)

        return pub_foundable_but_not_affiliated, details

class Ltime(AbstractRule):
    code = 'L-time'
    description = 'DOI is not within time range in global repo'

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        super().__init__(local_only, overlap, global_only, global_repo_from_missing_doi)

    def apply(self, publication):
        details = {}
        found_by_doi = self.global_repo_from_missing_doi.search_by_doi(publication.DOIs)
        if not found_by_doi:
            return False, details

        pub_found_outside_time_range = found_by_doi.year_issued < self.local_only.yr_min or found_by_doi.year_issued > self.local_only.yr_max

        # Record details
        if pub_found_outside_time_range:
            details['doi_found_global'] = publication.DOIs.intersection(found_by_doi.DOIs)
            details['year_issued_global'] = found_by_doi.year_issued

        return pub_found_outside_time_range, details

class LPrefix(AbstractRule):
    code = 'L-prefix'
    description = 'DOI not found in global repo, and has a DOI-prefix that is rarely in matched DOIs'

    PARAM_MIN_RECORDS = 100
    PARAM_ODDS_RATIO_TRESHOLD = 10

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        super().__init__(local_only, overlap, global_only, global_repo_from_missing_doi)

    def precompute(self):

        local_only_dois = set(doi for publications in self.local_only.publications for doi in publications.DOIs)
        overlap_dois = set(doi for publications in self.overlap.publications for doi in publications.DOIs)

        prefix_local_only = [doi.split('/')[0] for doi in local_only_dois]
        prefix_overlap = [doi.split('/')[0] for doi in overlap_dois]
        prefix_analysis = pd.DataFrame([{'prefix': x, 'is_overlap': True} for x in prefix_overlap] + [{'prefix': x, 'is_overlap': False} for x in prefix_local_only])
        prefix_analysis = prefix_analysis.pivot_table(index='prefix', columns='is_overlap', aggfunc='size', fill_value=0)


        prefix_analysis = self.add_odds_ratio(prefix_analysis)


        prefix_analysis = prefix_analysis[prefix_analysis['total'] >= self.PARAM_MIN_RECORDS]
        prefix_analysis = prefix_analysis[prefix_analysis['add_odds_ratio'] >= self.PARAM_ODDS_RATIO_TRESHOLD]
        self.prefix_analysis = prefix_analysis.reset_index()

        self.problematic_prefixes = prefix_analysis.index.tolist()


    def apply(self, publication):
        details = {}
        #print ('expected success rate', self.expected_success_rate)
        for doi in publication.DOIs:
            prefix = doi.split('/')[0]

            one_prefix_is_problematic = prefix in self.problematic_prefixes

            # Record details
            if one_prefix_is_problematic:
                details['problematic_prefix'] = prefix
                return True, details

        return False, details



class Lother(AbstractRule):
    code = 'L-other'
    description = 'DOI not found in global repo, and does not satisfy any other rule'

    def apply(self, publication):
        return True, {}


'''
################
OVERLAP RULES
################
'''

class Matched(AbstractRule):
    code = 'Matched'
    description = 'DOI is affiliated with institution and is within time range in global repo'

    def apply(self, publication):
        return True, {}

'''
################
GLOBAL RULES
################
'''

class Gprefix(AbstractRule):
    code = 'G-prefix'
    description = 'DOI is not in local list, and has a DOI-prefix that is rarely in matched DOIs'

    PARAM_MIN_RECORDS = 100
    PARAM_ODDS_RATIO_TRESHOLD = 10

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        super().__init__(local_only, overlap, global_only, global_repo_from_missing_doi)

    def precompute(self):

        prefix_in_local = pd.DataFrame(list(doi.split('/')[0] for publications in self.local_only.publications for doi in publications.DOIs))
        prefix_in_local['is_local_prefix'] = True
        prefix_in_global = pd.DataFrame(list(doi.split('/')[0] for publications in self.global_only.publications for doi in publications.DOIs))
        prefix_in_global['is_local_prefix'] = False
        prefix = pd.concat([prefix_in_local, prefix_in_global]).rename(columns={0: 'prefix'})

        prefix = prefix.pivot_table(index='prefix', columns='is_local_prefix', aggfunc='size', fill_value=0)

        prefix = self.add_odds_ratio(prefix)
        prefix = prefix[prefix['total'] >= self.PARAM_MIN_RECORDS]
        prefix = prefix[prefix['add_odds_ratio'] >= self.PARAM_ODDS_RATIO_TRESHOLD]

        self.prefix_analysis = prefix.reset_index()

        # We want to keep records where the success rate is significantly lower than the expected success rate
        self.problematic_prefixes = set(prefix.index.tolist())


    def apply(self, publication):
        details = {}

        prefix_in_publication = set(doi.split('/')[0] for doi in publication.DOIs)
        are_all_prefixes_problematic = prefix_in_publication.issubset(self.problematic_prefixes)

        if are_all_prefixes_problematic:
            details['problematic_prefix'] = prefix_in_publication
            return True, details

        return False, details

class Gtype(AbstractRule):
    code = 'G-type'
    description = 'DOI is not in local list, and has a publication type that is rarely in matched DOIs'

    PARAM_MIN_RECORDS = 100
    PARAM_ODDS_RATIO_TRESHOLD = 10

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        super().__init__(local_only, overlap, global_only, global_repo_from_missing_doi)

    def precompute(self):

        #type_with_success = set(doi for publications in self.global_only.publications for doi in publications.DOIs)
        type_with_success = [pub.type for pub in self.overlap.publications]
        type_without_success = [pub.type for pub in self.global_only.publications]

        type_analysis = pd.DataFrame([{'type': x, 'is_overlap': True} for x in type_with_success] + [{'type': x, 'is_overlap': False} for x in type_without_success])

        type_analysis = type_analysis.pivot_table(index='type', columns='is_overlap', aggfunc='size', fill_value=0)

        type_analysis = self.add_odds_ratio(type_analysis)

        self.type_analysis = type_analysis.reset_index().sort_values('add_odds_ratio', ascending=False)

        type_analysis = type_analysis[type_analysis['total'] >= self.PARAM_MIN_RECORDS]
        type_analysis = type_analysis[type_analysis['add_odds_ratio'] >= self.PARAM_ODDS_RATIO_TRESHOLD]



        # We want to keep records where the success rate is significantly lower than the expected success rate
        self.problematic_types = type_analysis.index.tolist()

    def apply(self, publication):
        details = {}
        if publication.type in self.problematic_types:

            # Record details
            details['problematic_type'] = publication.type
            return True, details

        return False, details

class Gauthors(AbstractRule):
    code = 'G-authors'
    description = 'DOI is not in local list, and authored by an author affiliated with the institution in the matched DOIs'

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
        details = {}
        authors_known_to_inst = set()
        year = publication.year_issued
        if year in self.known_authors:
            known_authors = self.known_authors[year]
            for author in publication.authors:
                if author.orcid in known_authors:
                    authors_known_to_inst.add(author.orcid)

        if len(authors_known_to_inst) > 0:
            details['authors_known_to_inst'] = authors_known_to_inst

        is_authored_by_known_author = len(authors_known_to_inst) > 0

        return is_authored_by_known_author, details

class Gother(AbstractRule):
    code = 'G-other'
    description = 'DOI is not in local list, and does not satisfy any other rule'

    def apply(self, publication):
        return True, {}

