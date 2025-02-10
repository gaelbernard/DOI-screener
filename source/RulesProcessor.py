import pandas as pd
import numpy as np

class RuleProcessor:
    '''
    This class is responsible for processing the rules of the publications.
    We used the chain of responsibility pattern to process the rules.
    It makes it easy to add new categories and to change the order of the categories.
    (instead of having a big if-else block)
    '''
    def __init__(self):
        '''
        The 'categories' are the all the categories that will be used to process the categorization.
        '''
        self.rules = []
        self.results = []

    def process(self, publications):
        '''
        This method will apply all the rules to the publication
        :param publication:
        :return:
        '''
        for publication in publications:
            for rule in self.rules:
                self.results.append(RuleResult(publication, rule))

    def get_single_rule_result(self, order):
        '''
        In some cases, we may want to explain a publication using only one rule.
        This is the case when we want to visualize the results in a bar chart.
        This method will return the first rule that is True for each publication.
        Hence, you need to provide a list that indicates in which order the rules should be applied.
        :param publication: the order of the rules (eg., ['L-inst', 'L-time', 'L-prefix', 'L-other'])
        :return: a list of RuleResult where each publication is explained by only one rule
        '''
        seen_rules = set(x.rule.code for x in self.results)
        assert seen_rules == set(order), f'You need to provide a list that indicates in which order the rules should be applied. The list should be composed of the following rules: {list(seen_rules)}'

        # Group by publication
        pubs = {}
        for r in self.results:
            if r.publication not in pubs:
                pubs[r.publication] = []
            pubs[r.publication].append(tuple([r, order.index(r.rule.code)]))

        # Sort the rules and return the first one that is True
        output = []
        for pub in pubs:
            sorted_tuples = sorted(pubs[pub], key=lambda tup: tup[1])

            found_one = False
            for rule_result, order in sorted_tuples:
                if rule_result.is_rule:
                    output.append(RuleResult(pub, rule_result.rule))
                    found_one = True
                    break
            assert found_one, 'A rule must be found (at least the default one)'
        return output

    def get_rule_by_code(self, name):
        '''
        Return a rule by its code
        :param name: the code of the rule
        :return: the rule
        '''
        for rule in self.rules:
            if rule.code == name:
                return rule
        return None


class AbstractRule():
    '''
    This is the abstract class for the rules.
    The method 'apply' should be implemented in the subclasses
    '''
    code = ''
    description = ''
    color = ''

    def __init__(self, local_only, overlap, global_only, global_repo_from_missing_doi):
        self.local_only = local_only
        self.overlap = overlap
        self.global_only = global_only
        self.global_repo_from_missing_doi = global_repo_from_missing_doi
        self.precompute()

    def precompute(self):
        '''
        This method is used to precompute some data that will be used when applying the rules.
        It is called once before the categorization starts.
        :return:
        '''
        pass

    def apply(self, publication):
        '''
        This method is used to test the rules to the application.
        It should return a tuple with two elements:
        - Bool: True if the rule applies to the publication
        - Dict: Details about the rules (eg., the prefix of the publication)
        :param publication:
        :return:
        '''
        raise NotImplementedError
        return True, {}

    def add_odds_ratio(self, df):
        assert set(df.columns.tolist()) == set([False, True])

        df['total'] = df.sum(axis=1)

        # Calculate expected fail rate
        expected_fail_rate = df[False].sum() / df.sum().sum()

        # Actual fail rate for each prefix
        df['actual_fail_rate'] = df[False] / df['total']

        # Convert rates to odds
        df['observed_odds'] = df['actual_fail_rate'] / (1 - df['actual_fail_rate'])
        expected_odds = expected_fail_rate / (1 - expected_fail_rate)

        # Compute actual odds ratio
        def safe_odds_ratio(obs_odds, exp_odds):
            if np.isinf(obs_odds):
                return np.inf
            return obs_odds / exp_odds

        df['add_odds_ratio'] = df['observed_odds'].apply(lambda x: safe_odds_ratio(x, expected_odds))

        df = df.sort_values(['add_odds_ratio',False], ascending=False)
        return df



class RuleResult:
    '''
    Store the result of a rule applied to a publication.
    '''
    def __init__(self, publication, rule):
        self.publication = publication
        self.rule = rule
        self.is_rule, self.details = rule.apply(publication)

