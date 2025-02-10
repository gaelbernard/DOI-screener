import pandas as pd
from source.Rules import LocalRuleProcessor, OverlapRuleProcessor, GlobalRuleProcessor
from source.BaseObjects import Repository, LocalRepository, OpenAlexRepository
import matplotlib.pyplot as plt
import os
import json

class OverlapProcessing():
    '''
    Class to process the overlap between the local and global repositories.
    Essentially, the user has to provide a list of DOIs of available in the local repository, the ROR ID and the year range.
    Then, the class will retrieve data from the global repository and compute the overlap between the two repositories.
    '''

    filename_local_only = 'local.pkl'
    filename_overlap = 'overlap.pkl'
    filename_global_only = 'global_only.pkl'
    filename_global_repo_from_missing_doi = 'global_from_missing_doi.pkl'

    def __init__(self, verbose=True):
        self.verbose = verbose

        # Variable to store the repositories
        self.local_only = None                      # Publications that we don't find in the global repository
        self.overlap = None                         # Publications that we find in both repositories
        self.global_only = None                     # Publications that we find in the global repository but not in the local repository
        self.global_repo_from_missing_doi = None    # Publications that we don't find in the global repository by requesting the ror_id for the time range but we can find them using the DOI (only used during the categorization)

        # Variable to store the rules
        self.local_rules = None            # Categorization of the local only publications
        self.overlap_rules = None          # Categorization of the overlap publications
        self.global_rules = None           # Categorization of the global only publications

    def prepare_repositories(self, internal_dois, yr_min, yr_max, ror_id, repo_name, path_to_save):
        '''
        Prepare the repositories for the overlap analysis. All we take as input is the list of DOIs and the ROR ID.
        We then prepare 4 repositories:
        - local_only: Publications that we don't find in the global repository
        - overlap: Publications that we find in both repositories (at least one DOI in common)
        - global_only: Publications that we find in the global repository but not in the local repository
        - global_repo_from_missing_doi: Publications that we don't find in the global repository
            by requesting the ror_id for the time range but we can find them using the DOI

        :param internal_dois: list of list of DOIs (list of list because we have multiple DOIs per publication)
        :param yr_min: min year used to extract the internal_dois
        :param yr_max: max year used to extract the internal_dois
        :param ror_id: Ror ID of the institution
        :param repo_name: openalex only for now
        :param path_to_save: path to save the repositories. The folder should not exist.
        '''

        assert not os.path.exists(path_to_save), 'The path indicated to save the repositories already exists. Remove it first.'
        assert type(internal_dois) == list, 'Internal DOIs must be a list'
        assert type(internal_dois[0]) == list, 'Internal DOIs must be a list of list (since we might have multiple DOIs per publication)'
        os.makedirs(path_to_save)

        # Load the local repository
        local_repo = LocalRepository(year_min=yr_min, year_max=yr_max, ror_id=ror_id)
        local_repo.load_from_dois(internal_dois)

        # Load the global repository from the ROR ID
        if repo_name == 'openalex':
            global_repo_from_ror_id = OpenAlexRepository(yr_min=yr_min, yr_max=yr_max, ror_id=ror_id)
        else:
            raise ValueError('Unknown type of repository')
        global_repo_from_ror_id.load_from_ror()

        # Step 1: Compute the overlap between the two repositories
        # It is enough to have a single DOI in common to consider the publication as overlapping
        local_only, overlap, global_only = self._compute_overlap(local_repo, global_repo_from_ror_id)

        if self.verbose:
            print(f"""
            Overlap analysis done. Here is the distribution:
                - {len(local_only.publications)} local-only publications
                - {len(overlap.publications)} overlapping publications
                - {len(global_only.publications)} global-only publications
            """)

        # Step 2: Take local only and see if we can find them in the global repository using the DOI
        # It could be that the publication exists but not affiliated with the institution
        # Or that the time range is out of the scope of the global repository.
        type_global = type(global_repo_from_ror_id)
        global_repo_from_missing_doi = type_global(yr_min=yr_min, yr_max=yr_max, ror_id=ror_id)
        doi_missing = set(doi for publications in local_only.publications for doi in publications.DOIs)
        global_repo_from_missing_doi.load_from_dois(doi_missing)

        # Save the repositories to disk
        local_only.save_to_disk(os.path.join(path_to_save, self.filename_local_only))
        overlap.save_to_disk(os.path.join(path_to_save, self.filename_overlap))
        global_only.save_to_disk(os.path.join(path_to_save, self.filename_global_only))
        global_repo_from_missing_doi.save_to_disk(os.path.join(path_to_save, self.filename_global_repo_from_missing_doi))

        self.local_only = local_only
        self.overlap = overlap
        self.global_only = global_only
        self.global_repo_from_missing_doi = global_repo_from_missing_doi

    def load_repositories(self, path):
        '''
        Load the repositories from disk
        :param path: folder where the repositories were saved
        :return:
        '''
        self.local_only = Repository.load_from_disk(os.path.join(path, self.filename_local_only))
        self.overlap = Repository.load_from_disk(os.path.join(path, self.filename_overlap))
        self.global_only = Repository.load_from_disk(os.path.join(path, self.filename_global_only))
        self.global_repo_from_missing_doi = Repository.load_from_disk(os.path.join(path, self.filename_global_repo_from_missing_doi))

    def apply_rules(self):
        '''
        Apply the rules to all publications from the three repositories
        :return:
        '''
        assert self.local_only and self.overlap and self.global_only and self.global_repo_from_missing_doi, 'The repositories must be prepared or loaded first (i.e., prepare_repositories, load_repositories)'

        self.local_rules = LocalRuleProcessor(self.local_only, self.overlap, self.global_only, self.global_repo_from_missing_doi)
        self.local_rules.process(self.local_only.publications)

        self.overlap_rules = OverlapRuleProcessor(self.local_only, self.overlap, self.global_only, self.global_repo_from_missing_doi)
        self.overlap_rules.process(self.overlap.publications)

        self.global_rules = GlobalRuleProcessor(self.local_only, self.overlap, self.global_only, self.global_repo_from_missing_doi)
        self.global_rules.process(self.global_only.publications)

    def build_bar_chart(self, local_order, matched_rule_order, global_rule_order, path=None, legend=True, color_mapping=None):
        '''
        Build a bar chart to visualize the results of the rules
        :param local_order: To keep the chart simple, we will only show the first rule that is True for each publication. Hence, you need to provide a list that indicates in which order the rules should be applied.
        :param matched_rule_order: Same as above but for the overlap publications
        :param global_rule_order: Same as above but for the global only publications
        :param path: path to save the chart
        :param legend: whether to show the legend or not
        :param color_mapping: dictionary to map the color of each rule (if not provided, we will use the default colors)
        :return:
        '''

        assert self.local_rules is not None, 'You must apply_rules first (i.e., apply_rules)'

        df = []
        default_color_mapping = {
            'L-inst': '#f58c87',
            'L-time': '#f58c87',
            'L-prefix': '#f58c87',
            'L-other': '#ee4037',
            'Matched': '#272261',
            'G-prefix': '#fccf8d',
            'G-type': '#fccf8d',
            'G-authors': '#fccf8d',
            'G-other': '#faaf41',
        }

        if not color_mapping:
            color_mapping = default_color_mapping
        else:
            # prioritize the provided color mapping but fill the missing ones with the default color
            for k in default_color_mapping:
                if k not in color_mapping:
                    color_mapping[k] = default_color_mapping[k]

        default_color = '#cccccc'

        # retrieve a single rule result for each sets and add it to the dataframe
        single_rule_result = self.local_rules.get_single_rule_result(local_order)
        for r in single_rule_result:
            df.append({
                'set': self.local_rules.name,
                'set-order': 1,
                'code': r.rule.code,
                'description': r.rule.description,
                'order': local_order.index(r.rule.code),
            })

        single_rule_result = self.overlap_rules.get_single_rule_result(matched_rule_order)
        for r in single_rule_result:
            df.append({
                'set': self.overlap_rules.name,
                'set-order': 2,
                'code': r.rule.code,
                'description': r.rule.description,
                'order': matched_rule_order.index(r.rule.code),
            })

        single_rule_result = self.global_rules.get_single_rule_result(global_rule_order)
        for r in single_rule_result:
            df.append({
                'set': self.global_rules.name,
                'set-order': 3,
                'code': r.rule.code,
                'description': r.rule.description,
                'order': global_rule_order.index(r.rule.code),
            })

        # Build dataframe and count
        df = pd.DataFrame(df)
        df['color'] = df['code'].map(color_mapping).fillna(default_color)
        df = df.groupby(['set', 'code', 'description','order','color','set-order']).size().reset_index().rename(columns={0: 'count'})
        data = df

        # We want to have a mirror effect, so reverse the order when the set-order is 1
        data.loc[data['set-order'] == 1, 'order'] = data.loc[data['set-order'] == 1, 'order'] * -1

        data = data.sort_values(by=['set-order', 'order'])

        data['pct'] = data['count'] / data['count'].sum() * 100
        data['pct'] = data['pct'].round(1)
        data['pct'] = data['pct'].astype(str) + '%'

        # Horizontal Stacked Bar Chart with Horizontal Text and Fixed Margins
        values = data['count'].astype(int)  # bar segment sizes
        pcts = data['pct']  # percent labels
        codes = data['code']  # rule codes
        colors = data['color']  # colors
        margin = .03 * data['count'].sum()

        # Compute left positions for each stacked segment
        left_positions = [0]
        for val in values[:-1]:
            left_positions.append(left_positions[-1] + val + margin)

        # Decide figure shape:
        # If legend=False: a single (10 x 2) figure
        # If legend=True : a taller (10 x 3) figure with 2 subplots
        if legend:
            fig, (ax_bar, ax_legend) = plt.subplots(
                nrows=2,
                figsize=(10, 3),
                gridspec_kw={'height_ratios': [2, 1]}
            )
        else:
            fig, ax_bar = plt.subplots(figsize=(10, 2))
            ax_legend = None  # not used if no legend

        # ------------ 1) Plot the Bar in ax_bar ------------
        bars = ax_bar.barh(
            [0],
            values,
            left=left_positions,
            color=colors,
            height=0.7
        )

        # Add text below each segment
        for bar, val, code, pct in zip(bars, values, codes, pcts):
            bar_center = bar.get_x() + bar.get_width() / 2
            bar_bottom = bar.get_y() - 0.12  # just below the bar
            ax_bar.text(
                bar_center,
                bar_bottom,
                #f"{code}\n{val}\n{pct}",
                f"{code}\n{val}\n",
                va='top',
                ha='center',
                fontsize=8,
                color='black',
            )

        # Add set lines/labels above the bar
        start_position = 0
        total = values.sum()
        for unique_set in data['set'].unique():
            ldf = data[data['set'] == unique_set]
            block_count = ldf['count'].sum()
            end_position = start_position + block_count + (margin * (len(ldf) - 1))

            ax_bar.hlines(
                y=0.4,
                xmin=start_position,
                xmax=end_position,
                color=ldf['color'].iloc[-1],
                linewidth=2
            )

            ax_bar.text(
                x=(start_position + end_position) / 2,
                y=0.3,
                s=(
                    f"{unique_set}\n"
                    f"{block_count}\n"
                    #   f"{(block_count / total) * 100:.1f}%"
                ),
                ha='center',
                va='bottom',
                fontsize=8,
                color='black',
            )

            start_position = end_position + margin

        # Style ax_bar
        ax_bar.set_yticks([])
        ax_bar.set_xticks([])
        ax_bar.set_ylim(-0.5, 1)  # ensure lower labels are visible
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

        # ------------ 2) Legend Subplot (only if legend=True) ------------
        if legend and ax_legend is not None:
            # Turn off the spines/ticks for the legend axis
            ax_legend.axis('off')

            # Build lines for the legend
            legend_lines = ["Legend"]
            for _, row in data[['code', 'description']].drop_duplicates().iterrows():
                legend_lines.append(f"{row['code']}:")
            legend_text = "\n".join(legend_lines)

            # Put text in the center of ax_legend
            ax_legend.text(
                0.1, 0.9,
                legend_text,
                ha='right',
                va='top',
                fontsize=8
            )

            legend_lines = [""]
            for _, row in data[['code', 'description']].drop_duplicates().iterrows():
                legend_lines.append(f"{row['description']}")
            legend_text = "\n".join(legend_lines)
            ax_legend.text(
                0.12, 0.9,
                legend_text,
                ha='left',
                va='top',
                fontsize=8
            )

        # Final layout & save/show
        plt.tight_layout()
        if path:
            if path.endswith('.png'):
                plt.savefig(path, bbox_inches='tight', dpi=600)
            else:
                plt.savefig(path, bbox_inches='tight')
        else:
            plt.show()

    def extract_detailed_stats(self, rule):
        assert self.local_rules is not None, 'You must apply_rules first (i.e., apply_rules)'

        orcids = []

        for result in self.local_rules.results+self.overlap_rules.results+self.global_rules.results:
            if result.rule.code != rule:
                continue
            print (result.is_rule, result.rule.code, result.rule.description, result.details)
            orcids.extend(result.details.get('authors_found_global', []))

        print (pd.Series(orcids).value_counts())


        exit()
        tot = data.shape[0]

        report = {}
        distribution_super = data.groupby('set').size().rename('count').reset_index()
        distribution_super['fraction'] = distribution_super['count'] / tot
        report['distribution-set'] = distribution_super.to_dict(orient='records')

        distribution = data.groupby(['set', 'code', 'description']).size().rename('count').reset_index()
        distribution['fraction'] = distribution['count'] / tot
        report['distribution-category'] = distribution.to_dict(orient='records')

        l_prefix = self.local_rules.get_rule_by_code('L-prefix').prefix_analysis.to_dict(orient='records')
        report['L-prefix-problematic-prefix'] = l_prefix

        l_time = data[data['code']=='L-time']['category_details'].apply(lambda x: x['year_issued_global']).value_counts().to_dict()
        report['L_time-year'] = l_time

        l_inst = data[data['code']=='L-inst']['DOIs'].to_list()
        # convert to list of list (to make it json serializable)
        l_inst = [list(x) for x in l_inst]
        report['L-inst-DOIs'] = l_inst

        g_prefix = self.global_rules.get_rule_by_code('G-prefix').prefix_analysis
        report['G-prefix-problematic-prefix'] = g_prefix.to_dict(orient='records')

        g_type = self.global_rules.get_rule_by_code('G-type').type_analysis
        report['G-prefix-problematic-type'] = g_type.to_dict(orient='records')

        # show sample of record for G-author
        l_inst = data[data['code']=='G-authors']['DOIs'].to_list()
        # convert to list of list (to make it json serializable)
        l_inst = [list(x) for x in l_inst]
        report['G-authors-DOIs'] = l_inst

        return report




    def _compute_overlap(self, internal_repo, global_repo_from_ror_id):
        '''
        Compute the overlap between the internal and global repositories.
        :param internal_repo: from the list of DOIs
        :param global_repo_from_ror_id: publications retrieved from the ROR ID
        :return: three repositories: local_only, overlap, global_only
        '''
        yr_min = internal_repo.yr_min
        yr_max = internal_repo.yr_max
        ror_id = internal_repo.ror_id

        local_only_repo = Repository(year_min=yr_min, year_max=yr_max, ror_id=ror_id)
        global_only_repo = Repository(year_min=yr_min, year_max=yr_max, ror_id=ror_id)
        overlap_repo = Repository(year_min=yr_min, year_max=yr_max, ror_id=ror_id)

        global_record_used = set()

        for publication_internal_repo in internal_repo.publications:

            # Search the local DOIs in the global repository
            publication_global_repo = global_repo_from_ror_id.search_by_doi(publication_internal_repo.DOIs)

            # Matching DOI found in the global repository
            if publication_global_repo:
                # Keep only DOIs that were found locally and globally

                publication_global_repo.DOIs = publication_internal_repo.DOIs.intersection(publication_global_repo.DOIs)

                # For the authors, we will not have authors from the local so we keep the ones from the global
                # Same for the year_issued and type
                overlap_repo.publications.append(publication_global_repo)

                # Keep track of the global record used so we exclude it from the global only
                global_record_used.add(publication_global_repo)
            else:
                local_only_repo.publications.append(publication_internal_repo)


        global_only_repo.publications = [x for x in global_repo_from_ror_id.publications if x not in global_record_used]

        return local_only_repo, overlap_repo, global_only_repo
