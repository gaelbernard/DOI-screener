from source.OverlapProcessing import OverlapProcessing

# local_dois is a list of potentially multiple DOIs because a single publication might have multiple DOIs
local_dois = [['10.1109/RO-MAN53752.2022.9900775'], ['10.1016/j.carbpol.2023.120622'], ['10.1016/j.ultramic.2021.113460'], ['10.1016/j.neurobiolaging.2019.08.007'], ['10.3390/s19020292'], ['10.1016/j.cma.2019.07.015'], ['10.1103/PhysRevB.101.024405'], ['10.1038/s41377-019-0174-6'], ['10.1016/j.jcou.2021.101881'], ['10.1016/j.scriptamat.2021.114490'], ['10.1016/j.cej.2022.134913'], ['10.1109/TRO.2022.3164789'], ['10.1038/s41567-019-0598-1'], ['10.1113/JP282609'], ['10.1002/anie.202001258'], ['10.1103/PhysRevB.100.104426'], ['10.3171/2019.1.JNS1995'], ['10.1186/s12984-019-0612-y'], ['10.1111/mafi.12358'], ['10.5075/epfl-thesis-7154'], ['10.1007/s10703-019-00338-9'], ['10.1116/6.0001686'], ['10.3390/genes10100750'], ['10.1093/mnras/stad2419'], ['10.1002/marc.202200196'], ['10.1016/j.spa.2018.03.013'], ['10.1109/MPE.2022.3230968'], ['10.3389/frobt.2023.1255666'], ['10.1002/admi.201900042'], ['10.1109/TASC.2022.3162175'], ['10.1109/ESSDERC53440.2021.9631804'], ['10.1145/3460120.3484565'], ['10.1109/EuroSOI-ULIS53016.2021.9560680'], ['10.1371/journal.pcbi.1007886'], ['10.1016/j.copbio.2019.09.008'], ['10.1007/s10543-020-00811-6']]

# The year range used to extract the local dois
yr_min = 2019
yr_max = 2023

# The ROR ID of the institution that maintains the local repository
ror_id = 'https://ror.org/02s376052'

# For now, the only option is 'openalex' but we plan to extend this to other global repositories such as OpenAIRE
global_repo_name = 'openalex'

# A folder where the retrieved data will be saved
# So that we can reuse the data without having to retrieve it again
path_export = 'exported_data/'

# Initialize the processor
# It might take several minutes for a large institution
processor = OverlapProcessing()
processor.prepare_repositories(local_dois, yr_min, yr_max, ror_id, global_repo_name, path_export)
processor.apply_rules()
processor.extract_detailed_stats()
processor.build_bar_chart('epfl_openalex_bar_chart.svg', True)