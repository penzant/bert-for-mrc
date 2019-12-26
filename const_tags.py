def get_dict(key, tags):
    return {tag: key for tag in tags}

ANON_TAG_DICT = {}
proper_tags = ['NNP', 'NNPS']
ANON_TAG_DICT.update(get_dict('noun', proper_tags))
noun_tags = ['NN', 'NNS']
ANON_TAG_DICT.update(get_dict('noun', noun_tags))
adj_tags = ['JJ', 'JJR', 'JJS']
ANON_TAG_DICT.update(get_dict('adjective', adj_tags))
adv_tags = ['RB', 'RBR', 'RBS']
ANON_TAG_DICT.update(get_dict('adverb', adv_tags))
verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
ANON_TAG_DICT.update(get_dict('verb', verb_tags))
number_tags = ['CD']
ANON_TAG_DICT.update(get_dict('number', number_tags))
punct_tags = ["''", '""', '``', '$', '-RRB-', '-LRB-', ',', ':', ';', '#', 'XX', 'SYM']
ANON_TAG_DICT.update(get_dict('punct', punct_tags))
period_tags = ['.']
ANON_TAG_DICT.update(get_dict('period', period_tags))

# new function words
wh_tags = ['WDT', 'WP', 'WP$', 'WRB']
ANON_TAG_DICT.update(get_dict('wh', wh_tags))

# to ignore in tokenization (used for keep inflection features)
POS_TAGS = proper_tags + noun_tags + adj_tags + adv_tags + verb_tags
POS_TAGS = [f'[{tag}]' for tag in POS_TAGS]

UNK_TOKEN = '[UNK]'

# from spacy.load('en_core_web_sm').tagger.labels
# ['$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '_SP', '``']
full_anon_tags = ['ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', 'SP']
## prp$ -> prp
## _sp -> sp
ANON_TAGS = list(set(ANON_TAG_DICT.values())) + ['entity', 'prep', 'other'] + [t.lower() for t in full_anon_tags]
