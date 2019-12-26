from collections import defaultdict


import spacy

nlp = spacy.load('en_core_web_sm')
# coref_nlp = spacy.load('en_coref_lg')
# nlp = None

from const_tags import *

import tokenization
tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

def anonymize_entity(
    tok_doc_tokens,
    tok_query_tokens,
    doc_cache=None,
    mode_open=False,
    target_tags='proper_noun',
    only_stem=False,
    add_inflection=True,
    query_first=True,  # fixed to True
    option_tokens_list=None,
):
    """
    Return tokens whose proper entities in context and query are anonymised.
    The same entities have the same id (@entity<id>).
    Entity ids are closed in a set of tokens: no cross-document references.

    input: higher tokenized doc and query tokens, cache of entity_dict (for doc)
    output: replace dictionary (tokenization-based indices)
    
    answer_extraction:
    orig_tokens
    -> anonymization for tokenized tokens and reconstruct
    -> ablation for tokenized tokens and reconstruct
    -> basic tokenization (make sure that tokenizer recognizes @entity tokens)
    -> wordpiece

    multiple_choice:
    orig_tokens
    -> basic_tokenize (higher_tokenization)
    -> ablation
    -> wordpiece

    this function does not solve corefernce.

    mode_open = use the same entity tag for the same token in all documents.
    mode_close = use different entity tag (0-indexed) in all documents

    add_inflection: add the information of inflections
    False = @{anon_tag}{id}
    True = @{anon_tag}{id} + [TAG]
    """

    anonymize_func_words = False
    anonymize_func_words_sub = False
    anonymize_func_words_only = False
    if target_tags == 'zero':
        target_poses = 'allcontent'
        anonymize_func_words = True
    elif target_tags == 'whprep':
        target_poses = 'allcontent'
        anonymize_func_words = True
    elif target_tags == 'strongzero':
        target_poses = 'allcontent'
        anonymize_func_words = True
    elif target_tags == 'invstrongzero':
        target_poses = 'allcontent'
        anonymize_func_words = True
        query_first = True
    elif target_tags == 'allfunction':
        anonymize_func_words = True
        anonymize_func_words_only = True

    # elif target_tags == 'noposzero':
    #     target_poses = 'allcontent'
    #     anonymize_func_words = True
    #     ANON_TAG_DICT = defaultdict(lambda: 'other')
    #     query_first = True

    if target_tags == 'proper_noun':
        target_poses = proper_tags
    elif target_tags == 'noun':
        target_poses = proper_tags + noun_tags
    elif target_tags == 'content':
        target_poses = proper_tags + noun_tags + adj_tags + adv_tags + number_tags
    elif target_tags == 'contentverb':
        target_poses = proper_tags + noun_tags + adj_tags + adv_tags + number_tags + verb_tags
    elif target_tags == 'allcontent':
        target_poses = proper_tags + noun_tags + adj_tags + adv_tags + number_tags + verb_tags + punct_tags + period_tags
    elif target_tags == 'subcontent' or target_tags == 'allfunction':
        target_poses = proper_tags + noun_tags + adj_tags + adv_tags + number_tags + verb_tags
    elif anonymize_func_words and not anonymize_func_words_only:
        target_poses = proper_tags + noun_tags + adj_tags + adv_tags + number_tags + verb_tags + punct_tags + period_tags
    else:
        ValueError(target_poses)

    if only_stem:
        add_inflection = True
    if add_inflection:
        inflection_tags = ['noun', 'verb', 'adjective', 'adverb']
    else:
        inflection_tags = []

    tag_counter = defaultdict(int)
    entity_dict, simple_entity_dict = {}, {}
    # entity_dict = {token: @verb3 [VBZ], ....}
    # # assumption: same token is same pos (the latest one is dominant)
    # simple_entity_dict = {token['pos[0]']: @verb3, ...}
    # # assumption: same token is same pos

    def anonymize_tokens(tokens_obj):
        for token_obj in tokens_obj:
            if token_obj.tag_ not in target_poses and not anonymize_func_words:
                continue
            if token_obj.tag_ in target_poses and anonymize_func_words_only:
                continue
            if target_tags == 'whprep' and (token_obj.tag_[0] == 'W' or token_obj.tag_ in ['IN' or 'TO']):
                continue
            tag = token_obj.tag_
            token = token_obj.text.lower()
            token_lemma = token_obj.lemma_.lower()
            simple_dict_tag = f'{token_lemma}[{tag[0]}]'
            # key pattern and expression
            # stem: {lemmaX: @tag}
            # inf: {lemmaX: @tag [pos]}
            # normal: {token: @tag}
            # where X is the first character of _.tag_
            # in all cases, same literal with diff class (noun, verb) has diff anon_tag & id
            # but to use cache, you need to have X and lemma... thus how about just have spacy result?
            if target_tags != 'noposzero' and anonymize_func_words and tag not in ANON_TAG_DICT:
                if tag == 'IN' or tag == 'TO':
                    ANON_TAG_DICT[tag] = 'prep'
                elif 'strongzero' in target_tags:
                    ANON_TAG_DICT[tag] = 'other'
                else:
                    ANON_TAG_DICT[tag] = tag.lower().replace('$', '').replace('_', '')
            if add_inflection and ANON_TAG_DICT[tag] in inflection_tags:
                if simple_dict_tag in simple_entity_dict:
                    anon_tag = simple_entity_dict[simple_dict_tag]
                    if only_stem:
                        token_tag = anon_tag
                    else:
                        token_tag = f'{anon_tag} [{tag}]'
                    entity_dict[token] = token_tag
                else:
                    token_tag = f'@{ANON_TAG_DICT[tag]}{tag_counter[ANON_TAG_DICT[tag]]}'
                    simple_entity_dict[simple_dict_tag] = token_tag
                    if only_stem:
                        # stem only
                        entity_dict[token] = token_tag
                    else:
                        # stem and inflection
                        entity_dict[token] = f'{token_tag} [{tag}]'
                    tag_counter[ANON_TAG_DICT[tag]] += 1                
            else:
                # no inflection and only_stem
                if token not in entity_dict:
                    token_tag = f'@{ANON_TAG_DICT[tag]}{tag_counter[ANON_TAG_DICT[tag]]}'
                    tag_counter[ANON_TAG_DICT[tag]] += 1
                    entity_dict[token] = token_tag

    if not doc_cache:
        doc = nlp(' '.join(tok_doc_tokens))
    else:
        doc = doc_cache

    query = nlp(' '.join(tok_query_tokens))

    if option_tokens_list:
        options = [nlp(' '.join(opt)) for opt in option_tokens_list]

    def generate_replace_dict(tokens):
        if anonymize_func_words and not anonymize_func_words_only:
            return {ti: entity_dict[token.lower()] if token.lower() in entity_dict else UNK_TOKEN
                    for ti, token in enumerate(tokens)}
        else:
            return {ti: entity_dict[token.lower()] for ti, token
                    in enumerate(tokens) if token.lower() in entity_dict}
    replace_dict = {}
    if query_first:
        anonymize_tokens(query)
        replace_dict['query'] = generate_replace_dict(tok_query_tokens)
        if option_tokens_list:
            replace_dict['options'] = []
            # replace_dict['options'] = [{}, {}, {}, {}]
            for opt, opt_tokens in zip(options, option_tokens_list):
                anonymize_tokens(opt)
                replace_dict['options'].append(generate_replace_dict(opt_tokens))
        anonymize_tokens(doc)
        replace_dict['doc'] = generate_replace_dict(tok_doc_tokens)
    else:
        anonymize_tokens(doc)
        replace_dict['doc'] = generate_replace_dict(tok_doc_tokens)
        anonymize_tokens(query)
        replace_dict['query'] = generate_replace_dict(tok_query_tokens)
        if option_tokens_list:
            replace_dict['options'] = []
            # for opt, opt_tokens in zip(options, option_tokens_list):
            #     anonymize_tokens(opt)
            #     replace_dict['options'].append(generate_replace_dict(opt_tokens))

    return replace_dict, doc  # entity_dict

