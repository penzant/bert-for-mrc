# coding=utf-8
"""
input and output: a raw json from corenlp (e.g., parsed doc)
to successively apply multiple functions
"""
import random
import pickle
import string
from collections import defaultdict

from nltk.tree import Tree
from nltk.corpus import stopwords

random.seed(42)

stops = stopwords.words("english")

interrogatives = ["what", "when", "where", "who", "which", "why", "whom", "how"]
question_stops = list(set(stops) - set(interrogatives))

from tokenization import UNK_TOKEN
from entity_anonymizer import ANON_TAGS


def check_anon_tags(token):
    for anon_tag in ANON_TAGS:
        if token.startswith(f'@{anon_tag}'):
            return anon_tag
    return None


def merge_sentences(data, ans_pos):
    """
    if the answer exists over sentence i and sentence i+1 (possibly),
    merge them into a single sentence
    """
    erase_idx = []
    offset = len(data[ans_pos[0][0]])
    for i in range(ans_pos[0][0] + 1, ans_pos[1][0] + 1):
        data[ans_pos[0][0]].extend(data[i])
        erase_idx.append(i)
    for i in erase_idx[:-1]:
        offset += len(data[i])
    for i in reversed(erase_idx):
        del data[i]
    ans_pos = (ans_pos[0], (ans_pos[0][0], ans_pos[1][1] + offset))
    return data, ans_pos


def shuffle_sentence_order(data, ans_pos=None):
    """shuffle the order of sentences"""
    if ans_pos and not ans_pos[0][0] == ans_pos[1][0]:
        data, ans_pos = merge_sentences(data, ans_pos)

    sentence_indices = list(range(len(data)))
    random.shuffle(sentence_indices)
    data = [data[i] for i in sentence_indices]
    if ans_pos:
        new_sent = sentence_indices.index(ans_pos[0][0])
        ans_pos = ((new_sent, ans_pos[0][1]), (new_sent, ans_pos[1][1]))
        return data, ans_pos, " ".join([str(i) for i in sentence_indices])
    else:
        return data, " ".join([str(i) for i in sentence_indices])


def segment_sequence(seq_len, span):
    """
    segnment sequence into some spans, based on the length and position of the given span.
    input: length of sequence, base span (pair of indices)
    output: list of pair of indices of spans
    e.g., seq = 9 = [a, b, c, d, e, f, g, h, i], span=(4,6)
    -> [[a], [b,c,d], [e,f,g], [h,i]]  # to shuffle this segment list
    return (0,0), (1,3), (4,6), (7,8)
    """
    segment_pairs = []
    span_len = span[1] - span[0] + 1
    if span_len >= seq_len:
        return [(0, seq_len - 1)]
    before_span_len = span[0]
    after_span_len = seq_len - span[1] - 1
    start_mod = before_span_len % span_len
    if start_mod > 0:
        segment_pairs.append((0, start_mod - 1))
    for i in range(before_span_len // span_len):
        pos = start_mod + i * span_len
        segment_pairs.append((pos, pos + span_len - 1))
    segment_pairs.append(span)
    for i in range(after_span_len // span_len):
        pos = span[1] + 1 + span_len * i
        segment_pairs.append((pos, pos + span_len - 1))
    if after_span_len > 0:
        end_mod = after_span_len % span_len
        if end_mod > 0:
            end_pos = seq_len - 1
            segment_pairs.append((end_pos - end_mod + 1, end_pos))
    # print(segment_pairs)
    return segment_pairs


def shuffle_sentence_words(data, ans_pos, fix_eos=False):
    """shuffle words in each sentence"""
    if ans_pos and not ans_pos[0][0] == ans_pos[1][0]:
        data, ans_pos = merge_sentences(data, ans_pos)

    ret_data = []
    ablation_info = []
    for si, sent in enumerate(data):
        if ans_pos[0][0] == si:
            segment_base_span = (ans_pos[0][1], ans_pos[1][1])
        else:
            segment_base_span = (0, ans_pos[1][1] - ans_pos[0][1])
        segment_indices = segment_sequence(len(sent), segment_base_span)
        if fix_eos:
            # if answer span is the last segment (the last token is eos like a period)
            if segment_indices[-1] == segment_base_span:
                segment_indices_except_eos = segment_indices[:-1]
                random.shuffle(segment_indices_except_eos)
                segment_indices[:-1] = segment_indices_except_eos
            else:
                segment_indices_except_eos = segment_indices[:-1]
                if segment_indices[-1][0] < len(sent) - 1:
                    segment_indices_except_eos.append(
                        (segment_indices[-1][0], len(sent) - 2)
                    )
                random.shuffle(segment_indices_except_eos)
                segment_indices = segment_indices_except_eos
                segment_indices.append((len(sent) - 1, len(sent) - 1))
        else:
            random.shuffle(segment_indices)

        sent = [sent[i] for seg in segment_indices for i in range(seg[0], seg[1] + 1)]

        if ans_pos[0][0] == si:
            pos = 0
            for span in segment_indices:
                if span == (ans_pos[0][1], ans_pos[1][1]):
                    break
                pos += span[1] - span[0] + 1
            new_ans_pos = ((si, pos), (si, pos + ans_pos[1][1] - ans_pos[0][1]))
        ret_data.append(sent)

        ablation_info.append(
            " ".join(["({},{})".format(seg[0], seg[1]) for seg in segment_indices])
        )
    return ret_data, new_ans_pos, "\n".join(ablation_info)


def shuffle_whole_words(data, doc_ans_pos):
    """shuffle tokens into single sentence"""
    ablation_info = ""
    seq_len = len(sum(data, []))
    segment_indices = segment_sequence(seq_len, doc_ans_pos)
    flatten_tokens = []
    for sent in data:
        for token in sent:
            flatten_tokens.append(token)
    random.shuffle(segment_indices)
    data = [
        flatten_tokens[i] for seg in segment_indices for i in range(seg[0], seg[1] + 1)
    ]
    ablation_info += " ".join(
        ["({},{})".format(seg[0], seg[1]) for seg in segment_indices]
    )
    pos = 0
    for span in segment_indices:
        if span == doc_ans_pos:
            break
        pos += span[1] - span[0] + 1
    new_doc_ans_pos = (pos, pos + doc_ans_pos[1] - doc_ans_pos[0])
    return data, new_doc_ans_pos, ablation_info


def shuffle_question_words(data):
    """shuffle tokens into single sentence"""
    ablation_info = ""
    word_indices = []
    for si, sent in enumerate(data):
        for ti in range(len(sent)):
            word_indices.append((si, ti))
        ablation_info += "sentence {}: {} tokens\n".format(si, ti)
    random.shuffle(word_indices)
    shuffled_words = []
    for (si, ti) in word_indices:
        shuffled_words.append(data[si][ti])
    data = [shuffled_words]
    ablation_info += " ".join(["{}_{}".format(i, j) for (i, j) in word_indices])
    return data, ablation_info


def tokenize_corenlp_data(data):
    tokens = []
    for sent in data["sentences"]:
        for token in sent["tokens"]:
            tokens.append(token["word"])
    return tokens


def tokenize_query_data(data, replace_dict):
    tokens = []
    token_count = -1
    for sent in data["sentences"]:
        for token in sent["tokens"]:
            token_count += 1
            tokens.append(replace_dict.get(token_count, token["word"]))
    return tokens


def drop_content_words(data):
    for sent in data["sentences"]:
        for token in sent["tokens"]:
            if token["word"].lower() not in stops:
                token["word"] = ""
    return data


def get_doc_answer_position(data, answer_position):
    """return sentence-based answer-span position"""
    start_pos, end_pos = answer_position
    count_tokens = -1
    doc_start_pos, doc_end_pos = None, None
    for si, sent in enumerate(data):
        for ti, token in enumerate(sent):
            count_tokens += 1
            if start_pos == (si, ti):
                doc_start_pos = count_tokens
            if end_pos == (si, ti):
                doc_end_pos = count_tokens
    if doc_start_pos is None:
        import pdb; pdb.set_trace()
    assert doc_start_pos is not None
    assert doc_end_pos is not None
    if doc_end_pos is None:
        import pdb

        pdb.set_trace()
    return (doc_start_pos, doc_end_pos)


def get_sentence_split_indices(tok_to_orig_index, orig_ans_pos, parsed_doc):
    """return a list of sequences of orig token ids (sequence = sentence)"""
    if len(tok_to_orig_index) > len(tokenize_corenlp_data(parsed_doc)):
        tok_to_orig_index = tok_to_orig_index[: len(tokenize_corenlp_data(parsed_doc))]
    if len(tok_to_orig_index) < len(tokenize_corenlp_data(parsed_doc)):
        tok_to_orig_index.extend(
            [tok_to_orig_index[-1]]
            * (len(tokenize_corenlp_data(parsed_doc)) - len(tok_to_orig_index))
        )
    # assert len(tok_to_orig_index) == len(tokenize_corenlp_data(parsed_doc))
    token_count = -1
    orig_sentences = []
    cn_start_pos, cn_end_pos = (0, 0), (0, 0)
    for si, sent in enumerate(parsed_doc["sentences"]):
        sentence = []
        for ti, token in enumerate(sent["tokens"]):
            token_count += 1
            if not tok_to_orig_index[token_count] in sentence:
                sentence.append(tok_to_orig_index[token_count])
        orig_sentences.append(sentence)

    for i in reversed(range(len(orig_sentences) - 1)):
        if orig_sentences[i][-1] == orig_sentences[i + 1][0]:
            orig_sentences[i].extend(orig_sentences[i + 1])
            orig_sentences[i] = sorted(set(orig_sentences[i]))
            del orig_sentences[i + 1]

    for i in range(len(orig_sentences)):
        sent = orig_sentences[i]
        if sent[-1] - sent[0] + 1 > len(sent):
            orig_sentences[i] = list(range(sent[0], sent[-1] + 1))

    token_count = -1
    for si, sent in enumerate(orig_sentences):
        for ti in range(len(sent)):
            token_count += 1
            if orig_ans_pos[0] == token_count:
                cn_start_pos = (si, ti)
            if orig_ans_pos[1] == token_count:
                cn_end_pos = (si, ti)

    return orig_sentences, (cn_start_pos, cn_end_pos)


def reconstruct_doc_tokens(
    orig_doc_tokens, tok_mask_dict, tok_to_orig_index, parsed_doc, mode=None
):
    if isinstance(parsed_doc, dict):
        tokenized_doc = [
            token["word"]
            for sentence in parsed_doc["sentences"]
            for token in sentence["tokens"]
        ]
    else:
        tokenized_doc = parsed_doc

    cand_orig_pos = [
        tok_to_orig_index[i] for i in tok_mask_dict.keys() if len(tok_to_orig_index) > i
    ]
    cand_orig_tokens = {}
    for orig_index in cand_orig_pos:
        cand_orig_tokens[orig_index] = [
            ti for ti, oi in enumerate(tok_to_orig_index) if oi == orig_index
        ]

    doc_mask_dict = {}
    for orig_index, tok_indices in cand_orig_tokens.items():
        if len(tok_indices) == 1:
            if mode == 'sout':
                doc_mask_dict[orig_index] = (
                    "\sout{" + "{0}".format(tokenized_doc[tok_indices[0]]) + "}"
                )
                # doc_mask_dict[orig_index] = '<span class="reducted">' + '{0}'.format(tokenized_doc[tok_indices[0]]) + '</span>'
            else:
                doc_mask_dict[orig_index] = tok_mask_dict[tok_indices[0]]
        else:
            # new_subtokens = [tok_mask_dict.get(tok_index, tokenized_doc[tok_index])
            #                  for tok_index in tok_indices]
            new_subtokens = []
            for tok_index in tok_indices:
                tok = tok_mask_dict.get(tok_index, tokenized_doc[tok_index])
                if tok == UNK_TOKEN and False:
                    tok = "\sout{" + "{0}".format(tokenized_doc[tok_index]) + "}"
                # if tok == UNK_TOKEN and False:
                #     tok = '<span class="reducted">{0}</span>'.format(tokenized_doc[tok_index])
                new_subtokens.append(tok)
            for i in range(len(new_subtokens) - 1):
                # if new_subtokens[i].startswith("@entity"):
                if check_anon_tags(new_subtokens[i]):
                    new_subtokens[i] += " "

            new_token = "".join(new_subtokens)

            new_token = new_token.replace(UNK_TOKEN, "") or UNK_TOKEN
            doc_mask_dict[orig_index] = new_token

    ret_doc_tokens = [
        doc_mask_dict.get(pos, token) for pos, token in enumerate(orig_doc_tokens)
    ]
    return ret_doc_tokens


def mask_tokens(parsed_doc, mask_tokens, complement=False, target_poses=None):
    mask_dict = {}
    token_count = -1
    for sent in parsed_doc["sentences"]:
        for token in sent["tokens"]:
            token_count += 1
            lower_token = token["word"].lower()
            if target_poses and not token["pos"] in target_poses:
                continue
            if isinstance(mask_tokens, defaultdict):
                dummy = mask_tokens[lower_token]
            if (lower_token in mask_tokens and not complement) or (
                lower_token not in mask_tokens and complement
            ):
                if isinstance(mask_tokens, dict):
                    mask_dict[token_count] = mask_tokens[lower_token]
                else:
                    mask_dict[token_count] = UNK_TOKEN
    return mask_dict


def drop_except_most_similar_sentences(parsed_doc, query_tokens):
    overlap_count = []
    for sent in parsed_doc["sentences"]:
        count = 0
        for token in sent["tokens"]:
            lower_token = token["word"].lower()
            if lower_token in query_tokens:
                count += 1
        overlap_count.append(count)

    most_similar_indices = [
        si for si, c in enumerate(overlap_count) if c == max(overlap_count)
    ]
    mask_dict = {}
    token_count = -1
    for si, sent in enumerate(parsed_doc["sentences"]):
        if si not in most_similar_indices:
            for token in sent["tokens"]:
                token_count += 1
                mask_dict[token_count] = UNK_TOKEN
        else:
            token_count += len(sent["tokens"])

    return mask_dict


def generate_ablated_input(
    option,
    orig_doc_tokens,
    orig_answer_position,
    tok_to_orig_index,
    parsed_doc,
    parsed_example,
):
    """main function for the generation of ablation input."""

    # lexicon-agnostic methods
    if option == "shuffle_sentence_order":
        orig_sent_indices, cn_ans_pos = get_sentence_split_indices(
            tok_to_orig_index, orig_answer_position, parsed_doc
        )
        shuffled, cn_ans_pos, ab_info = shuffle_sentence_order(
            orig_sent_indices, cn_ans_pos
        )
        doc_ans_pos = get_doc_answer_position(shuffled, cn_ans_pos)
        ablated_input = {
            "doc_tokens": [orig_doc_tokens[i] for i in sum(shuffled, [])],
            "start_position": doc_ans_pos[0],
            "end_position": doc_ans_pos[1],
            "ablation_info": ab_info,
        }

    elif option == "shuffle_sentence_words":
        orig_sent_indices, cn_ans_pos = get_sentence_split_indices(
            tok_to_orig_index, orig_answer_position, parsed_doc
        )
        shuffled, cn_ans_pos, ab_info = shuffle_sentence_words(
            orig_sent_indices, cn_ans_pos, fix_eos=True
        )
        doc_ans_pos = get_doc_answer_position(shuffled, cn_ans_pos)
        # if ' '.join(orig_doc_tokens[orig_answer_position[0]:orig_answer_position[1]+1]) != ' '.join([orig_doc_tokens[i] for i in sum(shuffled, [])][doc_ans_pos[0]:doc_ans_pos[1]+1]):
        #     print('nothing...')
        ablated_input = {
            "doc_tokens": [orig_doc_tokens[i] for i in sum(shuffled, [])],
            "start_position": doc_ans_pos[0],
            "end_position": doc_ans_pos[1],
            "ablation_info": ab_info,
        }

    elif option == "shuffle_sentence_words_without_fix_eos":
        orig_sent_indices, cn_ans_pos = get_sentence_split_indices(
            tok_to_orig_index, orig_answer_position, parsed_doc
        )
        shuffled, cn_ans_pos, ab_info = shuffle_sentence_words(
            orig_sent_indices, cn_ans_pos, fix_eos=False
        )
        doc_ans_pos = get_doc_answer_position(shuffled, cn_ans_pos)
        ablated_input = {
            "doc_tokens": [orig_doc_tokens[i] for i in sum(shuffled, [])],
            "start_position": doc_ans_pos[0],
            "end_position": doc_ans_pos[1],
            "ablation_info": ab_info,
        }

    elif option == "shuffle_sentence_words_and_order":
        ab_info = ""
        orig_sent_indices, cn_ans_pos = get_sentence_split_indices(
            tok_to_orig_index, orig_answer_position, parsed_doc
        )
        shuffled, cn_ans_pos, tmp_info = shuffle_sentence_words(
            orig_sent_indices, cn_ans_pos
        )
        ab_info += tmp_info
        shuffled, cn_ans_pos, tmp_info = shuffle_sentence_order(shuffled, cn_ans_pos)
        ab_info += tmp_info
        doc_ans_pos = get_doc_answer_position(shuffled, cn_ans_pos)
        ablated_input = {
            "doc_tokens": [orig_doc_tokens[i] for i in sum(shuffled, [])],
            "start_position": doc_ans_pos[0],
            "end_position": doc_ans_pos[1],
            "ablation_info": ab_info,
        }

    elif option == "shuffle_document_words":
        orig_sent_indices, _ = get_sentence_split_indices(
            tok_to_orig_index, orig_answer_position, parsed_doc
        )
        shuffled, doc_ans_pos, ab_info = shuffle_whole_words(
            orig_sent_indices, orig_answer_position
        )
        ablated_input = {
            "doc_tokens": [orig_doc_tokens[i] for i in shuffled],
            "start_position": doc_ans_pos[0],
            "end_position": doc_ans_pos[1],
            "ablation_info": ab_info,
        }

    elif option == "shuffle_question_words":
        shuffled_query, ab_info = shuffle_question_words(parsed_example["query"])
        ablated_input = {
            "query_tokens": tokenize_corenlp_data(shuffled_query),
            "ablation_info": ab_info,
        }

    elif option.startswith("question_first_"):
        query_tokens = tokenize_corenlp_data(parsed_example["query"])
        query_tokens = query_tokens[: int(option.replace("question_first_", ""))]
        ablated_input = {"query_tokens": query_tokens, "ablation_info": option}

    # lexicon-dependent methods
    elif option.startswith("drop_function_words") or option == "drop_content_words":
        import pdb

        pdb.set_trace()
        if (
            "unknown" in orig_doc_tokens
            and "yes" in orig_doc_tokens
            and "no" in orig_doc_tokens
        ):
            if option.startswith("drop_content_words"):
                stops.extend(["unknown", "yes"])
            elif option.startswith("drop_function_words"):
                if "no" in stops:
                    del stops[stops.index("no")]
        if option == "drop_function_words_with_punct":
            tok_mask_dict = mask_tokens(parsed_doc, stops + list(string.punctuation))
        elif option == "drop_content_words":
            tok_mask_dict = mask_tokens(
                parsed_doc, stops + list(string.punctuation), complement=True
            )
        else:
            tok_mask_dict = mask_tokens(parsed_doc, stops, complement=False)
        ablated_doc_tokens = reconstruct_doc_tokens(
            orig_doc_tokens, tok_mask_dict, tok_to_orig_index, parsed_doc
        )
        ablated_input = {
            "doc_tokens": ablated_doc_tokens,
            "ablation_info": option,
            "refer_to_orig_doc": True,
            "drop_dict": tok_mask_dict,
        }

    elif option == "drop_punctuations":
        tok_mask_dict = mask_tokens(parsed_doc, list(string.punctuation))
        ablated_doc_tokens = reconstruct_doc_tokens(
            orig_doc_tokens, tok_mask_dict, tok_to_orig_index, parsed_doc
        )
        ablated_input = {
            "doc_tokens": ablated_doc_tokens,
            "ablation_info": option,
            "refer_to_orig_doc": True,
            "drop_dict": tok_mask_dict,
        }

    elif option == "drop_logical_words":
        logical_words = [
            "not",
            "n't",
            "all",
            "any",
            "each",
            "every",
            "few",
            "if",
            "more",
            "most",
            "no",
            "nor",
            "other",
            "same",
            "some",
            "than",
        ]  # v1
        if (
            "unknown" in orig_doc_tokens
            and "yes" in orig_doc_tokens
            and "no" in orig_doc_tokens
        ):
            del logical_words[logical_words.index("no")]
        # logical_words = ['not', "n't", 'all', 'any', 'each', 'every', 'few',
        #                  'if', 'more', 'most', 'no', 'nor', 'other', 'same',
        #                  'some', 'than', 'and', 'or']  # v2, FraCaS
        tok_mask_dict = mask_tokens(parsed_doc, logical_words)
        ablated_doc_tokens = reconstruct_doc_tokens(
            orig_doc_tokens, tok_mask_dict, tok_to_orig_index, parsed_doc
        )
        ablated_input = {
            "doc_tokens": ablated_doc_tokens,
            "ablation_info": option,
            "refer_to_orig_doc": True,
            "drop_dict": tok_mask_dict,
        }

    elif option == "drop_causal_words":
        causal_words = ["because", "why", "therefore", "cause", "reason", "as", "since"]
        tok_mask_dict = mask_tokens(parsed_doc, causal_words)
        ablated_doc_tokens = reconstruct_doc_tokens(
            orig_doc_tokens, tok_mask_dict, tok_to_orig_index, parsed_doc
        )
        ablated_input = {
            "doc_tokens": ablated_doc_tokens,
            "ablation_info": option,
            "refer_to_orig_doc": True,
            "drop_dict": tok_mask_dict,
        }

    elif option == "mask_pronouns":
        # pronoun_mask = {
        #     'i': 'cat',
        #     'we': 'rabbits',
        #     'you': 'ludwig',
        #     'he': 'albert',
        #     'she': 'emily',
        #     'they': 'turtles',
        #     'it': 'pen',
        # }
        pronouns = """i you he she we they it her his mine my our ours their thy your
        hers herself him himself hisself itself me myself one oneself ours
        ourselves ownself self thee theirs them themselves thou thy us""".split()

        tok_mask_dict = mask_tokens(parsed_doc, pronouns, target_poses=["PRP", "PRP$"])
        ablated_doc_tokens = reconstruct_doc_tokens(
            orig_doc_tokens, tok_mask_dict, tok_to_orig_index, parsed_doc
        )
        ablated_input = {
            "doc_tokens": ablated_doc_tokens,
            "ablation_info": option,
            "refer_to_orig_doc": True,
            "drop_dict": tok_mask_dict,
        }

    elif option == "mask_numerics":
        dummy_num_mask = defaultdict(lambda: str(random.randint(1000, 10000)))
        tok_mask_dict = mask_tokens(parsed_doc, dummy_num_mask, target_poses=["CD"])
        ablated_doc_tokens = reconstruct_doc_tokens(
            orig_doc_tokens, tok_mask_dict, tok_to_orig_index, parsed_doc
        )
        ablated_input = {
            "doc_tokens": ablated_doc_tokens,
            "ablation_info": option,
            "refer_to_orig_doc": True,
            "drop_dict": tok_mask_dict,
        }

    elif option == "drop_question_overlaps":
        query_tokens = [
            x.lower() for x in tokenize_corenlp_data(parsed_example["query"])
        ]
        tok_mask_dict = mask_tokens(parsed_doc, query_tokens)
        ablated_doc_tokens = reconstruct_doc_tokens(
            orig_doc_tokens, tok_mask_dict, tok_to_orig_index, parsed_doc
        )
        ablated_input = {
            "doc_tokens": ablated_doc_tokens,
            "ablation_info": option,
            "refer_to_orig_doc": True,
            "drop_dict": tok_mask_dict,
        }

    elif option == "drop_except_most_similar_sentences":
        query_tokens = [
            x.lower() for x in tokenize_corenlp_data(parsed_example["query"])
        ]
        tok_mask_dict = drop_except_most_similar_sentences(parsed_doc, query_tokens)
        ablated_doc_tokens = reconstruct_doc_tokens(
            orig_doc_tokens, tok_mask_dict, tok_to_orig_index, parsed_doc
        )
        ablated_input = {
            "doc_tokens": ablated_doc_tokens,
            "ablation_info": option,
            "refer_to_orig_doc": True,
            "drop_dict": tok_mask_dict,
        }

    elif option in [
        "drop_question_content_words",
        "drop_question_function_words",
        "drop_question_function_words+",  # true function
        "drop_question_content_words+",
    ]:  # true content
        is_complement = option.startswith("drop_question_content_words")
        tok_mask_dict = mask_tokens(
            parsed_example["query"],
            question_stops if not option[-1] == "+" else stops,
            complement=is_complement,
        )
        dropped_query = tokenize_query_data(parsed_example["query"], tok_mask_dict)
        ablated_input = {
            "query_tokens": dropped_query,
            "ablation_info": option,
            "drop_dict": tok_mask_dict,
        }

    elif option == "drop_question_except_interrogatives":
        tok_mask_dict = mask_tokens(
            parsed_example["query"], interrogatives, complement=True
        )
        dropped_query = tokenize_query_data(parsed_example["query"], tok_mask_dict)
        len(tok_mask_dict)
        ablated_input = {
            "query_tokens": dropped_query,
            "ablation_info": option,
            "drop_dict": tok_mask_dict,
        }

    elif option == "drop_question_words":
        ablated_input = {"query_tokens": [], "ablation_info": option}

    else:
        raise ValueError("Invalid input-ablation option: {}".format(option))

    return ablated_input


if __name__ == "__main__":
    data = pickle.load(open("./corenlp_race/race_dev.pkl", "rb"))
    ks = [x for x in data.keys() if not x[-2] == "Q"]
    kss = random.sample(ks, 1)
    test_tokens = tokenize_corenlp_data(data[kss[0]])
    cn_ans_pos = ((0, 14), (2, 3))
    doc_ans_pos = get_doc_answer_position(data[kss[0]], cn_ans_pos)
    print(" ".join(test_tokens))
    print(
        "answer: {}".format(" ".join(test_tokens[doc_ans_pos[0] : doc_ans_pos[1] + 1]))
    )
    # print('answer: {}'.format(' '.join(test_tokens[10:12+1])))
    print("-----------------------------")
    res, new_pos, ab_info = shuffle_sentence_order(data[kss[0]], cn_ans_pos)
    # res, new_pos, ab_info = shuffle_sentence_words(data[kss[0]], (10, 12))
    new_pos = get_doc_answer_position(res, new_pos)
    new_test_tokens = tokenize_corenlp_data(res)
    print(" ".join(new_test_tokens))
    print(
        "new answer: {}".format(" ".join(new_test_tokens[new_pos[0] : new_pos[1] + 1]))
    )
    print(ab_info)
