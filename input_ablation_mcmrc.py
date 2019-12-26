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

UNK_TOKEN = "[UNK]"


def shuffle_sentence_order(data):
    sentence_indices = list(range(len(data["sentences"])))
    random.shuffle(sentence_indices)
    data["sentences"] = [data["sentences"][i] for i in sentence_indices]
    return data, " ".join([str(i) for i in sentence_indices])


def shuffle_sentence_words(data, fix_eos=False):
    ablation_info = []
    for sent in data["sentences"]:
        if fix_eos:
            word_indices = list(range(len(sent["tokens"]) - 1))
            random.shuffle(word_indices)
            word_indices.append(len(sent["tokens"]) - 1)
        else:
            word_indices = list(range(len(sent["tokens"])))
            random.shuffle(word_indices)
        sent["tokens"] = [sent["tokens"][i] for i in word_indices]
        ablation_info.append(" ".join([str(i) for i in word_indices]))
    return data, "\n".join(ablation_info)


def shuffle_whole_words(data):
    """shuffle tokens into single sentence"""
    ablation_info = ""
    word_indices = []
    for si, sent in enumerate(data["sentences"]):
        for ti in range(len(sent["tokens"])):
            word_indices.append((si, ti))
        ablation_info += "sentence {}: {} tokens\n".format(si, ti)
    random.shuffle(word_indices)
    shuffled_words = []
    for (si, ti) in word_indices:
        shuffled_words.append(data["sentences"][si]["tokens"][ti])
    data["sentences"] = [{"tokens": shuffled_words}]
    ablation_info += " ".join(["{}_{}".format(i, j) for (i, j) in word_indices])
    return data, ablation_info


def repr_corenlp_data(data):
    sentences = []
    for sent in data["sentences"]:
        sentence = []
        for token in sent["tokens"]:
            sentence.append(token["word"].lower())
        sentences.append(" ".join(sentence))
    return " ".join(sentences)


def tokenize_corenlp_data(data):
    tokens = []
    for sent in data["sentences"]:
        for token in sent["tokens"]:
            tokens.append(token["word"])
    return tokens


def mask_tokens(data, mask_tokens, complement=False, target_poses=None):
    mask_dict = {}
    token_count = -1
    for sent in data["sentences"]:
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
                    token["word"] = mask_tokens[lower_token]
                else:
                    mask_dict[token_count] = UNK_TOKEN
                    if True:
                        token["word"] = "\sout{" + token["word"] + "}"
                    else:
                        token["word"] = UNK_TOKEN
    return data, mask_dict


def drop_except_most_similar_sentences(data, query_tokens):
    overlap_count = []
    for sent in data["sentences"]:
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
    token_counter = -1
    for si, sent in enumerate(data["sentences"]):
        for token in sent["tokens"]:
            token_counter += 1
            if si not in most_similar_indices:
                token["word"] = UNK_TOKEN
                mask_dict[token_counter] = UNK_TOKEN

    return data, " ".join([str(i) for i in most_similar_indices]), mask_dict


def generate_ablated_input(option, parsed_doc, parsed_example):
    if option == "shuffle_sentence_order":
        shuffled, ab_info = shuffle_sentence_order(parsed_doc)
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(shuffled),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": ab_info,
        }

    elif option == "shuffle_sentence_words":
        shuffled, ab_info = shuffle_sentence_words(parsed_doc, fix_eos=True)
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(shuffled),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": ab_info,
        }

    elif option == "shuffle_sentence_words_without_fix_eos":
        shuffled, ab_info = shuffle_sentence_words(parsed_doc)
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(shuffled),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": ab_info,
        }

    elif option == "shuffle_sentence_words_and_order":
        ab_info = ""
        shuffled_doc, tmp_info = shuffle_sentence_words(parsed_doc)
        ab_info += tmp_info
        shuffled, tmp_info = shuffle_sentence_order(shuffled_doc)
        ab_info += tmp_info
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(shuffled),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": ab_info,
        }

    elif option == "shuffle_document_words":
        shuffled, ab_info = shuffle_whole_words(parsed_doc)
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(shuffled),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": None,
            "ans_idx": None,
            "ablation_info": ab_info,
        }

    elif option == "shuffle_question_words":
        shuffled_query, ab_info = shuffle_whole_words(parsed_example["query"])
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(parsed_doc),
            "query_tokens": tokenize_corenlp_data(shuffled_query),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": ab_info,
        }

    elif option == "shuffle_option_words":
        shuffled_results = [
            shuffle_whole_words(opt) for opt in parsed_example["options"]
        ]
        shuffled_options = [x[0] for x in shuffled_results]
        ab_info = "\n".join([x[1] for x in shuffled_results])
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(parsed_doc),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in shuffled_options
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": ab_info,
        }

    elif option.startswith("question_first_"):
        query_tokens = tokenize_corenlp_data(parsed_example["query"])
        query_tokens = query_tokens[: int(option.replace("question_first_", ""))]
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(parsed_doc),
            "query_tokens": query_tokens,
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": option,
        }

    elif option.startswith("drop_function_words") or option == "drop_content_words":
        if option == "drop_function_words_with_punct":
            dropped_doc, mask_dict = mask_tokens(
                parsed_doc, stops + list(string.punctuation)
            )
        elif option == "drop_content_words":
            dropped_doc, mask_dict = mask_tokens(
                parsed_doc, stops + list(string.punctuation), complement=True
            )
        else:
            dropped_doc, mask_dict = mask_tokens(parsed_doc, stops, complement=False)
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(dropped_doc),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": option,
            "drop_dict": mask_dict,
        }

    elif option == "drop_punctuations":
        dropped_doc, mask_dict = mask_tokens(parsed_doc, list(string.punctuation))
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(dropped_doc),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": option,
            "drop_dict": mask_dict,
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
        ]
        dropped_doc, mask_dict = mask_tokens(parsed_doc, logical_words)
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(dropped_doc),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": option,
            "drop_dict": mask_dict,
        }

    elif option == "drop_causal_words":
        causal_words = ["because", "why", "therefore", "cause", "reason", "as", "since"]
        dropped_doc, mask_dict = mask_tokens(parsed_doc, causal_words)
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(dropped_doc),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": option,
            "drop_dict": mask_dict,
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
        dropped_doc, mask_dict = mask_tokens(
            parsed_doc, pronouns, target_poses=["PRP", "PRP$"]
        )
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(dropped_doc),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": option,
            "drop_dict": mask_dict,
        }

    elif option == "mask_numerics":
        dummy_num = defaultdict(lambda: str(random.randint(10, 100)))
        masked_doc, mask_dict = mask_tokens(parsed_doc, dummy_num, target_poses=["CD"])
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(masked_doc),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": option,
            "drop_dict": mask_dict,
        }

    elif option == "drop_question_overlaps":
        query_tokens = [
            x.lower() for x in tokenize_corenlp_data(parsed_example["query"])
        ]
        masked_doc, mask_dict = mask_tokens(parsed_doc, query_tokens)
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(masked_doc),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": option,
            "drop_dict": mask_dict,
        }

    elif option == "drop_except_most_similar_sentences":
        query_tokens = [
            x.lower() for x in tokenize_corenlp_data(parsed_example["query"])
        ]
        dropped_doc, ab_info, mask_dict = drop_except_most_similar_sentences(
            parsed_doc, query_tokens
        )
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(dropped_doc),
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": ab_info,
            "drop_dict": mask_dict,
        }

    elif option == "drop_all_words":
        ablated_input = {
            "doc_tokens": [],
            "query_tokens": tokenize_corenlp_data(parsed_example["query"]),
            "option_tokens_list": None,
            "ans_idx": None,
            "ablation_info": option,
        }

    elif option in [
        "drop_question_content_words",
        "drop_question_function_words",
        "drop_question_function_words+",  # true function
        "drop_question_content_words+",
    ]:  # true content
        is_complement = option.startswith("drop_question_content_words")
        dropped_query, mask_dict = mask_tokens(
            parsed_example["query"],
            question_stops if not option[-1] == "+" else stops,
            complement=is_complement,
        )
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(parsed_doc),
            "query_tokens": tokenize_corenlp_data(dropped_query),
            "option_tokens_list": None,
            "ans_idx": None,
            "ablation_info": option,
            "drop_dict": mask_dict,
        }

    elif option == "drop_question_except_interrogatives":
        dropped_query, mask_dict = mask_tokens(
            parsed_example["query"], interrogatives, complement=True
        )
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(parsed_doc),
            "query_tokens": tokenize_corenlp_data(dropped_query),
            "option_tokens_list": None,
            "ans_idx": None,
            "ablation_info": option,
            "drop_dict": mask_dict,
        }

    elif option == "drop_all_question_words":
        ablated_input = {
            "doc_tokens": tokenize_corenlp_data(parsed_doc),
            "query_tokens": [],
            "option_tokens_list": [
                tokenize_corenlp_data(opt) for opt in parsed_example["options"]
            ],
            "ans_idx": parsed_example["answer"],
            "ablation_info": option,
        }

    elif option == "drop_except_options":
        ablated_input = {
            "doc_tokens": [],
            "query_tokens": [],
            "option_tokens_list": None,
            "ans_idx": None,
            "ablation_info": option,
        }

    else:
        raise ValueError("Invalid input-ablation option: {}".format(option))

    return ablated_input


if __name__ == "__main__":
    data = pickle.load(
        open("../pytorch-pretrained-bert/corenlp_race/race_train.pkl", "rb")
    )
    ks = [x for x in data.keys()]
    kss = random.sample(ks, 1)
    segment_into_clauses(data[kss])
