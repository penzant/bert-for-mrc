# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on Answer Extraction MRC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import logging
import json
import math
import os
import random
import six
from tqdm import tqdm, trange
import pickle
import glob

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import arguments
import processor as dataset_processor
import tokenization
import output_ans_extr
from entity_anonymizer import anonymize_entity
from modeling import BertConfig, BertForQuestionAnswering
from optimization import BERTAdam
from input_ablation_ans_extr import generate_ablated_input, reconstruct_doc_tokens
from input_example import InputTokenizedAnswerExtractionExample, InputAnswerExtractionFeature
from vocabulary_selection import vocab_selection

import score_functions
dev_score_function = {  # TODO
    "hotpot": score_functions.HotpotQAEvaluator
}["hotpot"]('./data/hotpot/hotpot_dev_distractor_v1.json')
score_function = dev_score_function

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

entity_set = set()
entity_cache = {}


def tokenize_with_positions(
    doc_tokens, tokenizer, start_position=None, end_position=None
):
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if not start_position or not end_position:
        return all_doc_tokens, tok_to_orig_index

    tok_start_position = orig_to_tok_index[start_position]
    if end_position < len(doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[end_position + 1] - 1
    else:
        tok_end_position = len(all_doc_tokens) - 1
    return all_doc_tokens, tok_start_position, tok_end_position, tok_to_orig_index


def generate_tokenized_examples(
    examples,
    input_ablation=None,
    corenlp_cache=None,
    tokenizer=None,
    entity_anonymization=None,
):
    """
    input: example given by dataset processor
    output: example in which original doc/query tokens are not tokenized, just modified
            to preserve the positions of answer span
    """
    global entity_set
    
    tokenized_examples = []
    for ei, example in enumerate(tqdm(examples)):
        ablation_info = ""
        refer_to_orig_doc = False

        doc_tokens = example.doc_tokens[:]
        start_position, end_position = example.start_position, example.end_position
        query_tokens = example.query_text

        # standard setting
        if entity_anonymization or example.input_ablation:
            basic_higher_tokenizer = tokenization.BasicTokenizer(do_lower_case=False)
            tok_doc_tokens, tok_to_orig_index = tokenize_with_positions(
                doc_tokens, basic_higher_tokenizer
            )
            tok_query_tokens, query_tok_to_orig_index = tokenize_with_positions(
                query_tokens.split(), basic_higher_tokenizer
            )
            query_tokens = query_tokens.split()

        if entity_anonymization:
            refer_to_orig_doc = True
            anon_tags = None
            add_inf = False
            stemming = False
            ent_anon_spec = entity_anonymization.split('_')
            if len(ent_anon_spec) > 1:
                anon_tags = ent_anon_spec[1]
                if len(ent_anon_spec) == 3:
                    add_inf = ent_anon_spec[2] == 'inf'
                    stemming = ent_anon_spec[2] == 'stem'
            if entity_anonymization.startswith("close"):
                # use cache for each example.did (entity_cache)
                # use different vocab for different question (query_anonym_dict)
                anonym_dicts, entity_dict = anonymize_entity(
                    tok_doc_tokens,
                    tok_query_tokens,
                    entity_cache.get(example.did, None),
                    target_tags=anon_tags,
                    only_stem=stemming,
                    add_inflection=add_inf,
                )
                if example.did not in entity_cache:
                    entity_cache[example.did] = entity_dict
                entity_set.update(list(anonym_dicts['doc'].values()))
                entity_set.update(list(anonym_dicts['query'].values()))
            elif entity_anonymization.startswith("open"):
                anonym_dict, query_anonym_dict, update_entity_dict = anonymize_entity(
                    tok_doc_tokens,
                    tok_query_tokens,
                    entity_cache,
                    mode_open=True,
                    target_tags=entity_tags,
                )

                entity_cache.update(update_entity_dict)

            doc_tokens = reconstruct_doc_tokens(
                doc_tokens, anonym_dicts['doc'], tok_to_orig_index, tok_doc_tokens
            )

            query_tokens = reconstruct_doc_tokens(
                example.query_text.split(),
                anonym_dicts['query'],
                query_tok_to_orig_index,
                tok_query_tokens,
            )

        if example.input_ablation and example.input_ablation != "original":
            if example.did == example.qid:
                doc_cache = corenlp_cache[example.qid]["doc"]
            else:
                doc_cache = corenlp_cache[example.did]
            answer_position = (start_position, end_position)
            ablated_example = generate_ablated_input(
                example.input_ablation,
                doc_tokens,
                answer_position,
                tok_to_orig_index,
                doc_cache,
                corenlp_cache[example.qid],
            )
            doc_tokens = ablated_example.get("doc_tokens", doc_tokens)
            query_tokens = ablated_example.get("query_tokens", query_tokens)
            start_position = ablated_example.get("start_position", start_position)
            end_position = ablated_example.get("end_position", end_position)
            ablation_info = ablated_example.get(
                "drop_dict", ablated_example.get("ablation_info", "")
            )
            refer_to_orig_doc = ablated_example.get("refer_to_orig_doc", False)
            if ei < 10:
                print(ablation_info)

            # set 'is_impossible' to mix_input_ablation
            # if input_ablation == 'mix_input_ablation':
            #     start_position = 0
            #     end_position = 0
            #     example.is_impossible = True

        tokenized_examples.append(
            InputTokenizedAnswerExtractionExample(
                example.did,
                example.qid,
                doc_tokens,
                query_tokens,
                start_position,
                end_position,
                example.orig_answer_text,
                ablation_info,
                example.doc_tokens,
                example.query_text,
                example.orig_answer_texts,
                example.is_impossible,
                refer_to_orig_doc,
            )
        )

    if entity_anonymization and tokenizer and len(entity_cache) > 0:
        if entity_anonymization.startswith("open"):
            entity_set.update(list(entity_cache.values()))
        if len(entity_anonymization.split('_')) > 2:
            if entity_anonymization.split('_')[2] == 'inf':
                new_entity_set = []
                for ent in entity_set:
                    if ' ' in ent:
                        new_entity_set.extend(ent.split())
                    else:
                        new_entity_set.append(ent)
                entity_set = set(new_entity_set)
        print("vocabulary updated")
        print(sorted(entity_set))
        tokenizer.vocab_update(sorted(entity_set))

    return tokenized_examples


def convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    is_output_example=False,
    # is_output_example=True,
    ignore_out_of_span=False,
):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    tokenize_fn = tokenizer.tokenize

    features = []
    for (example_index, example) in enumerate(
        tqdm(examples) if is_output_example else examples
    ):
        if type(example.query_tokens) == list:
            query_tokens = [
                t for token in example.query_tokens for t in tokenize_fn(token)
            ]
        else:
            query_tokens = tokenize_fn(example.query_tokens)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenize_fn(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens,
                tok_start_position,
                tok_end_position,
                tokenize_fn,
                example.orig_answer_text,
            )

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"]
        )
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index
                )
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if (
                    example.start_position < doc_start
                    or example.end_position < doc_start
                    or example.start_position > doc_end
                    or example.end_position > doc_end
                ):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    if ignore_out_of_span:
                        continue
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 5 and is_output_example:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("orig_doc_tokens %s" % ' '.join(example.orig_doc_tokens))
                logger.info("query_text %s" % example.orig_query_tokens)
                logger.info(
                    "tokens: %s"
                    % " ".join([tokenization.printable_text(x) for x in tokens])
                )
                logger.info(
                    "token_to_orig_map: %s"
                    % " ".join(
                        [
                            "%d:%d" % (x, y)
                            for (x, y) in six.iteritems(token_to_orig_map)
                        ]
                    )
                )
                logger.info(
                    "token_is_max_context: %s"
                    % " ".join(
                        [
                            "%d:%s" % (x, y)
                            for (x, y) in six.iteritems(token_is_max_context)
                        ]
                    )
                )
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position : (end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (tokenization.printable_text(answer_text))
                    )
                if is_training and example.is_impossible:
                    logger.info("impossible example")

            features.append(
                InputAnswerExtractionFeature(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible,
                )
            )
            unique_id += 1

    return features


def _improve_answer_span(
    doc_tokens, input_start, input_end, tokenize_fn, orig_answer_text
):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenize_fn(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits"]
)


def run_eval_model(model, eval_features, args, device, show_evaluating=False):
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long
    )
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_example_index
    )
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    if show_evaluating:
        eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")

    all_results = []
    for input_ids, input_mask, segment_ids, example_indices in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(
                input_ids, segment_ids, input_mask
            )
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits,
                )
            )

    return all_results


# for debug or distractor_ablation
def get_predictions(model, examples, eval_features, args, device, show_tqdm=False, get_nbest=0):
    all_results = run_eval_model(model, eval_features, args, device, show_tqdm)

    # ref: function write_prediction
    example_index_to_features = collections.defaultdict(list)
    for feature in eval_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {result.unique_id: result for result in all_results}

    # logits
    answer_logits = []
    for (example_index, example) in enumerate(examples):
        features = example_index_to_features[example_index]
        ans_logits = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            if not 0 <= feature.start_position < len(result.start_logits):
                continue
            ans_start_logit = result.start_logits[feature.start_position]
            if not 0 <= feature.end_position < len(result.end_logits):
                continue
            ans_end_logit = result.end_logits[feature.end_position]
            ans_logits.append(ans_start_logit + ans_end_logit)
        answer_logits.append(max(ans_logits))

    # for example, feature in zip(examples, eval_features):
    arg_nsdt = args.null_score_diff_threshold if args.allow_impossible else None
    predictions = write_predictions(
        examples,
        eval_features,
        all_results,
        get_nbest or 20,
        30,
        True,
        null_score_diff_threshold=arg_nsdt,
        return_prediction=True if not get_nbest else False,
        return_nbest=True if get_nbest else False,
    )

    return answer_logits, predictions


def write_predictions(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file=None,
    output_nbest_file=None,
    output_null_log_odds_file=None,
    null_score_diff_threshold=None,
    verbose_logging=False,
    return_prediction=False,
    return_nbest=False,
):
    """Write final predictions to the json file."""
    if output_prediction_file:
        logger.info("Writing predictions to: %s" % (output_prediction_file))
    if output_nbest_file:
        logger.info("Writing nbest to: %s" % (output_nbest_file))

    allow_impossible = null_score_diff_threshold is not None

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if allow_impossible:  # should be args.allow_impossible
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )

        if allow_impossible:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True,
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction",
            [
                "text",
                "start_logit",
                "end_logit",
                "orig_start_position",
                "orig_end_position",
            ],
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                if example.refer_to_orig_doc:
                    orig_tokens = example.orig_doc_tokens[
                        orig_doc_start : (orig_doc_end + 1)
                    ]
                else:
                    orig_tokens = example.doc_tokens[
                        orig_doc_start : (orig_doc_end + 1)
                    ]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text, orig_text, do_lower_case, verbose_logging
                )
                if final_text in seen_predictions:
                    continue
            else:
                orig_doc_start = -1
                orig_doc_end = -1
                final_text = ""

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    orig_start_position=orig_doc_start,
                    orig_end_position=orig_doc_end,
                )
            )

        if allow_impossible and "" not in seen_predictions:
            nbest.append(
                _NbestPrediction(
                    text="",
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                    orig_start_position=-1,
                    orig_end_position=-1,
                )
            )
            seen_predictions[""] = True

        if not nbest or (
            allow_impossible and "" in seen_predictions and len(nbest) == 1
        ):
            nbest.append(
                _NbestPrediction(
                    text="empty",
                    start_logit=0.0,
                    end_logit=0.0,
                    orig_start_position=-1,
                    orig_end_position=-1,
                )
            )

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["orig_start_position"] = entry.orig_start_position
            output["orig_end_position"] = entry.orig_end_position
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not allow_impossible:
            all_predictions[example.qid] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = (
                score_null
                - best_non_null_entry.start_logit
                - (best_non_null_entry.end_logit)
            )
            scores_diff_json[example.qid] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qid] = ""
            else:
                all_predictions[example.qid] = best_non_null_entry.text

        all_nbest_json[example.qid] = nbest_json

    if output_prediction_file:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    # if output_nbest_file:
    #     with open(output_nbest_file, "w") as writer:
    #         writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if output_null_log_odds_file:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    if return_prediction:
        return [all_nbest_json[ex.qid][0] for ex in all_examples]

    if return_nbest:
        return [all_nbest_json[ex.qid] for ex in all_examples]


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text,
                tok_ns_text,
            )
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(
        named_params_optimizer, named_params_model
    ):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(
    named_params_optimizer, named_params_model, test_nan=False
):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(
        named_params_optimizer, named_params_model
    ):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(
                    param_opti.data.new().resize_(*param_opti.data.size())
                )
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def main():
    args = arguments.get_argparse("answer_extraction")
    logger.info(json.dumps(args.__dict__))

    if args.eval_on_train and not args.log_spec:
        args.log_spec = "on_train"

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
        if args.fp16:
            logger.info(
                "16-bits training currently not supported in distributed training"
            )
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits trainiing: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError(
            "At least one of `do_train` or `do_eval` or `do_test` must be True."
        )
    if (args.do_train or args.do_eval) and args.do_test:
        raise ValueError("Runing test must be independent of running train and/or dev")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d"
            % (args.max_seq_length, bert_config.max_position_embeddings)
        )

    processors = {
        "squad": dataset_processor.SquadProcessor,
        "squad2": dataset_processor.SquadProcessor,
        "hotpot": dataset_processor.HotpotProcessor,
        "coqa": dataset_processor.CoqaProcessor,
        "duorc": dataset_processor.DuoRCProcessor,
    }

    if args.small_debug:
        args.output_dir = 'debug'
        args.cache_dir = 'debug_cache'
    if os.path.exists(args.output_dir):
        if not os.listdir(args.output_dir) == ["args_log.txt"] and not args.small_debug:
            raise ValueError("Output directory () already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    args_log = os.path.join(args.output_dir, "args_log.txt")
    with open(args_log, "w") as writer:
        writer.write(json.dumps(args.__dict__))

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    if "{}" in args.corenlp_cache_dir:
        args.corenlp_cache_dir = args.corenlp_cache_dir.format(task_name)

    if task_name == "squad2" and args.dataset_option is None:
        args.dataset_option = "v2.0"
    processor = processors[task_name](args.data_dir, args.dataset_option)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case
    )

    train_examples = None
    num_train_steps = None

    if args.mix_input_ablation:
        args.input_ablation = "mix_input_ablation"

    if args.entity_anonymization:
        if 'cache' in args.cache_dir and 'ent_anon' not in args.cache_dir:
            args.cache_dir = args.cache_dir.replace('cache', f'ent_anon_{args.entity_anonymization}_cache')
            os.makedirs(args.cache_dir, exist_ok=True)
            print(f'updated cache dir: {args.cache_dir}')

    def cache_features(examples, split_name):
        if args.input_ablation or args.output_statistics:
            if task_name == "hotpot" and split_name == "train":
                pathes = glob.glob(
                    os.path.join(
                        args.corenlp_cache_dir,
                        "{}_{}_*_*.pkl".format(task_name, split_name),
                    )
                )
                corenlp_cache = {}
                for p in pathes:
                    c = pickle.load(open(p, "rb"))
                    corenlp_cache.update(c)
                updata = pickle.load(
                    open(
                        os.path.join(
                            args.corenlp_cache_dir,
                            "{}_{}_append.pkl".format(task_name, split_name),
                        ),
                        "rb",
                    )
                )
                print("updata: {}".format(len(updata)))
                corenlp_cache.update(updata)
                print("loaded hotpot corenlp cache: {}".format(len(corenlp_cache)))
            else:
                corenlp_cache_path = os.path.join(
                    args.corenlp_cache_dir, "{}_{}.pkl".format(task_name, split_name)
                )
                corenlp_cache = pickle.load(open(corenlp_cache_path, "rb"))
        else:
            corenlp_cache = None


        if args.output_statistics:
            output_ans_extr.output_statistics(examples, corenlp_cache)

        tokenized_examples = None
        to_save_tokenized_examples = False

        if args.entity_anonymization and not args.input_ablation and not args.no_cache:  # because it is heavy
            tokenized_examples_cache_filepath = os.path.join(
                args.cache_dir, f"{task_name}_{split_name}_ent_anon_{args.entity_anonymization}.pkl"
            )
            if os.path.exists(tokenized_examples_cache_filepath) and not args.no_cache:
                tokenized_examples = pickle.load(open(tokenized_examples_cache_filepath, "rb"))
            elif not args.no_cache:
                to_save_tokenized_examples = True

        if tokenized_examples is None:
            # examples = examples[10000:20000]
            tokenized_examples = generate_tokenized_examples(
                examples,
                args.input_ablation,
                corenlp_cache,
                tokenizer,
                entity_anonymization=args.entity_anonymization,
            )
            if to_save_tokenized_examples:
                with open(tokenized_examples_cache_filepath, 'wb') as f:
                    pickle.dump(tokenized_examples, f)

        if split_name == 'dev' and (args.output_mturk or args.output_examples):
            for ex in examples:
                ex.input_ablation = "original"
            original_examples = generate_tokenized_examples(
                examples, None, None, tokenizer
            )

            ids = random.sample(range(len(tokenized_examples)), 50)

            if args.output_examples:
                output_ans_extr.output_examples(
                    tokenized_examples,
                    original_examples,
                    task_name,
                    'ent_anon' if args.entity_anonymization else args.input_ablation,
                )

            if args.output_mturk:
                output_ans_extr.output_mturk(
                    tokenized_examples,
                    original_examples,
                    task_name,
                    args.input_ablation
                )

            exit(1)

        del corenlp_cache

        option_spec = (
            "" if not args.input_ablation else "_{}".format(args.input_ablation)
        )
        feature_cache_filepath = os.path.join(
            args.cache_dir, "{}_{}{}.pkl".format(task_name, split_name, option_spec)
        )
        if os.path.exists(feature_cache_filepath) and not args.no_cache:
            features = pickle.load(open(feature_cache_filepath, "rb"))
        else:
            features = convert_examples_to_features(
                examples=tokenized_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=split_name == "train",
                ignore_out_of_span=args.ignore_out_of_span,
                is_output_example=True,
            )
            if not args.no_cache:
                with open(feature_cache_filepath, "wb") as f:
                    pickle.dump(features, f)

        return features, tokenized_examples

    if args.do_train:
        train_examples = processor.get_train_examples()
        if args.small_debug:
            train_examples = train_examples[:5000]
        num_train_steps = int(
            len(train_examples)
            / args.train_batch_size
            / args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        if args.mix_input_ablation:
            print("[mix input ablation]")
            example_size = len(train_examples)
            ablation_list = []
            augument = False
            if args.mix_input_ablation[0] == "+":
                augument = True
            for ablation_spec in args.mix_input_ablation.replace("+", "").split(":"):
                spec = ablation_spec.split("=")
                ablation_list.append((spec[0], float(spec[1]) / 100))
            assignments = []
            for ablation in ablation_list:
                assignments.extend([ablation[0]] * int(example_size * ablation[1]))
                print("{}: {}".format(ablation[0], int(example_size * ablation[1])))
            assert len(assignments) <= example_size
            print("original: {}".format(example_size - len(assignments)))
            assignments.extend(["original"] * (example_size - len(assignments)))
            random.shuffle(assignments)
            for example, assignment in zip(train_examples, assignments):
                example.input_ablation = assignment
                if augument and not assignment == "original":
                    orig_example = copy.deepcopy(example)
                    orig_example.input_ablation = "original"
                    train_examples.append(orig_example)
        elif args.input_ablation:
            for example in train_examples:
                example.input_ablation = args.input_ablation

        train_features, train_exapmles = cache_features(train_examples, "train")

    if args.do_eval:
        if args.eval_on_train:
            eval_examples = processor.get_train_examples()
            eval_features, eval_examples = cache_features(eval_examples, "train")
        else:
            eval_examples = processor.get_dev_examples()
            if args.small_debug:
                eval_examples = eval_examples[:1000]
            if not args.mix_input_ablation and args.input_ablation:
                for example in eval_examples:
                    example.input_ablation = args.input_ablation
            eval_features, eval_examples = cache_features(eval_examples, "dev")

    if args.do_test:
        eval_examples = processor.get_test_examples()
        eval_features, eval_examples = cache_features(eval_examples, "test")

    global entity_set
    if args.entity_anonymization:
        if len(entity_set) == 0:
            anon_tag_cache_file = os.path.join(
                args.cache_dir, f'{task_name}_anon_tags_{args.entity_anonymization}.pkl'
            )
            if os.path.exists(anon_tag_cache_file):
                entity_set = pickle.load(open(anon_tag_cache_file, 'rb'))
            else:
                if args.entity_anonymization == 'close':
                    entity_size = 190
                elif args.entity_anonymization == 'close_noun':
                    entity_size = 286
                else:
                    ValueError('unknown specification for anonymization tags')
                entity_set = [f'@entity{i}' for i in range(entity_size)] #TODO for hotpotqa: 190
            tokenizer.vocab_update(sorted(entity_set))
        else:
            anon_tag_cache_file = os.path.join(
                args.cache_dir, f'{task_name}_anon_tags_{args.entity_anonymization}.pkl'
            )
            if not os.path.exists(anon_tag_cache_file):
                with open(anon_tag_cache_file, 'wb') as f:
                    pickle.dump(entity_set, f)
    
    model = BertForQuestionAnswering(bert_config)

    if args.init_checkpoint is not None:
        load_state_dict = torch.load(args.init_checkpoint, map_location="cpu")
        if list(load_state_dict)[0].startswith("bert."):
            # finetuned on some target dataset
            state_dict = model.state_dict()
            state_dict.update(load_state_dict)
            model.load_state_dict(state_dict)
        else:
            # pretrained language model
            state_dict = model.bert.state_dict()
            state_dict.update(load_state_dict)
            model.bert.load_state_dict(state_dict)

    if args.entity_anonymization and len(entity_set):
        model.bert.embeddings.extend_word_embeddings(len(entity_set))

    if args.limit_vocab_size or args.limit_vocab_freq:
        use_vocab, train_features, eval_features = vocab_selection(
            train_features,
            eval_features,
            args.cache_dir,
            args.output_dir,
            task_name,
            tokenizer,
            args.entity_anonymization,
            args.limit_vocab_size,
            args.limit_vocab_freq,
        )
        model.bert.embeddings.limit_vocab(use_vocab)

    if args.fp16:
        model.half()

    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [
            (n, param.clone().detach().to("cpu").float().requires_grad_())
            for n, param in model.named_parameters()
        ]
    elif args.optimize_on_cpu:
        param_optimizer = [
            (n, param.clone().detach().to("cpu").requires_grad_())
            for n, param in model.named_parameters()
        ]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if n not in no_decay],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if n in no_decay],
            "weight_decay_rate": 0.0,
        },
    ]
    optimizer = BERTAdam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        t_total=num_train_steps,
    )

    global_step = 0

    if args.enter_debugger:
        model.eval()
        # features = convert_ans_extr_examples_to_features(
        #     eval_examples,
        #     tokenizer,
        #     args.max_seq_length,
        #     args.doc_stride,
        #     args.max_query_length,
        #     True,
        #     is_output_example=False,
        # )
        # logits, preds = get_predictions(
        #     model, eval_examples, features, args, device
        # )
        print("in debugger")
        def get_predictions_internal(
                model,
                examples,
                args,
                device,
                tokenizer,
                compute_score=True,
                show_tqdm=False,
                get_nbest=0,
        ):
            results = []
            features = convert_examples_to_features(
                examples,
                tokenizer,
                args.max_seq_length,
                args.doc_stride,
                args.max_query_length,
                True,
                is_output_example=False,
            )

            logits, predictions = get_predictions(
                model, examples, features, args, device, show_tqdm, get_nbest
            )
            for logit, prediction, example in zip(logits, predictions, examples):
                if get_nbest:
                    pred_str = prediction[0]["text"]
                else:
                    pred_str = prediction["text"]
                # score = example.get_score(pred_str) if compute_score else 0
                score = score_function.get_score(
                    pred_str, example.orig_answer_text
                ) if compute_score else 0
                result = {
                    "logit": logit,
                    "prediction": pred_str,
                    "logits": prediction,
                    "score": score,
                }
                results.append(result)
            return results

        def get_prediction(ex):
            return get_predictions_internal(model, [ex], args, device, tokenizer)

        print('call get_prediction(example) to see the prediction for the example')
        import pdb; pdb.set_trace()
    
    def eval_func(num_epoch=-1, num_step=-1, log_spec=None):
        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        logger.info("Start evaluating")
        all_results = run_eval_model(
            model, eval_features, args, device, show_evaluating=True
        )
        model.train()

        output_spec = ""
        if num_epoch > -1 and num_step > -1:
            output_spec = "_{}_{}".format(num_epoch, num_step)
        elif log_spec:
            output_spec += "_{}".format(log_spec)

        output_prediction_file = os.path.join(
            args.output_dir, "eval_predictions{}.json".format(output_spec)
        )
        output_nbest_file = os.path.join(
            args.output_dir, "eval_nbest_predictions{}.json".format(output_spec)
        )

        output_null_log_odds_file = os.path.join(
            args.output_dir, "eval_null_odds{}.json".format(output_spec)
        ) if args.allow_impossible else None
        nsdt = args.null_score_diff_threshold if args.allow_impossible else None

        write_predictions(
            eval_examples,
            eval_features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            nsdt,
            args.verbose_logging,
        )

    if args.do_train:
        # train_features = convert_examples_to_features(
        #     examples=train_examples,
        #     tokenizer=tokenizer,
        #     max_seq_length=args.max_seq_length,
        #     doc_stride=args.doc_stride,
        #     max_query_length=args.max_query_length,
        #     is_training=True)
        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long
        )
        all_start_positions = torch.tensor(
            [f.start_position for f in train_features], dtype=torch.long
        )
        all_end_positions = torch.tensor(
            [f.end_position for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_start_positions,
            all_end_positions,
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size
        )

        model.train()
        for i in trange(int(args.num_train_epochs), desc="Epoch"):
            tmp_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(
                        t.to(device) for t in batch
                    )  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions = (
                    batch
                )
                if len(input_ids) != args.train_batch_size:
                    continue  # ???
                loss = model(
                    input_ids, segment_ids, input_mask, start_positions, end_positions
                )

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(
                            param_optimizer, model.named_parameters(), test_nan=True
                        )
                        if is_nan:
                            logger.info(
                                "FP16 TRAINING: Nan in gradients, reducing loss scaling"
                            )
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(
                            model.named_parameters(), param_optimizer
                        )
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

                    if (
                        global_step % args.save_checkpoints_steps == 0
                        and global_step > 0
                    ):
                        output_model_file = os.path.join(
                            args.output_dir,
                            "pytorch_model_step_{}.bin".format(global_step),
                        )
                        if n_gpu > 1:
                            torch.save(model.module.state_dict(), output_model_file)
                        else:
                            torch.save(model.state_dict(), output_model_file)

                    tmp_loss += loss.item()
                    if (
                        args.loss_report_steps > 0
                        and global_step > 0
                        and global_step % args.loss_report_steps == 0
                    ):
                        logger.info(
                            "Step loss: {}".format(tmp_loss / args.loss_report_steps)
                        )
                        tmp_loss = 0

                    if (
                        args.eval_steps > 0
                        and global_step > 0
                        and global_step % args.eval_steps == 0
                        and args.do_eval
                    ):
                        eval_func(i, global_step, args.log_spec)

            output_model_file = os.path.join(
                args.output_dir, "pytorch_model_epoch_{}.bin".format(i)
            )
            if n_gpu > 1:
                torch.save(model.module.state_dict(), output_model_file)
            else:
                torch.save(model.state_dict(), output_model_file)
            if args.do_eval:
                eval_func(i, global_step, args.log_spec)

    if not args.do_train and args.do_eval:
        eval_func(log_spec=args.log_spec or "dev")

    if args.do_test:
        eval_func(log_spec=args.log_spec or "test")


if __name__ == "__main__":
    main()
