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
"""Run BERT on Multiple Choice MRC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os
import logging
import random
import json
import pickle
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import arguments
import processor as dataset_processor
import tokenization
import output_mcmrc
from modeling import BertConfig, BertForMultipleChoice
from optimization import BERTAdam
from input_ablation_mcmrc import generate_ablated_input
from input_example import InputTokenizedMultipleChoiceExample, InputMultipleChoiceFeature
from vocabulary_selection import vocab_selection
from entity_anonymizer import anonymize_entity

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

entity_set = set()
entity_cache = {}


def generate_tokenized_examples(
    examples, tokenizer, input_ablation=None, corenlp_cache=None, entity_anonymization=None
):
    global entity_set
    
    tokenized_examples = []
    for ei, example in enumerate(tqdm(examples)):
        ablation_info = ""
        if not input_ablation:
            doc_tokens = tokenizer.basic_tokenizer.tokenize(example.doc_text)
            query_tokens = tokenizer.basic_tokenizer.tokenize(example.query_text)
            option_tokens_list = [
                tokenizer.basic_tokenizer.tokenize(opt)
                for opt in example.option_text_list
            ]

        if entity_anonymization:
            anon_tags = None
            add_inf = False
            stemming = False
            ent_anon_spec = entity_anonymization.split('_')
            if len(ent_anon_spec) > 1:
                anon_tags = ent_anon_spec[1]
                if len(ent_anon_spec) == 3:
                    add_inf = ent_anon_spec[2] == 'inf'
                    stemming = ent_anon_spec[2] == 'stem'
            # use cache for each example.did (entity_cache)
            # use different vocab for different question (query_anonym_dict)
            anonym_dicts, entity_dict = anonymize_entity(
                doc_tokens,
                query_tokens,
                entity_cache.get(example.did, None),
                target_tags=anon_tags,
                only_stem=stemming,
                add_inflection=add_inf,
                option_tokens_list=option_tokens_list,
            )
            if example.did not in entity_cache:
                entity_cache[example.did] = entity_dict
            entity_set.update(list(anonym_dicts['doc'].values()))
            entity_set.update(list(anonym_dicts['query'].values()))
            entity_set.update(sum([list(t.values()) for t in anonym_dicts['options']], []))

            def mask_tokens(tokens, mask_dict):  # key=position
                return [mask_dict.get(i, token) for i, token in enumerate(tokens)]

            doc_tokens = mask_tokens(doc_tokens, anonym_dicts['doc'])
            query_tokens = mask_tokens(query_tokens, anonym_dicts['query'])
            option_tokens_list = [mask_tokens(opt, anonym_dicts['options'][i])
                                  for i, opt in enumerate(option_tokens_list)]

        elif input_ablation:
            if example.did == example.qid:
                doc_cache = copy.deepcopy(corenlp_cache[example.qid]["doc"])
            else:
                doc_cache = copy.deepcopy(corenlp_cache[example.did])
            query_cache = copy.deepcopy(corenlp_cache[example.qid])
            ablated_example = generate_ablated_input(
                input_ablation, doc_cache, query_cache
            )
            doc_tokens = ablated_example["doc_tokens"]
            query_tokens = ablated_example["query_tokens"]
            option_tokens_list = ablated_example["option_tokens_list"] or [
                tokenizer.basic_tokenizer.tokenize(opt)
                for opt in example.option_text_list
            ]
            ablation_info = ablated_example.get(
                "drop_dict", ablated_example.get("ablation_info", "")
            )
            if ei < 10:
                print(ablation_info)

        tokenized_examples.append(
            InputTokenizedMultipleChoiceExample(
                example.did,
                example.qid,
                doc_tokens,
                query_tokens,
                option_tokens_list,
                example.ans_idx,
                ablation_info,
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
    num_options,
    max_seq_length,
    max_query_length,
    max_option_length,
    tokenizer,
    is_output_example=True,
):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(
        tqdm(examples) if is_output_example else examples
    ):

        def wordpiece_tokenize(tokens):
            wordpiece_tokens = []
            for t in tokens:
                for subt in tokenizer.wordpiece_tokenizer.tokenize(t):
                    wordpiece_tokens.append(subt)
            return wordpiece_tokens

        doc_tokens = wordpiece_tokenize(example.doc_tokens)
        query_tokens = wordpiece_tokenize(example.query_tokens)
        option_tokens_list = [
            wordpiece_tokenize(opt) for opt in example.option_tokens_list
        ]

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        for i in range(num_options):
            doc_end_pos = max_seq_length
            doc_end_pos -= len(query_tokens) + len(option_tokens_list[i]) + 3
            truncated_doc_tokens = doc_tokens[:doc_end_pos]
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in truncated_doc_tokens:
                tokens.append(token)
                segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for token in option_tokens_list[i]:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)

        label_id = example.ans_idx
        if ex_index < 5 and is_output_example:
            logger.info("*** Example ***")
            logger.info("qid: %s" % (example.qid))
            logger.info(
                "tokens: %s"
                % " ".join([tokenization.printable_text(x) for x in tokens])
            )
            # logger.info(
            #     "input_ids: %s" % " ".join([str(x) for x in input_ids]))
            # logger.info(
            #     "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            # logger.info(
            #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("answer index: %d" % (label_id))

        features.append(
            InputMultipleChoiceFeature(
                input_ids=input_ids_list,
                input_mask=input_mask_list,
                segment_ids=segment_ids_list,
                label_id=label_id,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then
    # each token that's truncated likely contains more information
    # than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


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
    """
    Utility function for optimize_on_cpu and 16-bits training.
    Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(
        named_params_optimizer, named_params_model
    ):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(
                param_opti.data.new().resize_(*param_opti.data.size())
            )
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan


def get_predictions(model, eval_examples, eval_features, args, device):
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in eval_features], dtype=torch.long
    )
    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # for output
    output_logits = []
    output_predictions = []

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss, logits = model(
                input_ids, segment_ids, input_mask, label_ids
            )

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to("cpu").numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        output_logits.extend(logits)
        output_predictions.extend(np.argmax(logits, axis=1))

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    return output_logits, output_predictions, eval_loss, eval_accuracy


def main():
    args = arguments.get_argparse("multiple_choice")
    logger.info(json.dumps(args.__dict__))

    if args.eval_on_train and not args.log_spec:
        args.log_spec = "on_train"

    processors = {
        "race": dataset_processor.RaceProcessor,
        "mctest": dataset_processor.MCTestProcessor,
        "swag": dataset_processor.SwagProcessor,
        "squad": dataset_processor.SquadProcessor,
        "openbookqa": dataset_processor.OpenBookQAProcessor,
        "multirc": dataset_processor.MultiRCProcessor,
        "arc": dataset_processor.ARCProcessor,
        "qa4mre": dataset_processor.QA4MREProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend
        # which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
        if args.fp16:
            logger.info(
                """16-bits training currently not supported in distributed
                training"""
            )
            # (see https://github.com/pytorch/pytorch/pull/13496)
            args.fp16 = False
    logger.info(
        "device %s n_gpu %d distributed training %r",
        device,
        n_gpu,
        bool(args.local_rank != -1),
    )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            """Invalid gradient_accumulation_steps parameter: {},
            should be >= 1""".format(
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
            """At least one of `do_train` or `do_eval` or `do_test` must
            be True."""
        )
    if (args.do_train or args.do_eval) and args.do_test:
        raise ValueError("Runing test must be independent of running train and/or dev")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            """Cannot use sequence length {} because the BERT model was only
            trained up to sequence length {}""".format(
                args.max_seq_length, bert_config.max_position_embeddings
            )
        )

    if args.small_debug:
        args.output_dir = 'debug'
    if os.path.exists(args.output_dir):
        if not os.listdir(args.output_dir) == ["args_log.txt"] and not args.small_debug:
            raise ValueError("Output directory already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    args_log = os.path.join(args.output_dir, "args_log.txt")
    if not os.path.exists(args_log):
        with open(args_log, "w") as writer:
            writer.write(json.dumps(args.__dict__))
    else:
        print("args_log.txt already exists")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    if "{}" in args.corenlp_cache_dir:
        args.corenlp_cache_dir = args.corenlp_cache_dir.format(task_name)

    processor = processors[task_name](args.data_dir, args.dataset_option)
    num_options = processor.get_num_options()

    if args.convert_from_ans_extr:
        if args.do_train:
            if args.train_predictions:
                processor.set_candidates("train", args.train_predictions)
            else:
                raise ValueError("train prediction file is missing")
        if args.do_eval:
            if args.eval_predictions:
                processor.set_candidates("dev", args.eval_predictions)
            else:
                raise ValueError("eval prediction file is missing")

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case
    )

    train_examples = None
    num_train_steps = None

    def cache_features(examples, split_name):
        cache_spec_cand = [
            task_name,
            args.dataset_option,
            split_name,
            args.input_ablation,
        ]
        cache_spec = "_".join([str(x) for x in cache_spec_cand if x is not None])
        cache_path = os.path.join(args.cache_dir, "{}.pkl".format(cache_spec))

        if os.path.exists(cache_path) and not args.no_cache:
            features = pickle.load(open(cache_path, "rb"))
        else:
            if args.input_ablation or args.output_statistics:
                corenlp_cache_path = os.path.join(
                    args.corenlp_cache_dir, "{}_{}.pkl".format(task_name, split_name)
                )
                corenlp_cache = pickle.load(open(corenlp_cache_path, "rb"))
            else:
                corenlp_cache = None

            if args.output_statistics:
                output_mcmrc.output_statistics(examples, corenlp_cache)

            tokenized_examples = generate_tokenized_examples(
                examples,
                tokenizer,
                args.input_ablation,
                corenlp_cache,
                args.entity_anonymization
            )

            if args.output_mturk or args.output_examples:
                for ex in examples:
                    ex.input_ablation = "original"
                original_examples = generate_tokenized_examples(
                    examples, tokenizer, None, None
                )

                if args.output_examples:
                    output_mcmrc.output_examples(
                        tokenized_examples,
                        original_examples,
                        task_name,
                        'ent_anon' if args.entity_anonymization else args.input_ablation,
                    )

                if args.output_mturk:
                    output_mcmrc.output_mturk(
                        tokenized_examples,
                        original_examples,
                        task_name,
                        args.input_ablation
                    )

                exit(1)

            features = convert_examples_to_features(
                tokenized_examples,
                num_options,
                args.max_seq_length,
                args.max_query_length,
                args.max_option_length,
                tokenizer,
            )

            if not args.no_cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(features, f)

        # assert len(examples) == len(features)
        return features

    if args.do_train:
        train_examples = processor.get_train_examples()
        if args.small_debug:
            train_examples = train_examples[:6000]
        num_train_per_epoch = len(train_examples)
        num_train_per_epoch /= args.train_batch_size
        num_train_per_epoch /= args.gradient_accumulation_steps
        num_train_steps = int(num_train_per_epoch * args.num_train_epochs)
        train_features = cache_features(train_examples, "train")

    if args.do_eval:
        if args.eval_on_train:
            eval_examples = processor.get_train_examples()
            eval_features = cache_features(eval_examples, "train")
        else:
            eval_examples = processor.get_dev_examples()
            if args.small_debug:
                eval_examples = eval_examples[:1000]
            eval_features = cache_features(eval_examples, "dev")

    if args.do_test:
        eval_examples = processor.get_test_examples()
        eval_features = cache_features(eval_examples, "test")

    global entity_set
    if args.entity_anonymization:
        if len(entity_set) == 0:
            anon_tag_cache_file = os.path.join(
                args.cache_dir, f'{task_name}_anon_tags_{args.entity_anonymization}.pkl'
            )
            if not os.path.exists(anon_tag_cache_file):
                raise ValueError("vocabulary cache cannot be loaded")
            entity_set = pickle.load(open(anon_tag_cache_file, 'rb'))
            tokenizer.vocab_update(sorted(entity_set))
        else:
            anon_tag_cache_file = os.path.join(
                args.cache_dir, f'{task_name}_anon_tags_{args.entity_anonymization}.pkl'
            )
            if not os.path.exists(anon_tag_cache_file):
                with open(anon_tag_cache_file, 'wb') as f:
                    pickle.dump(entity_set, f)    

    # Prepare model
    model = BertForMultipleChoice(bert_config, num_options)
    if args.init_checkpoint is not None:
        state_dict = torch.load(args.init_checkpoint, map_location="cpu")
        if list(state_dict)[0].startswith("bert."):
            # finetuned on some target dataset
            model.load_state_dict(state_dict)
        else:
            # pretrained language model
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
            num_options=num_options,
        )
        id_to_token = {v:k for k, v in tokenizer.vocab.items()}
        use_tokens = [id_to_token[i] for i in use_vocab]
        logger.info(sorted(use_tokens))
        logger.info(f'{len(use_tokens)}')
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
        # features = convert_examples_to_features(
        #     eval_examples,
        #     num_options,
        #     args.max_seq_length,
        #     args.max_query_length,
        #     args.max_option_length,
        #     tokenizer,
        # )
        # output = get_predictions(
        #     model, eval_examples, features, args, device
        # )
        # output_logits, output_predictions, eval_loss, eval_accuracy = output
        print("in debugger")        
        import pdb; pdb.set_trace()
    
    def eval_func(num_epoch=-1, num_step=-1, log_spec=None):
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()

        output = get_predictions(model, eval_examples, eval_features, args, device)
        output_logits, output_predictions, eval_loss, eval_accuracy = output

        model.train()

        output_qids = [e.qid for e in eval_examples]
        output_answers = [e.ans_idx for e in eval_examples]

        result = {
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "global_step": global_step,
        }

        output_spec = ""
        if num_epoch > -1 and num_step > -1:
            output_spec = "_{}_{}".format(num_epoch, num_step)
        elif log_spec:
            output_spec += "_{}".format(log_spec)

        output_eval_file = os.path.join(
            args.output_dir, "eval_results{}.json".format(output_spec)
        )
        result["spec"] = output_spec
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        with open(output_eval_file, "w") as writer:
            writer.write(json.dumps(result))

        output_pred_file = os.path.join(
            args.output_dir, "eval_preds{}.jsonl".format(output_spec)
        )
        with open(output_pred_file, "w") as f:
            for qid, ans, pred, logit in zip(
                output_qids, output_answers, output_predictions, output_logits
            ):
                result = {
                    "qid": qid,
                    "answer": chr(ans + ord("A")),
                    "prediction": chr(pred + ord("A")),
                    "logits": logit.tolist(),
                }
                f.write(json.dumps(result) + "\n")

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
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
        all_label_ids = torch.tensor(
            [f.label_id for f in train_features], dtype=torch.long
        )
        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
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
            tr_loss = 0
            tmp_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/..
                    # ..sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                param.grad.data /= args.loss_scale
                        is_nan = set_optimizer_params_grad(
                            param_optimizer, model.named_parameters(), test_nan=True
                        )
                        if is_nan:
                            logger.info(
                                """FP16 TRAINING: Nan in gradients,
                                reducing loss scaling"""
                            )
                            args.loss_scale /= 2
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

                    if global_step % args.save_model_steps == 0:
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
