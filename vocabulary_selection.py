import collections
import json
import os
import pickle
import random
from tqdm import *

def vocab_selection(
        train_features,
        eval_features,
        cache_dir,
        output_dir,
        task_name,
        tokenizer,
        entity_anonymization=None,
        limit_vocab_size=None,
        limit_vocab_freq=None,
        num_options=None,
):
    if not (limit_vocab_size or limit_vocab_freq):
        raise ValueError("should be args.limit_vocab_size or args.limit_vocab_freq")
    
    if limit_vocab_freq:
        vocab_spec = f'vocabfreq{limit_vocab_freq}'
    else:
        vocab_spec = f'vocab{limit_vocab_size}'

    if entity_anonymization:
        vocab_spec += f'ent_anon_{entity_anonymization}'

    unk_id = 100
    vocab_file = os.path.join(cache_dir, f'{task_name}_{vocab_spec}.pkl')
    if os.path.exists(vocab_file):
        use_vocab = pickle.load(open(vocab_file, 'rb'))
        # return use_vocab
    else:
        def get_freq_vocab(features, top=10000, limit_freq=None):
            vocab_counter = collections.defaultdict(int)
            for feature in features:
                if num_options:
                    for oi in range(num_options):
                        for i in feature.input_ids[oi]:
                            vocab_counter[i] += 1
                else:
                    for i in feature.input_ids:
                        vocab_counter[i] += 1
            sorted_vocab = sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)
            if limit_freq:
                limit_idx = len(sorted_vocab)
                for wi, (word, freq) in enumerate(sorted_vocab):
                    if freq < limit_freq:
                        limit_idx = wi
                        break
            else:
                limit_idx = top
            use_vocab = [x[0] for x in sorted_vocab[:limit_idx]]
            if unk_id not in use_vocab:
                use_vocab.append(unk_id)
            return sorted(use_vocab)
        print('choosing vocabulary...')
        use_vocab = get_freq_vocab(
            train_features, limit_vocab_size, limit_vocab_freq
        )
        # top_vocab = get_freq_vocab(train_features, bert_config.vocab_size)
        with open(vocab_file, 'wb') as f:
            pickle.dump(use_vocab, f)

    if not use_vocab:
        raise ValueError("could not get a vocabulary to use")

    id_to_token = {v:k for k, v in tokenizer.vocab.items()}
    if True:
        use_tokens = [id_to_token[i] for i in use_vocab]
        if len(use_tokens) < 1000:
            print(f'{len(use_tokens)}')
            print(sorted(use_tokens))
            print(f'{len(use_tokens)}')
            # logger.info(sorted(use_tokens))
            # logger.info(f'{len(use_tokens)}')
            vocab_log = os.path.join(output_dir, "vocab_used.log")
            with open(vocab_log, "w") as writer:
                writer.write(json.dumps(sorted(use_tokens)))

    vocab_map = {x: xi for xi, x in enumerate(use_vocab)}

    # org_id = [0,...,30522,...]
    # mapped_id = [0,...,len(used_vocab)]
    # id_to_token = {id: token} from inv vocab
    def show_tokens(ids):
        inv_vocab_map = {v:k for k, v in vocab_map.items()}
        tokens = [id_to_token[inv_vocab_map.get(i, unk_id)] for i in ids]
        # for token in tokens:
        #     if token[0] not in ['@', '[']:
        #         print(f'error?: {token}')
        if '[PAD]' in tokens:
            tokens = tokens[:tokens.index('[PAD]')]
        return tokens

    train_vocab_feature_file = os.path.join(
        cache_dir, f'{task_name}_train_feature_{vocab_spec}.pkl'
    )
    if not os.path.exists(train_vocab_feature_file):
        for feature in tqdm(train_features):
            if num_options:
                for oi in range(num_options):
                    feature.input_ids[oi] = [
                        vocab_map.get(x, vocab_map[unk_id]) for x in feature.input_ids[oi]
                    ]
            else:
                feature.input_ids = [
                    vocab_map.get(x, vocab_map[unk_id]) for x in feature.input_ids
                ]
        with open(train_vocab_feature_file, 'wb') as f:
            pickle.dump(train_features, f)
    else:
        train_features = pickle.load(open(train_vocab_feature_file, 'rb'))

    # unk_rates = []
    # for feature in tqdm(train_features):
    #     unks = 0
    #     tokens = 0
    #     if num_options:
    #         l = sum(feature.input_ids, [])
    #     else:
    #         l = feature.input_ids
    #     for token_id in l:
    #         if token_id == 0:
    #             break
    #         tokens += 1
    #         if token_id == vocab_map[unk_id]:
    #             unks += 1
    #     unk_rates.append(unks / tokens)

    # print(f'unknown rate for traning set: {sum(unk_rates)/len(unk_rates)}')

    for train_feat in random.sample(train_features, 10):
        if num_options:
            print(' '.join(show_tokens(train_feat.input_ids[0])))
        else:
            print(' '.join(show_tokens(train_feat.input_ids)))

    eval_vocab_feature_file = os.path.join(
        cache_dir, f'{task_name}_dev_feature_{vocab_spec}.pkl'
    )
    if not os.path.exists(eval_vocab_feature_file):
        for feature in tqdm(eval_features):
            if num_options:
                for oi in range(num_options):
                    feature.input_ids[oi] = [vocab_map.get(x, vocab_map[unk_id]) for x in feature.input_ids[oi]]
            else:
                feature.input_ids = [vocab_map.get(x, vocab_map[unk_id]) for x in feature.input_ids]
        with open(eval_vocab_feature_file, 'wb') as f:
            pickle.dump(eval_features, f)
    else:
        eval_features = pickle.load(open(eval_vocab_feature_file, 'rb'))

    print('vocabulary selected: {}'.format(len(vocab_map)))

    return use_vocab, train_features, eval_features
