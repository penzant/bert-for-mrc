import csv
import glob
import json
import os
import pickle
import string
from collections import defaultdict

from tqdm import tqdm

import tokenization
from input_example import InputMultipleChoiceExample, InputAnswerExtractionExample


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        return self._create_examples('train')

    def get_dev_examples(self):
        return self._create_examples('dev')

    def get_test_examples(self):
        return self._create_examples('test')

    def _create_examples(self, data_split):
        raise NotImplementedError()

    def get_examples(self, data_split, dataset_type='*'):
        return self._create_examples(data_split, dataset_type)


class RaceProcessor(DataProcessor):
    """Processor for the RACE dataset."""

    def __init__(self, data_dir, dataset_type=None):
        self.data_dir = data_dir
        self.dataset_type = dataset_type or '*' # high, middle
        assert self.dataset_type in ['high', 'middle', '*']

    def get_train_examples(self):
        return self._create_examples('train')

    def get_dev_examples(self):
        return self._create_examples('dev', self.dataset_type)

    def get_test_examples(self):
        return self._create_examples('test', self.dataset_type)

    def get_num_options(self):
        return 4

    def _create_examples(self, data_split, dataset_type='*'):
        glob_dir = glob.glob(os.path.join(
            self.data_dir, data_split, dataset_type, '*.txt'))

        examples = []
        for item in tqdm([json.load(open(path, 'r')) for path in glob_dir]):
            for e in range(len(item['questions'])):
                doc_text = tokenization.convert_to_unicode(item['article'])
                query_text = tokenization.convert_to_unicode(item['questions'][e])
                option_text_list = [tokenization.convert_to_unicode(opt)
                                    for opt in item['options'][e]]
                ans_idx = ord(item['answers'][e]) - ord('A')
                qid = item['id'] + '-Q{0}'.format(e)
                examples.append(
                    InputMultipleChoiceExample(did=item['id'], qid=qid,
                                 doc_text=doc_text, query_text=query_text,
                                 option_text_list=option_text_list, ans_idx=ans_idx))
        return examples


class MCTestProcessor(DataProcessor):
    """Processor for the MCTest dataset."""

    def __init__(self, data_dir, dataset_type=None):
        self.data_dir = data_dir
        self.dataset_type = dataset_type or '*' # high, middle
        assert self.dataset_type in ['mc160', 'mc500', '*']

    def get_train_examples(self):
        return self._create_examples('train', '*')

    def get_dev_examples(self):
        return self._create_examples('dev', self.dataset_type)

    def get_test_examples(self):
        return self._create_examples('test', self.dataset_type)

    def get_num_options(self):
        return 4

    def _create_examples(self, data_split, dataset_type='*'):
        source_data, answer_data = [], []
        target = ['mc160', 'mc500'] if dataset_type == '*' else [dataset_type]
        for t in target:
            filepath = os.path.join(self.data_dir, '{}.{}.tsv'.format(t, data_split))
            with open(filepath, 'r') as f:
                source_data += f.readlines()
            filepath = os.path.join(self.data_dir, '{}.{}.ans'.format(t, data_split))
            with open(filepath, 'r') as f:
                answer_data += f.readlines()

        def processing_context(context):
            context = context.replace('\\newline', ' ').replace('\\tab', ' ')
            return context

        examples = []
        for item, answer_line in zip(source_data, answer_data):
            item = item.strip().split('\t')
            answer_list = answer_line.strip().split('\t')
            doc_text = tokenization.convert_to_unicode(processing_context(item[2]))
            doc_id = item[0]
            for i in range(4):
                query_text = item[3+i*5].replace('multiple: ', '').replace('one: ', '')
                query_text = tokenization.convert_to_unicode(query_text)
                option_text_list = [tokenization.convert_to_unicode(opt)
                                    for opt in item[4+i*5:4+i*5+4]]
                ans_idx = ord(answer_list[i]) - ord('A')
                qid = doc_id + '-Q{0}'.format(i)
                examples.append(
                    InputMultipleChoiceExample(did=doc_id, qid=qid, doc_text=doc_text, query_text=query_text,
                                               option_text_list=option_text_list, ans_idx=ans_idx))
        return examples


class SwagProcessor(DataProcessor):
    """Processor for the SWAG dataset."""

    def __init__(self, data_dir, dataset_type=None):
        self.data_dir = data_dir

    def get_num_options(self):
        return 4

    def get_examples(self, data_split):
        return self._create_examples(data_split)

    def _create_examples(self, data_split):
        if data_split == 'dev':
            data_split = 'val'
        filepath = os.path.join(self.data_dir, '{}.csv'.format(data_split))

        rows = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                rows.append(row)

        examples = []
        for row in tqdm(rows):
            qid = '{}-{}'.format(data_split, row[0])
            doc_text = tokenization.convert_to_unicode(row[4])  # sent1
            query_text = tokenization.convert_to_unicode(row[5])  # sent2
            option_text_list = [tokenization.convert_to_unicode(r) for r in row[7:7+4]]
            if len(row) == 12:  # regular set
                ans_idx = int(row[-1])
            else:
                ans_idx = -1
            examples.append(
                InputMultipleChoiceExample(did=qid,
                                           qid=qid,
                                           doc_text=doc_text,
                                           query_text=query_text,
                                           option_text_list=option_text_list,
                                           ans_idx=ans_idx))
        return examples


class OpenBookQAProcessor(DataProcessor):
    """Processor for the OpenBookQA  dataset."""
    def __init__(self, data_dir, dataset_type=None):
        self.data_dir = data_dir

    def get_num_options(self):
        return 4

    def _create_examples(self, data_split):
        filepath = os.path.join(self.data_dir, '{}.jsonl'.format(data_split))
        jsons = []
        for line in  open(filepath, 'r'):
            jsons.append(json.loads(line))

        examples = []
        for item in tqdm(jsons):
            qid = item['id']
            doc_text = ''
            query_text = item['question']['stem']
            option_text_list = [choice['text'] for choice in item['question']['choices']]
            ans_idx = ord(item['answerKey']) - ord('A')
            examples.append(
                InputMultipleChoiceExample(did=qid,
                                           qid=qid,
                                           doc_text=doc_text,
                                           query_text=query_text,
                                           option_text_list=option_text_list,
                                           ans_idx=ans_idx))
        return examples


class ARCProcessor(DataProcessor):
    """Processor for the ARC dataset."""
    def __init__(self, data_dir, dataset_type=None):
        self.data_dir = data_dir
        self.dataset_type = dataset_type or 'challenge'
        assert self.dataset_type in ['easy', 'challenge']

    def get_num_options(self):
        return 4

    def get_examples(self, data_split):
        return self._create_examples(data_split, self.dataset_type)

    def _create_examples(self, data_split, dataset_type='challenge'):
        filepath = os.path.join(
            self.data_dir, dataset_type, '{}.jsonl'.format(data_split))
        jsons = []
        for line in  open(filepath, 'r'):
            jsons.append(json.loads(line))

        examples = []
        for item in tqdm(jsons):
            qid = item['id']
            doc_text = item['para']
            query_text = item['question']['stem']
            option_text_list = [choice['text'] for choice in item['question']['choices']]
            if len(option_text_list) < 4:
                option_text_list.extend(['']*(4-len(option_text_list)))
            if len(option_text_list) > 4:
                continue
            answer_key = item['answerKey']
            if answer_key.upper() in string.ascii_uppercase:
                ans_idx = ord(answer_key) - ord('A')
            elif answer_key in string.digits:
                ans_idx = int(answer_key) - 1
            else:
                ans_idx = 0
            examples.append(
                InputMultipleChoiceExample(did=qid,
                                           qid=qid,
                                           doc_text=doc_text,
                                           query_text=query_text,
                                           option_text_list=option_text_list,
                                           ans_idx=ans_idx))
        return examples


class MultiRCProcessor(DataProcessor):
    """
    Processor for the MultiRC dataset.
    This dataset allows multiple options to be correct.
    Thus we cast a question to 1/0 classification of each option.
    """

    def __init__(self, data_dir, dataset_type=None):
        self.data_dir = data_dir

    def get_num_options(self):
        return 2

    def get_examples(self, data_split):
        return self._create_examples(data_split)

    def _create_examples(self, data_split):
        if data_split == 'train':
            filepath = os.path.join(self.data_dir, 'train_456-fixedIds.json')
        elif data_split == 'dev':
            filepath = os.path.join(self.data_dir, 'dev_83-fixedIds.json')

        examples = []
        data = json.load(open(filepath, 'r'))
        for para in data['data']:
            paragraph = para['paragraph']
            para_id = para['id']
            doc_text = paragraph['text']
            doc_text = doc_text.replace('<br>', ' ')
            doc_text = doc_text.replace('<b>', '').replace('</b>', '')
            for qas in paragraph['questions']:
                query_text = qas['question']
                query_id = qas['idx']
                for ai, ans in enumerate(qas['answers']):
                    answer = ans['text']
                    is_answer = ans['isAnswer']
                    qid = '{}-{}-{}'.format(para_id, query_id, ai)
                    query_text_with_answer = ' '.join([query_text, answer])
                    option_text_list = ['0', '1']
                    ans_idx = 1 if is_answer else 0
                    examples.append(
                        InputMultipleChoiceExample(
                            did=para_id,
                            qid=qid,
                            doc_text=doc_text,
                            query_text=query_text_with_answer,
                            option_text_list=option_text_list,
                            ans_idx=ans_idx))
        return examples


class SquadProcessor(DataProcessor):
    """Processor for the SQuAD dataset."""

    def __init__(self, data_dir, dataset_type=None):
        self.data_dir = data_dir
        self.dataset_type = dataset_type or 'v1.1'
        assert self.dataset_type in ['v1.1', 'v2.0']
        self.candidate_filepath = defaultdict(str)

    def get_train_examples(self):
        return self._create_examples('train', self.dataset_type)

    def get_dev_examples(self):
        return self._create_examples('dev', self.dataset_type)

    def get_test_examples(self):
        return self._create_examples('test', self.dataset_type)

    def set_candidates(self, data_split, pred_filepath):
        self.candidate_filepath[data_split] = pred_filepath

    def get_examples(self, data_split):
        return self._create_examples(data_split, self.dataset_type)

    def _create_examples(self, data_split, dataset_type='v1.1'):
        input_file = os.path.join(self.data_dir,
                                  '{}-{}.json'.format(data_split, dataset_type))
        with open(input_file, "r") as reader:
            input_data = json.load(reader)["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or ord(c) == 160:
                return True
            return False

        doc_count = 0
        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                doc_count += 1
                doc_id = '{}-{}'.format(data_split, doc_count)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    # note: use the first answer for all dataset splits.
                    # if len(qa["answers"]) != 1:
                    #     raise ValueError(
                    #         "For training, each question should have exactly 1 answer.")
                    if dataset_type == 'v2.0':
                        is_impossible = qa['is_impossible']
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        orig_answer_texts = [a['text'] for a in qa["answers"]]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            print("Could not find answer: '%s' vs. '%s'",
                                  actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ''
                        orig_answer_texts = []

                    example = InputAnswerExtractionExample(
                        did=doc_id,
                        qid=qas_id,
                        query_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        orig_answer_texts=orig_answer_texts,
                        is_impossible=is_impossible)
                    examples.append(example)

        if self.candidate_filepath[data_split]:
            return mc_converter(self.candidate_filepath[data_split], examples)
        else:
            return examples


class HotpotProcessor(DataProcessor):
    """Processor for the HotpotQA dataset."""
    """examples and eval_exapmles are the output of process_file func in prepro.py"""

    def __init__(self, data_dir, dataset_type=None):
        self.data_dir = data_dir
        self.dataset_type = dataset_type or 'distractor'
        assert self.dataset_type in ['distractor', 'simple', 'fullwiki']

    def get_train_examples(self, dataset_type='simple'):
        return self._create_examples('train', dataset_type)

    def get_dev_examples(self, dataset_type='distractor'):
        return self._create_examples('dev', dataset_type)

    def get_test_examples(self):
        return self._create_examples('test', dataset_type)

    def _create_examples(self, data_split, dataset_type='distractor'):
        input_file_1 = os.path.join(self.data_dir, 'hotpot_{}_examples.pkl'.format(data_split))
        with open(input_file_1, "rb") as reader:
            input_examples = pickle.load(reader)

        # input_file_2 = os.path.join(
        #     self.data_dir, 'hotpot_{}_eval_examples.pkl'.format(data_split))
        # with open(input_file_2, "rb") as reader:
        #     input_eval_examples = pickle.load(reader)

        def is_whitespace(token):
            for c in token:
                # u200b is not removed in context_tokens...
                if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) in [160, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF] or 0x2000 <= ord(c) <= 0x200B:
                    continue
                else:
                    return False
            return True

        def no_distractor(context_tokens, y1s, y2s, facts):
            para_starts = [
                ti for ti, token in enumerate(context_tokens) if token == '<t>']
            para_starts.append(len(context_tokens))
            para_spans = [(ps, para_starts[i+1])
                          for i, ps in enumerate(para_starts[:-1])]
            is_evidence_para = [0] * len(para_spans)  # 1 for evi and no ans, 2 for evi and ans
            for pi, (para_start, para_end) in enumerate(para_spans):
                if para_start <= y1s < para_end and para_start < y2s <= para_end:
                    is_evidence_para[pi] = 2
                    continue
                for fact_start, fact_end, is_evidence in facts:
                    if not is_evidence: continue
                    if para_start <= fact_start < para_end and para_start < fact_end <= para_end:
                        is_evidence_para[pi] = 1
                        break

            if not (y1s == y2s and y1s in [-1, -2]):
                ans_para = is_evidence_para.index(2)
                drop_para_spans = [ps for pi, ps in enumerate(para_spans[:ans_para])
                                   if is_evidence_para[pi] == 0]
                offset = sum([ps[1]-ps[0] for ps in drop_para_spans])
                y1s, y2s = y1s - offset, y2s - offset

            context_tokens = [token for ps, ie in zip(para_spans, is_evidence_para)
                              for token in context_tokens[ps[0]:ps[1]] if ie > 0]
            return context_tokens, y1s, y2s

        examples = []
        for item in tqdm(input_examples):
            qid = item['id']
            context_tokens = item['context_tokens']

            if dataset_type == 'simple':
                context_tokens, y1s, y2s = no_distractor(context_tokens,
                                                         item['y1s'][0], item['y2s'][0],
                                                         item['start_end_facts'])
            else:
                y1s, y2s = item['y1s'][0], item['y2s'][0]

            new_to_orig_index = []
            doc_tokens = ['yes', 'no']
            new_to_orig_index = [-1, -2]
            for i, token in enumerate(context_tokens):
                if not is_whitespace(token):
                    token = token.replace('\u200b', '')
                    new_to_orig_index.append(i)
                    doc_tokens.append(token)
            start_position = new_to_orig_index.index(y1s)
            end_position = new_to_orig_index.index(y2s)
            answer_text = ' '.join(doc_tokens[start_position:end_position+1])
            # orig_answer_text = input_eval_examples[item['id']]['answer']

            question_tokens = item['ques_tokens']
            question_text = ' '.join(question_tokens)

            example = InputAnswerExtractionExample(
                did=qid,
                qid=qid,
                query_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=answer_text,
                start_position=start_position,
                end_position=end_position)
            examples.append(example)
        return examples


class CoqaProcessor(DataProcessor):
    """Processor for the COQA dataset."""

    def __init__(self, data_dir, dataset_type=None):
        self.data_dir = data_dir
        self.dataset_type = dataset_type or 1 # history size. all = -1
        assert type(self.dataset_type) == int

    def get_train_examples(self):
        return self._create_examples('train', self.dataset_type)

    def get_dev_examples(self):
        return self._create_examples('dev', self.dataset_type)

    def get_test_examples(self):
        return self._create_examples('test', self.dataset_type)

    def _create_examples(self, data_split, history_size=1):
        input_file = os.path.join(self.data_dir,
                                  'coqa.{}.json'.format(data_split))
        with open(input_file, "r") as reader:
            input_data = json.load(reader)["data"]

        # to detect yes/no answer
        def normalize_answer(ans_str):
            exclude = set(string.punctuation)
            return ''.join([c for c in ans_str.lower() if c not in exclude]).strip()

        constant_answers = ['unknown', 'yes', 'no']

        examples = []
        for passage in tqdm(input_data):
            context_tokens = passage['annotated_context']['word']
            if not context_tokens[-1] == 'unknown':
                context_tokens.extend(constant_answers)
            else:
                context_tokens.extend(constant_answers[1:])

            doc_tokens = []
            orig_to_doc_index = []  # map the orig answer position to doc tokens
            for i, token in enumerate(context_tokens):
                orig_to_doc_index.append(len(doc_tokens))
                for subtoken in token.split('\xa0'):
                    doc_tokens.append(subtoken)

            for qi, qas in enumerate(passage['qas']):
                question_tokens = qas['annotated_question']['word']
                norm_ans = normalize_answer(qas['answer'])
                if norm_ans in constant_answers:
                    context_end = len(doc_tokens) - len(constant_answers)
                    ans_pos = context_end + constant_answers.index(norm_ans)
                    answer_span = [ans_pos, ans_pos]
                else:
                    answer_span = [orig_to_doc_index[qas['answer_span'][0]],
                                   orig_to_doc_index[qas['answer_span'][1]]]
                orig_answer_text = qas['answer']

                context_with_history = doc_tokens[:]
                history_from = 0 if history_size == -1 else max(0, qi - history_size)
                for qqas in passage['qas'][history_from:qi]:
                    context_with_history.extend(qqas['annotated_question']['word'])
                    context_with_history.extend(qqas['annotated_answer']['word'])

                qid = '-'.join([passage['source'], passage['id'], str(qas['turn_id'])])

                example = InputAnswerExtractionExample(
                    did=qid,
                    qid=qid,
                    query_text=' '.join(question_tokens),
                    doc_tokens=context_with_history,
                    orig_answer_text=orig_answer_text,
                    start_position=answer_span[0],
                    end_position=answer_span[1])
                examples.append(example)

        return examples


class DuoRCProcessor(DataProcessor):
    """
    Processor for the DuoRC dataset.
    We only use the first occurrence of an answer.
    """

    def __init__(self, data_dir, dataset_type=None):
        self.data_dir = data_dir
        self.dataset_type = dataset_type or 'prepro_pararc'
        assert self.dataset_type in ['pararc', 'prepro_pararc', 'selfrc']
        self.candidate_filepath = defaultdict(str)

    def get_train_examples(self):
        return self._create_examples('train', self.dataset_type)

    def get_dev_examples(self):
        return self._create_examples('dev', self.dataset_type)

    def get_test_examples(self):
        return self._create_examples('test', self.dataset_type)

    def set_candidates(self, data_split, pred_filepath):
        self.candidate_filepath[data_split] = pred_filepath

    def get_examples(self, data_split):
        return self._create_examples(data_split, self.dataset_type)

    def _create_examples(self, data_split, dataset_type='pararc'):
        data_file = os.path.join(self.data_dir, dataset_type,
                                 'data_{}.json'.format(data_split))
        shared_file = os.path.join(self.data_dir, dataset_type,
                                   'shared_{}.json'.format(data_split))
        with open(data_file, "r") as reader:
            data = json.load(reader)
        with open(shared_file, "r") as reader:
            shared_data = json.load(reader)

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or ord(c) == 160:
                return True
            return False

        examples = []
        for index in range(len(data['q'])):
            qid = data['ids'][index]
            query_tokens = data['q'][index]
            query_text = ' '.join(' '.join(query_tokens).split())

            orig_answer_text = data['answerss'][index][0]
            x_idx = data['*x'][index][0]
            doc_tokens = shared_data['x'][x_idx][0][0]
            ans_span = data['y'][index][0]
            start_position, end_position = ans_span[0][1], ans_span[1][1] - 1

            assert 0 <= start_position <= end_position < len(doc_tokens)

            example = InputAnswerExtractionExample(
                did=qid,
                qid=qid,
                query_text=query_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position)
            examples.append(example)

        return examples


class QA4MREProcessor(DataProcessor):
    """Processor for the QA4MRE dataset."""

    def __init__(self, data_dir, dataset_type=None):
        self.data_dir = data_dir
        self.dataset_type = None

    def get_train_examples(self):
        return self._create_examples('train')

    def get_dev_examples(self):
        return self._create_examples('dev')

    def get_test_examples(self):
        return self._create_examples('test')

    def get_examples(self, data_split):
        return self._create_examples(data_split)

    def get_num_options(self):
        return 5
        
    def _create_examples(self, data_split):
        if data_split == 'train':
            raise ValueError('There is no training data for QA4MRE')

        data_path = os.path.join(self.data_dir, 'qa4mre_gold.xml')

        import xml.etree.ElementTree as etree
        raw_data = etree.parse(data_path)
        elms = raw_data.getroot()

        examples = []
        for topic in elms.findall('topic'):
            topic_attrib = topic.attrib
            topic_id = topic_attrib['t_id'] + '_' + topic_attrib['t_name']
            for rtest in topic.findall('reading-test'):
                rtest_id = rtest.attrib['r_id']
                context = rtest.find('doc').text.replace("\u2019", "'")
                qs = rtest.findall('q')
                for q in qs:
                    question = q.find('q_str').text
                    options = []
                    for a in q.findall('answer'):
                        if 'correct' in a.attrib.keys():
                            ans_idx = len(options)
                        options.append(a.text.replace('\u2019', "'"))
                    qid = '{0}_{1}'.format(topic_id, rtest_id)
                    example = InputMultipleChoiceExample(
                        did=qid,
                        qid=qid,
                        query_text=question,
                        doc_text=context,
                        option_text_list=options,
                        ans_idx=ans_idx)
                    examples.append(example)

        return examples

