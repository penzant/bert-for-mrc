import numpy as np

import tokenization
from eval_ans_extr import f1_score, exact_match_score, metric_max_over_ground_truths

score_fn = {"f1": f1_score, "em": exact_match_score}


class InputTokenizedExample(object):
    def __init__(
        self,
        did,
        qid,
        doc_tokens,
        query_tokens,
        option_tokens_list=None,
        ans_idx=None,
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        tok_to_orig_index=None,
        orig_to_tok_index=None,
        ablation_info="",
        orig_answer_texts=None,
        is_impossible=None,
    ):
        # original example: self.doc_tokens = orig_doc_tokens
        # sub example: self.doc_tokens = (dropped) orig_doc_tokens
        # sub exapmle: self.orig_doc_tokens = orig_doc_tokens
        # orig_doc_tokens are basic-tokenized in generating feature
        self.did = did
        self.qid = qid
        self.doc_tokens = doc_tokens
        self.orig_doc_tokens = doc_tokens[:]
        self.query_tokens = query_tokens
        self.option_tokens_list = option_tokens_list
        self.ans_idx = ans_idx
        self.orig_answer_text = orig_answer_text
        self.orig_answer_texts = orig_answer_texts or [orig_answer_text]
        self.start_position = start_position
        self.end_position = end_position
        self.ablation_info = ablation_info
        self.orig_logit = None  # logit of answer
        self.orig_pred = None  # prediction
        self.current_output = []
        if option_tokens_list and ans_idx is not None:
            self.ans_type = "multi_choice"
            self.orig_logits = []
        elif start_position is not None and end_position is not None:
            self.ans_type = "ans_extr"
            self.tok_to_orig_index = tok_to_orig_index
            self.orig_to_tok_index = orig_to_tok_index
            self.is_impossible = is_impossible
        else:
            raise ValueError
        self.keep_indices = []
        self.drop_indices = []
        self.corenlp_doc = None
        self.corenlp_query = None
        self.idx_to_leaf_span = {}
        self.doc_tree = None
        self.sent_spans = []
        self.orig_score = None
        self.refer_to_orig_doc = True

        self.cand_idx = None
        self.pdoc = None

    def get_score(self, prediction, scoring="f1"):
        if self.ans_type == "multi_choice":
            return int(np.argmax(prediction) == self.ans_idx)
        elif self.ans_type == "ans_extr":
            if self.is_impossible or self.orig_answer_texts == []:
                return 1 if prediction == "" and self.is_impossible else 0
            else:
                return metric_max_over_ground_truths(
                    score_fn[scoring], prediction, self.orig_answer_texts
                )

    def get_orig_score(self, prediction):
        if not self.orig_score:
            if self.ans_type == "multi_choice":
                self.orig_logits = prediction
                self.orig_score = self.get_score(self.orig_logits)
            elif self.ans_type == "ans_extr":
                self.orig_pred = prediction
                self.orig_score = self.get_score(self.orig_pred)
        return self.orig_score

    def check_stop(self, new_output):
        return new_output["logit"] <= self.orig_logit
        # return new_output['logit'] <= self.current_output['logit']

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        to_print = tokenization.printable_text
        s = f"""
        qas_id: {to_print(self.qid)}
        question_text: {to_print(" ".join(self.query_tokens))}
        doc_tokens: {to_print(" ".join(self.doc_tokens))}
        """
        if self.ans_type == "multi_choice":
            option_text = [" ".join(opt) for opt in self.option_tokens_list]
            s += f"\n options: {to_print(' / '.join(option_text))}"
            s.append("answer: {}".format(self.ans_idx))
        elif self.ans_type == "ans_extr":
            s += f"\n answer: {' / '.join(self.orig_answer_texts)}"
            if self.start_position:
                s += f"\n start_position: {self.start_position}"
            if self.end_position:
                s += f"\n end_position: {self.end_position}"

        return s

