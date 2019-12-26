import tokenization


class InputMultipleChoiceExample(object):
    """A single training/test example for multiple choice question."""

    def __init__(self, did, qid, doc_text, query_text, option_text_list, ans_idx):
        self.did = did
        self.qid = qid
        self.doc_text = doc_text
        self.query_text = query_text
        self.option_text_list = option_text_list
        self.ans_idx = ans_idx
        assert 0 <= self.ans_idx < len(self.option_text_list)
        self.input_ablation = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "question_id: %s" % (tokenization.printable_text(self.qid))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.query_text))
        s += ", option_texts: %s" % (
            ' / '.join([tokenization.printable_text(opt)
                        for opt in self.option_text_list]))
        s += ", doc_text: [%s]" % (self.doc_text)
        return s


class InputTokenizedMultipleChoiceExample(object):
    def __init__(
        self,
        did,
        qid,
        doc_tokens,
        query_tokens,
        option_tokens_list,
        ans_idx,
        ablation_info="",
    ):
        self.did = did
        self.qid = qid
        self.doc_tokens = doc_tokens
        self.query_tokens = query_tokens
        self.option_tokens_list = option_tokens_list
        self.ans_idx = ans_idx
        self.ablation_info = ablation_info

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        option_text = " / ".join(
            [
                tokenization.printable_text(" ".join(opt))
                for opt in self.option_tokens_list
            ]
        )
        to_print = tokenization.printable_text
        s = f"""
        question_id: {to_print(self.qid)}
        question: {to_print(' '.join(self.query_tokens))}
        option: {option_text}
        answer: {self.ans_idx}
        doc_tokens: {to_print(' '.join(self.doc_tokens))}
        ablation setting: {self.ablation_info}
        """
        return s


class InputMultipleChoiceFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputAnswerExtractionExample(object):
    """A single training/test example for answer extraction question."""

    def __init__(self, did, qid, query_text, doc_tokens, orig_answer_text=None,
                 start_position=None, end_position=None, orig_answer_texts=None,
                 is_impossible=None):
        self.did = did
        self.qid = qid
        self.query_text = query_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.orig_answer_texts = orig_answer_texts
        self.is_impossible = is_impossible
        self.input_ablation = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "doc_id: %s" % (tokenization.printable_text(self.did))
        s += "qas_id: %s" % (tokenization.printable_text(self.qid))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.query_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputTokenizedAnswerExtractionExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
        self,
        doc_id,
        qas_id,
        doc_tokens,
        query_tokens,
        start_position=None,
        end_position=None,
        orig_answer_text=None,
        ablation_info=None,
        orig_doc_tokens=None,
        orig_query_text=None,
        orig_answer_texts=None,
        is_impossible=None,
        refer_to_orig_doc=False,
    ):
        self.did = doc_id
        self.qid = qas_id
        self.doc_tokens = doc_tokens
        self.orig_doc_tokens = orig_doc_tokens or doc_tokens[:]
        self.query_tokens = query_tokens
        self.orig_query_tokens = orig_query_text or query_tokens[:]
        self.orig_answer_text = orig_answer_text
        self.orig_answer_texts = orig_answer_texts
        self.start_position = start_position
        self.end_position = end_position
        self.ablation_info = ablation_info
        self.is_impossible = is_impossible
        self.refer_to_orig_doc = refer_to_orig_doc

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        to_print = tokenization.printable_text
        s = f"""
        qas_id: {to_print(self.qid)}
        question_text: {to_print(" ".join(self.query_tokens))}
        doc_tokens: {to_print(" ".join(self.doc_tokens))}
        """
        if self.start_position:
            s += f"\n start_position: {self.start_position}"
        if self.start_position:
            s += f"\n end_position: {self.end_position}"
        return s


class InputAnswerExtractionFeature(object):
    """A single set of features of data."""

    def __init__(
        self,
        unique_id,
        example_index,
        doc_span_index,
        tokens,
        token_to_orig_map,
        token_is_max_context,
        input_ids,
        input_mask,
        segment_ids,
        start_position=None,
        end_position=None,
        is_impossible=None,
        query_token_to_orig_map=None,
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.query_token_to_orig_map = query_token_to_orig_map

