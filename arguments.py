import argparse

def get_argparse(answer_type="answer_extraction", return_parser=False):
    parser = argparse.ArgumentParser()

    # setup/logging parameters
    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--vocab_file",
        default=None,
        type=str,
        required=True,
        help="The vocabulary file that the BERT model was trained on.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.",
    )
    parser.add_argument(
        "--dataset_option",
        default=None,
        type=str,
        help=(
            "dataset-specific option. "
            "RACE={'high', 'middle'}, "
            "MCTest={'mc160', 'mc500'}"
        ),
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,
        help="Where to save cache of features."
    )
    parser.add_argument(
        "--corenlp_cache_dir",
        default="corenlp_{}",
        type=str,
        help="directory of corenlp caches",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="Where to get the dataset data."
    )
    parser.add_argument(
        "--log_spec",
        default=None,
        type=str,
        help="specification for logging filename.",
    )
    parser.add_argument(
        "--no_cache",
        default=False,
        action="store_true",
        help="Never use feature cache."
    )
    parser.add_argument(
        "--verbose_logging",
        default=False,
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    # run parameters
    parser.add_argument(
        "--do_train",
        default=False,
        action="store_true",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--do_eval",
        default=False,
        action="store_true",
        help="Whether to run eval on the dev set.",
    )
    parser.add_argument(
        "--do_test",
        default=False,
        action="store_true",
        help="Whether to run eval on the dev set.",
    )
    parser.add_argument(
        "--eval_on_train",
        default=False,
        action="store_true",
        help="Evaluate on the training set.",        
    )
    parser.add_argument(
        "--data_split",
        default="",
        type=str,
    )

    # processing parameters
    parser.add_argument(
        "--do_lower_case",
        default=True,
        action="store_true",
        help="Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.",
    )
    parser.add_argument(
        "--max_seq_length",
        default={
            'answer_extraction': 384,
            'multiple_choice':128
        }[answer_type],
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default={
            'answer_extraction': 64,
            'multiple_choice': 23
        }[answer_type],
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )

    # model parameters
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        help="Initial checkpoint (usually from a pre-trained BERT model).",
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=64,  # 32
        type=int,
        help="Total batch size for predictions.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
        "of training.",
    )
    parser.add_argument(
        "--save_checkpoints_steps",
        default=200,
        type=int,
        help="How often to save the model checkpoint.",
    )
    parser.add_argument(
        "--save_model_steps",
        default=200,
        type=int,
        help="How often to save the model checkpoint.",
    )
    parser.add_argument(
        "--loss_report_steps",
        default=0,
        type=int,
        help="How often to report the loss."
    )
    parser.add_argument(
        "--eval_steps",
        default=200,
        type=int,
        help="How often to eval the model."
    )
    parser.add_argument(
        "--iterations_per_loop",
        default=1000,
        type=int,
        help="How many steps to make in each estimator call.",
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json "
        "output file.",
    )
    parser.add_argument(
        "--no_cuda",
        default=False,
        action="store_true",
        help="Whether not to use CUDA when available",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--optimize_on_cpu",
        default=False,
        action="store_true",
        help="Whether to perform optimization and keep the optimizer averages on CPU",
    )
    parser.add_argument(
        "--fp16",
        default=False,
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=128,
        help="Loss scaling, positive power of 2 values can improve fp16 convergence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization"
    )

    # method parameters
    parser.add_argument(
        "--input_ablation",
        default=None,
        type=str,
        help="input ablation: shuffle_sentences",
    )

    # output options
    parser.add_argument("--output_statistics", default=False, action="store_true")
    parser.add_argument("--output_mturk", default=False, action="store_true")
    parser.add_argument("--output_examples", default=False, action="store_true")
    parser.add_argument("--enter_debugger", default=False, action="store_true")

    # debug
    parser.add_argument("--debug_counter", default=-1, type=int)
    parser.add_argument("--debug_start_counter", default=-1, type=int)

    parser.add_argument("--small_debug", default=False, action="store_true")

    # vocabulary modifications
    parser.add_argument(
        "--entity_anonymization",
        # choices=["open", "close", "close_noun", "close_content", "close_contentverb"],
        default=None,
        type=str,
        help=(
            "Entity anonymization. close: use the same id for the same entity"
            "across context documents. open: use a different id."
        )
    )

    parser.add_argument(
        "--limit_vocab_size",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--limit_vocab_freq",
        default=None,
        type=int,
    )

    if answer_type == "answer_extraction":
        parser.add_argument(
            "--mix_input_ablation",
            default=None,
            type=str,
            help="example: shuff_document_words=10:shuffle_sentence_words=10",
        )        
        parser.add_argument("--ignore_out_of_span", default=False, action="store_true")
        parser.add_argument("--allow_impossible", default=False, action="store_true")
        parser.add_argument("--null_score_diff_threshold", default=0.0, type=float)

    elif answer_type == "multiple_choice":
        parser.add_argument(
            "--max_option_length",
            default=17,
            type=int,
            help="17 is used in GPTv1 on RACE."
        )
        parser.add_argument(
            "--convert_from_ans_extr",
            default=False,
            action="store_true",
            help="convert examples from answer extraction",
        )
        parser.add_argument(
            "--train_predictions",
            default=None,
            type=str,
            help="predictions for train examples",
        )
        parser.add_argument(
            "--eval_predictions",
            default=None,
            type=str,
            help="predictions for eval examples",
        )

    if return_parser:
        return parser #.parse_args()
    else:
        return parser.parse_args()
