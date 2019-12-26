import collections
import csv
import json
import random


def strize(obj):
    if isinstance(obj, list):
        return " ".join(obj)
    else:
        return obj


def output_statistics(exs, cache, tokenized_examples, task_name, input_ablation):
    result = collections.defaultdict(list)
    for example in exs:
        if example.qid == example.did:
            doc_cache = cache[example.qid]["doc"]
        else:
            doc_cache = cache[example.did]
        result["sentence_num"].append(len(doc_cache["sentences"]))
        result["doc_length"].append(
            sum([len(s["tokens"]) for s in doc_cache["sentences"]])
        )
        queries = [
            len(s["tokens"]) for s in cache[example.qid]["query"]["sentences"]
        ]
        result["query_length"].append(sum(queries))

    for k in ["sentence_num", "doc_length", "query_length"]:
        print("{}: {:.2f}".format(k, sum(result[k]) / len(result[k])))

    if isinstance(tokenized_examples[0].ablation_info, dict):
        stats = {}
        counter = []
        for ex in tokenized_examples:
            counter.append(len(ex.ablation_info))
        print("dataset: {}".format(task_name))
        print("length: {}".format(len(counter)))
        stats["length"] = len(counter)

        no_drop = len([x for x in counter if x == 0])
        print("no drop: {}".format(100.0 * no_drop / len(counter)))
        stats["no_drop"] = no_drop

        exist_drop_r = 100.0 * (len(counter) - no_drop) / len(counter)
        print("exist drop: {}".format(exist_drop_r))

        print("all average: {}".format(1.0 * sum(counter) / len(counter)))
        stats["token_average"] = 1.0 * sum(counter) / len(counter)

        output_path = "stats/{task_name}_{input_ablation}.json"
        with open(output_path, "w") as f:
            json.dump(stats, f)

        
def output_examples(tokenized_examples, original_examples, task_name, input_ablation):
    def read_jsonl(ff):
        d = {}
        for l in ff:
            line = json.loads(l)
            d[line["qid"]] = line
        return d

    from eval_ans_extr import metric_max_over_ground_truths as scoring, f1_score

    get_score = lambda prediction, ground_truths: scoring(
        f1_score, prediction, ground_truths
    )

    org_data_path = f"final_output/{task_name}/original/eval_preds_dev.jsonl"
    with open(org_data_path, "r") as f:
        org_data = read_jsonl(f)

    abl_data_path = f"final_output/{task_name}/{input_ablation}/eval_preds_dev.jsonl"
    with open(abl_data_path, "r") as f:
        ablation_data = read_jsonl(f)

    output_path = f"mturk/{task_name}_{input_ablation}_full.json"

    data_dict = {}
    for ex in tokenized_examples:
        answer = ex.orig_answer_texts or [ex.orig_answer_text]
        data_dict[ex.qid] = {
            "doc": " ".join(ex.doc_tokens),
            "query": strize(ex.query_tokens),
            "answer": answer,
            "org_pred": org_data[ex.qid],
            "new_pred": ablation_data[ex.qid],
            "org_score": get_score(org_data[ex.qid], answer),
            "new_score": get_score(ablation_data[ex.qid], answer),
        }

    with open(output_path, "w") as f:
        json.dump(data_dict, f)

    print("wrote: {}".format(output_path))


def output_mturk(tokenized_examples, original_examples, task_name, input_ablation):
    sample_num = 1750  # squad dev
    output_path = f"mturk/{task_name}_{input_ablation}_full.csv"
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        org_fields = [
            "q_id",
            "context",
            "question",
            "ground_truth_answer",
            "input_type",
        ]
        fields = [
            "{}_{}".format(i, field)
            for i in range(1, 1 + 6)
            for field in org_fields
        ]
        writer.writerow(fields)
        target_indices = random.sample(
            range(len(original_examples)), 6 * sample_num
        )
        for i in range(sample_num):
            line = []
            org_target = random.sample(range(6), 2)
            for j in range(6):
                if j in org_target:
                    add_example = original_examples[
                        target_indices[6 * i + j]
                    ]
                    input_type = "none"
                else:
                    add_example = tokenized_examples[
                        target_indices[6 * i + j]
                    ]
                    input_type = input_ablation
                line_item = [
                    add_example.qid,
                    " ".join(add_example.doc_tokens),
                    strize(add_example.query_tokens),
                    " | ".join(add_example.orig_answer_texts),
                    input_type,
                ]
                line.extend(line_item)
            writer.writerow(line)
    print("wrote: {}".format(output_path))
