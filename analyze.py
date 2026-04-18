import json
import os
import sys
import pandas as pd

SEED = [1234, 308, 1028]
# SHOT = [0, 1, 2, 5, 10]
SHOT = [0]
FEWSHOT_LABELS = [f"{x}shot" for x in SHOT] 
METHOD = ["cloze", "symbol", "api"]
CATEGORIES = ['standard_negation', 'local_negation', 'contradiction', 'paraphrase']

def remove_prefix_suffix(filename: str):
    name = filename.removeprefix("results_").removesuffix(".json")
    return name

def _read_samples_api(sample_data):
    temp = {}

    if sample_data["doc"]["standard_negation"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["standard_negation"])
        if idx != -1:
            temp[idx] = "standard_negation"

    if sample_data["doc"]["local_negation"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["local_negation"])
        if idx != -1:
            temp[idx] = "local_negation"

    if sample_data["doc"]["contradiction"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["contradiction"])
        if idx != -1:
            temp[idx] = "contradiction"

    if sample_data["doc"]["paraphrase"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["paraphrase"])
        if idx != -1:
            temp[idx] = "paraphrase"

    # Sort by appearance order in the query (A, B, C, D)
    sorted_items = sorted(temp.items(), key=lambda x: x[0])
    resps = sample_data["filtered_resps"][0]

    choices = ["A", "B", "C", "D"]
    choice_map = {}

    for i, (_, label) in enumerate(sorted_items):
        choice_map[choices[i]] = label

    return choice_map.get(resps, None)

def _read_samples_symbol(sample_data):
    temp = {}

    if sample_data["doc"]["standard_negation"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["standard_negation"])
        if idx != -1:
            temp[idx] = "standard_negation"

    if sample_data["doc"]["local_negation"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["local_negation"])
        if idx != -1:
            temp[idx] = "local_negation"

    if sample_data["doc"]["contradiction"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["contradiction"])
        if idx != -1:
            temp[idx] = "contradiction"

    if sample_data["doc"]["paraphrase"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["paraphrase"])
        if idx != -1:
            temp[idx] = "paraphrase"

    # Sort by appearance order in the query (A, B, C, D)
    sorted_items = sorted(temp.items(), key=lambda x: x[0])

    # Extract mapping of choices in order (A→..., B→..., C→..., D→...)
    mapping = [item[1] for item in sorted_items]

    # Extract model scores
    resps = [float(resp[0][0]) for resp in sample_data["resps"]]

    # Identify the index of the highest-scoring option (0~3)
    chosen_idx = resps.index(max(resps))
    
    return mapping[chosen_idx]

def _read_samples_cloze(sample_data):
    choices = ['standard_negation', 'local_negation', 'contradiction', 'paraphrase']
    resps = [float(resp[0][0]) for resp in sample_data["resps"]]

    return choices[resps.index(max(resps))]

def analyze(root_dir: str, method: str, fewshot: int):
    
    # Initialize counters for each choice type
    if fewshot == 0:
        result = {
            model_name: {
                'standard_negation': 0, 
                'local_negation': 0, 
                'contradiction': 0, 
                'paraphrase': 0}
            for model_name in os.listdir(f'{root_dir}/{fewshot}shot')
        }
    else:
        result = {
            model_name: 
                {seed:
                {'standard_negation': 0, 
                    'local_negation': 0, 
                    'contradiction': 0, 
                    'paraphrase': 0} for seed in SEED}
                
                for model_name in os.listdir(f'{root_dir}/{fewshot}shot')
                }

    
    for model_name in os.listdir(f'{root_dir}/{fewshot}shot'):
        base = f'{root_dir}/{fewshot}shot/{model_name}'
        for dirpath, dirnames, filenames in os.walk(base):
            for filename in filenames:
                # find the json files which starts with 'results'
                if filename.startswith("results") and filename.endswith(".json"):
                    result_file_path = os.path.join(dirpath, filename)
        
                    try:
                        with open(result_file_path, "r", encoding="utf-8") as r:
                            result_data = json.load(r)
                        
                        seed = result_data["config"]["random_seed"]
                        name = remove_prefix_suffix(filename)

                        sample_name = f"samples_ko_nubench_{method}_{name}.jsonl"
                        sample_file_path = os.path.join(dirpath, sample_name)
                        
                        try:
                            with open(sample_file_path, "r", encoding="utf-8") as s:
                                for line in s:
                                    if not line.strip():
                                        continue

                                    sample_data = json.loads(line)
                                    if method == "symbol":
                                        chosen = _read_samples_symbol(sample_data)
                                    elif method == "cloze":
                                        chosen = _read_samples_cloze(sample_data)
                                    elif method == "api":
                                        chosen = _read_samples_api(sample_data)
                                    
                                    # Increase count for the predicted choice type
                                    if fewshot == 0:
                                        if chosen == None:
                                            pass
                                        result[model_name][chosen] += 1
                                    else:
                                        if chosen == None:
                                            pass
                                        result[model_name][seed][chosen] += 1

                        except Exception as e_s:
                            print(f"failed to read sample file: {sample_file_path} → {e_s}")  
                                        
                    except Exception as e_r:
                        print(f"failed to read result file: {result_file_path} → {e_r}")
                

    return result

def analyze_cloze(data_path: str):
    result = {'standard_negation': 0, 'local_negation': 0, 'contradiction': 0, 'paraphrase': 0}
    choices = ['standard_negation', 'local_negation', 'contradiction', 'paraphrase']

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line) 
            resps = [float(resp[0][0]) for resp in data["resps"]]
            result[choices[resps.index(max(resps))]] += 1


    return result

def merge_fewshot(agg: dict, shot: int, src: dict) -> dict:
    shot_label = f"{shot}shot"
    for model, seeds in src.items():
        m = agg.setdefault(model, {})
        s = m.setdefault(shot_label, {})
        for seed, counts in seeds.items():
            s[seed] = counts
    return agg

def save_zeroshot_csv(method_name: str, zero_dict: dict):
    """
    zero_dict: zero_shot_results[method] 구조
        {
          "modelA": {"standard_negation": int, "local_negation": int, "contradiction": int, "paraphrase": int},
          "modelB": {...},
          ...
        }
    CSV: 행=model_name, 열=Z_CATEGORIES
    """
    if not zero_dict:
        print(f"[WARN] No zeroshot data for method={method_name}")
        return

    rows = []
    idx = []

    for model, counts in zero_dict.items():
        row = [int(counts.get(cat, 0)) for cat in CATEGORIES]
        rows.append(row)
        idx.append(model)

    df = pd.DataFrame(rows, index=idx, columns=CATEGORIES).sort_index()

    out_csv = f"analyze/{setting}/{method_name}_0shot.csv"
    df.to_csv(out_csv, encoding='utf-8-sig')
    print(f"Saved CSV: {out_csv}, shape={df.shape}")

def save_fewshot_csv(method_name: str, merged_dict: dict):
    """
    merged_dict: merged_fewshots[method] structure
        {
            "modelA": {
                "1shot": {seed: {4개 category}}, 
                "2shot": ...
            },
            "modelB": {...}
        }
    """
    # ---- generate 'MultiIndex' column (seed → category) ----
    col_tuples = []
    for seed in SEED:
        for cat in CATEGORIES:
            col_tuples.append((seed, cat))
    columns = pd.MultiIndex.from_tuples(col_tuples, names=['seed', 'type'])

    row_index = []
    row_data = []

    # ---- row generation (model → fewshot) ----
    for model, shots in merged_dict.items():
        for shot in SHOT:
            shot_label = f"{shot}shot"
            seed_dict = shots.get(shot_label, {}) or {}

            row_vals = []
            for seed in SEED:
                counts = seed_dict.get(seed, {}) or {}
                for cat in CATEGORIES:
                    row_vals.append(int(counts.get(cat, 0)))

            row_index.append((model, shot_label))
            row_data.append(row_vals)

    # ---- DataFrame Generation ----
    if not row_data:
        print(f"[WARN] No data to save for method={method_name}")
        return

    df = pd.DataFrame(
        row_data,
        index=pd.MultiIndex.from_tuples(row_index, names=['model', 'fewshot']),
        columns=columns
    )

    # ---- Sorting the data by the number of shots ----
    df = df.reindex(
        pd.MultiIndex.from_product(
            [sorted(df.index.levels[0]), FEWSHOT_LABELS],
            names=['model','fewshot']
        )
    )

    df = df.dropna(how='all')

    # ---- save to CSV file ----
    out_csv = f"analyze/{setting}/{method_name}_fewshot.csv"
    df.to_csv(out_csv, encoding='utf-8-sig')
    print(f"Saved CSV: {out_csv}, shape={df.shape}")


if __name__ == "__main__":
    setting = sys.argv[1] # 'baseline' or 'sft'
    if setting != "baseline" and setting!= "sft":
        raise ValueError("invalid configuration: sys.argv[1] must be 'baseline' or 'sft'.")
    root_dir = "~/Thunder-KoNUBench"
    
    #   - zero_shot_results[method]: model -> counts
    #   - merged_fewshots[method]:   model -> fewshot -> seed -> counts
    zero_shot_results = {}
    merged_fewshots = {}

    for method in METHOD:
        zero_shot_results[method] = {}
        merged_fewshots[method] = {}

        for shot in SHOT:
            res = analyze(root_dir=root_dir, method=method, fewshot=shot)

            if shot == 0:
                zero_shot_results[method] = res
            else:
                if setting == 'baseline':
                    merged_fewshots[method] = merge_fewshot(merged_fewshots[method], shot, res)

    
    for method in METHOD:
        os.makedirs(f"analyze/{setting}", exist_ok=True)
        zero_out_name = f"analyze/{setting}/{method}_0shot.json"
        with open(zero_out_name, "w", encoding="utf-8") as f:
            json.dump(zero_shot_results[method], f, ensure_ascii=False, indent=2)
            save_zeroshot_csv(method, zero_shot_results[method])
        
        if setting == 'baseline':
            few_out_name = f"analyze/{setting}/{method}_fewshot.json"
            with open(few_out_name, "w", encoding="utf-8") as f:
                json.dump(merged_fewshots[method], f, ensure_ascii=False, indent=2)
                save_fewshot_csv(method, merged_fewshots[method])