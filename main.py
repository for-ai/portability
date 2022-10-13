from datasets import load_dataset
from functools import partial
import json
import function_lists
import re
from tqdm import tqdm
import code_tokenize as ctok

import_keyword = {
    "torch": "Name: torch",
    "tensorflow": "Name: tensorflow",
    "jax": "Name: jax"
}

pytorch_set = set(function_lists.pytorch_functions)
tensorflow_set = set(function_lists.tensorflow_functions)

error_files = []
false_negatives = []


def all_imports(node):
    for subscope in node.iter_funcdefs():
        yield from all_imports(subscope)
    for subscope in node.iter_classdefs():
        yield from all_imports(subscope)
    yield from node.iter_imports()


def contains_framework(framework, item):
    code_content = item['content']
    import_regex = r"(from.*" + re.escape(framework) + \
        r"|import.*" + re.escape(framework) + r")"
    matches = re.findall(import_regex, code_content)
    tensorflow_authors = r".*Copyright 20.. The TensorFlow Authors"
    if len(matches) > 0:
        tensorflow_authors_matches = re.findall(tensorflow_authors, code_content)
        if framework == "tensorflow" and len(tensorflow_authors_matches) > 0:
            return False
        else:
            return True
    elif framework in code_content:
        false_negatives.append(item["content"])


def build_dictionary(set):
    dictionary = {}
    for item in set:
        dictionary[item] = 0
    return dictionary


def get_name_frequencies(freq_dict, set, framework, file_data, debug=False):
    string = file_data["content"]
    freq_dict = freq_dict.copy()
    function_list = []
    module_list = []
    valid_index_list = []
    from_started = False
    if debug:
        with open("example.py", "w") as f:
            f.write(string)

    try:
        ast = ctok.tokenize(string, lang="python")
        for i in range(len(ast)):
            word = ast[i]
            # if i != 0 and ast[i - 1].type == "from" and framework == word.text:
            #     from_started = True
            #     # import code; code.interact(local=dict(globals(), **locals()))
            # elif ast[i].type == "import":
            #     if from_started:
            #         # import code; code.interact(local=dict(globals(), **locals()))
            #         module_list.append(ast[i + 1].text)
            #         from_started = False
            #         print("FROM")
            #         # import code; code.interact(local=dict(globals(), **locals()))

            #     elif framework == ast[i + 1].text:
            #         offset = 2
            #         while ast[i + offset].type == ".":
            #             offset += 2
            #         if ast[i + offset].type == "as":
            #             module_list.append(ast[i + offset + 1].text)
            #             print("AS")
            #             # import code; code.interact(local=dict(globals(), **locals()))

            #         else:
            #             offset = 1
            #             while ast[i + offset].type == ".":
            #                 offset += 2

            #             module_list.append(ast[i + 1].text)
            #             print("UNQUALIFIED")
            #             # import code; code.interact(local=dict(globals(), **locals()))
                    

            if word.type == "identifier" and word.text in set:
                if i != len(ast) - 1 and ast[i + 1].text == "(" and ((i != 0 and ast[i - 1].text != "def") or i == 0):
                    # if (word in module_list or (i >= 2 and (ast[i - 2].text in module_list or (ast[i - 2] == ")" and i - 4 in valid_index_list)))):
                        # print(word)

                        # print("VALID", word, ast[i - 2])
                        word = word.text
                        # if word not in dict:
                        #     dict[word] = 0

                        freq_dict[word] += 1
                        valid_index_list.append(i)
                    #     if i >= 4 and ast[i - 3].type == "=":
                    #         print("ASSIGNMENT")
                    #         import code; code.interact(local=dict(globals(), **locals()))
                    #         module_list.append(ast[i - 4].text)
                    # else:
                    #     print("VALID BUT NOT IN MODULE LIST", word)
                    #     import code; code.interact(local=dict(globals(), **locals()))
    except SyntaxError as e:
        error_files.append((file_data["repo_name"], file_data["path"]))
        pass
    if debug:
        with open("example.py", "w") as f:
            f.write(string)
        filtered = {}
        for key, value in freq_dict.items():
            if value > 0:
                filtered[key] = value

        print(filtered)
        import code
        code.interact(local=locals())
    return {"frequencies": freq_dict}


def main():
    ds = load_dataset("codeparrot/codeparrot-clean",
                      streaming=False, split="train")
    framework = "tensorflow"
    f = open(framework + '_flat_functions.json')
    
    function_list = json.load(f)
    f.close()

    # filters for files only containing framework imports
    ds = ds.filter(partial(contains_framework, framework))
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(3001))
    # tokenizes and gets frequencies of tokens

    starting_dict = build_dictionary(function_list)
    print("function list length: ", len(function_list))

    ds = ds.map(partial(get_name_frequencies, starting_dict, function_list, framework, debug=False), batched=False, remove_columns=[
                "alpha_frac", "autogenerated", "content", 'copies', 'hash', 'license', 'line_max', 'line_mean', 'size'])
    counts = build_dictionary(framework)

    counts = []
    for result in tqdm(ds):
        counts.append(result)
    for item in counts:
        new_item = {}
        for key, value in item["frequencies"].items():
            if value > 0:
                new_item[key] = value
        item["frequencies"] = new_item

    print(error_files)
    f = open("individual" + framework + "_frequencies.json", "w")
    f.write(json.dumps(counts, indent=4, sort_keys=True))
    f.close()

    result = {}
    for count in counts:
        for key, value in count["frequencies"].items():
            if key not in result:
                result[key] = 0
            result[key] += value

    f = open(framework + "_frequencies.json", "w")
    f.write(json.dumps(result, indent=4, sort_keys=True))
    f.close()

    f = open("false_negatives.txt", "w")
    for item in false_negatives:
        f.write(item)
    f.close()


main()
