from datasets import load_dataset
from parso import parse
from parso.tree import BaseNode
from functools import partial
import json
from jedi import Script
import function_lists
import re
from tqdm import tqdm

import_keyword = {
    "torch": "Name: torch",
    "tensorflow": "Name: tensorflow",
    "jax": "Name: jax"
}


def all_imports(node):
    for subscope in node.iter_funcdefs():
        yield from all_imports(subscope)
    for subscope in node.iter_classdefs():
        yield from all_imports(subscope)
    yield from node.iter_imports()


def contains_framework(framework, item):
    code = item['code']
    ast = parse(code)
    generator = all_imports(ast)
    import_match = import_keyword[framework]
    for im in generator:
        # gets the first path from all imports
        paths = im.get_paths()
        if len(paths[0]) > 0:
            name = im.get_paths()[0][0]
            if import_match in name.__repr__():
                return True
    return False


def match_with_line_num(code_string):
    re_newline = re.compile(r'\n')
    count = 0
    for m in re.finditer("Variable", code_string):
        count += 1
        line_count = 1
        last_match = None
        for line in re_newline.finditer(code_string, 0, m.start()):
            line_count += 1
            last_match = line

        start_column = m.start() - last_match.end()

        end_line = len(re_newline.findall(code_string, 0, m.end()))+1
        yield line_count, start_column


def build_dictionary(framework):
    dictionary = {}
    for item in function_lists.pytorch_functions:
        dictionary[item] = 0
    return dictionary


def _get_code_for_children(self, children, include_prefix):
    if include_prefix:
        return "".join(c.get_code() for c in children)
    else:
        first = children[0].get_code(include_prefix=False)
        return first + "".join(c.get_code(include_prefix=False) for c in children[1:])


BaseNode._get_code_for_children = _get_code_for_children


def main():
    ds = load_dataset("codeparrot/github-code", streaming=True,
                      split="train", languages=["Python"])

    ds = ds.filter(partial(contains_framework, "torch"))
    counts = build_dictionary("torch")

    files = 100
    for i in tqdm(ds.take(files)):
        # Removes comments and whitespace from the file
        code_file = parse(i['code']).get_code(include_prefix=False)
        for fn in function_lists.pytorch_functions:
            counts[fn] += len(re.findall(fn, code_file))

    print(counts)
    f = open("frequencies.json", "a")
    f.write(json.dumps(counts))
    f.close()


main()
