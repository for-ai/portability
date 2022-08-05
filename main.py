from datasets import load_dataset
from parso import parse

ds = load_dataset("codeparrot/github-code", streaming=True,
                  split="train", languages=["Python"])

iterator = iter(ds)
count = 0


def all_imports(node):
    for subscope in node.iter_funcdefs():
        yield from all_imports(subscope)
    for subscope in node.iter_classdefs():
        yield from all_imports(subscope)
    yield from node.iter_imports()


def contains_framework(ast):
    generator = all_imports(ast)
    for im in generator:
        # gets the first path from all imports
        name = im.get_paths()[0][0]
        if "torch" in name.__repr__() or "tensorflow" in name.__repr__() or "jax" in name.__repr__():
            return True
    return False


def main():
    files = []
    for i in range(100000):
        code_data = next(iterator)
        if "torch" in code_data['code'] or "tensorflow" in code_data['code'] or "jax" in code_data['code']:
            files.append(code_data['code'])

    print("TEXT PASS LENGTH", len(files))
    filtered_files = [
        file for file in files if contains_framework(parse(file))
    ]

    print("AST PASS LENGTH", len(filtered_files))


main()
