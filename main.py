from datasets import load_dataset
from parso import parse


def all_imports(node):
    for subscope in node.iter_funcdefs():
        yield from all_imports(subscope)
    for subscope in node.iter_classdefs():
        yield from all_imports(subscope)
    yield from node.iter_imports()


def contains_framework(item):
    code = item['code']
    ast = parse(code)
    generator = all_imports(ast)
    for im in generator:
        # gets the first path from all imports
        name = im.get_paths()[0][0]
        if "torch" in name.__repr__() or "tensorflow" in name.__repr__() or "jax" in name.__repr__():
            return True
    return False


def text_match(item):
    code = item['code']
    return "torch" in code or "tensorflow" in code or "jax" in code


def main():
    ds = load_dataset("codeparrot/github-code", streaming=True,
                      split="train", languages=["Python"])

    ds = ds.filter(contains_framework)

    f = open("files.txt", "a")
    for i in ds.take(10):
        f.write(i['code'])
        f.write("\n")

    f.close()


main()
