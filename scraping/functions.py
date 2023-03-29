import requests
from bs4 import BeautifulSoup
import json

torch_functions = 'https://pytorch.org/docs/stable/nn'
tensor_functions = 'https://pytorch.org/docs/stable/tensors'
jax_functions = 'https://jax.readthedocs.io/en/latest'
list_modules =  [
    "torch",
    "nn",
    "nn.functional",
    "amp",
    "autograd",
    "library",
    "cuda",
    "backends",
    "distributed",
    "distributed.algorithms.join",
    "distributed.elastic",
    "fsdp",
    "distributed.optim",
    "distributions",
    "fft",
    "futures",
    "fx",
    "hub",
    "jit",
    "linalg",
    "monitor",
    "special",
    "torch.overrides",
    "package",
    "profiler",
    "nn.init",
    "onnx",
    "optim",
    "random",
    "nested",
    "sparse",
    "Storage",
    "testing",
    "benchmark_utils",
    "utils.bottleneck",
    "utils.checkpoint",
    "utils.cpp_extension",
    "utils.data",
    "utils.dlpack",
    "utils.mobile_optimizer",
    "utils.model_zoo",
    "utils.tensorboard",
    "__config__mod",
]

# Scrape the torch functions page 
def scrape_torch_functions(url):
    r = requests.get(url)
    # soup = BeautifulSoup("<p>Some</p>bad<i>HTML")
    soup = BeautifulSoup(r.text, 'html.parser')
    # find all p tags with class longtable docutils colwidths-auto align-default
    table = soup.find_all('table', class_='longtable')
    # find all span inner html with class pre
    print(len(table))
    res = [] 

    for t in table:

        rows = t.find_all('tr')
        for row in rows:
            td = row.find('td')
            function = td.find('span')
            # print(function.text)
            # add to array 
            res.append(function.get_text())

    return res

#  save to file 
def save_torch(functions):
    with open('scraping/torch.json', 'w') as f:
 
        f.write(json.dumps(
            functions, indent=4, sort_keys=True))
result = {}
for module in list_modules:
    print(module)
    arr = scrape_torch_functions(f'https://pytorch.org/docs/stable/{module}')
    result[module] = arr

save_torch(result)