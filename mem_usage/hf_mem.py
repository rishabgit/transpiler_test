import time
import ivy
from time import perf_counter
from transpiler.transpiler import transpile
import torch
import jax
import transformers
import transformers
from colorama import Fore
import psutil
import warnings

warnings.filterwarnings("ignore")


jax.config.update('jax_enable_x64', True)

mem = psutil.virtual_memory()
total_ram = mem.total
print(f"Total RAM: {total_ram / (1024 ** 3):.2f} GB")

mem = psutil.virtual_memory()
print(f"Memory usage at start: {mem.used / (1024 ** 3):.2f} GB")

# model_name = 'mosaicml/mpt-7b'
# model_name = 'mosaicml/mpt-1b-redpajama-200b'
# model_name = 'replit/replit-code-v1-3b'
model_name = 'openlm-research/open_llama_3b'

# model_name = 'michellejieli/emotion_text_classifier'

print(f'Loading {model_name}...')
model = transformers.AutoModelForCausalLM.from_pretrained(
  model_name,
  trust_remote_code=True
)
# model = transformers.AutoModelForSequenceClassification.from_pretrained(
#   model_name,
#   trust_remote_code=True
# )

mem = psutil.virtual_memory()
print(f"Memory usage after loading model: {mem.used / (1024 ** 3):.2f} GB")

id = model.main_input_name
expanded_dummy = model._expand_inputs_for_generation(expand_size=1, input_ids=model.dummy_inputs.get(id))
model_input = {id: expanded_dummy[0]}

mem = psutil.virtual_memory()
print(f"Memory usage after creating input tensors: {mem.used / (1024 ** 3):.2f} GB")


print('transpiling...')
graph = transpile(model.__call__, source="torch", to="jax", kwargs=model_input)

mem = psutil.virtual_memory()
print(f"Memory usage after eager transpiling: {mem.used / (1024 ** 3):.2f} GB")


def fn(x):
    return graph(**x).logits

jit_inputs = {}
for key, value in model_input.items():
        jit_inputs[key] = jax.numpy.array(value.cpu().numpy())

jitted = jax.jit(fn)
jitted(jit_inputs).block_until_ready()

mem = psutil.virtual_memory()
print(f"Memory usage after first jitted call: {mem.used / (1024 ** 3):.2f} GB")

jax_times = []
for _ in range(10):
    s = perf_counter()
    jitted(jit_inputs).block_until_ready()
    jax_times.append(perf_counter() - s)
jax_avg = sum(jax_times) / len(jax_times)

print(Fore.GREEN +f"Result {jax_avg}"+ Fore.RESET)

mem = psutil.virtual_memory()
print(f"Memory usage at end: {mem.used / (1024 ** 3):.2f} GB")
