# test.py
# import tracemalloc
# tracemalloc.start()
import time
import logging

logger = logging.getLogger(__name__)
logger.warning(f"**********************Starting: {time.asctime()}")

from patch import wrap_show, statistic

from functools import partial
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from intel_npu_acceleration_library import NPUAutoModel as AutoModel
from functools import partial
from intel_npu_acceleration_library.compiler import CompilerConfig
config = CompilerConfig()
wrap_show(AutoModel, "from_pretrained")
wrap_show(AutoProcessor, "from_pretrained")

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', config, trust_remote_code=True, 
                                  torch_dtype=torch.float16,
                                  revision="320a581d2195ad4a52140bb427a07f7207aeac6e",
                                  proxies={"https": "http://127.0.0.1:1080"},
                                  export=False)

# model = model.to(device='opengl')
# model = torch.compile(model, backend="npu")
# model = model.to(device='npu')
# model = intel_npu_acceleration_library.compile(model, dtype=torch.float16)
# model = model.to(device='cpu', dtype=torch.bfloat16)
# exit(0)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True,
                                          revision="320a581d2195ad4a52140bb427a07f7207aeac6e",
                                          proxies={"https": "http://127.0.0.1:1080"},
                                        #   dtype=torch.bfloat16
                                          )   # Vincent: Can't download tokenizer automatically, must download them manually
model.eval()

image = Image.open('/home/vv/python/llm/eval.jpg').convert('RGB')
question = 'Print out sentences in the image.'
msgs = [{'role': 'user', 'content': question}]


# res = model.chat(
#     image=image,
#     msgs=msgs,
#     tokenizer=tokenizer,
#     sampling=True, # if sampling=False, beam_search will be used by default
#     temperature=0.7,
#     proxies={"https": "http://127.0.0.1:1080"},
#     # system_prompt='' # pass system_prompt if needed
#     revision="320a581d2195ad4a52140bb427a07f7207aeac6e" # Vincent added
# )

# logger.warning(f"<<<<<<<chat Finished: {time.asctime()}")
# print(res)

# def patching_embedding(method):
#     func = partial(method.__func__, method.__self__)
#     def _wrapper(*args, **kwargs):
#       print(f">>>>>>>embedding start: {time.asctime()}")
#       ret = func(*args, **kwargs)
#       print(f">>>>>>>embedding end: {time.asctime()}")
#       return ret
#     return _wrapper
# model.get_vllm_embedding = patching_embedding(model.get_vllm_embedding)

wrap_show(model, "get_vllm_embedding")
wrap_show(model, "chat")

logger.warning(f">>>>>>>Starting chat: {time.asctime()}")

## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7,
    stream=True,
    revision="320a581d2195ad4a52140bb427a07f7207aeac6e" # Vincent added
)

generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')

logger.warning(f"<<<<<<<chat Finished: {time.asctime()}")
statistic.show()
