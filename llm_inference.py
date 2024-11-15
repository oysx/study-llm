# llm-inference.py
# import tracemalloc
# tracemalloc.start()
from config import using_intel_npu_acceleration_library, using_huggingface_accelerator, using_statistic, using_streaming

import time
import logging
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

logger = logging.getLogger(__name__)
logger.warning(f"**********************Starting: {time.asctime()}")

device = 'cpu'
if using_huggingface_accelerator:
  from accelerate import Accelerator
  accelerator = Accelerator()
  device = accelerator.device

from patch import wrap_show, statistic
if using_intel_npu_acceleration_library:
  from intel_npu_acceleration_library import NPUAutoModel as AutoModel
  from intel_npu_acceleration_library.compiler import CompilerConfig
  config = CompilerConfig()

wrap_show(AutoModel, "from_pretrained")
wrap_show(AutoProcessor, "from_pretrained")

if using_intel_npu_acceleration_library:
  model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', config, 
                                    trust_remote_code=True, 
                                    torch_dtype=torch.float16,
                                    revision="320a581d2195ad4a52140bb427a07f7207aeac6e",
                                    proxies={"https": "http://127.0.0.1:1080"},
                                    export=False)
else:
  model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5',
                                    trust_remote_code=True, 
                                    torch_dtype=torch.float16,
                                    revision="320a581d2195ad4a52140bb427a07f7207aeac6e",
                                    proxies={"https": "http://127.0.0.1:1080"},)

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True,
                                          revision="320a581d2195ad4a52140bb427a07f7207aeac6e",
                                          proxies={"https": "http://127.0.0.1:1080"},
                                        #   dtype=torch.bfloat16
                                          )   # Vincent: Can't download tokenizer automatically, must download them manually

# model = model.to(device='cpu', dtype=torch.bfloat16)
model.to(device)
model, tokenizer = accelerator.prepare(model, tokenizer)
model.eval()

image = Image.open('/home/vv/python/llm/eval.jpg').convert('RGB')
question = 'Print out sentences in the image.'
msgs = [{'role': 'user', 'content': question}]

wrap_show(model, "get_vllm_embedding")
wrap_show(model, "chat")

logger.warning(f">>>>>>>Starting chat: {time.asctime()}")

if not using_streaming:
  res = model.chat(
      image=image,
      msgs=msgs,
      tokenizer=tokenizer,
      sampling=True, # if sampling=False, beam_search will be used by default
      temperature=0.7,
      proxies={"https": "http://127.0.0.1:1080"},
      # system_prompt='' # pass system_prompt if needed
      revision="320a581d2195ad4a52140bb427a07f7207aeac6e" # Vincent added
  )
  print(res)
else:
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
if using_statistic:
  statistic.show()
