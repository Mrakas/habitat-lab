from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = '/mnt/data5/ghx/workplace/Huggingface/intervl'
image = load_image('/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug/t1192_f37.png')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
