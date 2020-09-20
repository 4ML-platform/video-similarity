
from jina.executors.encoders.frameworks import BaseTFEncoder
import sys
import os

cd = os.path.dirname(__file__)
print(cd)
sys.path.append(cd+"/visil")
from model.visil import ViSiL
from datasets import load_video

class VisilVideoEncoder(BaseTFEncoder):
    def __init__(self, model_dir:str="ckpt/resnet", batch_size=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dir = model_dir
        self.batch_size = batch_size

    def post_init(self):
        self.model = ViSiL(self.model_dir)

    def encode(self, data, *args, **kwargs):
        features = self.model.extract_features(data, batch_sz=self.batch_size)
        return features


