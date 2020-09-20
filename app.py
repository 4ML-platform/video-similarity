import glob

from jina.flow import Flow


def load():
    videos = glob.glob("data/*")
    for video_path in videos:
        print(f"indexing {video_path}")
        yield video_path.encode()


f = Flow().load_config('flow-index.yml')
with f:
    print(glob.glob("data/*"))
    f.index(load,batch=1)
