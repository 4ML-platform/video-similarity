import glob

from jina.flow import Flow


def load():
    videos = glob.glob("./index-videos/*.mp4")
    for video_path in videos:
        print(f"indexing {video_path}")
        yield video_path.encode()


def main():
    f = Flow().load_config('flow-index.yml')
    with f:
        f.index(load, batch=1)

if __name__ == "__main__":
    main()
