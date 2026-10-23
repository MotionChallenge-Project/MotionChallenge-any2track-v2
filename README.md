# Any2Track Generalist v2

Independent Motion Challenge adapter for [Any2Track Generalist v2](https://github.com/GalaxyGeneralRobotics/OpenTrack).

```sh
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-any2track-v2.git
cd motion-challenge-any2track-v2
uv sync
uv run evaluate.py
```

The root `motion-challenge` submodule owns datasets, physics, failure rules and scoring.
Local weights are reused; downloads are hash-checked. Results are stored outside Git.
Use `--weights PATH --data PATH --offline` for cached artifacts.

[Provenance](provenance.json)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.

Run all frames after tracking failure: `uv run evaluate.py --termination none`.
This records full-motion errors under a separate protocol; success still requires no failure.

## Checkpoint source

[GalaxyGeneralRobotics/OpenTrack](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/README.md#play-pretrained-checkpoints) · [LAFAN1 generalist v2 / checkpoints/model.onnx](https://drive.google.com/drive/folders/1wDL4Chr6sGQiCx1tbvhf9DowN73cP_PF). Author-released checkpoint.

The author README links the release folder. Its recorded v2 subfolder contains Drive file 1XG_-GP5svz_4TGh484X900dbctibZGAm; SHA256 pins the downloaded model.
