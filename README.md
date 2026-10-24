# Any2Track Generalist v2

Independent MotionChallenge adapter for [Any2Track Generalist v2](https://github.com/GalaxyGeneralRobotics/OpenTrack).

```sh
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-any2track-v2.git
cd MotionChallenge-any2track-v2
uv sync
uv run evaluate.py
```

The root `MotionChallenge-Core` submodule owns datasets, physics, failure rules and scoring.
Local weights are reused; downloads are hash-checked. Results are stored outside Git.
Use `--weights PATH --data PATH --offline` for cached artifacts.

[Provenance](provenance.json)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.


## Checkpoint source

[GalaxyGeneralRobotics/OpenTrack](https://github.com/GalaxyGeneralRobotics/OpenTrack/blob/cb9b751993a2483e5d1805a2565ddbfe950c04c9/README.md#play-pretrained-checkpoints) · [LAFAN1 generalist v2 / checkpoints/model.onnx](https://drive.google.com/drive/folders/1wDL4Chr6sGQiCx1tbvhf9DowN73cP_PF). Author-released checkpoint.

The author README links the release folder. Its recorded v2 subfolder contains Drive file 1XG_-GP5svz_4TGh484X900dbctibZGAm; SHA256 pins the downloaded model.

One `uv run evaluate.py` produces one verified `result.json`: success, completion,
three MPBPE alignments through first failure and the same three over the full motion.
Both scopes use the same checkpoint and source revision; no branch switching is needed.

## Observations

Ideal proprioception, without robot world-position or contact feedback. [Checkpoint requirements](admission.json) · [Rules](MotionChallenge-Core/docs/OBSERVATIONS.md).

Evaluation is unavailable until this checkpoint has a verified author-supported
inference path under these rules. The admission record describes the gap.
