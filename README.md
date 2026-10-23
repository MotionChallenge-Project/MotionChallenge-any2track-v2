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

[Provenance](provenance.json) · [Website](https://motion-challenge-project.github.io/motion-challenge-project/)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.
