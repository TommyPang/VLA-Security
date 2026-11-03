# CPU demonstration of the patch objective

This synthetic reasoning/action surrogate demonstrates patch compositing and
joint, action-only, and CoT-only optimization. Its numerical results do not
measure a real VLA model or reproduce the paper's tables.

```bash
cd toy
python -m pip install -r requirements.txt
python examples/toy_reproduction.py
python -m unittest discover -s tests -v
```

Use `--steps 20 --eval-scenes 64` for a short run. Outputs go to the ignored
`outputs/toy_reproduction/` directory under the current working directory.
