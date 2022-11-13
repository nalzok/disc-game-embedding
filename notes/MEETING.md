+ Matrix completion code
    + Merge Zilai's code into our codebase, at `main/completion.py`.
        + The original notebook can be found at `scratches/MC.ipynb` (I fixed a typo in the objective).
    + Improve the performance of `MC_GD` by 6x times with JAX (1m4s -> 11s).
    + Add support for Adam with weight decay, a cosine learning rate schedule (with warmup), gradient clipping, and other fancy stuff.
        + With the tricks, I can achieve a slightly lower objective (~4200 -> ~4100).
    + Opinion: matrix completion/imputation should be an integral part of the embedding algorithm, not a pre-processing step.
        + This is not an imputation method, as it may change the value of known entries.
            + Patrick is not happy with that.
        + The current method overlooks the information in `X`.
        + For `MC_GD`, we should return `A, B` instead of forming `A @ B.T - B @ A.T`.
