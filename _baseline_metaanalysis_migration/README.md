# covmar migration baseline

This folder freezes what the unmodified phd fitting code produced on the real
study data: the inputs, the expected results, the calibration, the data
containers and the specs of 34 cases. The covmar package, a copy of that code
being refactored, is tested against it.

**Private.** It holds real study data and results. Nothing in it may be copied
into the covmar repository, which will be public. covmar's baseline tests read
it in place through the `COVMAR_BASELINE_DIR` environment variable and are
skipped when that is unset.

The rules for working here are in `CLAUDE.md`. Everything else (every file,
field and term, how the baseline was made, and how to read a failing test) is
in covmar's `dev/baseline_reference.md`
(`C:\Users\clemettn\Documents\pypackages\covmar\dev\baseline_reference.md`).
