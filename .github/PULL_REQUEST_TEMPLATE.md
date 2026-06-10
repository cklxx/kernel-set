# Summary

<!-- What does this PR change and why? Link any related issue (e.g. Closes #123). -->

## Type of change

- [ ] Bug fix
- [ ] New kernel / operator
- [ ] New / updated binding
- [ ] Auto-selection / routing change
- [ ] Docs / benchmarks / CI
- [ ] Other:

## Verification

<!-- Tick what you ran. Most checks run without a GPU. -->

- [ ] `python scripts/check_abi.py`
- [ ] `python scripts/gen_baselines.py --check` and `python scripts/gen_optimal.py --check`
- [ ] `PYTHONPATH=bindings/python python -m pytest bindings/python/tests/test_dispatch.py bindings/python/tests/test_optimal.py -q`
- [ ] Built with `nvcc` (`cmake --build`) — arch(es): <!-- e.g. sm_80, sm_89, sm_90 -->
- [ ] `ctest --test-dir build` on a GPU — device: <!-- e.g. L4 / A100 / none -->
- [ ] Added / updated a `kernels/tests/test_*.cu` for changed kernels

## Contract compliance

- [ ] I did **not** change any signature in `include/`
- [ ] I followed the file-ownership rules in [`CONTRACT.md`](../blob/main/CONTRACT.md)
- [ ] I added a `CHANGELOG.md` entry under "Unreleased"

## Notes for reviewers

<!-- Anything non-obvious: tuning left as follow-up, known limitations, etc. -->
