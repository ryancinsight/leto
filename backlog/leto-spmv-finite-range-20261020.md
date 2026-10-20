<a id="LETO-SPMV-FINITE-RANGE-20261020"></a>
# LETO-SPMV-FINITE-RANGE-20261020 — Recover finite CSR row ranges [patch] — blocked
- priority: correctness; scope: native CSR recovery; target: Leto/LetoOps 0.46.3.
- [x] Confirm current main, registry slot and ownership; preserve separate 0.47 work.
- [x] Reproduce native f32/f64 and real GMRES failures before production changes.
- [x] Retry stored order for nonfinite results; preserve finite lane bits and wins.
- [x] Pass native/caller controls, full/strict/docs/minimal/MSRV and patch API gates.
- [x] Pass 5,160 differential cases/width/profile plus strict allocation controls.
- [x] Complete ABBA kernel/application controls, retaining all samples and anomalies.
- [ ] Complete standard package/source gates and assess all applicable hosted gates.
- [ ] Obtain exact-head independent numerical and performance acceptance.
- [ ] Coordinate existing private PyO3 advisories with FFI owner; no waiver/release.
- Preserve Athena126, Harmonia performance, future FFI and other release holds.
- Basis: [verification and numerical limits](../docs/verification/spmv-finite-range.md).
