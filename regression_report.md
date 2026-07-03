# Regression Report

- **Golden file**: `tests\golden\atmospheric_m107.json`
- **Generated at**: 2026-07-03T06:03:46.350309+00:00
- **Thresholds**: rtol=0.02, atol=1e-06
- **New platform result**: placeholder (MVP not ready)
- **Overall**: FAIL

## Scalar comparison

| key | golden | actual | abs_err | rel_err | status |
|-----|--------|--------|---------|---------|--------|
| geodetic_range_m | 17959.803776 | 18107.059954 | 1.472562e+02 | 0.008199 | FAIL |
| impact_angle_deg | 60.163735 | 59.888291 | 2.754446e-01 | 0.004578 | FAIL |
| landed | 1.000000 | 1.002936 | 2.936429e-03 | 0.002936 | PASS |
| max_alt_m | 6024.912537 | 6027.760795 | 2.848257e+00 | 0.000473 | FAIL |
| range_m | 17996.854417 | 18143.784256 | 1.469298e+02 | 0.008164 | FAIL |
| tof_s | 69.080878 | 68.760799 | 3.200793e-01 | 0.004633 | FAIL |
| v_impact_m_s | 327.762609 | 329.200122 | 1.437513e+00 | 0.004386 | FAIL |

> **Note**: 当前使用占位结果。待 ``ballistic_sim.simulator`` 实现后，请在 ``_try_real_new_result`` 中替换为真实调用。
