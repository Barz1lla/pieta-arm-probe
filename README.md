# pieta-arm-probe

Answers one question on real Arm64 silicon: does the CPU advertise **i8mm** (Armv8.6 int8 matmul), and does the i8mm
code path compute correctly? Companion to a DotProd control test.

Acceptance is a negative-capable test: if `i8mm` is absent from `/proc/cpuinfo`, the i8mm build fails by design rather
than silently falling back to scalar.
