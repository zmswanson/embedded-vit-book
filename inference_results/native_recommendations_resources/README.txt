Per-engine runtime memory and module energy measurements

Scope: the five Table 11 engines, batch 1, 96x96, MAXN_SUPER, CPU/GPU clocks
locked at 1.728/1.02 GHz. The script runs each trial in a fresh process with
resident input/output buffers and a captured CUDA Graph, processing 64 serial
inferences between synchronizations. No image decode, gallery matching or
host/device input transfer is included in the sustained measurement window.

Each configuration has three 45-second windows after 10 seconds of warm-up.
Engine order is shuffled across passes with seed 73. One real preprocessed
face input per pass is shared across configurations and recorded by hash.
An eight-second unloaded idle baseline precedes each fresh process. A
five-second loaded-idle interval is recorded separately. Neither baseline
is subtracted from the primary total module energy measurement.

VDD_IN is read from the onboard INA3221 monitor every 100 ms. Voltage in mV
times current in mA divided by 1e6 gives watts. Trapezoidal integration uses
the exact inference start/end boundaries, interpolating endpoint samples.
Joules divided by completed inferences gives joules per inference. This is
total module input energy, including baseline module draw, not GPU-only or
AC wall energy. Each compressed CSV preserves timestamps and raw readings.

Peak incremental system-memory demand is the idle median MemAvailable minus
the lowest MemAvailable during the sustained inference window. It measures
additional memory pressure relative to the existing idle system, rather than
the process's complete resident set or a sum of CPU and GPU memory figures.
Baseline variation, startup peak, host RSS/high-water values, TensorRT's
tracked device-allocation peak, and IO-buffer sizes are recorded separately.
CUDA memGetInfo is diagnostic only because it is unreliable on unified-memory
platforms. No host/device counters are added together.

The summary records the range of incremental-memory peaks across three trials,
and total integrated energy divided by all completed inferences. Energy
ranges across trials and allocator/host-memory details remain in the CSV/JSON
supporting record. TensorRT allocator callbacks cover weights and context
memory, while the system-memory measurement also captures runtime overhead.

Script: scripts/measure_native_resources.py. No clocks, power settings,
sensor configuration, system packages or engine binaries were changed.
The short pilot is kept separately on the Jetson and is not used in results.

NVIDIA reference for VDD_IN module scope:
https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html
