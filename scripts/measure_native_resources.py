#!/usr/bin/env python3
"""Measure isolated native TensorRT memory and module energy on Jetson.

Each trial uses a fresh process, preloaded input, batch 1, and CUDA Graph.
System MemAvailable is sampled relative to the preceding idle baseline;
TensorRT allocations and host RSS are recorded separately, never summed.
VDD_IN voltage/current readings measure total module energy, including idle
draw. Energy is integrated over the completed-inference window only.
"""

import argparse
import bisect
import csv
import ctypes
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time


LABELS = ("swin_tiny_kd", "swin_tiny_kd_blocks2", "resnet50d",
          "deit3_small_fp16", "deit3_small_int8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clocks():
    cpu, gpu = {}, {}
    for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq"):
        cpu[p.parent.name] = {k: (p / k).read_text().strip()
                              for k in ("scaling_min_freq", "scaling_max_freq", "scaling_cur_freq")}
    for p in Path("/sys/class/devfreq").glob("*gpu*"):
        gpu[p.name] = {k: (p / k).read_text().strip() for k in ("min_freq", "max_freq", "cur_freq")}
    assert cpu and all(v["scaling_min_freq"] == v["scaling_max_freq"] == "1728000" for v in cpu.values())
    assert gpu and all(v["min_freq"] == v["max_freq"] == "1020000000" for v in gpu.values())
    return {"cpu": cpu, "gpu": gpu}


class Cuda:
    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so.12")
        ptr, size = ctypes.c_void_p, ctypes.c_size_t
        signatures = {
            "cudaSetDevice": [ctypes.c_int], "cudaMalloc": [ctypes.POINTER(ptr), size],
            "cudaFree": [ptr], "cudaMemcpy": [ptr, ptr, size, ctypes.c_int],
            "cudaStreamCreate": [ctypes.POINTER(ptr)], "cudaStreamDestroy": [ptr],
            "cudaStreamSynchronize": [ptr], "cudaStreamBeginCapture": [ptr, ctypes.c_int],
            "cudaStreamEndCapture": [ptr, ctypes.POINTER(ptr)],
            "cudaGraphInstantiateWithFlags": [ctypes.POINTER(ptr), ptr, ctypes.c_ulonglong],
            "cudaGraphLaunch": [ptr, ptr], "cudaGraphDestroy": [ptr], "cudaGraphExecDestroy": [ptr],
            "cudaMemGetInfo": [ctypes.POINTER(size), ctypes.POINTER(size)],
        }
        for name, types in signatures.items():
            getattr(self.lib, name).argtypes = types
            getattr(self.lib, name).restype = ctypes.c_int
        self.call("cudaSetDevice", 0)

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name} returned CUDA error {code}")

    def malloc(self, size):
        ptr = ctypes.c_void_p()
        self.call("cudaMalloc", ctypes.byref(ptr), size)
        return ptr.value

    def memory(self):
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.call("cudaMemGetInfo", ctypes.byref(free), ctypes.byref(total))
        return {"free_bytes": free.value, "total_bytes": total.value}


def pointer_value(value):
    if value is None or isinstance(value, int):
        return value or 0
    if isinstance(value, ctypes.c_void_p):
        return value.value or 0
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.argtypes, get_name.restype = [ctypes.py_object], ctypes.c_char_p
    get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
    get_ptr.argtypes, get_ptr.restype = [ctypes.py_object, ctypes.c_char_p], ctypes.c_void_p
    return get_ptr(value, get_name(value))


def worker(args):
    import gc
    import zipfile
    import numpy as np
    import tensorrt as trt
    from evaluate_native_deployment import preprocess

    def event(name, **values):
        print(json.dumps({"event": name, "time": time.monotonic(), **values}), flush=True)

    assert sha(args.engine) == args.engine_sha256
    cuda = Cuda()

    class Allocator(trt.IGpuAllocator):
        def __init__(self):
            trt.IGpuAllocator.__init__(self)
            self.allocations, self.events = {}, []
            self.current, self.peak = 0, 0

        def allocate(self, size, alignment, flags):
            if not size:
                return None
            address = cuda.malloc(size)
            assert not alignment or address % alignment == 0
            self.allocations[address] = size
            self.current += size
            self.peak = max(self.current, self.peak)
            self.events.append({"time": time.monotonic(), "action": "allocate", "bytes": size,
                                "live_bytes": self.current})
            return address

        def deallocate(self, memory):
            address = pointer_value(memory)
            if address:
                size = self.allocations.pop(address)
                cuda.call("cudaFree", address)
                self.current -= size
                self.events.append({"time": time.monotonic(), "action": "free", "bytes": size,
                                    "live_bytes": self.current})
            return True

        def allocate_async(self, size, alignment, flags, stream):
            return self.allocate(size, alignment, flags)

        def deallocate_async(self, memory, stream):
            return self.deallocate(memory)

        def reallocate(self, address, alignment, new_size):
            # The default runtime does not need resizable allocations.
            return None

    allocator = Allocator()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    runtime.gpu_allocator = allocator
    cuda_baseline = cuda.memory()
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError("Could not deserialize engine")
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    assert len(inputs) == len(outputs) == 1
    inp, out = inputs[0], outputs[0]
    assert context.set_input_shape(inp, (1, 3, 96, 96))
    with zipfile.ZipFile(args.images) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        entry = manifest["images"][args.image_index]
        image = preprocess(archive.read(entry["file"]))
    input_host = image.astype(trt.nptype(engine.get_tensor_dtype(inp)), copy=False)
    output_shape = tuple(context.get_tensor_shape(out))
    assert all(d > 0 for d in output_shape)
    output_host = np.empty(output_shape, dtype=trt.nptype(engine.get_tensor_dtype(out)))
    input_device, output_device = cuda.malloc(input_host.nbytes), cuda.malloc(output_host.nbytes)
    cuda.call("cudaMemcpy", input_device, input_host.ctypes.data, input_host.nbytes, 1)
    assert context.set_tensor_address(inp, input_device) and context.set_tensor_address(out, output_device)
    stream, graph, executable = (ctypes.c_void_p() for _ in range(3))
    cuda.call("cudaStreamCreate", ctypes.byref(stream))
    assert context.execute_async_v3(stream.value)
    cuda.call("cudaStreamSynchronize", stream)
    cuda.call("cudaMemcpy", output_host.ctypes.data, output_device, output_host.nbytes, 2)
    assert np.isfinite(output_host).all() and np.linalg.norm(output_host) > 0
    cuda.call("cudaStreamBeginCapture", stream, 0)
    assert context.execute_async_v3(stream.value)
    cuda.call("cudaStreamEndCapture", stream, ctypes.byref(graph))
    cuda.call("cudaGraphInstantiateWithFlags", ctypes.byref(executable), graph, 0)
    gc.collect()
    event("loaded_idle_start")
    time.sleep(args.loaded_idle_seconds)
    event("loaded_idle_end")

    def run_for(seconds):
        start, count = time.monotonic(), 0
        while True:
            for _ in range(64):
                cuda.call("cudaGraphLaunch", executable, stream)
            cuda.call("cudaStreamSynchronize", stream)
            count += 64
            end = time.monotonic()
            if end - start >= seconds:
                return {"start": start, "end": end, "completed_inferences": count,
                        "elapsed_seconds": end - start}

    event("warmup_start")
    run_for(args.warmup_seconds)
    event("warmup_end")
    event("measurement_start")
    measured = run_for(args.duration)
    event("measurement_end")
    cuda.call("cudaMemcpy", output_host.ctypes.data, output_device, output_host.nbytes, 2)
    assert np.isfinite(output_host).all()
    result = {**measured, "tensorrt_version": trt.__version__,
              "input_file": entry["file"], "input_tensor_sha256": hashlib.sha256(input_host.tobytes()).hexdigest(),
              "tensor_rt_peak_allocation_bytes": allocator.peak,
              "tensor_rt_live_allocation_bytes": allocator.current,
              "tensor_rt_allocation_events": allocator.events,
              "io_allocation_bytes": input_host.nbytes + output_host.nbytes,
              "context_device_memory_requirement_bytes": engine.device_memory_size_v2,
              "cuda_memory_baseline_diagnostic": cuda_baseline,
              "cuda_memory_loaded_diagnostic": cuda.memory(),
              "output_dimension": int(output_host.size),
              "output_norm": float(np.linalg.norm(output_host))}
    event("worker_result", result=result)
    cuda.call("cudaGraphExecDestroy", executable)
    cuda.call("cudaGraphDestroy", graph)
    del context, engine, runtime
    cuda.call("cudaFree", input_device)
    cuda.call("cudaFree", output_device)
    cuda.call("cudaStreamDestroy", stream)


class Sampler:
    def __init__(self, sensor, interval):
        self.sensor, self.interval = sensor, interval
        self.samples, self.pid, self.error = [], None, None
        self.stop = threading.Event()
        self.gpu = next(Path("/sys/class/devfreq").glob("*gpu*"))
        self.temperatures = {p.parent.name: p.parent / "temp"
                             for p in Path("/sys/class/thermal").glob("thermal_zone*/type")
                             if p.read_text().strip() in ("cpu-thermal", "gpu-thermal")}
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        start = time.monotonic()
        voltage = int((self.sensor / "in1_input").read_text())
        current = int((self.sensor / "curr1_input").read_text())
        info = {line.split(":")[0]: int(line.split()[1])
                for line in Path("/proc/meminfo").read_text().splitlines()}
        status = {}
        if self.pid:
            try:
                status = {line.split(":")[0]: int(line.split()[1])
                          for line in Path(f"/proc/{self.pid}/status").read_text().splitlines()
                          if line.startswith(("VmRSS:", "VmHWM:"))}
            except FileNotFoundError:
                pass
        result = {"time": (start + time.monotonic()) / 2,
                  "voltage_mV": voltage, "current_mA": current,
                  "module_power_W": voltage * current / 1e6,
                  "mem_available_kib": info["MemAvailable"], "swap_free_kib": info["SwapFree"],
                  "process_rss_kib": status.get("VmRSS", 0),
                  "process_hwm_kib": status.get("VmHWM", 0),
                  "gpu_clock_hz": int((self.gpu / "cur_freq").read_text())}
        result.update({key + "_mC": int(p.read_text()) for key, p in self.temperatures.items()})
        return result

    def loop(self):
        try:
            while not self.stop.is_set():
                self.samples.append(self.sample())
                self.stop.wait(self.interval)
        except BaseException as exc:
            self.error = str(exc)


def integral(samples, start, end, field="module_power_W"):
    times = [s["time"] for s in samples]
    assert times[0] <= start < end <= times[-1]

    def interpolate(t):
        i = bisect.bisect_right(times, t) - 1
        if times[i] == t:
            return samples[i][field]
        a, b = samples[i], samples[i + 1]
        weight = (t - a["time"]) / (b["time"] - a["time"])
        return a[field] + weight * (b[field] - a[field])

    points = [(start, interpolate(start))]
    points.extend((s["time"], s[field]) for s in samples if start < s["time"] < end)
    points.append((end, interpolate(end)))
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(points, points[1:]))


def run(args):
    records = json.loads(args.engines.read_text())["engines"]
    engines = dict(zip(LABELS, records))
    assert len(records) == 5 and len(engines) == 5
    if args.only:
        engines = {label: engines[label] for label in args.only}
    sensor = next(p.parent for p in Path("/sys/bus/i2c/drivers/ina3221/1-0040").glob("hwmon/hwmon*/in1_label")
                  if p.read_text().strip() == "VDD_IN")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode = subprocess.check_output(["nvpmodel", "-q"], text=True).strip()
    assert "MAXN_SUPER" in mode
    report = {"measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "nvpmodel": mode, "clocks_before": clocks(), "input_shape": [1, 3, 96, 96],
              "script_sha256": sha(__file__), "inference_helper_sha256": sha(Path(__file__).with_name("evaluate_native_deployment.py")),
              "engine_manifest_sha256": sha(args.engines), "image_archive_sha256": sha(args.images),
              "duration_seconds": args.duration, "warmup_seconds": args.warmup_seconds,
              "idle_baseline_seconds": args.idle_seconds, "loaded_idle_seconds": args.loaded_idle_seconds,
              "sample_interval_seconds": args.sample_interval, "seed": args.seed,
              "power_scope": "VDD_IN total module input power; includes idle draw; excludes carrier/wall-adapter losses",
              "memory_scope": "Peak decrease in system MemAvailable during steady inference relative to preceding idle median; fresh process per trial; no RSS/GPU summation",
              "gpu_allocation_scope": "TensorRT runtime allocator peak plus separately recorded IO; excludes CUDA driver/code allocations",
              "sensor_path": str(sensor), "sensor_update_interval": (sensor / "update_interval").read_text().strip(),
              "sensor_samples_setting": (sensor / "samples").read_text().strip(),
              "execution": "Batch 1, resident input/output buffers, CUDA Graph, 64 launches per synchronization",
              "trials": []}
    rng = random.Random(args.seed)
    image_indices = rng.sample(range(8171), args.repeats)
    for repeat in range(args.repeats):
        order = list(engines)
        rng.shuffle(order)
        for label in order:
            record = engines[label]
            assert sha(record["engine_path"]) == record["sha256"]
            name = f"{label}_{repeat + 1}"
            print("Starting", name, flush=True)
            before = clocks()
            sampler = Sampler(sensor, args.sample_interval)
            sampler.thread.start()
            time.sleep(args.idle_seconds)
            launched = time.monotonic()
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--engine", record["engine_path"],
                       "--engine-sha256", record["sha256"], "--images", str(args.images),
                       "--image-index", str(image_indices[repeat]), "--duration", str(args.duration),
                       "--loaded-idle-seconds", str(args.loaded_idle_seconds), "--warmup-seconds", str(args.warmup_seconds)]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            sampler.pid = process.pid
            events, lines = [], []
            for line in process.stdout:
                lines.append(line)
                if line.startswith('{"event":'):
                    events.append(json.loads(line))
            code = process.wait()
            time.sleep(args.sample_interval * 2)
            sampler.stop.set()
            sampler.thread.join()
            (args.output_dir / (name + ".log")).write_text("".join(lines))
            if code or sampler.error:
                raise RuntimeError(f"{name} failed: process={code}, sampler={sampler.error}; inspect its log")
            result = next(e["result"] for e in events if e["event"] == "worker_result")
            after = clocks()
            samples = sampler.samples
            baseline = [s for s in samples if s["time"] < launched]
            active = [s for s in samples if result["start"] <= s["time"] <= result["end"]]
            assert len(active) >= args.duration / args.sample_interval * .8
            base_available = statistics.median(s["mem_available_kib"] for s in baseline)
            joules = integral(samples, result["start"], result["end"])
            idle_start = next(e["time"] for e in events if e["event"] == "loaded_idle_start")
            idle_end = next(e["time"] for e in events if e["event"] == "loaded_idle_end")
            loaded_idle_watts = integral(samples, idle_start, idle_end) / (idle_end - idle_start)
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=samples[0])
            writer.writeheader(); writer.writerows(samples)
            trace = name + "_samples.csv.gz"
            (args.output_dir / trace).write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))
            trial = {"label": label, "repeat": repeat + 1, **record, **result,
                     "trace_file": trace, "trace_sha256": sha(args.output_dir / trace),
                     "events": events, "command": command,
                     "clocks_before": before, "clocks_after": after,
                     "active_samples": len(active), "module_energy_joules": joules,
                     "mean_module_power_W": joules / result["elapsed_seconds"],
                     "module_energy_mJ_per_inference": 1000 * joules / result["completed_inferences"],
                     "throughput_inferences_per_second": result["completed_inferences"] / result["elapsed_seconds"],
                     "loaded_idle_module_power_W": loaded_idle_watts,
                     "baseline_module_power_W": statistics.mean(s["module_power_W"] for s in baseline),
                     "baseline_mem_available_kib": base_available,
                     "baseline_mem_noise_MiB": (max(s["mem_available_kib"] for s in baseline) - min(s["mem_available_kib"] for s in baseline)) / 1024,
                     "peak_incremental_system_RAM_MiB": (base_available - min(s["mem_available_kib"] for s in active)) / 1024,
                     "startup_peak_incremental_system_RAM_MiB": (base_available - min(s["mem_available_kib"] for s in samples if s["time"] >= launched)) / 1024,
                     "peak_host_RSS_MiB": max(s["process_rss_kib"] for s in active) / 1024,
                     "tensor_rt_peak_allocation_MiB": result["tensor_rt_peak_allocation_bytes"] / 1024**2,
                     "minimum_observed_gpu_clock_hz": min(s["gpu_clock_hz"] for s in active)}
            report["trials"].append(trial)
            report["clocks_after"] = clocks()
            (args.output_dir / "native_resources.json").write_text(json.dumps(report, indent=2) + "\n")
            print(name, "W", round(trial["mean_module_power_W"], 3),
                  "mJ/query", round(trial["module_energy_mJ_per_inference"], 3),
                  "RAM MiB", round(trial["peak_incremental_system_RAM_MiB"], 1), flush=True)
    print("Completed", len(report["trials"]), "isolated trials", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--images", type=Path, required=True)
    common.add_argument("--duration", type=float, default=45)
    common.add_argument("--warmup-seconds", type=float, default=10)
    common.add_argument("--loaded-idle-seconds", type=float, default=5)
    work = commands.add_parser("worker", parents=[common])
    work.add_argument("--engine", type=Path, required=True)
    work.add_argument("--engine-sha256", required=True)
    work.add_argument("--image-index", type=int, required=True)
    run_parser = commands.add_parser("run", parents=[common])
    run_parser.add_argument("--engines", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--only", action="append", choices=LABELS)
    run_parser.add_argument("--repeats", type=int, default=3)
    run_parser.add_argument("--seed", type=int, default=73)
    run_parser.add_argument("--idle-seconds", type=float, default=8)
    run_parser.add_argument("--sample-interval", type=float, default=.1)
    args = parser.parse_args()
    assert args.duration > 0 and args.warmup_seconds > 0 and args.loaded_idle_seconds > 0
    if args.command == "worker":
        worker(args)
    else:
        assert args.repeats > 0 and args.idle_seconds > 0 and args.sample_interval > 0
        run(args)


if __name__ == "__main__":
    main()
