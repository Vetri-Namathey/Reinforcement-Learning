"""Profile local GPU power/temperature telemetry while running an Ollama
LLM workload, saving the trace to data/local_gpu_trace.csv."""

import threading
import time

import ollama
import pandas as pd
import pynvml

from configs import DATA_DIR

TRACE_PATH = DATA_DIR / "local_gpu_trace.csv"

is_running = True
telemetry_data = []


def monitor_gpu():
    """Background thread polling the NVIDIA GPU every 0.1 seconds."""
    global is_running
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    start_time = time.time()

    while is_running:
        current_time = time.time() - start_time
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

        telemetry_data.append({
            "Time (s)": round(current_time, 2),
            "Power (W)": power_w,
            "Temperature (C)": temp,
            "GPU Utilization (%)": util
        })

        time.sleep(0.1)

    pynvml.nvmlShutdown()


def run_ollama_workload():
    print("Starting LLM Generation...")

    # A heavy prompt forces a distinct prefill power spike
    prompt = "Explain the complete history of artificial intelligence, including major breakthroughs and winters, in extreme detail."
    response = ollama.generate(model='mistral:7b', prompt=prompt)

    print("\nGeneration Complete!")
    print(f"Generated {len(response['response'].split())} words.")


if __name__ == "__main__":
    print("Initializing NVIDIA Telemetry...")

    monitor_thread = threading.Thread(target=monitor_gpu)
    monitor_thread.start()

    # Give the monitor a second to capture idle baseline power
    time.sleep(1)

    run_ollama_workload()

    is_running = False
    monitor_thread.join()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(telemetry_data)
    df.to_csv(TRACE_PATH, index=False)

    print(f"Telemetry saved to {TRACE_PATH} with {len(df)} data points.")
    print(f"Peak Power Draw: {df['Power (W)'].max()} W")
