import numpy as np
import time
from openvino.runtime import Core

# --- Load OpenVINO model ---
ie = Core()
compiled_model = ie.compile_model(
    "openvino_ir/student_model.xml",
    device_name="CPU",
    config={"PERFORMANCE_HINT": "THROUGHPUT"}
)
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# --- Prepare 128x128 dummy input (shape: 1, 3, 128, 128) ---
input_tensor = np.random.rand(1, 3, 128, 128).astype(np.float32)

# --- Warm-up (important for JIT/backends) ---
for _ in range(5):
    _ = compiled_model([input_tensor])

# --- Measure FPS over N runs ---
N = 100
start = time.time()
for _ in range(N):
    _ = compiled_model([input_tensor])[output_layer]
end = time.time()

fps = N / (end - start)
print(f"\n✅ OpenVINO Inference FPS @128x128: {fps:.2f}\n")
