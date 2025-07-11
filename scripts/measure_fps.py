import onnxruntime as ort
import numpy as np
import time

def measure_onnx_fps(onnx_path="student_model.onnx", input_shape=(1, 3, 128, 128), num_runs=100):
    # Create an ONNX inference session
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Dummy input
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Warm-up
    for _ in range(10):
        _ = session.run([output_name], {input_name: dummy_input})

    # Time the inference
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = session.run([output_name], {input_name: dummy_input})
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    print(f"Average inference time: {avg_time * 1000:.2f} ms")
    print(f"Estimated FPS: {fps:.2f}")
    return fps

if __name__ == "__main__":
    measure_onnx_fps()
