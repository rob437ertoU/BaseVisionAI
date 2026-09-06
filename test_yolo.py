from ultralytics import YOLO
import torch

print("=" * 60)
print("BaseVision AI - YOLO GPU Test")
print("=" * 60)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print("\nCargando modelo YOLO...")

model = YOLO("yolo26n.pt")

print("Modelo cargado correctamente.")

print("\nEjecutando inferencia en GPU...")

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=0,
    save=True,
    verbose=True
)

print("\n" + "=" * 60)
print("TEST COMPLETADO")
print("=" * 60)
print("BaseVision AI está listo para Computer Vision.")