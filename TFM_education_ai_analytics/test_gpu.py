import tensorflow as tf
import sys

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detectadas: {len(gpus)}")
    for gpu in gpus:
        print(f" - {gpu.name}")
        
    print("\nRealizando prueba básica de tensores en GPU...")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("Resultado de la prueba:\n", c.numpy())
        print("✅ ¡La GPU funciona y NO ha crasheado con JIT-PTX al compilar!")
else:
    print("❌ No se ha detectado ninguna GPU. Verifica tu configuración de Docker/NVIDIA.")
