import os


_MIN_BLACKWELL_CUDA = (12, 8)


def resolve_execution_device(default_device):
    requested_device = os.getenv("EXECUTION_DEVICE", str(default_device)).strip().lower()
    if requested_device not in {"gpu", "cpu"}:
        raise ValueError("EXECUTION_DEVICE must be 'gpu' or 'cpu'.")
    return requested_device


def _parse_version_tuple(version):
    parts = []
    for token in str(version).split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        if len(parts) == 2:
            break
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts[:2])


def _format_compute_capability(value):
    if not value:
        return "unknown"
    major, minor = tuple(value[:2])
    return f"sm_{major}{minor}"


def _hide_gpus(tf):
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def configure_tensorflow_runtime(tf, requested_device, logger):
    tf.config.optimizer.set_jit(False)

    if requested_device == "cpu":
        _hide_gpus(tf)
        logger.info("🖥️ Modo de ejecución: CPU (EXECUTION_DEVICE=cpu)")
        return "cpu"

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.warning("⚠️ EXECUTION_DEVICE=gpu pero no se detectó ninguna GPU física.")
        return "cpu"

    build_info = tf.sysconfig.get_build_info()
    cuda_version = _parse_version_tuple(build_info.get("cuda_version", "0.0"))
    incompatible = []

    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
        except Exception:
            details = {}
        compute_capability = tuple(details.get("compute_capability", (0, 0)))
        if compute_capability >= (12, 0) and cuda_version < _MIN_BLACKWELL_CUDA:
            device_name = details.get("device_name", gpu.name)
            incompatible.append((device_name, compute_capability))

    if incompatible:
        devices = ", ".join(
            f"{name} ({_format_compute_capability(compute_capability)})"
            for name, compute_capability in incompatible
        )
        message = (
            "⚠️ TensorFlow GPU incompatible con la GPU detectada. "
            f"Build CUDA={build_info.get('cuda_version', 'unknown')} y GPU={devices}. "
            "Esta combinación termina en 'PTX version 8.5 does not support target sm_120' "
            "durante model.fit()."
        )
        if os.getenv("TFM_GPU_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError(message)
        _hide_gpus(tf)
        logger.warning(message + " Fallback automático a CPU. Usa TFM_GPU_STRICT=1 para fallar en seco.")
        return "cpu"

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"✅ GPU Activa: {gpus}")
    except RuntimeError as exc:
        logger.error(f"Error configuración GPU: {exc}")
    return "gpu"