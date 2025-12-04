def create_check(default_device, default_dtype):
    def check_tensor(tensor, name, shape=None, dtype=None, device=None):
        if not isinstance(shape, tuple):
            shape = (shape,)
        if device is None:
            device = default_device
        if tensor.device.type != default_device:
            raise TypeError(f"{name} must be on {default_device}, but got {tensor.device.type}")
        if dtype is None:
            dtype = default_dtype
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must be of type {dtype}, but got {tensor.dtype}")
        if shape is not None and tensor.shape != shape:
            raise ValueError(f"{name} must be of shape {shape}, but got {tensor.shape}")

    return check_tensor
