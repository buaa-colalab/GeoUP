import functools
from inspect import getfullargspec
from typing import Callable, Iterable, Optional

import torch
from torch import nn
from torch.cuda.amp import autocast


def cast_tensor_type(inputs, src_type: torch.dtype, dst_type: torch.dtype):
    """Recursively convert Tensor types in nested structures."""
    if isinstance(inputs, torch.Tensor):
        if inputs.dtype == src_type:
            return inputs.to(dst_type)
        return inputs
    elif isinstance(inputs, (list, tuple)):
        return type(inputs)(cast_tensor_type(item, src_type, dst_type) for item in inputs)
    elif isinstance(inputs, dict):
        return {k: cast_tensor_type(v, src_type, dst_type) for k, v in inputs.items()}
    else:
        return inputs

def auto_bf16(
    apply_to: Optional[Iterable[str]] = None,
    out_fp32: bool = True,
    supported_types: tuple = (nn.Module,),
) -> Callable:
    """Decorator to enable automatic BFloat16 mixed precision training.
    This decorator converts specified input arguments from FP32 to BF16 before
    passing them to the decorated method. It uses PyTorch's `autocast` (for BF16)
    as the backend when available (PyTorch >= 1.10). The output can optionally
    be cast back to FP32.
    Args:
        apply_to (Iterable[str], optional): Argument names to convert to BF16.
            If None, all arguments are converted.
        out_fp32 (bool): Whether to cast the output back to FP32. Default True.
        supported_types (tuple): Types that can be decorated (e.g., nn.Module).
    Example:
        >>> class MyModule(nn.Module):
        >>>     @auto_bf16()
        >>>     def forward(self, x, y):
        >>>         return x + y
        >>>
        >>>     @auto_bf16(apply_to=('features',))
        >>>     def process(self, features, mask):
        >>>         return self.head(features)
    """
    def decorator(old_func: Callable) -> Callable:
        @functools.wraps(old_func)
        def new_func(*args, **kwargs) -> Callable:
            # 检查是否作用于合法类型
            if not isinstance(args[0], supported_types):
                raise TypeError(
                    f"@auto_bf16 can only decorate methods of classes {supported_types}, "
                    f"but got {type(args[0])}"
                )
            module = args[0]
            # 检查是否启用了 bf16 混合精度
            if not (hasattr(module, 'bf16_enabled') and module.bf16_enabled):
                return old_func(*args, **kwargs)
            # 获取函数参数名
            args_info = getfullargspec(old_func)
            arg_names = args_info.args[:len(args)]
            args_to_cast = set(args_to_cast for args_to_cast in (apply_to or args_info.args))
            # 转换 args
            new_args = []
            for i, arg_name in enumerate(arg_names):
                if arg_name in args_to_cast:
                    arg = args[i]
                    if isinstance(arg, torch.Tensor) and arg.dtype == torch.float32:
                        arg = arg.bfloat16()
                    elif isinstance(arg, (list, tuple, dict)):
                        arg = cast_tensor_type(arg, torch.float32, torch.bfloat16)
                    new_args.append(arg)
                else:
                    new_args.append(args[i])
            # 转换 kwargs
            new_kwargs = {}
            for k, v in kwargs.items():
                if k in args_to_cast:
                    if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                        v = v.bfloat16()
                    elif isinstance(v, (list, tuple, dict)):
                        v = cast_tensor_type(v, torch.float32, torch.bfloat16)
                    new_kwargs[k] = v
                else:
                    new_kwargs[k] = v
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                output = old_func(*new_args, **new_kwargs)
            # 将输出强制转回 FP32（通常推荐，因为梯度计算更稳定）
            if out_fp32:
                output = cast_tensor_type(output, torch.bfloat16, torch.float32)
            return output
        return new_func
    return decorator

# [MODIFIED] 创建一个更通用的 cast_tensor_type 的包装器
def cast_to_fp32(value):
    """A general-purpose caster to convert low-precision tensors to FP32."""
    if isinstance(value, torch.Tensor) and value.dtype in [torch.half, torch.bfloat16]:
        return value.to(torch.float)
    
    # 对于容器类型，递归地应用转换
    if isinstance(value, (list, tuple)):
        return type(value)(cast_to_fp32(v) for v in value)
    if isinstance(value, dict):
        return {k: cast_to_fp32(v) for k, v in value.items()}
        
    return value


def force_fp32(apply_to: Optional[Iterable] = None, out_fp16: bool = False) -> Callable:
    """Decorator to convert input arguments to fp32 in force.
    
    This is a modified version that supports both FP16 and BFloat16.
    """
    def force_fp32_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs) -> Callable:
            module = args[0]
            if not isinstance(module, torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the method of nn.Module')

            # [MODIFIED] 检查 fp16_enabled 或 bf16_enabled
            fp16_enabled = getattr(module, 'fp16_enabled', False)
            bf16_enabled = getattr(module, 'bf16_enabled', False)

            # 如果没有启用任何混合精度，则直接返回
            if not (fp16_enabled or bf16_enabled):
                return old_func(*args, **kwargs)

            # 获取需要转换的参数名
            args_info = getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            
            # [MODIFIED] 使用新的、更通用的转换逻辑
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_to_fp32(args[i]))
                    else:
                        new_args.append(args[i])

            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_to_fp32(arg_value)
                    else:
                        new_kwargs[arg_name] = arg_value

            # 在禁用 autocast 的上下文中执行原始函数
            with autocast(enabled=False):
                output = old_func(*new_args, **new_kwargs)

            # Full-BF16 modules may still use this decorator for numerically
            # sensitive geometry, but downstream activations should stay BF16.
            if bf16_enabled and getattr(module, 'force_bf16_output', False):
                output = cast_tensor_type(output, torch.float, torch.bfloat16)
            elif out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            
            return output

        return new_func

    return force_fp32_wrapper
