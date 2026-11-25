"""Freefield: serial native GGUF text encoding and diffusion loading.
Only used by the Recast and LTX compact recipes. No global memory profile changes.
"""
import gc
import importlib
import logging
import weakref
import torch
import gguf
import folder_paths
import nodes
import comfy.sd
import comfy.ops
import comfy.model_management as mm

def _scoped_loader():
    native = nodes.NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]
    mod = importlib.import_module(native.__module__)
    ops = importlib.import_module(mod.__package__ + ".ops")
    dq = importlib.import_module(mod.__package__ + ".dequant")
    dequantize, dequantize_functions = dq.dequantize, dq.dequantize_functions
    def _packed_inference_rows(weight):
        # The GGML blocks are row-independent only for a contiguous, block-aligned
        # logical matrix. Do not reinterpret patched, unsupported or oddly laid-out
        # weights. Keep all other loaders/operators on their original route.
        if weight is None or getattr(weight, "patches", ()) or torch.is_grad_enabled():
            return None
        qtype = getattr(weight, "tensor_type", None)
        shape = tuple(int(x) for x in getattr(weight, "tensor_shape", weight.shape))
        sizes = gguf.GGML_QUANT_SIZES.get(qtype)
        if qtype not in dequantize_functions or not sizes or len(shape) != 2:
            return None
        count, width = shape
        block_size, type_size = sizes
        if count <= 0 or width <= 0 or width % block_size or count * width * 2 < 512 * 1024**2:
            return None
        raw = weight.data
        if type(raw) is not torch.Tensor:
            raw = raw.as_subclass(torch.Tensor)
        row_bytes = width // block_size * type_size
        if raw.dtype not in (torch.uint8, torch.int8) or not raw.is_contiguous() or raw.numel() != count * row_bytes:
            return None
        return raw.view(torch.uint8).reshape(count, row_bytes), qtype, count, width


    def _decode_inference_rows(packed, qtype, width, dtype, dequant_dtype):
        decode_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(packed, qtype, (packed.shape[0], width), dtype=decode_dtype).to(dtype)


    def _gather_quantized_embedding_rows(spec, indices, dtype, dequant_dtype):
        raw, qtype, count, width = spec
        flat = indices.reshape(-1).to(device=raw.device, dtype=torch.long)
        # Gather packed rows before transfer/dequantization. If the table is
        # offloaded, CPU only performs the lookup/transfer of compressed bytes;
        # the original GPU device performs dequantization and inference.
        # Cap temporary dense decode to 128 MiB, preserving exact token order.
        chunk = max(1, (128 * 1024**2) // (width * 4))
        pieces = []
        for start in range(0, flat.numel(), chunk):
            packed = raw.index_select(0, flat[start:start+chunk]).to(indices.device)
            pieces.append(_decode_inference_rows(packed, qtype, width, dtype, dequant_dtype))
        if not pieces:
            return torch.empty((*indices.shape, width), device=indices.device, dtype=dtype)
        return torch.cat(pieces, dim=0).reshape(*indices.shape, width)


    def _bounded_quantized_linear(spec, input_tensor, bias, dequant_dtype):
        raw, qtype, count, width = spec
        # Output rows are independent; bound dense scratch while retaining the
        # original dequantizer and activation dtype. No approximation/requantization.
        chunk = max(1, (128 * 1024**2) // (width * max(4, input_tensor.element_size())))
        outputs = []
        for start in range(0, count, chunk):
            end = min(count, start + chunk)
            packed = raw[start:end].to(input_tensor.device)
            dense = _decode_inference_rows(packed, qtype, width, input_tensor.dtype, dequant_dtype)
            outputs.append(torch.nn.functional.linear(input_tensor, dense, None if bias is None else bias[start:end]))
            del dense, packed
        return torch.cat(outputs, dim=-1)

    class BoundedOps(ops.GGMLOps):
        class Linear(ops.GGMLOps.Linear):
            def forward_ggml_cast_weights(self, input):
                if input.device.type == "cuda" and not getattr(self.weight, "patches", ()) and not getattr(self.bias, "patches", ()):
                    spec = _packed_inference_rows(self.weight)
                    if spec is not None and input.shape[-1] == spec[3]:
                        bias = None
                        if self.bias is not None:
                            bias = self.get_weight(self.bias.to(input.device), input.dtype)
                            bias = comfy.ops.cast_to(bias, input.dtype, input.device, non_blocking=mm.device_supports_non_blocking(input.device), copy=False)
                        return _bounded_quantized_linear(spec, input, bias, self.dequant_dtype)
                return super().forward_ggml_cast_weights(input)
        class Embedding(ops.GGMLOps.Embedding):
            def forward_ggml_cast_weights(self, input, out_dtype=None):
                if input.device.type == "cuda" and self.max_norm is None and not self.scale_grad_by_freq and not self.sparse:
                    spec = _packed_inference_rows(self.weight)
                    if spec is not None:
                        dtype = out_dtype if out_dtype is not None else getattr(self, "dtype", torch.float32)
                        return _gather_quantized_embedding_rows(spec, input, dtype, self.dequant_dtype).to(dtype=out_dtype)
                return super().forward_ggml_cast_weights(input, out_dtype)
    class ScopedLoader(native):
        def load_patcher(self, clip_paths, clip_type, clip_data):
            clip = comfy.sd.load_text_encoder_state_dicts(
                clip_type=clip_type, state_dicts=clip_data,
                model_options={"custom_operations": BoundedOps,
                    "initial_device": torch.device("cpu"),
                    "load_device": mm.get_torch_device(),
                    "offload_device": torch.device("cpu")},
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                disable_dynamic=True)
            clip.patcher = mod.GGUFModelPatcher.clone(clip.patcher)
            return clip
    return ScopedLoader()

def _release():
    mm.reset_cast_buffers()
    mm.unload_all_models()
    gc.collect()
    mm.cleanup_models()
    mm.soft_empty_cache(force=True)

class FreefieldSequentialGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip_name": (folder_paths.get_filename_list("clip_gguf"),),
            "unet_name": (folder_paths.get_filename_list("unet_gguf"),),
            "family": (["wan", "ltxv"],),
            "text": ("STRING", {"multiline": True})},
            "optional": {"after_drive_mask": ("IMAGE",), "after_ref_mask": ("IMAGE",)}}
    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "load"
    CATEGORY = "freefield/loaders"
    def load(self, clip_name, unet_name, family, text, after_drive_mask=None, after_ref_mask=None):
        dev = mm.get_torch_device()
        if dev.type != "cuda":
            raise RuntimeError("FreefieldSequentialGGUF requires a GPU")
        for label, mask in (("driver", after_drive_mask), ("reference", after_ref_mask)):
            if mask is not None:
                blue = (mask[..., 2] > .88) & (mask[..., 0] < .12) & (mask[..., 1] < .12)
                if not bool(blue.flatten(1).any(dim=1).all().item()):
                    raise ValueError("No target in one or more " + label + " frames; choose a more specific target.")
        _release()
        before = torch.cuda.memory_allocated(dev)
        with torch.inference_mode():
            clip = _scoped_loader().load_clip(clip_name, family)[0]
            if clip.patcher.load_device.type != "cuda":
                raise RuntimeError("Text encoder not assigned to GPU")
            refs = (weakref.ref(clip.cond_stage_model), weakref.ref(clip.patcher))
            cond = nodes.CLIPTextEncode().encode(clip, text)[0]
            torch.cuda.synchronize(dev)
            # Conditioning is the only encoder result retained.
            del clip
            _release()
            if any(ref() is not None for ref in refs):
                raise RuntimeError("Text encoder retained: refusing overlapping model allocation")
            logging.info("Freefield sequential %s: encoder released, GPU bytes %d -> %d", family, before, torch.cuda.memory_allocated(dev))
            model = nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]().load_unet(unet_name)[0]
            return model, cond

NODE_CLASS_MAPPINGS = {"FreefieldSequentialGGUF": FreefieldSequentialGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"FreefieldSequentialGGUF": "Freefield Sequential GGUF (GPU)"}
