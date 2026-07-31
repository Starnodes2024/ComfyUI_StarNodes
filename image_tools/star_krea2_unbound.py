import torch
import comfy.patcher_extension

_WKEY = "star_unbound_enhancer"

_TAP_COUNT = 12
_TAP_DIM = 2560
_CHUNKS = 24
_CHUNK_DIM = 1280
_GAIN_PROFILE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0) * 2
_GLOBAL_BOOST = 15.0
_REL_CAP = 0.75


def _is_target(dm):
    return all(hasattr(dm, a) for a in ("txtfusion", "txtmlp", "blocks", "_unpack_context")) \
        and int(getattr(dm, "txtlayers", 0)) == _TAP_COUNT \
        and int(getattr(dm, "txtdim", 0)) == _TAP_DIM


def _forward_taps(tf, x, mask, topts):
    b, s, t, d = x.shape
    y = x.reshape(b * s, t, d)
    for blk in tf.layerwise_blocks:
        y = blk(y.contiguous(), mask=None, transformer_options=topts)
    proj = tf.projector(y.reshape(b, s, t, d).permute(0, 1, 3, 2).contiguous()).squeeze(-1)
    for blk in tf.refiner_blocks:
        proj = blk(proj, mask=mask, transformer_options=topts)
    return proj


def _enhanced_forward(tf, x, mask, topts):
    b, s, t, d = x.shape
    if t != _TAP_COUNT or d != _TAP_DIM:
        return tf._star_orig_fwd(x, mask=mask, transformer_options=topts)

    ref = _forward_taps(tf, x, mask, topts)

    gains = torch.tensor(_GAIN_PROFILE, device=x.device, dtype=torch.float32).to(x.dtype)
    boost = _GLOBAL_BOOST
    scaled = (x.reshape(b, s, _CHUNKS, _CHUNK_DIM) * gains.view(1, 1, _CHUNKS, 1) * boost).reshape_as(x)
    cand = _forward_taps(tf, scaled, mask, topts)

    delta = cand.detach().float() - ref.detach().float()
    base_rms = torch.sqrt(torch.mean(ref.detach().float() ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
    delta_rms = torch.sqrt(torch.mean(delta ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
    scale = (_REL_CAP / (delta_rms / base_rms)).clamp(max=1.0)
    return (ref.detach().float() + delta * scale).to(cand.dtype)


def star_unbound_wrapper(executor, *args, **kwargs):
    # 'x' ist in der Regel das erste positionelle Argument nach dem 'executor'
    x = kwargs.get("x", args[0] if args else None)
    
    # transformer_options wird von ComfyUI fast immer als Keyword-Argument uebergeben
    topts = kwargs.get("transformer_options", {})
    cfg = topts.get(_WKEY) if topts else None
    
    # BUGFIX: Auf explizit 'None' pruefen, da leere Dictionaries sonst als False gewertet werden.
    if cfg is None or cfg.get("_active"):
        return executor(*args, **kwargs)

    dm = executor.class_obj
    if not _is_target(dm):
        return executor(*args, **kwargs)

    tf = dm.txtfusion
    if hasattr(tf, "_star_orig_fwd"):
        tf.forward = tf._star_orig_fwd
        delattr(tf, "_star_orig_fwd")
    orig = tf.forward

    def patched_fwd(x_in, mask=None, transformer_options=None):
        tf._star_orig_fwd = orig
        try:
            return _enhanced_forward(tf, x_in, mask, transformer_options or {})
        finally:
            if hasattr(tf, "_star_orig_fwd"):
                delattr(tf, "_star_orig_fwd")

    try:
        cfg["_active"] = True
        tf.forward = patched_fwd
        return executor(*args, **kwargs)
    finally:
        cfg["_active"] = False
        tf.forward = orig


class StarKrea2Unbound:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    DESCRIPTION = "Unbound prompt adherence enhancer for Krea2 models."

    def apply(self, model):
        m = model.clone()
        to = m.model_options.setdefault("transformer_options", {})
        
        # BUGFIX: Dictionary mit Inhalt füllen, damit es nicht als leeres Element verpufft.
        to[_WKEY] = {"enabled": True}
        
        if hasattr(m, "remove_wrappers_with_key"):
            m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _WKEY)
        to.get("wrappers", {}).get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {}).pop(_WKEY, None)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _WKEY, star_unbound_wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {"StarKrea2Unbound": StarKrea2Unbound}
NODE_DISPLAY_NAME_MAPPINGS = {"StarKrea2Unbound": "⭐ Star Krea2 Unbound"}
