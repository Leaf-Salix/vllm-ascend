"""Compare only the native slots written by the current decode step."""


def cache_dump(impl, metadata, payload):
    names = ("compressed", "raw", "main_state", "inner_state", "index_key", "index_scale")
    groups = (0, 4, 1, 2, 3, 3)
    result = {}
    for i, name in enumerate(names):
        group = groups[i]
        if group in (0, 3):
            slots = impl._compute_compressor_metadata(metadata[group].decode)[2]
        else:
            slots = metadata[group].decode.slot_mapping
        slots = slots.long()
        valid = (slots[:, 0] >= 0) & (slots[:, 1] >= 0)
        slots = slots[valid]
        values = {}
        for path in ("native", "pto"):
            values[path] = payload[path][i][slots[:, 0], slots[:, 1]].detach().cpu()
        result[name] = {"slots": slots.detach().cpu(), **values}
    return result
