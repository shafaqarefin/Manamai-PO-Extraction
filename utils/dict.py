def combine_dicts(*dicts):
    """Combine dictionaries using ** unpacking"""
    result = {}
    for d in dicts:
        result = {**result, **d}
    return result


def remove_keys_from_dic(dic: dict[str, str], to_remove: list[str]) -> dict[str, str]:
    for key in to_remove:
        # Using pop with None to avoid KeyError if key doesn't exist
        dic.pop(key, None)
    return dic
