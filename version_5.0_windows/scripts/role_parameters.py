"""Role-named parameter inputs with read-only compatibility for old spellings."""


def role_parameter(params, name, legacy_name, default=None):
    # Presence, not truthiness: an explicit null/zero must not revive an old value.
    return params[name] if name in params else params.get(legacy_name, default)
