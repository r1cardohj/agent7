"""Layered application configuration."""


def resolve_config(defaults, file_values, environ, overrides):
    resolved = dict(defaults)
    resolved.update(overrides)
    resolved.update(file_values)
    for key in resolved:
        environment_key = f"APP_{key.upper()}"
        if environment_key in environ:
            resolved[key] = environ[environment_key]
    return resolved
