"""Session snapshot migration."""


def migrate_snapshot(snapshot):
    migrated = dict(snapshot)
    migrated["version"] = 2
    migrated["model_ref"] = migrated.pop("model", "")
    return migrated
