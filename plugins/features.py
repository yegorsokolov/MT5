from . import register_feature

# Example feature plugin placeholder
@register_feature
def add_dummy_feature(df):
    """Add a placeholder feature column if not present."""
    if "dummy" not in df.columns:
        df["dummy"] = 0
    return df
