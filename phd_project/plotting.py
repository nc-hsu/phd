def custom_log_formatter(x, pos):
    """
    Formats the tick label:
    - If x < 1, display as a decimal (e.g., 0.1, 0.01)
    - If x >= 1, display as an integer (e.g., 1, 10, 100)
    """
    # Check if the number is less than 1 (or 0.99999... for float safety)
    if x < 1.0:
        # Use a general format for floating point numbers
        # The .2f or similar might be too restrictive, so we check for magnitude.
        # Format based on the magnitude to avoid excessive trailing zeros.
        if x < 0.001:
            # If very small, revert to scientific notation for readability
            return f"{x:.0e}"
        else:
            # Use general float format, stripping insignificant zeros
            return f"{x:g}"
    else:
        # Use integer format for values >= 1
        return f"{int(round(x))}"