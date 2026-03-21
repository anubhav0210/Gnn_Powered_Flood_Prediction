"""
Flood warning system - Binary flood/safe classification
"""

def is_flood(current_level_percent, threshold=0.90):
    """
    Determine if water level indicates flood condition.
    
    Args:
        current_level_percent (float): Current water level as % of capacity (0-100)
        threshold (float): Flood threshold as % of capacity (0-1, default 0.90 = 90%)
    
    Returns:
        bool: True if flood, False if safe
    """
    # Convert percentage to decimal (0-1 range)
    current_decimal = current_level_percent / 100.0
    return current_decimal >= threshold


def get_status_text(is_flood_status):
    """
    Get status text based on flood condition.
    
    Args:
        is_flood_status (bool): True if flood, False if safe
    
    Returns:
        str: Status text with emoji
    """
    if is_flood_status:
        return "🔴 FLOOD"
    else:
        return "🟢 SAFE"


def get_status_color(is_flood_status):
    """
    Get color code based on flood condition.
    
    Args:
        is_flood_status (bool): True if flood, False if safe
    
    Returns:
        str: Hex color code
    """
    if is_flood_status:
        return "#d62728"  # Red
    else:
        return "#2ca02c"  # Green


def calculate_capacity_percentage(water_level, max_capacity):
    """
    Calculate water level as percentage of capacity.
    
    Args:
        water_level (float): Current water level in meters
        max_capacity (float): Maximum capacity in meters
    
    Returns:
        float: Percentage (0-100)
    """
    if max_capacity <= 0:
        return 0.0
    percentage = (water_level / max_capacity) * 100.0
    return min(percentage, 150.0)  # Cap at 150% for display purposes


def days_until_threshold(predictions, threshold, max_capacity):
    """
    Calculate days until water level reaches threshold.
    
    Args:
        predictions (list): List of predicted water levels for next 7 days
        threshold (float): Flood threshold (0-1, e.g., 0.90 for 90%)
        max_capacity (float): Maximum capacity in meters
    
    Returns:
        int or None: Days until threshold exceeded, or None if never exceeds
    """
    threshold_level = threshold * max_capacity
    
    for day, level in enumerate(predictions, start=1):
        if level >= threshold_level:
            return day
    
    return None  # Never exceeds threshold in forecast period


def get_flood_status_for_lake(current_level, max_capacity, predictions, threshold):
    """
    Get complete flood status information for a lake.
    
    Args:
        current_level (float): Current water level in meters
        max_capacity (float): Maximum capacity in meters
        predictions (list): List of predicted water levels for next 7 days
        threshold (float): Flood threshold (0-1)
    
    Returns:
        dict: Complete status information
    """
    current_percent = calculate_capacity_percentage(current_level, max_capacity)
    is_flood_status = is_flood(current_percent, threshold)
    days_to_flood = days_until_threshold(predictions, threshold, max_capacity)
    
    return {
        'is_flood': is_flood_status,
        'status_text': get_status_text(is_flood_status),
        'status_color': get_status_color(is_flood_status),
        'current_level': current_level,
        'current_percent': current_percent,
        'threshold_percent': threshold * 100,
        'days_to_flood': days_to_flood
    }
