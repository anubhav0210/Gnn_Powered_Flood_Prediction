"""
Map generation for flood warning visualization
"""

import folium
from folium import plugins
import streamlit as st
import config


def create_base_map(center=None, zoom_start=8):
    """
    Create base Folium map centered on Godavari Basin.
    
    Args:
        center (list): Center coordinates [lat, lon], default is Godavari Basin
        zoom_start (int): Initial zoom level
    
    Returns:
        folium.Map: Base map object
    """
    if center is None:
        center = config.MAP_CENTER
    
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    return m


def add_lake_markers(map_obj, lake_data, flood_status_dict):
    """
    Add lake markers to map with color coding based on flood status.
    
    Args:
        map_obj (folium.Map): Folium map object
        lake_data (dict): Lake information {lake_name: {current_level, capacity, predictions}}
        flood_status_dict (dict): Flood status for each lake {lake_name: status_dict}
    
    Returns:
        folium.Map: Map with markers added
    """
    for lake_name, coordinates in config.LAKE_COORDINATES.items():
        if lake_name not in flood_status_dict:
            continue
        
        status = flood_status_dict[lake_name]
        color = status['status_color']
        status_text = status['status_text']
        current_percent = status['current_percent']
        threshold_percent = status['threshold_percent']
        days_to_flood = status['days_to_flood']
        
        # Build popup text
        popup_text = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin: 5px 0;">{lake_name}</h4>
            <hr style="margin: 5px 0;">
            <b>Status:</b> {status_text}<br>
            <b>Current Level:</b> {current_percent:.1f}%<br>
            <b>Flood Threshold:</b> {threshold_percent:.1f}%<br>
        """
        
        if days_to_flood is not None and days_to_flood > 0:
            popup_text += f"<b>Days to Flood:</b> {days_to_flood} days<br>"
        else:
            popup_text += f"<b>Days to Flood:</b> Safe (no flood in 7 days)<br>"
        
        popup_text += "</div>"
        
        # Add marker
        folium.CircleMarker(
            location=coordinates,
            radius=20,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            weight=3,
            tooltip=f"{lake_name}: {status_text}"
        ).add_to(map_obj)
    
    return map_obj


def add_legend(map_obj):
    """
    Add simple legend to map showing Flood/Safe colors.
    
    Args:
        map_obj (folium.Map): Folium map object
    
    Returns:
        folium.Map: Map with legend added
    """
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px;
                border-radius: 5px;">
        <p style="margin: 5px 0; font-weight: bold;">Flood Warning Legend</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 5px 0;">
            <i style="background:#d62728; width: 15px; height: 15px; 
                     display: inline-block; border-radius: 50%;"></i> 
            FLOOD (≥ Threshold)
        </p>
        <p style="margin: 5px 0;">
            <i style="background:#2ca02c; width: 15px; height: 15px; 
                     display: inline-block; border-radius: 50%;"></i> 
            SAFE (< Threshold)
        </p>
    </div>
    '''
    map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    return map_obj


def create_flood_warning_map(lake_data, flood_status_dict):
    """
    Create complete flood warning map with all components.
    
    Args:
        lake_data (dict): Lake information
        flood_status_dict (dict): Flood status for each lake
    
    Returns:
        folium.Map: Complete map object
    """
    # Create base map
    m = create_base_map()
    
    # Add markers
    m = add_lake_markers(m, lake_data, flood_status_dict)
    
    # Add legend
    m = add_legend(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    return m
