"""
USGS Earthquake Data Gatherer

This script fetches earthquake data from the USGS API and saves it to a file.
Documentation: https://earthquake.usgs.gov/fdsnws/event/1/
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# USGS API base URL
USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def fetch_earthquake_data_from_location(start_time, end_time, min_magnitude=4.5, limit=500, 
                          latitude=None, longitude=None, radius=None):
    """
    Fetch earthquake data from USGS API
    
    Args:
        start_time (str): Start time in ISO format (YYYY-MM-DD)
        end_time (str): End time in ISO format (YYYY-MM-DD)
        min_magnitude (float): Minimum earthquake magnitude
        limit (int): Maximum number of results
        latitude (float): Center latitude for search area
        longitude (float): Center longitude for search area
        radius (float): Search radius in kilometers
        
    Returns:
        dict: JSON response from USGS API
    """
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minmagnitude": min_magnitude,
        "limit": limit,
        "orderby": "time"
    }
    
    # Add location parameters if provided
    if latitude is not None and longitude is not None and radius is not None:
        params["latitude"] = latitude
        params["longitude"] = longitude
        params["maxradiuskm"] = radius
        logger.info(f"Filtering by location: ({latitude}, {longitude}) with radius {radius} km")
    
    logger.info(f"Fetching earthquake data from {start_time} to {end_time} with magnitude >= {min_magnitude}")
    
    try:
        response = requests.get(USGS_API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return None


def fetch_earthquake_data(start_time, end_time, min_magnitude=4.5, limit=500):
    """
    Fetch earthquake data from USGS API
    
    Args:
        start_time (str): Start time in ISO format (YYYY-MM-DD)
        end_time (str): End time in ISO format (YYYY-MM-DD)
        min_magnitude (float): Minimum earthquake magnitude
        limit (int): Maximum number of results
        
    Returns:
        dict: JSON response from USGS API
    """
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minmagnitude": min_magnitude,
        "limit": limit,
        "orderby": "time"
    }
    
    logger.info(f"Fetching earthquake data from {start_time} to {end_time} with magnitude >= {min_magnitude}")
    
    try:
        response = requests.get(USGS_API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return None


def save_to_json(data, output_file):
    """Save data to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved raw JSON data to {output_file}")


def convert_to_csv(data, output_file):
    """Convert GeoJSON to CSV and save to file"""
    if not data or 'features' not in data:
        logger.error("Invalid data format")
        return
    
    records = []
    for feature in data['features']:
        props = feature['properties']
        geo = feature['geometry']
        
        record = {
            'id': feature['id'],
            'time': datetime.fromtimestamp(props['time']/1000).isoformat(),
            'updated': datetime.fromtimestamp(props['updated']/1000).isoformat() if 'updated' in props else None,
            'magnitude': props['mag'],
            'type': props['type'],
            'place': props['place'],
            'longitude': geo['coordinates'][0] if geo and 'coordinates' in geo else None,
            'latitude': geo['coordinates'][1] if geo and 'coordinates' in geo else None,
            'depth': geo['coordinates'][2] if geo and 'coordinates' in geo else None,
            'felt': props.get('felt'),
            'cdi': props.get('cdi'),
            'mmi': props.get('mmi'),
            'alert': props.get('alert'),
            'status': props.get('status'),
            'tsunami': props.get('tsunami'),
            'sig': props.get('sig'),
            'url': props.get('url'),
            'detail': props.get('detail')
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    logger.info(f"Converted data to CSV and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Fetch earthquake data from USGS API")
    parser.add_argument("-s", "--start", help="Start date (YYYY-MM-DD), defaults to 30 days ago")
    parser.add_argument("-e", "--end", help="End date (YYYY-MM-DD), defaults to today")
    parser.add_argument("-m", "--magnitude", type=float, default=4.5, help="Minimum magnitude (default: 4.5)")
    parser.add_argument("-l", "--limit", type=int, default=500, help="Maximum number of results (default: 500)")
    parser.add_argument("-o", "--output-dir", default="data", help="Output directory (default: 'data')")
    #location-based search parameters
    parser.add_argument("--lat", type=float, help="Latitude for center of search area")
    parser.add_argument("--lon", type=float, help="Longitude for center of search area")
    parser.add_argument("--radius", type=float, help="Search radius in kilometers")
    args = parser.parse_args()

    # Set default dates if not provided
    end_time = args.end or datetime.now().strftime("%Y-%m-%d")
    start_time = args.start or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_file = os.path.join(args.output_dir, f"earthquakes_{timestamp}.csv")
    
    # Fetch and save data
    earthquake_data = fetch_earthquake_data(
        start_time=start_time,
        end_time=end_time,
        min_magnitude=args.magnitude,
        limit=args.limit
    )
    
    if earthquake_data:
        count = len(earthquake_data['features']) if 'features' in earthquake_data else 0
        logger.info(f"Found {count} earthquakes")
        
        convert_to_csv(earthquake_data, csv_file)
    else:
        logger.error("No data retrieved")


if __name__ == "__main__":
    main()
