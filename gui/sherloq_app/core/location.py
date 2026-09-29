"""Exact ExifTool coordinate values and the historical map URL."""
import json
import math


def parse_location(data):
    documents = json.loads(data.decode('utf-8'))
    if not isinstance(documents, list) or len(documents) != 1 or not isinstance(documents[0], dict):
        raise ValueError('Invalid location response from ExifTool.')
    metadata = documents[0]
    if 'Composite:GPSLatitude' not in metadata or 'Composite:GPSLongitude' not in metadata:
        return None
    lat, lon = metadata['Composite:GPSLatitude'], metadata['Composite:GPSLongitude']
    for value, bound in ((lat, 90), (lon, 180)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not -bound <= value <= bound or not math.isfinite(value)):
            raise ValueError('Invalid GPS coordinates in image metadata.')
    url = (f'https://www.google.com/maps/place/{lat},{lon}/@{lat},{lon},17z/'
           f'data=!4m5!3m4!1s0x0:0x0!8m2!3d{lat}!4d{lon}')
    return lat, lon, url
