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

#configure logging
logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler('earthquake_data_gather.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
