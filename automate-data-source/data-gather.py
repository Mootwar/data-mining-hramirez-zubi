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