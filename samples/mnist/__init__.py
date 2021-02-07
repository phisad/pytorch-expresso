"""
Basic logging configuration for the whole project
"""
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)

