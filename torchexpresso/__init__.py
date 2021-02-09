"""
PT-EXPRESSO: PyTorch configurable EXPeRimEntS framewOrk
"""
# We need to set the initial import order here for comet_ml to work properly with auto-logging
import comet_ml
import torch

"""
Basic logging configuration for the whole project
"""
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)
