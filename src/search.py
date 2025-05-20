from src.fasta_data import FASTAData
from src.models import NEARResNet

from typing import Literal
import faiss

def search_against_index(model: NEARResNet, data: FASTAData, index: faiss.Index):
    pass