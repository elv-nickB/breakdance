import pytest
import os
import pickle as pkl

@pytest.fixture
def project_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def original_dataset(project_dir):
    with open(os.path.join(project_dir, 'datasets', 'original.pkl'), 'rb') as f:
        data = pkl.load(f)
    return data