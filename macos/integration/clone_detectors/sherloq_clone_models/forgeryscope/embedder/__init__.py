# Local SHERLOQ adaptation; original: third_party/research/clone_detectors/02_forgeryscope/forgeryscope/embedder/__init__.py
# Source SHA256: f199e5f6ce0714d075ead885b3ca3780349928c19babb72322218fa9ae9c667f
from sherloq_clone_models.forgeryscope.embedder.torch import Embedder
from sherloq_clone_models.forgeryscope.embedder.torch import TorchImageDataset

__all__ = ['Embedder', 'TorchImageDataset']