"""
conftest.py – shared pytest fixtures and patches.

Patches ConstructionMaterialClassifier to use pretrained=False by default
so that tests never block on network downloads of ImageNet weights.
"""
import pytest
import unittest.mock as mock

# ---------------------------------------------------------------------------
# Autouse patch: force pretrained=False for ALL tests that instantiate the
# classifier.  This keeps tests fast and completely offline.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_pretrained_download(monkeypatch):
    """Replace ConstructionMaterialClassifier.__init__ so that pretrained
    is always forced to False, preventing network weight downloads."""
    import models.efficientnet_finetune as m

    original_init = m.ConstructionMaterialClassifier.__init__

    def patched_init(self, num_classes=5, pretrained=True, dropout=0.2):
        # Force pretrained=False to avoid downloading ImageNet weights
        original_init(self, num_classes=num_classes, pretrained=False, dropout=dropout)

    monkeypatch.setattr(m.ConstructionMaterialClassifier, "__init__", patched_init)
