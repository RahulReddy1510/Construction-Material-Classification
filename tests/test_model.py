import unittest
import torch
from models.efficientnet_finetune import ConstructionMaterialClassifier

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = ConstructionMaterialClassifier(num_classes=5)
        self.dummy_input = torch.randn(1, 3, 224, 224)

    def test_forward_pass(self):
        output = self.model(self.dummy_input)
        self.assertEqual(output.shape, (1, 5))

    def test_unfreeze_logic(self):
        # Freeze all
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.freeze_backbone(unfreeze_last_n_blocks=1)
        # Check if classifier is trainable
        for param in self.model.backbone.classifier.parameters():
            self.assertTrue(param.requires_grad)
            
        # Check if last block is trainable
        last_block = list(self.model.backbone.features.children())[-1]
        for param in last_block.parameters():
            self.assertTrue(param.requires_grad)

    def test_model_size(self):
        size = self.model.get_model_size_mb()
        # EfficientNet-B0 FP32 should be around 15-25MB depending on state dict details
        self.assertTrue(15 < size < 30)

if __name__ == '__main__':
    unittest.main()
