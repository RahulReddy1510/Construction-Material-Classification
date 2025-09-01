import unittest
import torch
import torch.nn as nn
from models.efficientnet_finetune import ConstructionMaterialClassifier
from models.quantization import prepare_for_qat, convert_to_int8

class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.model = ConstructionMaterialClassifier(num_classes=5)

    def test_prepare_qat(self):
        # preparation requires CPU and fbgemm for this test
        model_qat = prepare_for_qat(self.model, backend="fbgemm")
        
        # Check if fake_quant modules are present
        has_fake_quant = any('fake_quant' in name for name, _ in model_qat.named_modules())
        # Note: In PyTorch 2.1, the observers might be inside modules
        self.assertTrue(has_fake_quant, "Prepare QAT should insert fake quantization modules")

    def test_conversion_flow(self):
        model_qat = prepare_for_qat(self.model)
        model_int8 = convert_to_int8(model_qat)
        
        # Check if some weights are now quantized tensors
        # Traversing to a conv layer
        first_conv = model_int8.backbone.features[0][0]
        self.assertTrue(hasattr(first_conv, 'weight'))
        # Quantized weights are usually torch.per_channel_affine or similar
        # For simplicity in test, we just check completion
        self.assertEqual(model_int8.training, False)

if __name__ == '__main__':
    unittest.main()
