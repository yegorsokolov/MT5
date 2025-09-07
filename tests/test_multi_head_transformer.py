import torch
from models.multi_head import MultiHeadTransformer


def test_dropout_training_vs_eval():
    torch.manual_seed(0)
    model = MultiHeadTransformer(3, num_symbols=1, dropout=0.5)
    x = torch.randn(8, 5, 3)
    model.train()
    out1 = model(x, 0)
    out2 = model(x, 0)
    assert not torch.allclose(out1, out2)
    model.eval()
    out3 = model(x, 0)
    out4 = model(x, 0)
    assert torch.allclose(out3, out4)


def test_output_shape_with_layer_norm():
    model = MultiHeadTransformer(
        4, num_symbols=2, layer_norm=True, ff_dim=32, dropout=0.0
    )
    x = torch.randn(7, 10, 4)
    out = model(x, 1)
    assert out.shape == (7,)
