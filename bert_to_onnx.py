import torch
from transformers import AutoModel

if __name__ == '__main__':
    Batch_size = 1
    seg_length = 8
    albert_model = AutoModel.from_pretrained("albert_tiny_pytorch")
    albert_model.eval()

    dummy_input0 = torch.zeros(Batch_size, seg_length).long()
    dummy_input1 = torch.zeros(Batch_size, seg_length).long()
    dummy_input2 = torch.zeros(Batch_size, seg_length).long()

    torch.onnx.export(albert_model,
                      (dummy_input0, dummy_input1, dummy_input2),
                      "output.onnx",
                      input_names=["input_ids", "token_type_ids", "attention_mask"],

                      opset_version=12)
