from transformers import BertTokenizer
from ONNXModel import ONNXModel

if __name__ == '__main__':
    rnet1 = ONNXModel("output.onnx")
    tokenizer = BertTokenizer.from_pretrained("albert_tiny_pytorch")
    input = tokenizer("网络休闲游戏", return_tensors="pt")
    print(input)
    r1, r2 = rnet1.forward((input["input_ids"].numpy(), input["token_type_ids"].numpy(), input["attention_mask"].numpy()))
    print(r1.shape, r2.shape)