import torch.nn as nn
from transformers import BertTokenizer, AutoModel


class AlbertClassification(nn.Module):
    def __init__(self):
        super(AlbertClassification, self).__init__()
        self.albert_model = AutoModel.from_pretrained("albert_tiny_pytorch")

    def forward(self, input):
        return self.albert_model(**input)


if __name__ == '__main__':
    albert = AlbertClassification()
    albert.eval()
    tokenizer = BertTokenizer.from_pretrained("albert_tiny_pytorch")
    input = tokenizer("网络休闲游戏", return_tensors="pt")

    print(input)
    print(albert(input))