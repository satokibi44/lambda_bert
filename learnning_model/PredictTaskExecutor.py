from learnning_model.Transformer import Transformer
import torch
from Bert import Bert

from transformers.tokenization_bert_japanese import BertJapaneseTokenizer

import pickle

MAX_SEQ_LEN = 75
class PredictTaskExecutor:

    def predict_task_executor(self, net, sentence):
        net.eval()
        tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sentence = tokenizer.encode(sentence, return_tensors='pt').to(device)
        tokens = torch.zeros(MAX_SEQ_LEN, dtype=torch.long).to(device)
        net = net.to(device)
        for i in range(len(text[0])):
            tokens[i] = round(text[0][i].item())

        with torch.set_grad_enabled(False):
            outputs = net(tokens.unsqueeze(0))
            print(outputs)
            _, preds = torch.max(outputs, 1)  # ラベルを予測
            pred2 = torch.softmax(outputs, 1)

        return pred2

    def main(self, sentence):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Bert()

        model_path = "/content/drive/MyDrive/Colab Notebooks/KusoripuMetrix2/model/best_epoche23"

        model.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))

        return self.predict_task_executor(model, sentence)
