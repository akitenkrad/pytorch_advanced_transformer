from pathlib import Path
import string
import re
import random
import warnings
import torchtext
from torchtext.vocab import Vectors

warnings.filterwarnings('ignore')

def preprocessing_text(text):
    text = re.sub('<br />', '', text)
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    return text

def tokenizer_punctuation(text):
    return text.strip().split()

def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret

def get_dataloaders(data_path, max_length=256, batch_size=24):
    # torchtextでコーパスをロードするための設定
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True,
                                batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    # データをロード
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(path=data_path, train='IMDb_train.tsv', test='IMDb_test.tsv',
                                                                 format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])

    # 学習データと検証データを分離
    train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))

    # 学習済みベクトルをロード
    english_fasttext_vectors = torchtext.vocab.FastText()
    
    # Vocabularyを構築
    TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=10)

    # DataLoaderを作成
    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)
    
    return train_dl, val_dl, test_dl, TEXT

def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)


def mk_html(index, batch, preds, normlized_weights_1, normlized_weights_2, TEXT):
    # indexの結果を抽出
    sentence = batch.Text[0][index]  # 文章
    label = batch.Label[index]  # ラベル
    pred = preds[index]  # 予測

    # indexのAttentionを抽出と規格化
    attens1 = normlized_weights_1[index, 0, :]  # 0番目の<cls>のAttention
    attens1 /= attens1.max()

    attens2 = normlized_weights_2[index, 0, :]  # 0番目の<cls>のAttention
    attens2 /= attens2.max()

    # ラベルと予測結果を文字に置き換え
    if label == 0:
        label_str = "Negative"
    else:
        label_str = "Positive"

    if pred == 0:
        pred_str = "Negative"
    else:
        pred_str = "Positive"

    # 表示用のHTMLを作成する
    html = '正解ラベル：{}<br>推論ラベル：{}<br><br>'.format(label_str, pred_str)

    # 1段目のAttention
    html += '[TransformerBlockの1段目のAttentionを可視化]<br>'
    for word, attn in zip(sentence, attens1):
        html += highlight(TEXT.vocab.itos[word], attn)
    html += "<br><br>"

    # 2段目のAttention
    html += '[TransformerBlockの2段目のAttentionを可視化]<br>'
    for word, attn in zip(sentence, attens2):
        html += highlight(TEXT.vocab.itos[word], attn)

    html += "<br><br>"

    return html
