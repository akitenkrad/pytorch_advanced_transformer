from pathlib import Path
import click
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext

from utils import get_dataloaders, mk_html
from models import TransformerClassification

# ネットワークの初期化を定義
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner層の初期化
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, save_dir):

    # GPUが使えるかを確認
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    print('-----start-------')
    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # データローダーからミニバッチを取り出すループ
            for batch in (dataloaders_dict[phase]):
                # batchはTextとLableの辞書オブジェクト

                # GPUが使えるならGPUにデータを送る
                inputs = batch.Text[0].to(device)  # 文章
                labels = batch.Label.to(device)  # ラベル

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    # mask作成
                    input_pad = 1  # 単語のIDにおいて、'<pad>': 1 なので
                    input_mask = (inputs != input_pad)

                    # Transformerに入力
                    outputs, _, _ = net(inputs, input_mask)
                    loss = criterion(outputs, labels)  # 損失を計算

                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(save_dir / 'weights.pth'))
    return net

@click.group()
def cli():
    pass

@cli.command()
@click.option('--data', type=click.Path(exists=True), help='path to dataset directory')
@click.option('--epochs', type=int, default=10, help='number of epochs')
@click.option('--batch-size', type=int, default=32, help='batch size')
@click.option('--learning-rate', type=float, default=2e-5, help='learning rate')
@click.option('--save-dir', type=click.Path(), help='directory to save model')
def train(data, epochs=10, batch_size=32, learning_rate=2e-5, save_dir='./weights'):
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # prepare data
    train_dl, val_dl, test_dl, TEXT = get_dataloaders(data_path=data, max_length=256, batch_size=64)
    dataloaders_dict = {"train": train_dl, "val": val_dl}

    # configure transformer model
    net = TransformerClassification(text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=256, output_dim=2)
    net.train()

    # initialize transformer blocks
    net.net3_1.apply(weights_init)
    net.net3_2.apply(weights_init)

    print('Finished model configuration.')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    net_trained = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=epochs, save_dir=save_dir)

@cli.command()
@click.option('--data', type=click.Path(exists=True), help='path to dataset directory')
@click.option('--weights', type=click.Path(exists=True), help='path to weights')
@click.option('--html-dir', type=click.Path(), help='output html dir')
def test(data, weights, html_dir):
    html_dir = Path(html_dir)
    html_dir.mkdir(parents=True, exist_ok=True)
    
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # prepare data
    train_dl, val_dl, test_dl, TEXT = get_dataloaders(data_path=data, max_length=256, batch_size=64)
    dataloaders_dict = {"train": train_dl, "val": val_dl}
    
    # configure model
    net_trained = TransformerClassification(text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=256, output_dim=2)
    net_trained.load_state_dict(torch.load(weights))
    
    net_trained.eval()
    net_trained.to(device)

    epoch_corrects = 0

    for batch_idx, batch in tqdm(enumerate((test_dl)), total=len(test_dl)):
        inputs = batch.Text[0].to(device)
        labels = batch.Label.to(device)

        with torch.set_grad_enabled(False):

            input_pad = 1
            input_mask = (inputs != input_pad)

            outputs, normalized_weights_1, normalized_weights_2 = net_trained(inputs, input_mask)
            _, preds = torch.max(outputs, 1)

            epoch_corrects += torch.sum(preds == labels.data)
            
            index = random.randint(0, len(batch.Text[0])-1)
            html_output = mk_html(index, batch, preds, normalized_weights_1, normalized_weights_2, TEXT)
            with open(str(html_dir / '{}-{}.html'.format(batch_idx, index)), 'w', encoding='utf-8') as f:
                f.write(html_output)

    epoch_acc = epoch_corrects.double() / len(test_dl.dataset)

    print('Data Size:{}   Accuracy: {:.4f}'.format(len(test_dl.dataset),epoch_acc))

    # save attention html
    batch = next(iter(test_dl))
    
    
if __name__ == '__main__':
    cli()
    