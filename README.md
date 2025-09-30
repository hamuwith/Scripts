# パズルゲームのCPU戦の強化学習

## 📖 ゲーム概要
中学生で習う化学式をそろえると消えるぷよぷよ風ゲーム。

## 📁 含まれるファイル
Pythonファイル(強化学習)の部分のみ記載
```
Python/
└── ChemicalChainEnv.py → パズルゲームの環境とStable-Baselines3のDQNを用いた強化学習。
├── ChemicalChainEnvOnehot.py → 入力データの改良
└── Rainbow.py → 強化学習のRainbowの実装
```

## 🛠 環境
- 言語: Python 3.10.16
- ゲームエンジン: Unity6000.0
- ライブラリ / フレームワーク:
  - Stable-Baselines3
  - Gymnasium
  - NumPy
  - PyTorch
 
## 💻 実装
- ChemicalChainEnv.py
  - ゲームの環境とDQNを実装しました。
  - 学習時はTensorBoardを利用し、可視化しながら実行を進めました。
  - 元素にそれぞれ番号を振っていたため、それっぽくは動くものの正しく学習することができませんでした。
    
- ChemicalChainEnvOnehot.py
  - 環境を改善し、元素それぞれにフィールドを持ち、ビットで表現できるようにしました。
    
- Rainbow.py
  - 改良した環境を使用し、Rainbowを実装しました。
  - 入力データが疎なため、うまく学習が進みませんでした。

- CPUManager.cs
  - 学習したモデルからONNXファイルを作成し、Unityに取り込みました。

## 🌱 改善策
- 入力データをベクトルに埋め込み、密なベクトルで元素を表現する。
- 最新の強化学習について調べ、適した手法を採用し実装していく。
