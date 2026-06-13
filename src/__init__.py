"""crypto-image-cnn: 市場のミクロ構造を画像として描画し CNN で学習することで、
短期の暗号資産リターンを予測する。

公開APIは意図的に小さく保っている — notebook は以下を通じてすべてを駆動する:

    from src.config import Config
    from src import pipeline, train, images
"""
from .config import Config

__all__ = ["Config"]
