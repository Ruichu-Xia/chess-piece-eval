import chess
import chess.svg
from IPython.display import display, SVG
from torch.utils.data import random_split

from config import data_settings


def _svg_url(fen):
    fen_board = fen.split()[0]
    return data_settings.svg_base_url + fen_board


def show_board_from_fen(fen):
    board = chess.Board(fen)
    svg_data = chess.svg.board(board)
    display(SVG(data=svg_data))
    # display(SVG(url=_svg_url(fen)))


def split_dataset(dataset, train_size, val_size):
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )
    return train_dataset, val_dataset, test_dataset
