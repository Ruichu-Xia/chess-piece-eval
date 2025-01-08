import torch
from torch.utils.data import Dataset


class ChessValueDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.piece_to_value = {
            'P': 1,  'N': 3.25, 'B': 3.5,  'R': 5,   'Q': 9,   'K': 20,
            'p': -1, 'n': -3.25, 'b': -3.5, 'r': -5,  'q': -9,  'k': -20
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, eval = self.data[idx]
        input_vector = self.fen_to_vector(fen)
        eval_tensor = torch.tensor(eval, dtype=torch.float32)
        return input_vector, eval_tensor

    def fen_to_vector(self, fen):
        """
        Convert FEN to a flattened vector representation.
        """
        position_part, turn, castling, _ = fen.split(' ')[:4]
        vector = torch.zeros(64 + 5, dtype=torch.float32)
        rank, file = 0, 0

        # Fill the board squares
        for char in position_part:
            if char == '/':
                rank += 1
                file = 0
            elif char.isdigit():
                file += int(char)
            else:
                piece_value = self.piece_to_value[char]
                vector[rank * 8 + file] = piece_value
                file += 1

        # Add global features
        vector[64] = 1 if turn == 'w' else 0  # Turn indicator
        vector[65] = 1 if 'K' in castling else 0  # White king-side
        vector[66] = 1 if 'Q' in castling else 0  # White queen-side
        vector[67] = 1 if 'k' in castling else 0  # Black king-side
        vector[68] = 1 if 'q' in castling else 0  # Black queen-side

        return vector

    def vector_to_fen(self, vector):
        """
        Convert a 69-dimensional vector representation back into a FEN string.
        """
        # Reverse mapping from piece values to FEN characters
        value_to_piece = {v: k for k, v in self.piece_to_value.items()}

        # Initialize the FEN components
        fen_rows = []

        # Convert the first 64 elements (board squares) back to FEN rows
        for rank in range(8):
            fen_row = ''
            empty_count = 0
            for file in range(8):
                index = rank * 8 + file
                value = vector[index].item()
                if value == 0:
                    empty_count += 1  # Count empty squares
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)  # Add empty square count
                        empty_count = 0
                    fen_row += value_to_piece[value]

            if empty_count > 0:
                fen_row += str(empty_count)  # Add trailing empty squares
            fen_rows.append(fen_row)

        position_part = '/'.join(fen_rows)

        # Convert the global features (last 5 elements)
        turn = 'w' if vector[64].item() == 1 else 'b'
        castling = ''
        castling += 'K' if vector[65].item() == 1 else ''
        castling += 'Q' if vector[66].item() == 1 else ''
        castling += 'k' if vector[67].item() == 1 else ''
        castling += 'q' if vector[68].item() == 1 else ''
        if castling == '':
            castling = '-'

        # Default values for en passant, halfmove clock, and fullmove number
        en_passant = '-'  # Not represented in this model
        halfmove_clock = '0'
        fullmove_number = '1'

        # Combine all components to form the FEN string
        fen = (
            f"{position_part} {turn} {castling} "
            f"{en_passant} {halfmove_clock} {fullmove_number}"
            )
        return fen
