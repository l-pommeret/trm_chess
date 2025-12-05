import os
import json
import numpy as np
import chess
import chess.pgn
from tqdm import tqdm
from argdantic import ArgParser
from pydantic import BaseModel
from typing import Optional

from common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    pgn_file: str = "lichess_db_standard_rated_2013-09.pgn"
    output_dir: str = "data/chess-gen"
    max_games: Optional[int] = 10000
    test_ratio: float = 0.1

def board_to_array(board: chess.Board) -> np.ndarray:
    # 8x8 board
    # 0: Empty
    # 1-6: White P, N, B, R, Q, K
    # 7-12: Black P, N, B, R, Q, K
    arr = np.zeros((8, 8), dtype=np.uint8)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            val = piece.piece_type
            if not piece.color: # Black
                val += 6
            # chess.Board uses rank-major (0 is a1, 7 is h1, 8 is a2...)
            # We want 8x8. Let's map 0->(7,0), 63->(0,7) to match visual board?
            # Or just simple reshaping.
            # i // 8 is rank (0-7), i % 8 is file (0-7).
            # Let's keep it simple: rank 0 is bottom.
            row = i // 8
            col = i % 8
            arr[row, col] = val
    return arr

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    print(f"Processing {config.pgn_file}...")
    
    inputs = []
    labels = []
    
    with open(config.pgn_file) as pgn:
        games_processed = 0
        pbar = tqdm(total=config.max_games)
        
        while True:
            if config.max_games and games_processed >= config.max_games:
                break
                
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
                
            board = game.board()
            
            # Iterate through moves
            for move in game.mainline_moves():
                # Input: Current board
                inp = board_to_array(board)
                
                # Make move
                board.push(move)
                
                # Label: Next board
                lbl = board_to_array(board)
                
                inputs.append(inp)
                labels.append(lbl)
            
            games_processed += 1
            pbar.update(1)
            
        pbar.close()

    print(f"Total examples: {len(inputs)}")
    
    # Split train/test
    total_samples = len(inputs)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    test_size = int(total_samples * config.test_ratio)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]
    
    def save_subset(subset_name, subset_indices):
        subset_inputs = [inputs[i] for i in subset_indices]
        subset_labels = [labels[i] for i in subset_indices]
        
        # Prepare data structure for TRM
        # It expects: inputs, labels, puzzle_indices, group_indices, puzzle_identifiers
        
        results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
        
        puzzle_id = 0
        example_id = 0
        
        results["puzzle_indices"].append(0)
        results["group_indices"].append(0)
        
        for inp, lbl in zip(subset_inputs, subset_labels):
            results["inputs"].append(inp)
            results["labels"].append(lbl)
            
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0) # Dummy identifier
            
            # Each example is its own group for now (no augmentation grouping)
            results["group_indices"].append(puzzle_id)
            
        # To Numpy
        def _seq_to_numpy(seq):
            if not seq:
                return np.array([])
            arr = np.stack(seq)
            return arr # No +1 needed if we used 0-12 and 0 is PAD? 
            # Wait, TRM usually reserves 0 for PAD.
            # My board_to_array uses 0 for empty.
            # If PAD is 0, then Empty square is same as PAD?
            # In Sudoku, 0 is PAD, and digits are 1-9.
            # So I should probably shift my values by 1?
            # 0: PAD
            # 1: Empty
            # 2-13: Pieces
            
        # Let's shift everything by 1 to avoid conflict with PAD=0
        results["inputs"] = np.stack(results["inputs"]) + 1
        results["labels"] = np.stack(results["labels"]) + 1
        
        results["group_indices"] = np.array(results["group_indices"], dtype=np.int32)
        results["puzzle_indices"] = np.array(results["puzzle_indices"], dtype=np.int32)
        results["puzzle_identifiers"] = np.array(results["puzzle_identifiers"], dtype=np.int32)
        
        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=64, # 8x8
            vocab_size=14, # 0(PAD) + 1(Empty) + 12(Pieces) = 14
            pad_id=0,
            ignore_label_id=0, # Ignore PAD? Or maybe we want to predict Empty?
            # If we want to predict Empty, we should not ignore it.
            # But PAD is usually ignored in loss.
            # Let's set ignore_label_id to something else if we don't want to ignore 0?
            # Actually, if 0 is PAD, and we never have 0 in our data (because we shifted +1), then it's fine.
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=len(results["group_indices"]) - 1,
            mean_puzzle_examples=1,
            total_puzzles=len(results["group_indices"]) - 1,
            sets=["all"]
        )
        
        # Save
        save_dir = os.path.join(config.output_dir, subset_name)
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
            
        for k, v in results.items():
            np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
            
    save_subset("train", train_indices)
    save_subset("test", test_indices)
    
    # Identifiers
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

if __name__ == "__main__":
    cli()
