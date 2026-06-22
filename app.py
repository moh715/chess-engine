# app.py
from flask import Flask, request, jsonify
import chess
import chess.engine
from player import Searcher , evaluate

app = Flask(__name__)
# Replace with your Stockfish path
board = chess.Board()
engine = Searcher(board, evaluate)

@app.route('/move', methods=['POST'])
def make_move():
    data = request.get_json()
    move_uci = data.get('move')
    # Create move object and push to board if legal
    move = chess.Move.from_uci(move_uci)
    if move in board.legal_moves:
        board.push(move)
        # Let engine think for 0.5 seconds then get best move
        result = engine.play()
        board.push(result.move)
    return jsonify({"fen": board.fen()})