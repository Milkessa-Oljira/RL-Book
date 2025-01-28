import 'package:flutter/material.dart';

class GameProvider extends ChangeNotifier {
  final int boardSize = 7;
  final int boardWidth = 3;

  List<List<int>> board = [];
  int currentPlayer = 1; // 1 for Human, -1 for AI
  int humanWins = 0;
  int aiWins = 0;
  bool isGameOver = false;

  GameProvider() {
    resetGame();
  }

  void resetGame() {
    board = List.generate(
      boardSize,
      (_) => List.generate(boardWidth, (_) => 0),
    );
    currentPlayer = 1;
    isGameOver = false;
    notifyListeners();
  }

  void makeMove(int row, int col, int newRow, int newCol) {
    if (isGameOver) return;

    board[newRow][newCol] = board[row][col];
    board[row][col] = 0;

    // Check for win condition
    if (_checkWin()) {
      isGameOver = true;
      if (currentPlayer == 1) {
        humanWins++;
      } else {
        aiWins++;
      }
    }

    // Switch turns
    currentPlayer = -currentPlayer;
    notifyListeners();
  }

  List<List<int>> getPossibleMoves(int row, int col) {
    // Add logic for valid moves
    return [
      [row + 1, col],
      [row + 1, col + 1],
    ];
  }

  bool _checkWin() {
    // Logic to check if the current player wins
    return false; // Replace with actual logic
  }
}
