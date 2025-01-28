import 'package:flutter/foundation.dart';

@immutable
class GameState {
  final List<List<int>> board;
  final int currentPlayer;
  final bool gameOver;
  final Map<String, int> gameStats;

  const GameState({
    required this.board, 
    required this.currentPlayer,
    this.gameOver = false,
    required this.gameStats
  });

  GameState copyWith({
    List<List<int>>? board,
    int? currentPlayer,
    bool? gameOver,
    Map<String, int>? gameStats,
  }) {
    return GameState(
      board: board ?? this.board,
      currentPlayer: currentPlayer ?? this.currentPlayer,
      gameOver: gameOver ?? this.gameOver,
      gameStats: gameStats ?? this.gameStats,
    );
  }
}