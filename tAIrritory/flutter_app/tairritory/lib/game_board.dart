import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'game_provider.dart';

class GameBoard extends StatelessWidget {
  const GameBoard({super.key});

  @override
  Widget build(BuildContext context) {
    final gameProvider = context.watch<GameProvider>();
    final board = gameProvider.board;

    return GridView.builder(
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3,
        childAspectRatio: 1.0,
      ),
      itemCount: board.length * board[0].length,
      itemBuilder: (context, index) {
        final row = index ~/ 3;
        final col = index % 3;
        final pieceValue = board[row][col];

        return GestureDetector(
          onTap: () => _handleTap(context, row, col),
          child: Container(
            margin: const EdgeInsets.all(4.0),
            decoration: BoxDecoration(
              color: _getTileColor(pieceValue),
              shape: BoxShape.rectangle,
              border: Border.all(color: Colors.black),
            ),
            child: Center(child: _buildPiece(pieceValue)),
          ),
        );
      },
    );
  }

  Widget _buildPiece(int value) {
    if (value == 0) return Container();
    final color = value > 0 ? Colors.red : Colors.blue;
    return CircleAvatar(backgroundColor: color);
  }

  void _handleTap(BuildContext context, int row, int col) {
    final gameProvider = context.read<GameProvider>();
    final possibleMoves = gameProvider.getPossibleMoves(row, col);

    if (possibleMoves.isEmpty) return;

    // Show possible moves or take action
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Select a Move'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: possibleMoves.map((move) {
              return ListTile(
                title: Text('Move to (${move[0]}, ${move[1]})'),
                onTap: () {
                  Navigator.pop(context);
                  gameProvider.makeMove(row, col, move[0], move[1]);
                },
              );
            }).toList(),
          ),
        );
      },
    );
  }

  Color _getTileColor(int pieceValue) {
    if (pieceValue > 0) return Colors.red.withOpacity(0.5);
    if (pieceValue < 0) return Colors.blue.withOpacity(0.5);
    return Colors.grey[300]!;
  }
}
