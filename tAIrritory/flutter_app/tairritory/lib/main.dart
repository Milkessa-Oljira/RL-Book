import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'game_provider.dart';
import 'game_board.dart';

void main() {
  runApp(
    ChangeNotifierProvider(
      create: (_) => GameProvider(),
      child: const MaterialApp(home: TAIrritoryApp()),
    ),
  );
}

class TAIrritoryApp extends StatelessWidget {
  const TAIrritoryApp({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("tAIrritory"),
        backgroundColor: Colors.blueGrey,
      ),
      body: Column(
        children: [
          // Scoreboard
          const ScoreBoard(),
          // Game Board
          Expanded(child: GameBoard()),
        ],
      ),
    );
  }
}

class ScoreBoard extends StatelessWidget {
  const ScoreBoard({super.key});

  @override
  Widget build(BuildContext context) {
    final gameProvider = context.watch<GameProvider>();
    return Container(
      color: Colors.grey[200],
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              const Icon(Icons.person, color: Colors.red, size: 24),
              const SizedBox(width: 8),
              Text("Humans: ${gameProvider.humanWins}",
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            ],
          ),
          Row(
            children: [
              Text("AI: ${gameProvider.aiWins}",
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              const SizedBox(width: 8),
              const Icon(Icons.computer, color: Colors.blue, size: 24),
            ],
          ),
        ],
      ),
    );
  }
}
