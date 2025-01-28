import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'game_models.dart';

class GameService {
  final WebSocketChannel channel;

  GameService(this.channel);

  void sendMove(List<int> move) {
    channel.sink.add(jsonEncode({
      'action': 'move',
      'move': move
    }));
  }

  void resetGame() {
    channel.sink.add(jsonEncode({
      'action': 'reset'
    }));
  }

  Stream<GameState> get gameUpdates {
    return channel.stream.map((message) {
      final data = jsonDecode(message);
      return GameState(
        board: (data['board'] as List<dynamic>).map((row) => (row as List<dynamic>).map((cell) => cell as int).toList()).toList(),
        currentPlayer: data['current_player'] as int,
        gameOver: data['game_over'] as bool? ?? false,
        gameStats: Map<String, int>.from(data['game_stats'] ?? {}),
      );
    });
  }
}
