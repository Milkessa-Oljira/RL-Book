from fastapi import FastAPI, WebSocket
from Game_Env import TAIrritoryEnv
import uvicorn
import numpy as np

app = FastAPI()
game = TAIrritoryEnv()
game_stats = {
    'human_wins': 0,
    'ai_wins': 0
}

@app.websocket("/play")
async def play_game(websocket: WebSocket):
    await websocket.accept()

    # Send the initial game state upon connection
    obs, _ = game.reset()  # Initialize the board
    response = {
        'board': obs.tolist(),
        'current_player': game.current_player,
        'game_stats': game_stats
    }
    await websocket.send_json(response)

    while True:
        data = await websocket.receive_json()
        
        if data['action'] == 'move':
            action = data['move']
            obs, reward, terminated, truncated, info = game.step(action)
            
            if terminated:
                game_stats['human_wins' if reward == -1 else 'ai_wins'] += 1
            
            response = {
                'board': obs.tolist(),
                'reward': reward,
                'game_over': terminated,
                'current_player': game.current_player,
                'game_stats': game_stats
            }
            
            await websocket.send_json(response)
        
        elif data['action'] == 'reset':
            obs, _ = game.reset()
            response = {
                'board': obs.tolist(),
                'current_player': game.current_player,
                'game_stats': game_stats
            }
            await websocket.send_json(response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)