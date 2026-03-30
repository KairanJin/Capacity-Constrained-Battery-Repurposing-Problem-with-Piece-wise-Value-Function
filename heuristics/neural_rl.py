import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import argparse
import os


# Placeholder for the environment (e.g., Gym, CartPole, Custom)
# Users should replace this with their actual environment wrapper
class SimpleEnv:
    def __init__(self):
        # Initialize environment here (e.g., load Gym env, setup CartPole)
        pass

    def step(self, action):
        # Execute action and return (next_state, reward, done)
        # This is a STUB. Replace with actual logic.
        # For testing/demonstration, return dummy data
        return np.zeros((self.state_dim,)), 0.0, True


class NeuralRL:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Initialize placeholder for the environment
        self.env = None
        self.device = None  # 'cpu' or 'cuda' or None

        # Load hyperparameters from config or use defaults
        if config:
            self.gamma = config.get('gamma', 0.99)
            self.buffer_size = config.get('buffer_size', 10000)
        else:
            self.gamma = 0.99
            self.buffer_size = 10000

        # 构建神经网络模型
        self.model = self.build_model()

        # 初始化经验回放缓冲区
        self.buffer = []

        # 初始化训练参数
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Add hardware configuration
        self.use_cuda = False  # Default to CPU

    def set_device(self, device):
        """Set hardware device ('cpu', 'cuda', or None)."""
        self.use_cuda = device

    #     if device == 'cuda':
    #             if not tf.config.list_physical_devices('GPU'):
    #                 raise RuntimeError("No CUDA GPU found.")
    #             self.device = tf.device('/device:GPU')
    #         else:
    #             self.device = None  # CPU

    def build_model(self):
        # Build model dynamically based on config/inputs
        model = Sequential()
        model.add(LSTM(128, input_shape=(1, self.state_dim)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))

        # Select optimizer based on device
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def act(self, state):
        # 执行动作
        # Normalize state if necessary (e.g., clip values for Atari/CartPole)
        # Assuming state is a flat array of 0-255 for pixel inputs
        # Or if state_space is already normalized by env.step
        if np.random.rand() <= self.epsilon:
            # 探索
            return np.random.randint(self.action_dim)
        else:
            # 利用
            # Reshape state to [Batch, 1, State_Dim]
            # Note: next_state used for target calc in replay must be pre-processed here if possible
            # For now, we assume state is compatible
            state_tensor = np.reshape(state, [1, 1, self.state_dim])
            act_values = self.model.predict(state_tensor, verbose=0)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        # 经验回放 (Experience Replay)
        if len(self.buffer) < batch_size:
            return

        # Sample a minibatch
        minibatch = random.sample(self.buffer, batch_size)

        # Prepare data for batch training
        states = []
        targets = []

        for experience in minibatch:
            state, action, reward, next_state, done = experience

            # Normalize next_state if necessary (Crucial for correct target calculation)
            # Assuming next_state from buffer needs same normalization as 'state' in act()
            # But next_state here is raw env output, likely needs normalization
            # We will skip explicit normalization here for the stub,
            # but the user MUST ensure next_state matches model input dims (1, 1, state_dim)

            states.append(state)

            # Q-Learning Target Calculation
            # Calculate target for the WHOLE minibatch first
            # The standard Q-Learning target is: r + gamma * max_a(Q(s', a'))
            # We approximate the max Q-value for next_state using the current model
            # This is a heuristic since we don't have the full Q-table.

            # 1. Get the max Q-value for the NEXT state
            # We need to predict Q-values for the next_state to find max_a
            next_state_tensor = np.reshape(next_state, [1, 1, self.state_dim])
            q_values_next = self.model.predict(next_state_tensor, verbose=0)[0]

            # 2. Calculate target
            # If the transition is NOT done (done=False), target = reward
            # If done (done=True), target = reward + self.gamma * np.max(q_values_next)

            # We create a target vector for the current 'state'
            target_f = self.model.predict(state_tensor, verbose=0)

            # Apply the calculated target value to the 'action' taken
            target_f[0][action] = target

            targets.append(target_f)

        # Perform batch training on the entire collected batch
        if states and targets:
            stacked_states = np.stack(states)
            stacked_targets = np.stack(targets)
            self.model.fit(stacked_states, stacked_targets, epochs=10, verbose=0)

        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def add_to_buffer(self, state, action, reward, next_state, done):
        # 添加经验到缓冲区
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def save_model(self, filepath):
        """Save the trained model weights to a file."""
        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model weights from a file."""
        try:
            self.model.load_weights(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def run(self, env, episodes=1000):
        # 运行算法
        rewards = []

        # Check environment
        if self.env is None:
            print("Warning: env is None. Using SimpleEnv stub.")
            self.env = SimpleEnv()

        for episode in range(episodes):
            state = env.reset()  # Assuming env.reset() returns initial state

            total_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done = env.step(action)

                self.add_to_buffer(state, action, reward, next_state, done)

                # Perform replay every N steps to speed up learning (optional)
                if episode % 5 == 0:
                    self.replay(32)  # Use a larger batch for replay

                state = next_state
                total_reward += reward

            rewards.append(total_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {episode}: Total Reward = {total_reward}, Avg Reward (last 100) = {avg_reward:.2f}")

        return rewards


if __name__ == "__main__":
    # CLI entry point
    parser = argparse.ArgumentParser(description="NeuralRL Training Script")
    parser.add_argument("--state_dim", type=int, default=4,
                        help="Dimension of the state space (e.g., pixel count for Atari)")
    parser.add_argument("--action_dim", type=int, default=2,
                        help="Dimension ofthe action space (number of possible actions)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Max size of experience buffer")
    parser.add_argument("--save_model", type=str, default="model.h5", help="Path to save the trained model")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to load a pre-trained model to continue training")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Hardware device: 'cpu', 'cuda', or None")

    args = parser.parse_args()

    # Initialize model
    config = {
        'gamma': args.gamma,
        'lr': args.lr,
        'buffer_size': args.buffer_size
    }

    model = NeuralRL(state_dim=args.state_dim, action_dim=args.action_dim, learning_rate=args.lr, config=config)

    if args.device:
        model.set_device(args.device)

    # Load model if requested
    if args.load_model:
        model.load_model(args.load_model)

    # Run training
    rewards = model.run(env, episodes=args.episodes)

    print(f"Training complete. Avg Reward: {np.mean(rewards):.2f}")
