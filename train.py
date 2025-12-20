"""Training script for baseball state prediction model"""
from baseball_states.training import ModelConfig, train_model


if __name__ == "__main__":
    config = ModelConfig()
    model, tokenizer = train_model(config)