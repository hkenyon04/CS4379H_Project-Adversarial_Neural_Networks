import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_network(message_length=16, key_length=16):
    """Build Alice, Bob, and Eve networks using modern TensorFlow 2.x."""
    
    # Input layers
    msg_input = layers.Input(shape=(message_length, 1), name='message')
    key_input = layers.Input(shape=(key_length, 1), name='key')
    
    # Alice (Encryption)
    alice_input = layers.Concatenate(axis=1)([msg_input, key_input])
    alice = layers.Flatten()(alice_input)
    alice = layers.Dense(message_length, activation='relu')(alice)
    alice = layers.Dense(message_length, activation='relu')(alice)
    alice_output = layers.Reshape((message_length, 1), name='alice_output')(alice)
    
    alice_model = keras.Model(inputs=[msg_input, key_input], outputs=alice_output, name='Alice')
    
    # Bob (Decryption)
    cipher_input = layers.Input(shape=(message_length, 1), name='ciphertext')
    bob_input = layers.Concatenate(axis=1)([cipher_input, key_input])
    bob = layers.Flatten()(bob_input)
    bob = layers.Dense(message_length, activation='relu')(bob)
    bob = layers.Dense(message_length, activation='tanh')(bob)
    bob_output = layers.Reshape((message_length, 1), name='bob_output')(bob)
    
    bob_model = keras.Model(inputs=[cipher_input, key_input], outputs=bob_output, name='Bob')
    
    # Eve (Adversary - no key)
    eve = layers.Flatten()(cipher_input)
    eve = layers.Dense(message_length, activation='relu')(eve)
    eve = layers.Dense(message_length, activation='tanh')(eve)
    eve_output = layers.Reshape((message_length, 1), name='eve_output')(eve)
    
    eve_model = keras.Model(inputs=cipher_input, outputs=eve_output, name='Eve')
    
    return alice_model, bob_model, eve_model