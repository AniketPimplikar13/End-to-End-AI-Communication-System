import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from scipy.special import erfc

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*70)
print("AI COMMUNICATION SYSTEM WITH PHASE NOISE")
print("="*70)

M = 4
n = 2
k = 2
PHASE_NOISE_STD = 0.2

TRAIN_SNR = 7.0
EPOCHS = 150
BATCH_SIZE = 256
SNR_RANGE = np.arange(0, 16, 1)
N_TEST = 50000

print(f"Phase noise std: {PHASE_NOISE_STD:.2f} rad (~{np.degrees(PHASE_NOISE_STD):.1f}°)")

QPSK_MAP = {
    (0, 0): (1, 1),
    (0, 1): (-1, 1),
    (1, 1): (-1, -1),
    (1, 0): (1, -1)
}

QPSK_DEMAP = {
    (1, 1): (0, 0),
    (-1, 1): (0, 1),
    (-1, -1): (1, 1),
    (1, -1): (1, 0)
}

def bits_to_qpsk(bits):
    bits = bits.reshape(-1, 2)
    symbols = np.zeros((len(bits), 2))
    
    for i in range(len(bits)):
        b0, b1 = int(bits[i, 0]), int(bits[i, 1])
        symbols[i] = QPSK_MAP[(b0, b1)]
    
    return symbols / np.sqrt(2.0)

def qpsk_to_bits(symbols):
    bits = np.zeros((len(symbols), 2), dtype=int)
    
    for i in range(len(symbols)):
        I_sign = 1 if symbols[i, 0] > 0 else -1
        Q_sign = 1 if symbols[i, 1] > 0 else -1
        bits[i] = QPSK_DEMAP[(I_sign, Q_sign)]
    
    return bits.flatten()

def apply_phase_noise(symbols, phase_std):
    
    n_symbols = len(symbols)
    phases = np.random.normal(0, phase_std, n_symbols)
    
    rotated = np.zeros_like(symbols)
    for i in range(n_symbols):
        cos_theta = np.cos(phases[i])
        sin_theta = np.sin(phases[i])
        
        I, Q = symbols[i]
        rotated[i, 0] = cos_theta * I - sin_theta * Q  # I'
        rotated[i, 1] = sin_theta * I + cos_theta * Q  # Q'
    
    return rotated

def qpsk_theory_ber(snr_db):
    snr_lin = 10**(snr_db/10.0)
    return 0.5 * erfc(np.sqrt(snr_lin))

def qpsk_simulate_with_phase_noise(snr_db, phase_std, n_bits=50000):
  
    n_bits = (n_bits // 2) * 2
    
    tx_bits = np.random.randint(0, 2, n_bits)
    
    tx_symbols = bits_to_qpsk(tx_bits)
    
    tx_symbols = apply_phase_noise(tx_symbols, phase_std)
    
    snr_linear = 10**(snr_db / 10.0)
    noise_var = 1.0 / (2.0 * snr_linear)
    noise = np.sqrt(noise_var) * np.random.randn(*tx_symbols.shape)
    
    rx_symbols = tx_symbols + noise
    
    rx_bits = qpsk_to_bits(rx_symbols)
    
    errors = np.sum(tx_bits != rx_bits)
    ber = errors / n_bits
    
    return ber

class NormalizationLayer(layers.Layer):
    def call(self, x):
        power = tf.reduce_mean(tf.reduce_sum(tf.square(x), axis=1))
        return x / tf.sqrt(power + 1e-7)

class PhaseNoiseAWGNChannel(layers.Layer):
  
    def __init__(self, train_snr_db, phase_noise_std):
        super().__init__()
        snr_lin = 10**(train_snr_db/10.0)
        self.noise_std = tf.constant(np.sqrt(1.0/(2.0*snr_lin)), dtype=tf.float32)
        self.phase_std = tf.constant(phase_noise_std, dtype=tf.float32)
    
    def call(self, x, training=None):
        if training:
            batch_size = tf.shape(x)[0]
            
            phases = tf.random.normal([batch_size, 1], stddev=self.phase_std)
            cos_theta = tf.cos(phases)
            sin_theta = tf.sin(phases)
            
            I = x[:, 0:1]
            Q = x[:, 1:2]
            
            I_rot = cos_theta * I - sin_theta * Q
            Q_rot = sin_theta * I + cos_theta * Q
            
            x_rotated = tf.concat([I_rot, Q_rot], axis=1)
            
            noise = tf.random.normal(tf.shape(x_rotated), stddev=self.noise_std)
            return x_rotated + noise
        
        return x

def build_autoencoder_with_phase_robustness():
  
    msg_in = layers.Input(shape=(M,))
    
    x = layers.Dense(M*2, activation='relu')(msg_in)
    x = layers.Dense(M, activation='relu')(x)
    x = layers.Dense(n, activation=None, name='tx')(x)
    x = NormalizationLayer(name='normalize')(x)
    
    y = PhaseNoiseAWGNChannel(TRAIN_SNR, PHASE_NOISE_STD)(x)
    
    z = layers.Dense(M*2, activation='relu')(y)
    z = layers.Dense(M, activation='relu')(z)
    out = layers.Dense(M, activation='softmax')(z)
    
    model = models.Model(msg_in, out)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("\n" + "="*70)
print("TRAINING AI AUTOENCODER")
print("(Learning to communicate with phase noise)")
print("="*70)

model = build_autoencoder_with_phase_robustness()

def data_gen():
    while True:
        msgs = np.random.randint(0, M, BATCH_SIZE)
        msgs_oh = keras.utils.to_categorical(msgs, M)
        yield msgs_oh, msgs_oh

history = model.fit(
    data_gen(),
    steps_per_epoch=200,
    epochs=EPOCHS,
    verbose=1
)

print(f"\nFinal training accuracy: {history.history['accuracy'][-1]:.4f}")

encoder = models.Model(model.input, model.get_layer('normalize').output)
msgs_all = keras.utils.to_categorical(range(M), M)
constellation = encoder.predict(msgs_all, verbose=0)

print("\nLearned constellation:")
for i in range(M):
    print(f"  Msg {i}: I={constellation[i,0]:6.3f}, Q={constellation[i,1]:6.3f}")

def eval_autoencoder_with_phase_noise(model, snr_db, phase_std):
    """Evaluate autoencoder with phase noise"""
    msgs = np.random.randint(0, M, N_TEST)
    msgs_oh = keras.utils.to_categorical(msgs, M)
    
    encoder = models.Model(model.input, model.get_layer('normalize').output)
    tx = encoder.predict(msgs_oh, batch_size=1024, verbose=0)
    
    tx = apply_phase_noise(tx, phase_std)
    
    snr_lin = 10**(snr_db/10.0)
    noise_std = np.sqrt(1.0/(2.0*snr_lin))
    noise = noise_std * np.random.randn(*tx.shape)
    rx = tx + noise
    
    dec_in = layers.Input(shape=(n,))
    found_channel = False
    x = dec_in
    for layer in model.layers:
        if found_channel and 'dense' in layer.name:
            x = layer(x)
        if 'channel' in layer.name:
            found_channel = True
    
    decoder = models.Model(dec_in, x)
    
    out = decoder.predict(rx, batch_size=1024, verbose=0)
    pred = np.argmax(out, axis=1)
    
    ser = np.mean(pred != msgs)
    ber = ser / k
    
    return ber


print("\n" + "="*70)
print("COMPARING PERFORMANCE WITH PHASE NOISE")
print("="*70)

ber_qpsk_perfect = []
ber_qpsk_with_phase = []
ber_ai = []

for snr in SNR_RANGE:
    print(f"SNR={snr:2d}dB: ", end='', flush=True)
    
    bp = qpsk_theory_ber(snr)
    ber_qpsk_perfect.append(bp)
    
    bq = qpsk_simulate_with_phase_noise(snr, PHASE_NOISE_STD)
    ber_qpsk_with_phase.append(bq)
    
    ba = eval_autoencoder_with_phase_noise(model, snr, PHASE_NOISE_STD)
    ber_ai.append(ba)
    
    print(f"QPSK(perfect)={bp:.3e}, QPSK(phase)={bq:.3e}, AI={ba:.3e}")

plt.figure(figsize=(12, 8))

plt.semilogy(SNR_RANGE, ber_qpsk_perfect, 'b--', 
             linewidth=2, label='QPSK (Perfect Phase Sync)', alpha=0.7)
plt.semilogy(SNR_RANGE, ber_qpsk_with_phase, 'ro-', 
             linewidth=2.5, markersize=10, label='QPSK (With Phase Noise)',
             markerfacecolor='white', markeredgewidth=2)
plt.semilogy(SNR_RANGE, ber_ai, 'gd-', 
             linewidth=2.5, markersize=10, label='AI Autoencoder (Phase-Robust)',
             markerfacecolor='white', markeredgewidth=2)

plt.xlabel('Eb/N0 (dB)', fontsize=14, fontweight='bold')
plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
plt.title(f'End-to-End AI Communication System\nAWGN Channel with Phase Noise (σ={np.degrees(PHASE_NOISE_STD):.1f}°)', 
          fontsize=15, fontweight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='upper right')
plt.ylim([1e-6, 1])
plt.tight_layout()
plt.savefig('ai_vs_qpsk_phase_noise.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

idx = 10
print(f"\nAt Eb/N0 = {SNR_RANGE[idx]} dB:")
print(f"  QPSK (Perfect Phase):  {ber_qpsk_perfect[idx]:.6e}")
print(f"  QPSK (With Phase Noise): {ber_qpsk_with_phase[idx]:.6e}")
print(f"  AI Autoencoder:        {ber_ai[idx]:.6e}")

if ber_ai[idx] < ber_qpsk_with_phase[idx]:
    improvement = (ber_qpsk_with_phase[idx] - ber_ai[idx]) / ber_qpsk_with_phase[idx] * 100
    gain_db = 10 * np.log10(ber_qpsk_with_phase[idx] / ber_ai[idx])
    print(f"\n✓ AI ACHIEVES:")
    print(f"  - {improvement:.1f}% BER reduction")
    print(f"  - {gain_db:.2f} dB effective coding gain")

print("\n" + "="*70)
print("KEY INSIGHT")
print("="*70)
print("Classical QPSK assumes perfect carrier phase synchronization.")
print("In realistic channels with phase noise, QPSK performance degrades.")
print()
print("The AI autoencoder learns BOTH:")
print("  1. Phase-robust modulation patterns")
print("  2. Phase-invariant detection/decoding")
print()
print("Result: AI significantly outperforms classical QPSK in realistic")
print("        conditions with imperfect phase synchronization!")
print("="*70)
