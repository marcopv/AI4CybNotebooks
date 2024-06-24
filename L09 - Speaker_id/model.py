import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from layers import ZScoreNormalization, LogMelgramLayer

from resnet.resnet import ResNet18

PARAMS = {
    'sample_rate': 16000,
    'stft_window_seconds': 0.025,
    'stft_hop_seconds': 0.010,
    'mel_bands': 128,
    'mel_min_hz': 125.0,
    'mel_max_hz': 7500.0,
}


def SpeakerID(input_shape, checkpoint_path, n_classes=1024):

    # Input
    inputs = Input(shape=input_shape)

    window_length_samples = int(
        round(PARAMS['sample_rate'] * PARAMS['stft_window_seconds']))
    hop_length_samples = int(
        round(PARAMS['sample_rate'] * PARAMS['stft_hop_seconds']))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    num_spectrogram_bins = fft_length // 2 + 1


    x = tf.keras.layers.Lambda(lambda x: x / tf.math.reduce_max(x,-2,keepdims=True))(inputs)

    # Mel Spectrogram
    x = LogMelgramLayer(
        num_fft=fft_length,
        window_length=window_length_samples,
        hop_length=hop_length_samples,
        sr=PARAMS['sample_rate'],
        mel_bins=PARAMS['mel_bands'],
        spec_bins=num_spectrogram_bins,
        fmin=PARAMS['mel_min_hz'],
        fmax=PARAMS['mel_max_hz']
    )(x)

    # Normalize along coeffients and time
    x = ZScoreNormalization(axis=[1, 2])(x)
    
    # Backbone
    input_shape = (None, PARAMS['mel_bands'])
    backbone = ResNet18(input_shape, classes=n_classes)
    backbone.load_weights(checkpoint_path) 
    y = backbone(x)

    # Final Model
    model = Model(inputs=inputs, outputs=y)

    return model
