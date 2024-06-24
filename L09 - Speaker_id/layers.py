import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

class SpectrogramLayer(Layer):
    def __init__(self, num_fft, window_length, hop_length, spec_bins, sr, log=False, log_offset=0.001, pad_end=True ,**kwargs):
        super(SpectrogramLayer, self).__init__(**kwargs)
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.window_length = window_length
        self.log = log
        self.num_spec_bins = spec_bins
        self.sr=sr
        self.log_offset = log_offset
        self.pad_end = pad_end

        assert num_fft // 2 + 1 == self.num_spec_bins        

    def build(self, input_shape):
        #self.non_trainable_weights.append(self.lin_to_mel_matrix)
        super(SpectrogramLayer, self).build(input_shape)

    def call(self, x):
        """
        Args:
            x (tensor): Batch of mono waveform, shape: (None, N)
        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)
        """

        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator
      
        # tf.signal.stft seems to be applied along the last axis
        stfts = tf.signal.stft(
            x, frame_length=self.window_length, frame_step=self.hop_length, pad_end=self.pad_end
        )
        mag_stfts = tf.abs(stfts)

        if self.log:
            return _tf_log10(mag_stfts + self.log_offset)

        return mag_stfts
    
    def compute_output_shape(self, input_shape):
      if input_shape[1] != None:
        random_waveform = tf.convert_to_tensor(np.random.rand(input_shape[-1]), dtype=tf.float32)
        stfts = tf.signal.stft(
              random_waveform, frame_length=self.window_length, frame_step=self.hop_length, pad_end=self.pad_end
          )
        stfts_frame = stfts.shape[0].value
      else:
        stfts_frame = None
      #int(round(((input_shape[-1] - (self.window_length - 1) - 1) / self.hop_length) + 1)) NOT CONSIDERING PADDING
      return (input_shape[0], stfts_frame, self.num_spec_bins)

    
    def get_config(self):
        config = super().get_config()
        config["num_fft"] = self.num_fft
        config["hop_length"] = self.hop_length
        config["window_length"] = self.window_length
        config["num_spec_bins"] = self.num_spec_bins
        config["sr"] = self.sr
        config["log"] = self.log
        config["log_offset"] = self.log_offset
        config["pad_end"] = self.pad_end
        return config

class LogMelgramLayer(Layer):
    def __init__(self, num_fft, window_length, hop_length, mel_bins, spec_bins, sr, fmin, fmax, log_offset=0.001, pad_end=True ,**kwargs):
        super(LogMelgramLayer, self).__init__(**kwargs)
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.window_length = window_length
        self.num_mel_bins = mel_bins
        self.num_spec_bins = spec_bins
        self.sr=sr
        self.f_min = fmin
        self.f_max = fmax
        self.log_offset = log_offset
        self.pad_end = pad_end

        assert num_fft // 2 + 1 == self.num_spec_bins
        self.lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=self.num_spec_bins,
            sample_rate=self.sr,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max,
        )
        

    def build(self, input_shape):
        #self.non_trainable_weights.append(self.lin_to_mel_matrix)
        super(LogMelgramLayer, self).build(input_shape)

    def call(self, x):
        """
        Args:
            x (tensor): Batch of mono waveform, shape: (None, N)
        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)
        """

        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator
      
        # tf.signal.stft seems to be applied along the last axis
        stfts = tf.signal.stft(
            x[:,:,0], frame_length=self.window_length, frame_step=self.hop_length, pad_end=self.pad_end
        )
        mag_stfts = tf.abs(stfts)

        melgrams = tf.tensordot(tf.square(mag_stfts), self.lin_to_mel_matrix, axes=[2, 0])
        log_melgrams = _tf_log10(melgrams + self.log_offset)
        return log_melgrams
    
    def compute_output_shape(self, input_shape):
      if input_shape[1] != None:
        random_waveform = tf.convert_to_tensor(np.random.rand(input_shape[-1]), dtype=tf.float32)
        stfts = tf.signal.stft(
              random_waveform, frame_length=self.window_length, frame_step=self.hop_length, pad_end=self.pad_end
          )
        stfts_frame = stfts.shape[0].value
      else:
        stfts_frame = None
      #int(round(((input_shape[-1] - (self.window_length - 1) - 1) / self.hop_length) + 1)) NOT CONSIDERING PADDING
      return (input_shape[0], stfts_frame, self.num_mel_bins)

    
    def get_config(self):
        config = super().get_config()
        config["num_fft"] = self.num_fft
        config["hop_length"] = self.hop_length
        config["window_length"] = self.window_length
        config["mel_bins"] = self.num_mel_bins
        config["spec_bins"] = self.num_spec_bins
        config["sr"] = self.sr
        config["fmin"] = self.f_min
        config["fmax"] = self.f_max
        config["log_offset"] = self.log_offset
        config["pad_end"] = self.pad_end
        return config


class AmplitudeToDB(Layer):
    def __init__(self, log_offset):
        self.log_offset = log_offset
        super(AmplitudeToDB, self).__init__()

    def call(self, inputs):
        numerator = tf.math.log(inputs+self.log_offset)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

class ZScoreNormalization(Layer):
    def __init__(self, axis=[-1, -2], scale_variance=True, eps=1e-07, **kwargs):
        self.axis = axis
        self.scale = scale_variance
        self.eps = eps
        super(ZScoreNormalization, self).__init__(**kwargs)

    def get_config(self):
        return {
            "axis": self.axis,
            "scale_variance": self.scale,
            "eps": self.eps
        }

    def build(self, input_shape):
        super(ZScoreNormalization, self).build(input_shape)

    def call(self, input_tensor):
        mean_values = tf.math.reduce_mean(
            input_tensor, axis=self.axis, keepdims=True)

        if self.scale:
            dev_std = tf.math.reduce_std(
                input_tensor, axis=self.axis, keepdims=True) + tf.constant(self.eps)
            norm_tensor = (input_tensor - mean_values)/dev_std
        else:
            norm_tensor = input_tensor - mean_values

        return norm_tensor

    def compute_output_shape(self, input_shape):
        return input_shape
