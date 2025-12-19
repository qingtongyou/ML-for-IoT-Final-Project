/*
 * Integrated training and inference pipeline.
 *
 * - Training and inference use separate weight buffers
 * - Serial commands control training, inference, synchronization
 * - Supports BLE-based weight exchange
 * 
 * Weight buffers:
 * - WeightBiasPtr_training : updated during training
 * - WeightBiasPtr_inference: used during inference
 *
 *
 * Serial commands:
 * - "train" / "t"        : train one epoch (update training weights)
 * - "infer" / "i"        : enter inference mode (use inference weights)
 * - "sync" / "s"         : sync training weights to inference weights
 * - "exit"               : exit inference mode
 * - "help"               : show help message
 * 
 */

#include<stdio.h>
#include<stdlib.h>
#include<string.h>  // for memcpy
#include<math.h>
#include<stdint.h>
#include<TinyMLShield.h>
#include <PDM.h>
#include <ArduinoBLE.h>

// ==================== Performance flags ====================
#define DEBUG_MODE
#define ENABLE_ENERGY_PRINT

// LED pin for on/off control
#define LED_PIN 11  // pin LED(D11)

// Threshold for treating prediction as "unknown"
const float SOFTMAX_UNKNOWN_THRESHOLD = 0.6f;

// Check NaN/Inf in debug mode, detect invalid float values
#ifdef DEBUG_MODE
  inline bool isInvalidNumber(float x) {
    return isnan(x) || isinf(x);
  }
#else
  #define isInvalidNumber(x) (false)
#endif
// =========================================================


// NN parameters
#define DATA_TYPE_FLOAT
#define LEARNING_RATE 0.005
#define EPOCH 50
#define DEBUG 0

extern const int first_layer_input_cnt;
extern const int classes_cnt;

#include "mfe_data_normalized.h"

// NN structure configuration
static const unsigned int NN_def[] = {first_layer_input_cnt, 32, 16, classes_cnt};

// Stored training mean + std, for realtime inference
#include "normalization_params_user2_mode0.h"   // Use this if single-user mode
// #include "normalization_params_global_a0.65.h"     // Use this if FL mode

#include "NN_functions_modified.h"

int iter_cnt = 0;  // count epoch

// ==================== Serial command control ====================
bool serial_train_request = false;  // train request
bool serial_infer_request = false;  // enter inference mode
bool serial_exit_request = false;   // exit inference mode
// =========================================================

// ==================== BLE configuration  ====================
BLEService weightService("19B10000-E8F2-537E-4F6C-D104768A1214"); // service UUID
BLECharacteristic weightNotifyChar("19B10001-E8F2-537E-4F6C-D104768A1214", BLENotify, 512); // Notify (max 512 bytes)
BLECharacteristic weightWriteChar("19B10002-E8F2-537E-4F6C-D104768A1214", BLEWrite, 512); // Write (max 512 bytes)
bool ble_send_enabled = false;  // control BLE weights sending
// =========================================================

// ============= MFE Feature Extraction parmas ================
#define SAMPLE_RATE 16000
#define NUM_MELS 15
#define FRAME_LEN_MS 25.0f
#define FRAME_HOP_MS 200.0f

#define FRAME_LEN_SAMPLES (int)(FRAME_LEN_MS * SAMPLE_RATE / 1000.0f)  // 400 samples
#define N_FFT 512                 // 512-FFT
#define HOP_LENGTH (int)(FRAME_HOP_MS * SAMPLE_RATE / 1000.0f)
#define TARGET_FEATURE_DIM 75     // 15 mels × 5 frames = 75
// =========================================================

// ==================== Energy-based trigger parameters ====================
#define ENERGY_WINDOW_MS 50.0f    // Monitoring window (50 ms)
#define ENERGY_WINDOW_SAMPLES (int)(ENERGY_WINDOW_MS * SAMPLE_RATE / 1000.0f)  // 800 samples @ 16kHz
#define ENERGY_THRESHOLD 500.0f   // Energy threshold
#define ENERGY_CHECK_INTERVAL_MS 20  // Check energy every 20 ms

// Audio record params
#define PRE_TRIGGER_MS 20.0f       // Frozen audio before trigger (20ms)
#define PRE_TRIGGER_SAMPLES (int)(PRE_TRIGGER_MS * SAMPLE_RATE / 1000.0f)  // 320 samples @ 16kHz
#define POST_TRIGGER_MS 480.0f     // Audio recorded after trigger (480 ms)
#define POST_TRIGGER_SAMPLES (int)(POST_TRIGGER_MS * SAMPLE_RATE / 1000.0f)  // 7680 samples @ 16kHz
#define TOTAL_INFERENCE_SAMPLES (PRE_TRIGGER_SAMPLES + POST_TRIGGER_SAMPLES)  // 8000 samples = 500 ms
// =========================================================

// Audio buffer (for x ms pre-buffer during real-time inference)
int16_t* audio_buffer = NULL;
float* audio_float = NULL;

// MFE feature
float mfe_features[TARGET_FEATURE_DIM];

// Energy monitoring buffer
int16_t energy_buffer[ENERGY_WINDOW_SAMPLES];
volatile int energy_samples_read = 0;
volatile bool energy_monitoring = false;

// Ring buffer for pre-trigger audio, continuously updated
int16_t pre_trigger_buffer[PRE_TRIGGER_SAMPLES];
volatile int pre_trigger_write_idx = 0;
volatile bool pre_trigger_ready = false;

// Operating mode
enum Mode {
  MODE_IDLE,        
  MODE_TRAINING,    
  MODE_INFERENCE
};
Mode currentMode = MODE_IDLE;

// ==================== Decoupled training and inference weights ====================
DATA_TYPE* WeightBiasPtr_training = NULL;
DATA_TYPE* WeightBiasPtr_inference = NULL;
int weights_bias_cnt = 0;     // total number of weights
// =========================================================

// ==================== Inference status ====================
// Store maximum softmax probability of last inference
float g_max_softmax_prob = 0.0f;
// =========================================================

// ==================== MFE feature extraction ====================

// Hann window
void createHannWindow(float* window, int n) {
  for (int i = 0; i < n; i++) {
    window[i] = 0.5f * (1.0f - cos(2.0f * M_PI * i / (n - 1)));
  }
}

// Hz to Mel
float hz_to_mel(float hz) {
  return 2595.0f * log10(1.0f + hz / 700.0f);
}

// Mel to Hz
float mel_to_hz(float mel) {
  return 700.0f * (pow(10.0f, mel / 2595.0f) - 1.0f);
}

// Create Mel filterbank from 512-point FFT (257 bins)
void createMelFilterbank(float* filterbank, int n_mels, int n_fft, int sr) {
  float fmax = sr / 2.0f;
  float mel_min = hz_to_mel(0.0f);
  float mel_max = hz_to_mel(fmax);
  
  float mel_points[n_mels + 2];
  float hz_points[n_mels + 2];
  int fft_bins[n_mels + 2];
  int num_freq_bins = n_fft / 2 + 1;  // 257 for 512-point FFT
  
  for (int i = 0; i < n_mels + 2; i++) {
    mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    hz_points[i] = mel_to_hz(mel_points[i]);
    fft_bins[i] = (int)(num_freq_bins * hz_points[i] / fmax);
    if (fft_bins[i] >= num_freq_bins) fft_bins[i] = num_freq_bins - 1;
  }
  
  for (int i = 0; i < n_mels * num_freq_bins; i++) {
    filterbank[i] = 0.0f;
  }
  
  for (int i = 0; i < n_mels; i++) {
    int left = fft_bins[i];
    int center = fft_bins[i + 1];
    int right = fft_bins[i + 2];
    
    for (int k = left; k < center; k++) {
      if (k < num_freq_bins) {
        filterbank[i * num_freq_bins + k] = (float)(k - left) / (center - left);
      }
    }
    
    for (int k = center; k < right; k++) {
      if (k < num_freq_bins) {
        filterbank[i * num_freq_bins + k] = (float)(right - k) / (right - center);
      }
    }
  }
}

// ==================== FFT-512 ====================

void bitReverse(float* x, int n) {
  int j = 0;
  for (int i = 1; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      float temp_real = x[i * 2];
      float temp_imag = x[i * 2 + 1];
      x[i * 2] = x[j * 2];
      x[i * 2 + 1] = x[j * 2 + 1];
      x[j * 2] = temp_real;
      x[j * 2 + 1] = temp_imag;
    }
  }
}

void fft512(float* x) {
  const int n = 512;
  bitReverse(x, n);
  
  for (int len = 2; len <= n; len <<= 1) {
    float angle = -2.0f * M_PI / len;
    float wlen_real = cos(angle);
    float wlen_imag = sin(angle);
    
    for (int i = 0; i < n; i += len) {
      float w_real = 1.0f;
      float w_imag = 0.0f;
      
      for (int j = 0; j < len / 2; j++) {
        int idx1 = (i + j) * 2;
        int idx2 = (i + j + len / 2) * 2;
        
        float u_real = x[idx1];
        float u_imag = x[idx1 + 1];
        float v_real = x[idx2];
        float v_imag = x[idx2 + 1];
        
        float t_real = w_real * v_real - w_imag * v_imag;
        float t_imag = w_real * v_imag + w_imag * v_real;
        
        x[idx1] = u_real + t_real;
        x[idx1 + 1] = u_imag + t_imag;
        x[idx2] = u_real - t_real;
        x[idx2 + 1] = u_imag - t_imag;
        
        float next_w_real = w_real * wlen_real - w_imag * wlen_imag;
        float next_w_imag = w_real * wlen_imag + w_imag * wlen_real;
        w_real = next_w_real;
        w_imag = next_w_imag;
      }
    }
  }
}

void computeFFT512(float* frame, float* real, float* imag) {
  static float fft_input[512 * 2];
  
  for (int i = 0; i < 512; i++) {
    fft_input[i * 2] = (i < FRAME_LEN_SAMPLES) ? frame[i] : 0.0f;
    fft_input[i * 2 + 1] = 0.0f;
  }
  
  fft512(fft_input);
  
  for (int i = 0; i < 257; i++) {
    real[i] = fft_input[i * 2];
    imag[i] = fft_input[i * 2 + 1];
  }
}

// Precomputed window and Mel filterbank
static float hann_window[FRAME_LEN_SAMPLES];
static float mel_filterbank[NUM_MELS * (N_FFT / 2 + 1)];
static bool mfe_initialized = false;

void initMFE() {
  if (mfe_initialized) return;
  
  createHannWindow(hann_window, FRAME_LEN_SAMPLES);
  createMelFilterbank(mel_filterbank, NUM_MELS, N_FFT, SAMPLE_RATE);
  
  mfe_initialized = true;
}

// compute MFE features, only used for online inference
void computeMFE(float* audio, float* features) {
  initMFE();
  
  int num_frames = 5;

  // Static buffers for FFT (avoid stack allocation)
  static float frame[512];
  static float real[257];
  static float imag[257];
  static float power[257];
  const int num_freq_bins = 257;
  
  for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
    int start = (int)(frame_idx * (TOTAL_INFERENCE_SAMPLES - FRAME_LEN_SAMPLES) / (num_frames - 1.0f));
    
    int i = 0;
    for (; i < FRAME_LEN_SAMPLES; i++) {
      int sample_idx = start + i;
      if (sample_idx < TOTAL_INFERENCE_SAMPLES) {
        frame[i] = audio[sample_idx] * hann_window[i];
      } else {
        frame[i] = 0.0f;
      }
    }
    for (; i < 512; i++) {
      frame[i] = 0.0f;
    }
    
    computeFFT512(frame, real, imag);
    
    for (int i = 0; i < num_freq_bins; i++) {
      power[i] = real[i] * real[i] + imag[i] * imag[i];
    }

    for (int mel = 0; mel < NUM_MELS; mel++) {
      float e = 0.0f;
      int offset = mel * num_freq_bins;

      for (int k = 0; k < num_freq_bins; k++) {
        e += mel_filterbank[offset + k] * power[k];
      }

      if (e < 1e-10f) e = 1e-10f;

      features[frame_idx * NUM_MELS + mel] = log(e);
    }
  }
}

// Normalization
void normalizeFeatures(float* features) {
  #if defined(ARDUINO_ARDUINO_NANO33BLE) || defined(ARDUINO_NANO33BLE)
    for (int i = 0; i < TARGET_FEATURE_DIM; i++) {
      #ifdef DEBUG_MODE
        if (isInvalidNumber(features[i])) {
          features[i] = 0.0f;
        }
        float mean_val = training_mean[i];
        float std_val = training_std[i];
        if (isInvalidNumber(mean_val)) mean_val = 0.0f;
        if (isInvalidNumber(std_val) || std_val <= 0.0f) std_val = 1.0f;
        features[i] = (features[i] - mean_val) / std_val;
        if (isInvalidNumber(features[i])) features[i] = 0.0f;
      #else
        features[i] = (features[i] - training_mean[i]) / training_std[i];
      #endif
    }
  #else
    for (int i = 0; i < TARGET_FEATURE_DIM; i++) {
      #ifdef DEBUG_MODE
        if (isInvalidNumber(features[i])) {
          features[i] = 0.0f;
        }
        float mean_val = pgm_read_float(&training_mean[i]);
        float std_val = pgm_read_float(&training_std[i]);
        if (isInvalidNumber(mean_val)) mean_val = 0.0f;
        if (isInvalidNumber(std_val) || std_val <= 0.0f) std_val = 1.0f;
        features[i] = (features[i] - mean_val) / std_val;
        if (isInvalidNumber(features[i])) features[i] = 0.0f;
      #else
        float mean_val = pgm_read_float(&training_mean[i]);
        float std_val = pgm_read_float(&training_std[i]);
        features[i] = (features[i] - mean_val) / std_val;
      #endif
    }
  #endif
}

// Perform inference
int inference(float* features) {
  #ifdef DEBUG_MODE
    for (unsigned int j = 0; j < IN_VEC_SIZE; j++) {
      if (isInvalidNumber(features[j])) {
        return -1;
      }
      input[j] = features[j];
    }
  #else
    for (unsigned int j = 0; j < IN_VEC_SIZE; j++) {
      input[j] = features[j];
    }
  #endif
  
  forwardProp();
  
  #ifdef DEBUG_MODE
    for (unsigned int j = 0; j < OUT_VEC_SIZE; j++) {
      if (isInvalidNumber(y[j])) {
        return -1;
      }
    }
  #endif
  
  int predicted_class = 0;
  DATA_TYPE max_prob = y[0];
  for (unsigned int j = 1; j < OUT_VEC_SIZE; j++) {
    if (y[j] > max_prob) {
      max_prob = y[j];
      predicted_class = j;
    }
  }

  // record max softmax probability of this inference
  g_max_softmax_prob = (float)max_prob;
  
  return predicted_class;
}

// Print inference result
void printInferenceResult(int predicted_class) {

  // Modify this array if **class applied changes**
  const char* class_names[] = {"on", "off"}; 

  const int class_names_size = sizeof(class_names) / sizeof(class_names[0]);
  
  if (predicted_class < 0 || predicted_class >= OUT_VEC_SIZE) {
    Serial.print("Predicted: <INVALID> (class ");
    Serial.print(predicted_class);
    Serial.print(", valid range: 0-");
    Serial.print(OUT_VEC_SIZE - 1);
    Serial.print(")");
  } else {
    // Confidence-based rejection
    if (g_max_softmax_prob <= SOFTMAX_UNKNOWN_THRESHOLD) {
      Serial.print("Predicted: unknown");
      Serial.print(" (max_prob=");
      Serial.print(g_max_softmax_prob, 4);
      Serial.print(")");
    } else {
      if (predicted_class < class_names_size) {
        Serial.print("Predicted: ");
        Serial.print(class_names[predicted_class]);
      } else {
        Serial.print("Predicted: <UNNAMED>");
      }
      Serial.print(" (class ");
      Serial.print(predicted_class);
      Serial.print(")");
      Serial.print(" | max_prob=");
      Serial.print(g_max_softmax_prob, 4);
    }
  }
  
  Serial.print(" | Probabilities: [");
  for (unsigned int j = 0; j < OUT_VEC_SIZE; j++) {
    Serial.print(y[j], 4);
    if (j < OUT_VEC_SIZE - 1) Serial.print(", ");
  }
  Serial.println("]");
}

// Control LED according to inference result and confidence
void handleLedAfterInference(int predicted_class) {
  // if confidence is not high enough, consider it unknown, do not change LED state
  if (g_max_softmax_prob <= SOFTMAX_UNKNOWN_THRESHOLD) {
    Serial.println("LED: prediction is unknown, LED state unchanged.");
    return;
  }

  //  0 -> "on", 1 -> "off"
  if (predicted_class == 0) {
    // light on
    digitalWrite(LED_PIN, HIGH);
    Serial.println("LED: ON (class = on)");
  } else if (predicted_class == 1) {
    //  light off
    digitalWrite(LED_PIN, LOW);
    Serial.println("LED: OFF (class = off)");
  } else {
    // other classes, not handled, keep current state
    Serial.print("LED: unhandled class index ");
    Serial.print(predicted_class);
    Serial.println(", LED state unchanged.");
  }
}

// ==================== Training Function ====================

void do_training() {
  #if DEBUG      
  Serial.println("Now Training");
  PRINT_WEIGHTS();
  #endif

  // Debug: print last 10 weights before training
  printLastWeights("TRAIN BEFORE - training buffer", WeightBiasPtr_training, 10);
  printLastWeights("TRAIN BEFORE - inference buffer", WeightBiasPtr_inference, 10);

  Serial.print("Epoch count (training count): ");
  Serial.print(++iter_cnt);
  Serial.println();
  Serial.println("Training one epoch...");

  shuffleIndx();
  
  // Train an epoch, update weights every sample
  for (int j = 0; j < numTrainData; j++) {
    generateTrainVectors(j);
    
    // Sample-wise SGD
    forwardProp();   // compute softmax prob
    backwardProp();  // compute gradient, then immediately update weights
  }

  // Training complete.
  Serial.println("Epoch training complete. Model weights updated.");
  
  // Sync updated model weights from L back to training buffer
  // for correct BLE transmission
  packUnpackVector(PACK);

  // Sync train to infer buffer
  syncTrainingToInference();

  // Debug: last 10 weights should be same
  printLastWeights("TRAIN AFTER - training buffer", WeightBiasPtr_training, 10);
  printLastWeights("TRAIN AFTER - inference buffer", WeightBiasPtr_inference, 10);
  
  // Send weights to PC aggregator if ble enabled and just finishing 3x epochs of training.
  if (ble_send_enabled && (iter_cnt % 3 == 0)) {
    sendWeightsViaBLE();
  }

  Serial.println("\n========================================");
  Serial.println("=== Training Complete - Accuracy ===");
  Serial.println("========================================");
  printAccuracy();
  Serial.println("========================================");
  Serial.println("Model has been updated with new weights.");
  Serial.println("Ready for inference using updated model.");
  Serial.println("========================================\n");
}

// ==================== PDM Callback Function ====================

volatile int samplesRead = 0;
volatile bool recording = false;

void onPDMdata() {
  int bytesAvailable = PDM.available();
  if (bytesAvailable <= 0) return;
  
  int16_t temp_buffer[512];
  int samplesToRead = min(bytesAvailable / sizeof(int16_t), 512);
  PDM.read(temp_buffer, samplesToRead * sizeof(int16_t));
  
  if (energy_monitoring && !recording) {
    for (int i = 0; i < samplesToRead; i++) {
      int energy_buffer_idx = energy_samples_read % ENERGY_WINDOW_SAMPLES;
      energy_buffer[energy_buffer_idx] = temp_buffer[i];
      energy_samples_read++;
      if (energy_samples_read >= ENERGY_WINDOW_SAMPLES * 2) {
        energy_samples_read = ENERGY_WINDOW_SAMPLES;
      }
      
      pre_trigger_buffer[pre_trigger_write_idx] = temp_buffer[i];
      pre_trigger_write_idx = (pre_trigger_write_idx + 1) % PRE_TRIGGER_SAMPLES;
      if (!pre_trigger_ready && pre_trigger_write_idx == 0) {
        pre_trigger_ready = true;
      }
    }
    return;
  }
  
  if (recording) {
    for (int i = 0; i < samplesToRead && samplesRead < POST_TRIGGER_SAMPLES; i++) {
      audio_buffer[samplesRead++] = temp_buffer[i];
    }
  }
}

// Compute audio energy using 64-bit accumulation to avoid overflow
// Return squared to reduce computational cost
uint32_t calculateEnergySquared(int16_t* samples, int num_samples) {
  if (num_samples == 0) return 0;
  
  uint64_t sum_squares = 0;
  int i = 0;
  for (; i < num_samples - 3; i += 4) {
    int32_t s0 = (int32_t)samples[i];
    int32_t s1 = (int32_t)samples[i+1];
    int32_t s2 = (int32_t)samples[i+2];
    int32_t s3 = (int32_t)samples[i+3];
    
    sum_squares += (uint64_t)(s0 * s0) + (uint64_t)(s1 * s1) + 
                   (uint64_t)(s2 * s2) + (uint64_t)(s3 * s3);
  }
  
  for (; i < num_samples; i++) {
    int32_t sample = (int32_t)samples[i];
    sum_squares += (uint64_t)(sample * sample);
  }
  
  uint64_t avg_squared = sum_squares / num_samples;
  // Safety check: clamp to UINT32_MAX if overflow occurs
  if (avg_squared > UINT32_MAX) {
    return UINT32_MAX;
  }
  return (uint32_t)avg_squared;
}

// Check energy threshold
bool checkEnergyThreshold() {
  if (!energy_monitoring) {
    return false;
  }
  
  if (energy_samples_read < ENERGY_WINDOW_SAMPLES) {
    return false;
  }
  
  uint32_t energy_squared = calculateEnergySquared(energy_buffer, ENERGY_WINDOW_SAMPLES);
  uint32_t threshold_squared = (uint32_t)(ENERGY_THRESHOLD * ENERGY_THRESHOLD);
  
  #ifdef ENABLE_ENERGY_PRINT
    float energy = sqrtf((float)energy_squared);
    static unsigned long last_print = 0;
    if (millis() - last_print > 100) {
      Serial.print("Energy: ");
      Serial.print(energy, 2);
      Serial.print(" | Threshold: ");
      Serial.print(ENERGY_THRESHOLD, 2);
      if (energy_squared > threshold_squared) {
        Serial.println(" >>> TRIGGERED! <<<");
      } else {
        Serial.println();
      }
      last_print = millis();
    }
  #endif
  
  return energy_squared > threshold_squared;
}

// Audio record
bool recordAudio() {
  #ifdef DEBUG_MODE
    Serial.println("\n=== Energy Threshold Exceeded ===");
  #endif
  
  int16_t frozen_pre_trigger[PRE_TRIGGER_SAMPLES];
  if (pre_trigger_ready) {
    for (int i = 0; i < PRE_TRIGGER_SAMPLES; i++) {
      int idx = (pre_trigger_write_idx + i) % PRE_TRIGGER_SAMPLES;
      frozen_pre_trigger[i] = pre_trigger_buffer[idx];
    }
  } else {
    int available = pre_trigger_write_idx;
    int zero_count = PRE_TRIGGER_SAMPLES - available;
    for (int i = 0; i < zero_count; i++) {
      frozen_pre_trigger[i] = 0;
    }
    for (int i = 0; i < available; i++) {
      frozen_pre_trigger[zero_count + i] = pre_trigger_buffer[i];
    }
  }
  
  energy_monitoring = false;
  samplesRead = 0;
  recording = true;
  
  int i = 0;
  for (; i < POST_TRIGGER_SAMPLES - 3; i += 4) {
    audio_buffer[i] = 0;
    audio_buffer[i+1] = 0;
    audio_buffer[i+2] = 0;
    audio_buffer[i+3] = 0;
  }
  for (; i < POST_TRIGGER_SAMPLES; i++) {
    audio_buffer[i] = 0;
  }
  
  delay(5);
  
  unsigned long startTime = millis();
  while (samplesRead < POST_TRIGGER_SAMPLES) {
    if (millis() - startTime > 2000) {
      recording = false;
      energy_monitoring = true;
      return false;
    }
    delay(5);
  }
  
  recording = false;
  
  const float scale = 1.0f / 32768.0f;
  
  i = 0;
  for (; i < PRE_TRIGGER_SAMPLES - 3; i += 4) {
    audio_float[i] = frozen_pre_trigger[i] * scale;
    audio_float[i+1] = frozen_pre_trigger[i+1] * scale;
    audio_float[i+2] = frozen_pre_trigger[i+2] * scale;
    audio_float[i+3] = frozen_pre_trigger[i+3] * scale;
  }
  for (; i < PRE_TRIGGER_SAMPLES; i++) {
    audio_float[i] = frozen_pre_trigger[i] * scale;
  }
  
  i = 0;
  for (; i < POST_TRIGGER_SAMPLES - 3; i += 4) {
    if (i < samplesRead) {
      audio_float[PRE_TRIGGER_SAMPLES + i] = audio_buffer[i] * scale;
      audio_float[PRE_TRIGGER_SAMPLES + i+1] = audio_buffer[i+1] * scale;
      audio_float[PRE_TRIGGER_SAMPLES + i+2] = audio_buffer[i+2] * scale;
      audio_float[PRE_TRIGGER_SAMPLES + i+3] = audio_buffer[i+3] * scale;
    } else {
      audio_float[PRE_TRIGGER_SAMPLES + i] = 0.0f;
      audio_float[PRE_TRIGGER_SAMPLES + i+1] = 0.0f;
      audio_float[PRE_TRIGGER_SAMPLES + i+2] = 0.0f;
      audio_float[PRE_TRIGGER_SAMPLES + i+3] = 0.0f;
    }
  }
  for (; i < POST_TRIGGER_SAMPLES; i++) {
    audio_float[PRE_TRIGGER_SAMPLES + i] = (i < samplesRead) ? (audio_buffer[i] * scale) : 0.0f;
  }
  
  return true;
}

// ==================== Weight synchronization ====================

// Copy training weights to inference weights
void syncTrainingToInference() {
  if (WeightBiasPtr_training && WeightBiasPtr_inference && weights_bias_cnt > 0) {
    memcpy(WeightBiasPtr_inference, WeightBiasPtr_training, weights_bias_cnt * sizeof(DATA_TYPE));
    Serial.println("Training weights synchronized to inference weights.");
  } else {
    Serial.println("ERROR: Weight buffers not initialized!");
  }
}

// Print last N elements of a weight buffer (debug only)
void printLastWeights(const char* tag, DATA_TYPE* buf, int n) {
  if (!buf || weights_bias_cnt == 0) {
    Serial.print(tag);
    Serial.println(": buffer not initialized");
    return;
  }

  int count = (n < weights_bias_cnt) ? n : weights_bias_cnt;
  int start = weights_bias_cnt - count;

  Serial.print(tag);
  Serial.print(" (last ");
  Serial.print(count);
  Serial.println(" weights):");

  for (int i = start; i < weights_bias_cnt; i++) {
    Serial.print((float)buf[i], 6);
    if (i < weights_bias_cnt - 1) Serial.print(", ");
  }
  Serial.println();
}

// ==================== BLE RX: global weights (static buffer) ====================

// Static buffer for receiving global weights

static const uint32_t MAX_GLOBAL_WEIGHTS_BYTES = 12000;  // ~10KB
static uint8_t g_global_weights_buf[MAX_GLOBAL_WEIGHTS_BYTES];

static bool     g_rx_active        = false;
static uint16_t g_rx_session_id    = 0;
static uint16_t g_rx_total_chunks  = 0;
static uint32_t g_rx_total_size    = 0;
static uint32_t g_rx_received_size = 0;
static uint16_t g_rx_received_chunks = 0;

void resetBleRxState() {
  g_rx_active          = false;
  g_rx_session_id      = 0;
  g_rx_total_chunks    = 0;
  g_rx_total_size      = 0;
  g_rx_received_size   = 0;
  g_rx_received_chunks = 0;
}

void handleIncomingWeights(const uint8_t* data, int length) {
  const int header_size = 11;
  if (length < header_size) {
    Serial.println("BLE RX: packet too short");
    return;
  }

  // Header (same with Tx)
  uint16_t session_id   = (data[0] << 8) | data[1];
  uint16_t chunk_id     = (data[2] << 8) | data[3];
  uint16_t total_chunks = (data[4] << 8) | data[5];
  uint32_t total_size   = ((uint32_t)data[6] << 24) |
                          ((uint32_t)data[7] << 16) |
                          ((uint32_t)data[8] << 8)  |
                          (uint32_t)data[9];
  uint8_t flags         = data[10];

  const uint8_t* payload = data + header_size;
  int payload_len = length - header_size;

  // Reset state on new session or session change
  if (!g_rx_active || g_rx_session_id != session_id) {
    resetBleRxState();

    if (total_size > MAX_GLOBAL_WEIGHTS_BYTES) {
      Serial.println("BLE RX: total_size exceeds MAX_GLOBAL_WEIGHTS_BYTES");
      return;
    }

    g_rx_active        = true;
    g_rx_session_id    = session_id;
    g_rx_total_chunks  = total_chunks;
    g_rx_total_size    = total_size;
    g_rx_received_size = 0;
    g_rx_received_chunks = 0;

    Serial.print("BLE RX: new session ");
    Serial.print(session_id);
    Serial.print(", total_chunks=");
    Serial.print(total_chunks);
    Serial.print(", total_size=");
    Serial.println(total_size);
  }

  if (!g_rx_active) {
    return;
  }

  // Compute offset based on chunk_id and fixed payload size
  // Must match 'MAX_CHUNK_PAYLOAD' on the PC side (ble_push_global.py)
  const uint32_t MAX_CHUNK_PAYLOAD_RX = 180;
  uint32_t offset = (uint32_t)chunk_id * MAX_CHUNK_PAYLOAD_RX;

  if (offset >= g_rx_total_size) {
    Serial.println("BLE RX: offset out of range, drop chunk");
    return;
  }
  if (offset + (uint32_t)payload_len > g_rx_total_size) {
    payload_len = (int)(g_rx_total_size - offset);
  }

  memcpy(g_global_weights_buf + offset, payload, payload_len);
  g_rx_received_size   += (uint32_t)payload_len;
  g_rx_received_chunks += 1;

  Serial.print("BLE RX: session ");
  Serial.print(session_id);
  Serial.print(" chunk ");
  Serial.print(chunk_id + 1);
  Serial.print("/");
  Serial.print(total_chunks);
  Serial.print(" (");
  Serial.print(payload_len);
  Serial.println(" bytes)");

  bool is_last = (flags & 0x01) != 0;
  if (is_last) {
    Serial.println("BLE RX: LAST_CHUNK flag set");
  }

  bool complete = (g_rx_received_chunks == g_rx_total_chunks) &&
                  (g_rx_received_size  == g_rx_total_size);

  if (!complete) {
    return;
  }

  Serial.println("BLE RX: session complete, applying global weights...");

  // check if size matched
  uint32_t expected_bytes = (uint32_t)weights_bias_cnt * (uint32_t)sizeof(DATA_TYPE);
  if (g_rx_total_size != expected_bytes) {
    Serial.print("BLE RX: size mismatch, expected ");
    Serial.print(expected_bytes);
    Serial.print(", got ");
    Serial.println(g_rx_total_size);
    resetBleRxState();
    return;
  }

  // Overwrite inference weights with global
  memcpy(WeightBiasPtr_inference,
         (const void*)g_global_weights_buf,
         expected_bytes);

  // Overwrite training weights with global model,
  // for subsequent local training
  memcpy(WeightBiasPtr_training,
         (const void*)g_global_weights_buf,
         expected_bytes);

  Serial.println("BLE RX: inference weights updated from global model.");

  // Sanity Check: print last 5 weights 
  float* w = (float*)WeightBiasPtr_inference;
  int start = weights_bias_cnt - 5;
  if (start < 0) start = 0;

  Serial.print("BLE RX: last 5 weights = ");
  for (int i = start; i < start + 5 && i < weights_bias_cnt; i++) {
    Serial.print(w[i], 6);
    if (i < start + 4 && i < weights_bias_cnt - 1) Serial.print(", ");
  }
  Serial.println();

  // Load training weights to network L and calculate accuracy
  Serial.println("\n=== Calculating Accuracy with Received Global Weights ===");
  WeightBiasPtr = WeightBiasPtr_training;
  packUnpackVector(UNPACK);
  printAccuracy();
  Serial.println("===========================================================\n");

  resetBleRxState();
}

// ==================== BLE functions ====================

// Session ID counter
static uint16_t ble_session_id = 0;

bool initBLE() {
  Serial.println("Initializing BLE...");
  
  delay(100);
  
  if (!BLE.begin()) {
    Serial.println("ERROR: BLE initialization failed!");
    return false;
  }
  
  // Set Name
  BLE.setLocalName("Sender_2");
  BLE.setDeviceName("Sender_2");
  
  // Set advertised service UUID
  BLE.setAdvertisedService(weightService);
  
  // Add characteristics to the service
  weightService.addCharacteristic(weightNotifyChar);
  weightService.addCharacteristic(weightWriteChar);
  
  // Add service to BLE stack
  BLE.addService(weightService);
  
  // Start advertising
  BLE.advertise();
  
  Serial.println("BLE initialized successfully!");
  Serial.println("Device name: Sender_2");
  Serial.println("BLE is now advertising...");
  
  return true;
}

// Send training weights via BLE
void sendWeightsViaBLE() {
  if (!ble_send_enabled || !WeightBiasPtr_training || weights_bias_cnt == 0) {
    return;
  }
  
  // Session ID
  ble_session_id++;
  if (ble_session_id == 0) ble_session_id = 1;
  
  Serial.println("\n=== Sending weights via BLE (epoch is multiple of 3) ===");
  Serial.print("Session ID: ");
  Serial.println(ble_session_id);
  Serial.print("Epoch: ");
  Serial.println(iter_cnt);
  Serial.print("Total weights: ");
  Serial.println(weights_bias_cnt);
  
  int total_bytes = weights_bias_cnt * sizeof(DATA_TYPE);
  Serial.print("Total data size: ");
  Serial.print(total_bytes);
  Serial.println(" bytes");
  
  // Packing
  // Header: session_id(2) + chunk_id(2) + total_chunks(2) + total_size(4) + flags(1) = 11 bytes
  const int header_size = 11;
  const int max_chunk_size = 220;  // Maximum 233 bytes
  uint8_t* weight_bytes = (uint8_t*)WeightBiasPtr_training;
  
  int chunks = (total_bytes + max_chunk_size - 1) / max_chunk_size;
  Serial.print("Sending in ");
  Serial.print(chunks);
  Serial.println(" chunks...");
  
  for (uint16_t chunk = 0; chunk < chunks; chunk++) {
    int offset = chunk * max_chunk_size;
    int chunk_size = min(max_chunk_size, total_bytes - offset);
    bool is_last_chunk = (chunk == chunks - 1);
    
    // Create packet
    uint8_t packet[512];
    
    // Header: session_id (2 bytes, big-endian)
    packet[0] = (ble_session_id >> 8) & 0xFF;
    packet[1] = ble_session_id & 0xFF;
    
    // chunk_id (2 bytes, big-endian)
    packet[2] = (chunk >> 8) & 0xFF;
    packet[3] = chunk & 0xFF;
    
    // total_chunks (2 bytes, big-endian)
    packet[4] = (chunks >> 8) & 0xFF;
    packet[5] = chunks & 0xFF;
    
    // total_size (4 bytes, big-endian)
    packet[6] = (total_bytes >> 24) & 0xFF;
    packet[7] = (total_bytes >> 16) & 0xFF;
    packet[8] = (total_bytes >> 8) & 0xFF;
    packet[9] = total_bytes & 0xFF;
    
    // flags (1 byte): bit 0 = is_last_chunk (1=last, 0=not last)
    packet[10] = is_last_chunk ? 0x01 : 0x00;
    
    memcpy(&packet[header_size], &weight_bytes[offset], chunk_size);
    
    int packet_size = header_size + chunk_size;
    weightNotifyChar.writeValue(packet, packet_size);
    
    Serial.print("Sent chunk ");
    Serial.print(chunk + 1);
    Serial.print("/");
    Serial.print(chunks);
    Serial.print(" (");
    Serial.print(packet_size);
    Serial.print(" bytes)");
    if (is_last_chunk) {
      Serial.print(" [LAST CHUNK]");
    }
    Serial.println();
    
    delay(500);  // delay between chunks, larger for more realiable transmission to PC
  }
  
  Serial.println("=== Weights sent via BLE successfully ===");
  Serial.print("Session ID: ");
  Serial.println(ble_session_id);
}


void toggleBLESend() {
  ble_send_enabled = !ble_send_enabled;
  if (ble_send_enabled) {
    Serial.println("BLE data transmission: ENABLED");
    Serial.println("Weights will be sent when epoch is a multiple of 3.");
  } else {
    Serial.println("BLE data transmission: DISABLED");
  }
}


void processSerialCommand() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toLowerCase();
    
    if (command == "train" || command == "t") {
      if (currentMode == MODE_IDLE) {
        serial_train_request = true;
        Serial.println("\n>>> Training requested via serial command");
      } else {
        Serial.println("ERROR: Cannot train while in inference mode. Type 'exit' first.");
      }
    } else if (command == "infer" || command == "i") {
      if (currentMode == MODE_IDLE) {
        serial_infer_request = true;
        Serial.println("\n>>> Inference mode requested via serial command");
      } else {
        Serial.println("Already in inference mode. Type 'exit' to leave.");
      }
    } else if (command == "exit") {
      if (currentMode == MODE_INFERENCE) {
        serial_exit_request = true;
        Serial.println("\n>>> Exiting inference mode requested via serial command");
      } else {
        Serial.println("Not in inference mode. Current mode: IDLE");
      }
    } else if (command == "sync" || command == "s") {

      syncTrainingToInference();

    } else if (command == "ble") {
      
      toggleBLESend();

    } else if (command == "help" || command == "?") {
      Serial.println("\n=== Available Serial Commands ===");
      Serial.println("  train / t        - Train 1 epoch (update model)");
      Serial.println("  infer / i        - Enter inference mode");
      Serial.println("  exit             - Exit inference mode");
      Serial.println("  sync / s         - Sync training weights to inference weights");
      Serial.println("  ble              - Enable/disable BLE data transmission on/off");
      Serial.println("  help / ?         - Show this help message");
      Serial.println("===================================\n");
    } else if (command.length() > 0) {
      Serial.print("Unknown command: '");
      Serial.print(command);
      Serial.println("'. Type 'help' for available commands.");
    }
  }
}

// ==================== Setup ====================

void setup() {
  Serial.begin(9600);
  
  #if defined(ARDUINO_ARDUINO_NANO33BLE) || defined(ARDUINO_NANO33BLE)
    delay(3000);
  #else
    delay(5000);
    while (!Serial);
  #endif
  
  Serial.println("=== Training and Inference Pipeline ===");

  // led initial setup
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);  // default off state at startup
  
  Serial.println("\n=== Initializing BLE ===");
  if (initBLE()) {
    Serial.println("BLE is ready. Use 'ble' command to enable/disable data transmission.");
  } else {
    Serial.println("WARNING: BLE initialization failed. BLE features will not work.");
  }
  // =========================================================
  
  initializeShield();
  
  // Assign audio buffer
  audio_buffer = (int16_t*)malloc(POST_TRIGGER_SAMPLES * sizeof(int16_t));
  audio_float = (float*)malloc(TOTAL_INFERENCE_SAMPLES * sizeof(float));
  
  if (!audio_buffer || !audio_float) {
    Serial.println("ERROR: Failed to allocate audio buffers!");
    while(1);
  }
  
  PDM.onReceive(onPDMdata);
  
  // Count weight size
  weights_bias_cnt = calcTotalWeightsBias();
  
  Serial.print("Total weights and bias: ");
  Serial.println(weights_bias_cnt);
  
  // =============== Decouple training buffer and infer buffer =================
  WeightBiasPtr_training = (DATA_TYPE*)calloc(weights_bias_cnt, sizeof(DATA_TYPE));
  WeightBiasPtr_inference = (DATA_TYPE*)calloc(weights_bias_cnt, sizeof(DATA_TYPE));
  
  if (!WeightBiasPtr_training || !WeightBiasPtr_inference) {
    Serial.println("ERROR: Failed to allocate weight buffers!");
    while(1);
  }
  
  // Specify fixed random seed **before setupNN**,
  // to ensure two devices have identical initial weights
  srand(0);
  
  // Create NN structure with randomly initialized weights
  setupNN(WeightBiasPtr_training);
  
  // Pack initial weights to training weight buffer
  packUnpackVector(PACK);
  
  // Copy training weight buffer to inference weight buffer
  memcpy(WeightBiasPtr_inference, WeightBiasPtr_training, weights_bias_cnt * sizeof(DATA_TYPE));
  
  Serial.println("Training and inference weights are decoupled into separate buffers.");
  Serial.print("Training buffer: ");
  Serial.print((unsigned long)WeightBiasPtr_training, HEX);
  Serial.print(", Inference buffer: ");
  Serial.println((unsigned long)WeightBiasPtr_inference, HEX);
  // =========================================================
  
  Serial.print("The accuracy before training: ");
  printAccuracy();
  
  initMFE();
  
  Serial.println("\n=== Ready ===");
  Serial.println("Serial command mode enabled");
  Serial.println("Type 'train' to train 1 epoch (update model)");
  Serial.println("Type 'infer' to enter inference mode");
  Serial.println("Type 'exit' to exit inference mode");
  Serial.println("Type 'ble' to toggle BLE data transmission on/off");
  Serial.println("Type 'help' for available commands");
  Serial.println("Note: Inference uses the latest trained model weights");
  Serial.println("Note: BLE will send weights when epoch is a multiple of 3 (if enabled)");
  Serial.println("=================================");
}

// ==================== Loop ====================

void loop() {

  processSerialCommand();

  BLE.poll();
  
  if (weightWriteChar.written()) {
    uint8_t buffer[512];
    int len = weightWriteChar.valueLength();
    if (len > 0 && len <= 512) {
      weightWriteChar.readValue(buffer, len);
      handleIncomingWeights(buffer, len);
    }
  }
  
  // Training request
  if (serial_train_request) {
    serial_train_request = false;
    
    Serial.println("\n========================================");
    Serial.println("=== Starting Training (1 Epoch) ===");
    Serial.println("========================================");

    currentMode = MODE_TRAINING;
    do_training();
    currentMode = MODE_IDLE;

    Serial.println("\n=== Ready for Next Operation ===");
    Serial.println("Type 'train' to train 1 epoch (update model)");
    Serial.println("Type 'infer' to enter inference mode");
    Serial.println("========================================\n");
  }
  
  // Inference request
  if (serial_infer_request) {
    serial_infer_request = false;
    
    Serial.println("\n=== Entering Inference Mode ===");
    Serial.println("Monitoring audio energy...");
    Serial.println("Type 'exit' to exit inference mode");

    currentMode = MODE_INFERENCE;
    
    // Switch to infer weights buffer and load them to network L
    WeightBiasPtr = WeightBiasPtr_inference;
    packUnpackVector(UNPACK);
    Serial.println("Inference weights loaded.");

    // Debug: print last 10 infer weights before entering infer mode
    printLastWeights("INFER BEFORE - inference buffer", WeightBiasPtr_inference, 10);
    
    // Start energy monitoring
    energy_monitoring = true;
    energy_samples_read = 0;
    pre_trigger_write_idx = 0;
    pre_trigger_ready = false;
    
    // Start PDM
    if (!PDM.begin(1, SAMPLE_RATE)) {
      Serial.println("ERROR: Failed to start PDM!");
      currentMode = MODE_IDLE;
      return;
    }
  }
  
  // Exiting inference model request
  if (serial_exit_request) {
    serial_exit_request = false;
    
    Serial.println("\n=== Exiting Inference Mode ===");
    PDM.end();
    energy_monitoring = false;
    currentMode = MODE_IDLE;
    
    // Clear BLE Write buffer
    if (weightWriteChar.written()) {
      uint8_t buffer[512];
      while (weightWriteChar.written()) {
        weightWriteChar.readValue(buffer, 512);
      }
      Serial.println("BLE Write buffer cleared.");
    }
    
    // Debug: print last 10 weights after inference end. Should be same to before entering.
    printLastWeights("INFER AFTER - inference buffer", WeightBiasPtr_inference, 10);
    Serial.println("Type 'train' to train 1 epoch (update model)");
    Serial.println("Type 'infer' to enter inference mode");
    Serial.println("========================================\n");
    return;
  }
  
  // Inference mode
  if (currentMode == MODE_INFERENCE) {
    // check threshold
    if (checkEnergyThreshold()) {
      #ifdef DEBUG_MODE
        Serial.println("\n>>> Energy threshold exceeded! Starting inference...");
      #endif
      
      // Audio recording
      if (!recordAudio()) {
        #ifdef DEBUG_MODE
          Serial.println("ERROR: Failed to record audio!");
        #endif
        delay(50);
        return;
      }
      
      // Extract MFE features
      #ifdef DEBUG_MODE
        Serial.println("Extracting MFE features...");
      #endif
      unsigned long t0 = millis();
      computeMFE(audio_float, mfe_features);
      #ifdef DEBUG_MODE
        Serial.print("MFE extraction took: ");
        Serial.print(millis() - t0);
        Serial.println("ms");
      #endif
      
      // Normalization
      #ifdef DEBUG_MODE
        Serial.println("Normalizing features...");
      #endif
      normalizeFeatures(mfe_features);
      
      // Inference
      #ifdef DEBUG_MODE
        Serial.println("Running inference...");
      #endif
      int predicted = inference(mfe_features);
      
      if (predicted >= 0) {
        printInferenceResult(predicted);
        // led control
        handleLedAfterInference(predicted);
      } else {
        #ifdef DEBUG_MODE
          Serial.println("ERROR: Inference failed!");
        #endif
      }
      
      // Resume energy monitoring
      energy_monitoring = true;
      energy_samples_read = 0;
      pre_trigger_write_idx = 0;
      pre_trigger_ready = false;
      
      delay(200);
    }
    
    delay(ENERGY_CHECK_INTERVAL_MS);
  } else {
    // idle
    delay(100);
  }
}

