#include "mfe.h"
#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FRAME_LEN_MS 25.0f
#define FRAME_HOP_MS 200.0f

#define FRAME_LEN_SAMPLES (int)(FRAME_LEN_MS * SAMPLE_RATE / 1000.0f)  // 400 samples
#define N_FFT 512                 // 512-point FFT (power of 2, fast)
#define HOP_LENGTH (int)(FRAME_HOP_MS * SAMPLE_RATE / 1000.0f)
#define TOTAL_INFERENCE_SAMPLES 8000   // 8000 samples = 0.5s @ 16kHz

static bool isInvalidNumber(float x) {
    return std::isnan(x) || std::isinf(x);
}

// ===== Hann & Mel =====
static float hann_window[FRAME_LEN_SAMPLES];  // 400-sample window
static float mel_filterbank[NUM_MELS * (N_FFT / 2 + 1)];  // 15 × 257
static bool mfe_initialized = false;

static void createHannWindow(float* window, int n) {
    for (int i = 0; i < n; i++) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (n - 1)));
    }
}

static float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// Create Mel filter bank (512-FFT, 257 freq bins)
static void createMelFilterbank(float* filterbank, int n_mels, int n_fft, int sr) {
    float fmax = sr / 2.0f;
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel(fmax);
    
    // Calculate mel points
    float mel_points[n_mels + 2];
    float hz_points[n_mels + 2];
    int fft_bins[n_mels + 2];
    int num_freq_bins = n_fft / 2 + 1;  // 257 bins for 512 FFT
    
    for (int i = 0; i < n_mels + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        hz_points[i] = mel_to_hz(mel_points[i]);
        fft_bins[i] = (int)(num_freq_bins * hz_points[i] / fmax);
        if (fft_bins[i] >= num_freq_bins) fft_bins[i] = num_freq_bins - 1;
    }
    
    // initialize filterbank
    for (int i = 0; i < n_mels * num_freq_bins; i++) {
        filterbank[i] = 0.0f;
    }
    
    // fill filterbank
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

// ==================== FFT512 Implementation (Radix-2, Fast) ====================

static void bitReverse(float* x, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            // Swap complex pair
            float temp_real = x[i * 2];
            float temp_imag = x[i * 2 + 1];
            x[i * 2] = x[j * 2];
            x[i * 2 + 1] = x[j * 2 + 1];
            x[j * 2] = temp_real;
            x[j * 2 + 1] = temp_imag;
        }
    }
}

// 512-point FFT
// Input: x (complex array)
// Output: modifies x in-place
static void fft512(float* x) {
    const int n = 512;
    
    // Bit reversal
    bitReverse(x, n);
    
    // FFT butterfly operation
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * M_PI / len;
        float wlen_real = std::cos(angle);
        float wlen_imag = std::sin(angle);
        
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
                
                // Twiddle factor multiplication: w * v
                float t_real = w_real * v_real - w_imag * v_imag;
                float t_imag = w_real * v_imag + w_imag * v_real;
                
                // Butterfly operation
                x[idx1] = u_real + t_real;
                x[idx1 + 1] = u_imag + t_imag;
                x[idx2] = u_real - t_real;
                x[idx2 + 1] = u_imag - t_imag;
                
                // Update twiddle factor: w = w * wlen
                float next_w_real = w_real * wlen_real - w_imag * wlen_imag;
                float next_w_imag = w_real * wlen_imag + w_imag * wlen_real;
                w_real = next_w_real;
                w_imag = next_w_imag;
            }
        }
    }
}

static void computeFFT512(float* frame, float* real, float* imag) {
    // Convert to complex format (interleaved) - optimized: use static buffer to avoid allocation
    static float fft_input[512 * 2];
    
    // Copy data and zero-pad (if frame has only 400 samples)
    for (int i = 0; i < 512; i++) {
        fft_input[i * 2] = (i < FRAME_LEN_SAMPLES) ? frame[i] : 0.0f;  // real part
        fft_input[i * 2 + 1] = 0.0f;  // imag part (initially 0)
    }
    
    // Perform FFT
    fft512(fft_input);
    
    // Extract first 257 frequency bins (0 to Nyquist)
    // For 512-point FFT, output 257 bins: 0, 1, 2, ..., 256
    // Corresponding frequencies: 0Hz, 31.25Hz, 62.5Hz, ..., 8000Hz
    for (int i = 0; i < 257; i++) {
        real[i] = fft_input[i * 2];
        imag[i] = fft_input[i * 2 + 1];
    }
}

// ===== Public API =====
void initMFE() {
    if (mfe_initialized) return;
    
    // Create window (400 samples, 25ms)
    createHannWindow(hann_window, FRAME_LEN_SAMPLES);
    
    // Create Mel filterbank (based on 512-point FFT, 257 frequency bins)
    createMelFilterbank(mel_filterbank, NUM_MELS, N_FFT, SAMPLE_RATE);
    
    mfe_initialized = true;
}

// Compute MFE features
// features: output MFE features (75-dim: 15 mels * 5 frames)
void computeMFE(float* audio, float* features) {
    // Initialize (executed once)
    initMFE();
    
    // Calculate number of frames (5 frames: 0, 0.2, 0.4, 0.6, 0.8s)
    int num_frames = 5;
    
    // Pre-allocate buffers (avoid allocation in loop)
    float frame[512];  // FFT512 needs 512 samples (first 400 are actual data, last 112 zero-padded)
    float real[257];   // 512-point FFT outputs 257 frequency bins
    float imag[257];
    float power[257];
    
    const int num_freq_bins = 257;  // Number of frequency bins for 512-point FFT
    
    // Use full 0.5s audio (TOTAL_INFERENCE_SAMPLES = 8000 samples)
    // 5 frame positions: evenly distributed within 0.5s, ensuring all frames are within audio range
    for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Frame position: evenly distributed from 0 to (8000-400), total 5 frames
        int start = (int)(frame_idx * (TOTAL_INFERENCE_SAMPLES - FRAME_LEN_SAMPLES) / (num_frames - 1.0f));
        
        // Extract frame and apply window (400 samples, 25ms)
        // Then zero-pad to 512 samples for FFT
        int i = 0;
        for (; i < FRAME_LEN_SAMPLES; i++) {
            int sample_idx = start + i;
            if (sample_idx < TOTAL_INFERENCE_SAMPLES) {
                frame[i] = audio[sample_idx] * hann_window[i];
            } else {
                frame[i] = 0.0f;  // Zero-pad outside boundary
            }
        }
        // Zero-pad to 512
        for (; i < 512; i++) {
            frame[i] = 0.0f;
        }
        
        // FFT512 - fast Radix-2 FFT, outputs 257 frequency bins
        computeFFT512(frame, real, imag);
        
        // Power spectrum: |fft_result|^2
        for (int i = 0; i < num_freq_bins; i++) {
            power[i] = real[i] * real[i] + imag[i] * imag[i];
        }

        // Mel energy
        for (int mel = 0; mel < NUM_MELS; mel++) {
            float e = 0.0f;
            int offset = mel * num_freq_bins;

            for (int k = 0; k < num_freq_bins; k++) {
                e += mel_filterbank[offset + k] * power[k];
            }

            if (e < 1e-10f) e = 1e-10f;

            features[frame_idx * NUM_MELS + mel] = std::log(e);
        }
    }
}
