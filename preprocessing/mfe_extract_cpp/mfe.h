// mfe.h
#pragma once

#define SAMPLE_RATE 16000
#define NUM_MELS 15
#define TARGET_FEATURE_DIM 75

// Normalize
#define MFE_AUTO_NORMALIZE 0

void computeMFE(float* audio, float* features);
