#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#include "data_user2_2class.h"   // raw int16
#include "mfe.h"    // MFE interface


// ======================== params ==========================
#define AUDIO_LEN 8000   // 8000 samples = 500ms @ 16kHz
#define FEATURE_DIM 75
// =======================================================

// raw int16 -> float
static float audio_float[AUDIO_LEN];
static float mfe_features[FEATURE_DIM];

static float training_mean[FEATURE_DIM];
static float training_std[FEATURE_DIM];


void extract_one_sample(const int16_t* src) {
    for (int i = 0; i < AUDIO_LEN; i++) {
        audio_float[i] = src[i] / 32768.0f;
    }
    computeMFE(audio_float, mfe_features);
}

void write_feature_array(
    FILE* f,
    const char* name,
    const float* data,
    int N,
    int D
) {
    fprintf(f, "const float %s[%d][%d] = {\n", name, N, D);
    for (int i = 0; i < N; i++) {
        fprintf(f, "  {");
        for (int j = 0; j < D; j++) {
            fprintf(f, "%.8f", data[i * D + j]);
            if (j < D - 1) fprintf(f, ", ");
        }
        fprintf(f, "}");
        if (i < N - 1) fprintf(f, ",");
        fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
}

void write_normalization_header(
    const char* filename,
    const float* mean,
    const float* std,
    int D
) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Failed to create %s\n", filename);
        return;
    }

    fprintf(f, "// Auto-generated normalization parameters (Z-score)\n\n");
    fprintf(f, "#ifndef NORMALIZATION_PARAMS_H\n");
    fprintf(f, "#define NORMALIZATION_PARAMS_H\n\n");

    fprintf(f, "const int feature_dim = %d;\n\n", D);

    fprintf(f, "const float training_mean[%d] = {\n  ", D);
    for (int i = 0; i < D; i++) {
        fprintf(f, "%.8f", mean[i]);
        if (i < D - 1) fprintf(f, ", ");
    }
    fprintf(f, "\n};\n\n");

    fprintf(f, "const float training_std[%d] = {\n  ", D);
    for (int i = 0; i < D; i++) {
        fprintf(f, "%.8f", std[i]);
        if (i < D - 1) fprintf(f, ", ");
    }
    fprintf(f, "\n};\n\n");

    fprintf(f, "#endif // NORMALIZATION_PARAMS_H\n");
    fclose(f);

    printf("normalization_params_user0_mode2.h generated!\n");
}

void normalize_dataset(
    float* data,
    int N,
    int D,
    const float* mean,
    const float* std
) {
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++) {
            data[i * D + d] =
                (data[i * D + d] - mean[d]) / std[d];
        }
    }
}


int main() {
    printf("=== Generating mfe_data.h from raw int16 data ===\n");

    static float* train_feat = new float[train_data_cnt * FEATURE_DIM];
    static float* val_feat   = new float[validation_data_cnt * FEATURE_DIM];
    static float* test_feat  = new float[test_data_cnt * FEATURE_DIM];

    // -------- TRAIN --------
    printf("Extracting TRAIN features...\n");
    for (int i = 0; i < train_data_cnt; i++) {
        extract_one_sample(train_data[i]);
        for (int j = 0; j < FEATURE_DIM; j++) {
            train_feat[i * FEATURE_DIM + j] = mfe_features[j];
        }
    }

    // ==================  compute mean / std  ==================
    printf("Computing training mean/std...\n");

    for (int d = 0; d < FEATURE_DIM; d++) {
        training_mean[d] = 0.0f;
        training_std[d] = 0.0f;
    }

    // mean
    for (int i = 0; i < train_data_cnt; i++) {
        for (int d = 0; d < FEATURE_DIM; d++) {
            training_mean[d] += train_feat[i * FEATURE_DIM + d];
        }
    }
    for (int d = 0; d < FEATURE_DIM; d++) {
        training_mean[d] /= train_data_cnt;
    }

    // std
    for (int i = 0; i < train_data_cnt; i++) {
        for (int d = 0; d < FEATURE_DIM; d++) {
            float diff = train_feat[i * FEATURE_DIM + d] - training_mean[d];
            training_std[d] += diff * diff;
        }
    }
    for (int d = 0; d < FEATURE_DIM; d++) {
        training_std[d] = std::sqrt(training_std[d] / train_data_cnt);
        if (training_std[d] < 1e-8f) training_std[d] = 1.0f;
    }


    // -------- VAL --------
    printf("Extracting VAL features...\n");
    for (int i = 0; i < validation_data_cnt; i++) {
        extract_one_sample(validation_data[i]);
        for (int j = 0; j < FEATURE_DIM; j++) {
            val_feat[i * FEATURE_DIM + j] = mfe_features[j];
        }
    }

    // -------- TEST --------
    printf("Extracting TEST features...\n");
    for (int i = 0; i < test_data_cnt; i++) {
        extract_one_sample(test_data[i]);
        for (int j = 0; j < FEATURE_DIM; j++) {
            test_feat[i * FEATURE_DIM + j] = mfe_features[j];
        }
    }

    // ================== normalize all datasets ==================
    printf("Normalizing TRAIN / VAL / TEST with computed mean/std...\n");

    normalize_dataset(train_feat, train_data_cnt, FEATURE_DIM,
                      training_mean, training_std);

    normalize_dataset(val_feat, validation_data_cnt, FEATURE_DIM,
                      training_mean, training_std);

    normalize_dataset(test_feat, test_data_cnt, FEATURE_DIM,
                      training_mean, training_std);


    // ================== generate mfe_data.h ==================

    write_normalization_header(
    "normalization_params_user0_mode2.h",
    training_mean,
    training_std,
    FEATURE_DIM
);

    FILE* f = fopen("mfe_data_normalized.h", "w");
    if (!f) {
        printf("Failed to create mfe_data_normalized.h\n");
        return -1;
    }

    fprintf(f, "// Automatically generated MFE feature data\n\n");
    fprintf(f, "#ifndef MFE_DATA_H\n#define MFE_DATA_H\n\n");

    fprintf(f, "const int first_layer_input_cnt = %d;\n", FEATURE_DIM);
    fprintf(f, "const int train_data_cnt = %d;\n", train_data_cnt);
    fprintf(f, "const int validation_data_cnt = %d;\n", validation_data_cnt);
    fprintf(f, "const int test_data_cnt = %d;\n", test_data_cnt);
    fprintf(f, "const int classes_cnt = %d;\n\n", classes_cnt);

    // -------- labels --------
    fprintf(f, "const int train_labels[%d] = {", train_data_cnt);
    for (int i = 0; i < train_data_cnt; i++) {
        fprintf(f, "%d", train_labels[i]);
        if (i < train_data_cnt - 1) fprintf(f, ", ");
    }
    fprintf(f, "};\n\n");

    fprintf(f, "const int validation_labels[%d] = {", validation_data_cnt);
    for (int i = 0; i < validation_data_cnt; i++) {
        fprintf(f, "%d", validation_labels[i]);
        if (i < validation_data_cnt - 1) fprintf(f, ", ");
    }
    fprintf(f, "};\n\n");

    fprintf(f, "const int test_labels[%d] = {", test_data_cnt);
    for (int i = 0; i < test_data_cnt; i++) {
        fprintf(f, "%d", test_labels[i]);
        if (i < test_data_cnt - 1) fprintf(f, ", ");
    }
    fprintf(f, "};\n\n");

    // -------- features --------
    write_feature_array(f, "train_data", train_feat, train_data_cnt, FEATURE_DIM);
    write_feature_array(f, "validation_data", val_feat, validation_data_cnt, FEATURE_DIM);
    write_feature_array(f, "test_data", test_feat, test_data_cnt, FEATURE_DIM);

    fprintf(f, "#endif // MFE_DATA_H\n");
    fclose(f);

    delete[] train_feat;
    delete[] val_feat;
    delete[] test_feat;

    printf("mfe_data.h generated successfully!\n");
    return 0;

}
