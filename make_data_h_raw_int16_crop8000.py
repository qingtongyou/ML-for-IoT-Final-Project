import os
import glob
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

# ======================= CONFIG ==========================
# user id: 1 and 2
# user 0 refers to combined version, only for performance test

# mode 0: only 'on' and 'off'
# mode 1: idle/on/off
# mode 2: unknown/on/off

# ======================= ENERGY DEBUG ==========================
ENERGY_DEBUG = False
# ===============================================================

USER = 2
MODE = 0

BASE_DIR = r""

if USER == 1:
    USER_DIR = "user1croped/"
elif USER == 2:
    USER_DIR = "user2croped/"
elif USER == 0:
    USER_DIR = "combined_wav_mcu"
else:
    raise ValueError(f"Unknown USER = {USER}")

TRAIN_DIR = os.path.join(BASE_DIR, USER_DIR, "training")
TEST_DIR  = os.path.join(BASE_DIR, USER_DIR, "test")

OUT_HEADER = f"arduino/data_user{USER}_mode{MODE}_raw_int16_cropped.h"

TARGET_LEN = 8000  # 500ms @ 16kHz

if MODE == 0:
    LABEL_MAP = {"on": 0, "off": 1}
elif MODE == 1:
    LABEL_MAP = {"idle": 0, "off": 1, "on": 2}
elif MODE == 2:
    LABEL_MAP = {"unknown": 0, "off": 1, "on": 2}
else:
    raise ValueError(f"Unknown MODE = {MODE}")

# =========================================================


def compute_energy_ms_int16(audio_int16):
    """
    audio_int16: np.ndarray, shape [8000], dtype int16
    return: mean square energy (float)
    """
    audio = audio_int16.astype(np.int32)
    return float(np.mean(audio * audio))


def extract_label(fname):
    prefix = fname.split(".")[0]

    IGNORE_MAP = {
        0: {"idle", "unknown"},
        1: {"unknown"},
        2: {"idle"},
    }

    if prefix in IGNORE_MAP.get(MODE, set()):
        print(f"⚠️ Ignore {prefix} document: {fname}")
        return None

    if prefix not in LABEL_MAP:
        raise ValueError(f"Unknown label prefix: {prefix} in {fname}")

    return LABEL_MAP[prefix]


def load_wav_exact(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != 16000:
        raise ValueError(f"{path}: sample rate must be 16k, got {sr}")

    if len(audio) != TARGET_LEN:
        raise ValueError(
            f"{path}: length must be EXACTLY 8000, got {len(audio)}"
        )

    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)

    return audio


def load_wav_dir(dir_path, tag):

    energy_per_class = {k: [] for k in LABEL_MAP.values()}

    paths = sorted(glob.glob(os.path.join(dir_path, "*.wav")))
    if not paths:
        raise RuntimeError(f"{tag}: Cannot find {dir_path}")

    X_list = []
    y_list = []

    print(f"\n=== proccessing {tag} ({len(paths)} documents) ===")

    skipped_idle = 0

    for p in paths:
        fname = os.path.basename(p)

        label = extract_label(fname)

        if label is None:
            skipped_idle += 1
            continue

        audio = load_wav_exact_16000(p)

        if ENERGY_DEBUG:
            energy = compute_energy_ms_int16(audio)
            energy_per_class[label].append(energy)
            print(f"[ENERGY] {tag} | {fname:40s} | label={label} | E_ms={energy:.2e}")

        X_list.append(audio)
        y_list.append(label)

    X = np.stack(X_list)
    y = np.array(y_list)

    print(f"✔ Complete {tag}: {X.shape[0]} valid samples")
    if skipped_idle > 0:
        print(f"⚠️ Ignore idle document count: {skipped_idle}")

    if ENERGY_DEBUG:
        print(f"\n--- {tag} Energy Mean (mean square) ---")
        inv_label_map = {v: k for k, v in LABEL_MAP.items()}
        for lbl, energies in energy_per_class.items():
            if len(energies) == 0:
                continue
            print(f"  {inv_label_map[lbl]:8s}: mean={np.mean(energies):.2e}, "
                  f"std={np.std(energies):.2e}, n={len(energies)}")

    return X, y


def write_data_h(train_X, train_y, val_X, val_y, test_X, test_y, out_path):
    train_N = train_X.shape[0]
    val_N   = val_X.shape[0]
    test_N  = test_X.shape[0]
    D = TARGET_LEN

    with open(out_path, "w") as f:
        f.write("// Automatically generated raw int16 data file\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"const int first_layer_input_cnt = {D};\n")
        f.write(f"const int train_data_cnt = {train_N};\n")
        f.write(f"const int validation_data_cnt = {val_N};\n")
        f.write(f"const int test_data_cnt = {test_N};\n")
        f.write(f"const int classes_cnt = {len(LABEL_MAP)};\n\n")

        # -------- labels --------
        def write_label_array(name, arr):
            f.write(f"const int {name}[{len(arr)}] = {{\n  ")
            f.write(", ".join(str(int(v)) for v in arr))
            f.write("\n};\n\n")

        write_label_array("train_labels", train_y)
        write_label_array("validation_labels", val_y)
        write_label_array("test_labels", test_y)

        # -------- data --------
        def write_data_array(name, arr):
            N = arr.shape[0]
            f.write(f"const int16_t {name}[{N}][{D}] = {{\n")
            for i in range(N):
                f.write("  {")
                f.write(", ".join(str(int(v)) for v in arr[i]))
                f.write("}")
                if i < N - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")

        write_data_array("train_data", train_X)
        write_data_array("validation_data", val_X)
        write_data_array("test_data", test_X)

    print(f"\n✅ data.h generated: {out_path}")


def print_label_distribution(y, tag):
    print(f"\n--- {tag} label distribution ---")
    unique, counts = np.unique(y, return_counts=True)

    inv_label_map = {v: k for k, v in LABEL_MAP.items()}

    total = len(y)
    for u, c in zip(unique, counts):
        name = inv_label_map[int(u)]
        ratio = 100.0 * c / total
        print(f"  {name:8s} ({int(u)}): {c:4d}  ({ratio:5.1f}%)")

    print(f"  Total: {total}\n")


def main():

    train_X_all, train_y_all = load_wav_dir(TRAIN_DIR, tag="TRAIN SET (ALL)")
    test_X, test_y = load_wav_dir(TEST_DIR, tag="TEST SET")


    print_label_distribution(train_y_all, "TRAIN (ALL, BEFORE SPLIT)")

    train_X, val_X, train_y, val_y = train_test_split(
        train_X_all,
        train_y_all,
        test_size=0.2,
        stratify=train_y_all,
        random_state=42
    )

    print_label_distribution(train_y, "TRAIN (FINAL)")

    print_label_distribution(val_y, "VALIDATION (FINAL)")

    print_label_distribution(test_y, "TEST (FINAL)")

    print("\n=== Total Samples ===")
    print("Train:", len(train_y))
    print("Val  :", len(val_y))
    print("Test :", len(test_y))

    os.makedirs(os.path.dirname(OUT_HEADER), exist_ok=True)
    write_data_h(train_X, train_y, val_X, val_y, test_X, test_y, OUT_HEADER)


if __name__ == "__main__":
    main()
