# Smart Light: A Distributed On-device Voice-controlled LED Prototype

This code was developed as part of the EITP40 Final Project by Group 7. This prototype is able to identify 'on' and 'off' speech commands and give real-time response by automatically turning on/off the LED on an externally connected breadboard.

**Key features:**
- Fully on-device training and real-time inference on Arduino Nano 33 BLE Sense
- Audio feature extraction with MFE
- Lightweight MLP model
- BLE-based communication between two MCUs and PC aggregator


## Dataset
Voice samples from two users were recorded using the built-in microphone on the MCUs. Data collection was assisted by Edge Impulse for quality inspection convenience. All audio samples were cropped to a fixed duration of 500 ms, and temporally aligned to the onset of the speech signal. 

The exported `.wav` files were used only for offline inspection and preprocessing, then converted into int16 C arrays (`.h` format) for loading on the MCUs.


## Feature Extraction and Normalization

To reduce the computational load on the MCUs, 75-dimensional MFE features of the collected samples were computed offline on the PC and stored as `.h` files. These features were used for **on-device training, validation, and testing** on the MCUs.

During real-time inference, the same MFE feature representation is computed on the MCU using an identical C++ implementation, with consistent normalization applied.


## System Architecture

The system consists of two Arduino Nano 33 BLE Sense devices and a PC-based FL aggregator.

Each MCU performs local audio recording, feature extraction, on-device training, and real-time inference. Model updates are sent to the PC via BLE and aggregated there, which then distributes the updated global model back to the devices.


## Hardware Setup

- Two Arduino Nano 33 BLE Sense boards
- An external LED connected via a breadboard


## How to Run

1. Flash the [Arduino code](arduino/bp_pipeline_BLE_user1) to two Nano 33 BLE Sense boards
2. Wait for BLE initialization
3. Run the Python [BLE aggregator](ble_fl_loop_fedprox.py) on PC
4. Trigger local training via serial command ('t')
5. Aggregation and distribution are handled automatically






