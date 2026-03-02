"# Visually Impaired Helper" 
# 🕶️ Visualy Impaired Helper

An AI-powered assistive system designed for Raspberry Pi and NVIDIA Jetson. It uses computer vision to narrate the world to visually impaired users using a "Command/Order" style.

## Features
- **Object Detection:** Uses YOLO11s to identify obstacles (cars, persons, dogs).
- **OCR Integration:** Reads text on signs (STOP, YIELD) and road markings (School Zone).
- **Smart Narration:** Combines objects, distance, and text into natural "Order-style" commands.
- **Hardware Agnostic:** Supports standard USB cameras, Sony IMX500 (AI Camera), and Jetson Nano/Orin.
- **Button Control:** Integrated support for 3-button assistive glasses.

## 🛠️ Hardware Requirements
- **Raspberry Pi 4/5** and **NVIDIA Jetson**.
- **Sony IMX500 AI Camera** 
- **Audio Output:** Bluetooth headphones or 3.5mm jack.

## 📦 Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/molkachouk/visualy-impared-helper.git](https://github.com/molkachouk/visualy-impared-helper.git)
   cd visualy-impared-helper