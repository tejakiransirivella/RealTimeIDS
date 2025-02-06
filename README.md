# **Real Time IDS**

Real Time IDS is an Intrusion Detection System designed to analyze network flow in real time and identify potential threats. By leveraging a machine learning-based detection layer, it provides accurate insights into network activity, allowing users to make informed security decisions. This system enhances network protection by detecting anomalies and giving users control over the next steps.

## **Highlights**
- Detects different threats like **DoS, Portscan, and Bot** from network packets.
- Simulates a **real-time IDS** using **pcap files** that may contain malicious packets.
- The entire **machine learning layer** is built from scratch using **C++ and CUDA** for high-performance detection.
- Utilizes **PyFlowmeter** to convert raw packets into flows and extract key network features.
- Provides detailed **IP address and host information** for potential threats.

## **Installation**

### **Prerequisites**
Ensure you have the following installed on your system:
- **Python 3**
- **G++** (GNU Compiler Collection)
- **CUDA Toolkit** (for GPU acceleration) – Install it using:
  \`\`\`bash
  sudo apt install nvidia-cuda-toolkit
  \`\`\`

### **Setting Up the Environment**
It is recommended to use a virtual environment for Python dependencies.

### **Running the Project**
Navigate to the `scripts/` directory and execute the `run.sh` script, providing only the name of the pcap file (which should be placed in the `data/` folder):

```bash
cd scripts
./run.sh <pcap_filename>
```
This script will automatically install all necessary Python dependencies and start the project.
