# Classification_from_Multi-Station
This repository implements a CNN-GNN hybrid neural network for seismic event classification. The model dynamically processes multi-station data without fixed station constraints, enhancing generalization and flexibility for real-world seismic monitoring.
# Model Overview
The model first uses a CNN to extract high-level features from each station's waveform. These features are then treated as nodes in a graph, where edges represent station pairs. A GNN processes this graph to capture spatial relationships and produce the final event classification.
# Trained Models
./model/model.pth
# Install conda and requirements
## Download repository
git clone https://github.com/ZHANG-EP/Classification_from_Multi-Station.git

cd Classification_from_Multi-Station
# Install to "myenv" virtual envirionment
conda create -n myenv python=3.12 -y

conda activate myenv
pip install -r requirements.txt
# Quick Inference Demo
#This script (Pred.py) loads the pre-trained model and runs inference on the sample Utah dataset located in example/test_data/.
cd example
python Pred.py
# License
This project is licensed under the MIT License - see the LICENSE file for details.
# Related paper
Hybrid CNN-GNN Framework for Discrimination Between Earthquakes and Anthropogenic Explosions from Multiple Stations. submitted (Jan 2026).
