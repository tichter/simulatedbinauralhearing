This repository contains all the code I used for my master thesis (Thesis.pdf). All the graphics can be found in the ThesisGrpahics folder.

To run the experiments, simply clone the repository and start training with trainnetwork.py. All the evaluations can be plotted by running testnetwork.py.

Note that you need to specify the network configurations in the two files.
- Simple: Layers3 = False, Split = False, hidden_size = 2
- Ausili: Layers3 = False, Split = False, hidden_size = 20
- Early Fusion: Layers3 = True, Split = False, hidden_size_L1 = 40, hidden_size_L2 = 4
- Late Fusion: Layers3 = True, Split = True, hidden_size_L1 = 40, hidden_size_L2 = 4

Additionally, you can choose to construct the training data with all frontal HRTFs (add_ele = True) or only with HRTFs with zero elevation (add_ele = False).
The doublepolar parameter, indicates whether the azimuth labels are in navigational or double polar coordinates.

The git_empty.txt files only have the purpose of uploading empty folders to git, such that the folder structure for the generated plots is preserved. 
