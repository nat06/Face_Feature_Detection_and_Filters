# INF573 - Project

*This GitHub repository is for the final project for Ecole Polytechnique's INF573 - Image Analysis and Computer Vision course.*

This project uses deep learning models and computer vision to detect face features in a webcam video and place AR objects on the face(s) seen. The filters are set off by pressing specific keys on your computer's keyboard.

To use the program, first run `conda env create -f environment.yml`. Once all the necessary packages have been installed, and the conda environment has been activated, run `python3 image-stream.py`. This will prompt a window with the webcam stream to launch. In order to enable a filter, click one of keys listed below.

List of keys:
* **c** : Displays 81 feature points on the faces found in the frame.
* **f** : Displays a coloured in version of the 81 feature points found on each face in the frame.
* **b** : Blurs and boxes the face of detected users.
* **e** : Detects and boxes the face and eyes of detected users.
* **p** : Applies a pig nose on all the faces found in the frame.
* **s** : Applies a septum piercing on all the faces found in the frame.
* **d** : Applies devil horns on all the heads found in the frame.
* **t** : Applies tears to all the faces found in the frame.
* ***spacebar***: Will remove the filter.
* **q** : Quits the program entirely. Can only be done when no filters are applied.

