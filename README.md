# Vision-A-deep-learning-approach-to-provide-walking-assistance-to-the-visually-impaired

This was my Capstone Project for my undergrad. This is a link to the paper: https://arxiv.org/abs/1911.08739

Visually impaired people face a lot of problems just in their day to day activities. Doing basic chores is a challenge in itself. In this project, we aim to provide a walking aid by making them aware of their surroundings through the help of audio devices such as headphones. We have identified 3 main tasks in this problem:
1. Object Detection
2. Monocular Depth Estimation
3. Text to Speech

First, we need to correctly detect and classify the objects around them. We have narrowed our scope to indoor objects for this project. We have used the YOLOv3 algorithm for object detection. We have used a pretrained Tensorflow model for this task.

Secondly, we need to inform the user only about objects that are close to the user, i.e. the objects that concern the user. A table that is 30 ft away does not immediately concern the user, and hence it's pointless to inform the user about that table. For this reason, we perform depth estimation in order to only inform the user about the objects that are close-by. While estimating depth using stereo vision is a comparatively easy task which uses simple geometry, the setup is very tedious. For that, we require 2 calibrated cameras and we need to mount them with caution. We also need to know the camera intrinsic parameters in order to estimate depth. This makes the overall setup tedious to use in a real use case. This is why we resort to monocular depth estimation. While this is an ill-posed problem, it is possible to achieve a decent depth map using deep learning based approaches. The main advantage of this approach is that it requires only one camera, which can even be an ordinary cell phone camera.

Lastly, whatever nearby objects that have been detected by the first 2 parts need to be delivered to the user in the form of an audio. Thus, we first construct a sentence of the form "There is a chair to your left", and then pass it to GTTS (Google Text to Speech) API. This audio is played to user through headphones.

### How to run the code:
Run main.py and it will call the all the necessary portions and play the audio.
