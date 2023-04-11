# Classifier Plans

# Train Classifier
- get cropped images of the turtles (painted/unpainted) from detector (+- some bounding box)
- create balanced dataset of cropped images for painted/unpainted turtles
- train classifier: for basic approach, can use ConvNet (see https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), or for SOTA, can use NFNets (https://github.com/benjs/nfnets_pytorch)
- during operation, can give it single image for classification, but better yet may be giving sequence of images (i.e. track), do classification on each image and then vote for classification over the sequence
- output: image/sequence is painted/unpainted turtle
