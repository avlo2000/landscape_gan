from skimage.viewer import ImageViewer
from skimage import io


def show_two_images(img1, img2):
    io.imshow(img1)
    io.show()
    io.imshow(img2)
    io.show()