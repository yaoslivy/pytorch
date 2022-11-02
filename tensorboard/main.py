
from tensorboardX import SummaryWriter
import numpy as np





if __name__ == "__main__":
        
    # Create summaryWriter
    writer = SummaryWriter()
    
    # Initialize some numbers randomly and then record them using add_scalar.
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    
    # Create an image, and then record it using add_image.
    img = np.zeros((3, 100, 100))
    img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

    writer.add_image('my_image', img, 0)
    writer.close()
