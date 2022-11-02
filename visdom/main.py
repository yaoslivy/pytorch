
from visdom import Visdom
import numpy as np
import time







if __name__ == "__main__":
    # Create a visdom
    viz = Visdom() 
    viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))

    # Initialize some numbers randomly and then record them using add_scalar.
    for n_iter in range(10):
        loss = 0.2 * np.random.randn() + 1
        viz.line([loss], [n_iter], win='train_loss', update='append')
        time.sleep(0.5)


    # Create an image, and then record it using visdom.image.
    img = np.zeros((3, 100, 100))
    img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    viz.image(img)
