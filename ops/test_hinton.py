import numpy as np
import matplotlib.pyplot as plt

def _blob(x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, colour, edgecolor=colour)

def hinton(W, maxweight=None, colors=['white','red']):
    """
    Draws a Hinton diagram for visualizing a weight matrix. 
    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.
    """
    reenable = False
    if plt.isinteractive():
        plt.ioff()
    
    plt.clf()
    height, width = W.shape
    if not maxweight:
        maxweight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))
        
    plt.fill(np.array([0, width, width, 0]), 
             np.array([0, 0, height, height]),
             'gray')
    
    plt.axis('off')
    plt.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y, x]
            if w > 0 and w < 0.5:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, w/maxweight),
                      'white')
            elif w > 0.8:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, w/maxweight),
                      colors[0])
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, w/maxweight)/4,
                      colors[1])
            elif w < 0:
                _blob(_x - 0.5,
                      height - _y + 0.5, 
                      min(1, -w/maxweight), 
                      'black')
    if reenable:
        plt.ion()
    
if __name__ == "__main__":
    hinton(np.random.randn(20, 10))
    plt.title('Example Hinton diagram - 20x20 random normal')
    plt.show()
    #hinton(np.random.randn(20, 20),colors=['red','white'])
    #plt.title('Example Hinton diagram - 20x20 random normal')
    #plt.show()