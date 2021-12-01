import sys
sys.path.append("..")

from helpers import *
from models import *

class Edgelet:
    def __init__(self, c1, c2):
        self.coords = [c1, c2]
        self.line = np.cross((c1[0], c1[1], 1), (c2[0], c2[1], 1)) # Homogeneous coordinate
        self.line = self.line / np.linalg.norm(self.line[0:2])
        self.center = np.array([(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2])
        self.direction = np.array([c2[0] - c1[0], c2[1] - c1[1]])
        self.direction = self.direction / np.linalg.norm(self.direction)
        self.strength = np.linalg.norm(np.array(c2) - np.array(c1))
    
    def drawLine(self, image, color):
        cv2.line(image, (self.coords[0][0], self.coords[0][1]), (self.coords[1][0], self.coords[1][1]), color, 2)

    def getModel(self, others):
        model = np.cross(self.line, others.line)
        model = model / (np.max(np.abs(model)) + 1e-5) # scaling
        return model

    def applyModel(self, model, threshold):
        vec = (model[0:2] - np.array(self.center) * model[2])
        vec = vec / (np.linalg.norm(vec) + 1e-5)
        angle = np.arccos(min(np.abs(np.dot(vec, self.direction)), 1))
        if angle < threshold * np.pi / 180:
            return self.strength 
        return 0
    
    def getLine(self):
        return self.line

def generateBaseModel(edgelets, candidateSize=100):
    # random sampling two edges to find out vanishing point, RANSAC
    bestModel = None
    bestModelIndex = 0
    for i in range(candidateSize):
        value = np.random.choice(len(edgelets), 2, replace=False)
        model = edgelets[value[0]].getModel(edgelets[value[1]])
        modelIndex = 0
        for j in range(len(edgelets)):
            modelIndex += edgelets[j].applyModel(model, 5)
        if bestModelIndex < modelIndex:
            bestModelIndex = modelIndex
            bestModel = model
    return bestModel

def refineModel(model, edgelets):
    # return refined model(=vanishing point) and remaining edgelets
    nextEdgelets = []
    lines = []
    for edgelet in edgelets:
        if edgelet.applyModel(model, 5) > 0:
            lines.append(edgelet.getLine())
        else:
            nextEdgelets.append(edgelet)
    lines = np.array(lines)
    _, _, vt = np.linalg.svd(lines)
    vanish = vt[-1, :]
    return vanish, nextEdgelets

def rectify(src, sigma=11, cannyLow=20, cannyHigh=50, voteThreshold=100, minLineLength=30, maxLineGap=3):
    """
    Args:
        src : source image (ndarray)
        sigma, cannyLow, cannyHigh : parameters for Canny edge detector
        voteThreshold, minLineLength, maxLineGap : parameters for Hough line detector
    Returns:
        rectified image
    """
    ratio = src.shape[0] / 1000
    srcResized = cv2.resize(src, (int(src.shape[1] / ratio), 1000))
    bsrc = cv2.GaussianBlur(srcResized, (sigma, sigma), 0)
    dst = cv2.Canny(bsrc, cannyLow, cannyHigh, None, 3)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, voteThreshold, None, minLineLength, maxLineGap)
    assert(len(linesP) > 4)

    size = len(linesP)
    edgelets = []
    for i in range(size):
        l = linesP[i][0]
        edgelet = Edgelet((l[0], l[1]), (l[2], l[3]))
        edgelets.append(edgelet)

    model = generateBaseModel(edgelets)
    vanish1, edgelets = refineModel(model, edgelets)
    model = generateBaseModel(edgelets)
    vanish2, edgelets = refineModel(model, edgelets)

    # Find horizontal / vertical vanishing point
    # vanish1 : vertical, vanish2 : horizontal 
    imageCenter = np.array([srcResized.shape[1] - 1, srcResized.shape[0] - 1]) / 2
    disp1 = vanish1[0:2] - imageCenter * vanish1[2]
    disp2 = vanish2[0:2] - imageCenter * vanish2[2]
    hor1 = np.arctan2(disp1[1], disp1[0])
    hor2 = np.arctan2(disp2[1], disp2[0])
    hor1 = min(min(abs(hor1), abs(np.pi - hor1)), abs(hor1 + np.pi))
    hor2 = min(min(abs(hor2), abs(np.pi - hor2)), abs(hor2 + np.pi))
    if hor1 < hor2:
        vanish1, vanish2 = vanish2, vanish1

    # Set homography matrix 
    H = np.identity(3)
    H[2] = np.cross(vanish1, vanish2)
    H[0, 0], H[0, 1] = vanish1[1], -vanish1[0]
    H[1, 0], H[1, 1] = vanish2[1], -vanish2[0]

    # Proper scaling (maintaining image ratio in center)
    imageCenter = np.append(imageCenter, 1)
    imageCenterTrans = H@imageCenter
    deltaCenter = (H[0:2, 0:2] * imageCenterTrans[2] - H[2, 0:2] * imageCenterTrans[0:2]) / imageCenterTrans[2] / imageCenterTrans[2]
    H[0] *= 1 / np.sign(deltaCenter[0, 0]) / np.linalg.norm(deltaCenter[0])
    H[1] *= 1 / np.sign(deltaCenter[1, 1]) / np.linalg.norm(deltaCenter[1])
    H = np.diag([ratio, ratio, 1])@H@np.diag([1 / ratio, 1 / ratio, 1])

    # Translate & crop to content
    corner = (np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]) * np.array([src.shape[1] - 1, src.shape[0] - 1, 1])).T
    cornerTransformed = np.matmul(H, corner)
    cornerTransformed = cornerTransformed[0:2] / cornerTransformed[2]
    xMin = np.max(cornerTransformed[0, (0, 3)])
    xMax = np.min(cornerTransformed[0, (1, 2)])
    yMin = np.max(cornerTransformed[1, (0, 1)])
    yMax = np.min(cornerTransformed[1, (2, 3)])
    H = np.matmul(np.array([[1, 0, -xMin], [0, 1, -yMin], [0, 0, 1]]), H)

    dstShape = (int(xMax - xMin), int(yMax - yMin))
    dst = cv2.warpPerspective(src, H, dstShape)
    
    return dst

def generate_row_image(src: Source) -> list:
    """
    Args:
        src : Source to split by rows
    Returns:
        list of RowImage
    """
    
    dst = src.img

    # Find horizontal edges
    scaledDst = cv2.resize(dst, (1000, int(dst.shape[0] / dst.shape[1] * 1000)))
    bDst = cv2.GaussianBlur(scaledDst, (5, 5), 0)
    edgeMask = cv2.Canny(bDst, 50, 70)
    kernel = np.zeros((5, 5), dtype=np.uint8)
    kernel[2, :] = 1
    edgeMaskEroded = cv2.erode(edgeMask, kernel, iterations=1)
    lines = cv2.HoughLines(edgeMaskEroded, 1, np.pi / 180, 150)

    
    '''
    # Debugging purpose
    scaledDst2 = scaledDst.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(scaledDst2, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    show_image(scaledDst2)
    '''

    # Grouping edges
    heights = []
    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            if abs(theta - np.pi / 2) < 0.1:
                height = (rho - (scaledDst.shape[1] - 1) / 2 * np.cos(theta)) / np.sin(theta)
                if height >= 0 and height < scaledDst.shape[0] - 1:
                    heights.append(height)

    heights.sort()
    gapRange = []
    gapRange.append((0, 0))
    if len(heights) > 0:
        gapStart = heights[0]
        for i in range(1, len(heights)):
            if heights[i] - heights[i-1] > 192:
                gapRange.append((gapStart, heights[i-1]))
                gapStart = heights[i]
        gapRange.append((gapStart, heights[-1]))
    gapRange.append((scaledDst.shape[0] - 1, scaledDst.shape[0] - 1))
    
    RowImages = []
    relativeFloor = 0
    for i in range(len(gapRange) - 1):
        if gapRange[i+1][1] - gapRange[i][0] > 192:
            y1 = int(gapRange[i][0] * dst.shape[0] / scaledDst.shape[0])
            y2 = int(gapRange[i+1][1] * dst.shape[0] / scaledDst.shape[0])
            RowImages.append(RowImage(dst[y1:y2, :], src, relativeFloor))
            relativeFloor += 1

    return RowImages

if __name__== '__main__':
    filepath = sys.argv[1]
    img = read_image(filepath)
    imgRect = rectify(img)
    src = Source(imgRect, filepath)
    rowImages = generate_row_image(src)

    for rowImage in rowImages:
        show_image(rowImage.img)

