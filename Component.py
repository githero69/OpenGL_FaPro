import numpy as np


class Component:

    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height
        self.array = np.full((length, width), height)

    def getArrayAt(self, x, y):
        return self.array[x][y]

    def remove(self, x, y, z):
        offset = self.length / 2
        if x >= -35 and x <= 34 and y >= -35 and y <= 34:
            self.array[int(x + offset)][int(y + offset)] = z

    def removeTopRight(self, stepX, stepY, pointX, pointY, z):
        endX = stepX + pointX
        endY = stepY + pointY
        for i in range(stepX, endX + 1):
            self.remove(i, endY, z)

    def removeTopLeft(self, stepX, stepY, pointX, pointY, z):
        k = stepX
        endX = stepX + pointX
        endY = stepY + pointY
        for i in range(stepX, endX + 1):
            self.remove(k, endY, z)
            k = k - 1

    def removeBotRight(self, stepX, stepY, pointX, pointY, z):
        endX = stepX + pointX
        endY = stepY - pointY + 1
        for i in range(stepX, endX + 1):
            self.remove(i, endY, z)

    def removeBotLeft(self, stepX, stepY, pointX, pointY, z):
        k = stepX
        endX = stepX + pointX
        endY = stepY - pointY + 1
        for i in range(stepX, endX + 1):
            self.remove(k, endY, z)
            k = k - 1

    def remove_circle(self, stepX, stepY, pointX, pointY, z):
        self.removeTopRight(stepX, stepY, pointX, pointY, z)
        self.removeTopLeft(stepX, stepY, pointX, pointY, z)
        self.removeBotRight(stepX, stepY, pointX, pointY, z)
        self.removeBotLeft(stepX, stepY, pointX, pointY, z)

    #def removeCircleNew(self, stepX, stepY, posX, posY, height):
    #    for i in range(0, posX):
    #        for j in range(0, posY):
    #            self.array[stepX + i][stepY + j] = height
    #    for i in range(0, posX):
    #        for j in range(0, posY):
    #            self.array[stepX + i][stepY + j] = height
    #    for i in range(0, posX):
    #        for j in range(0, posY):
    #            self.array[stepX + i][stepY + j] = height
    #    for i in range(0, posX):
    #        for j in range(0, posY):
    #            self.array[stepX + i][stepY + j] = height

    def printArray(self):
        print(self.array)

