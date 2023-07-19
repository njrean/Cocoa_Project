import numpy as np
from collections import OrderedDict

class Tracker():
    def __init__(self) -> None:
        self.nextID = 0
        self.objects = OrderedDict()

    def register(self, centroid):
        self.objects[self.nextID] = centroid
        self.nextID += 1

    def deregister(self, objectID):
        del self.objects[objectID]

    def update(self, centroidInput):
        if len(centroidInput) > self.nextID:
            self.objects[self.nextID] = centroidInput[0]