# import numpy as np
# from collections import OrderedDict

# class Tracker():
#     def __init__(self):
#         self.nextID = 0
#         self.objects = OrderedDict()
#         self.IDs = []
#         self.classes = []
#         self.centroids = []

#     def register(self, centroid):
#         self.objects[self.nextID] = centroid
#         self.nextID += 1

#     def deregister(self, objectID):
#         self.objects.pop(objectID)

#     def update(self, classes:list, centroidInput:list):
#         if len(centroidInput) == 0:
#             self.objects = OrderedDict()
#             self.IDs = []
#             self.classes = []
#             self.centroids = []

#         else:
#             centroidInput = centroidInput[::-1]
#             if len(self.objects.keys()) == 0:
#                 for i in range(len(centroidInput)):
#                     self.register(centroidInput[i])
                
#             else:
#                 #Find start ID
#                 for id in list(self.objects.keys()):
#                     if centroidInput[0][0] > self.objects[id][0]:
#                         ID = id
#                         break
#                     self.deregister(id)
#                 #Add new centroid
#                 for i in range(len(centroidInput)):
#                     if i<len(self.objects):
#                         self.objects[ID] = centroidInput[i]
#                         ID += 1
#                     else: self.register(centroidInput[i])

#             self.IDs = list(self.objects.keys())[::-1]
#             self.classes = classes
#             self.centroids = centroidInput


##
from collections import OrderedDict

class Tracker():
    def __init__(self):
        self.nextID = 0
        self.objects = OrderedDict()
        self.IDs = []

    def register(self, centroid):
        self.objects[self.nextID] = centroid
        self.nextID += 1

    def deregister(self, objectID):
        self.objects.pop(objectID)

    def update(self, centroidInput:list):
        if len(centroidInput) == 0:
            self.objects = OrderedDict()
            self.IDs = []

        else:
            centroidInput = centroidInput[::-1]
            if len(self.objects.keys()) == 0:
                for i in range(len(centroidInput)):
                    self.register(centroidInput[i])
                
            else:
                #Find start ID
                for id in list(self.objects.keys()):
                    if centroidInput[0][0] > self.objects[id][0]:
                        ID = id
                        break
                    self.deregister(id)
                #Add new centroid
                for i in range(len(centroidInput)):
                    if i<len(self.objects):
                        self.objects[ID] = centroidInput[i]
                        ID += 1
                    else: self.register(centroidInput[i])

            self.IDs = list(self.objects.keys())[::-1]

        # print(self.objects)
            