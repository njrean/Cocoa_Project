import numpy as np
import serial
import time
import heapq
from threading import Timer

from collections import defaultdict, OrderedDict

class Scheduler():
    def __init__(self, x_reference_point):
        super(Scheduler, self).__init__()
        # self.serial = serial.Serial('/dev/ttyACM1', 
        #                             baudrate=115200,
        #                             bytesize=8, 
        #                             parity="N", 
        #                             stopbits=1)
        self.data_send = 0b0
        self.id_class = defaultdict(list)
        self.id_xCentroid = OrderedDict()
        self.time_info = dict()
        self.heap_timestamp = []
        self.wait_beanID = 0 #bean ID which wait to set timestamp

        self.xpoint_ref = x_reference_point

        self.timeadd = 0.88 #delay time to reach stataion 0 from reference point
        self.timebt = 0.23 #delay time between station

        self.timestampInterupt = 0
        self.idInterupt = 0

        self.buffer_test = 0

        # self.timer = Timer(0, self.buffer_function)
        self.flag_timer = 0

    def collect(self, ids, classes, Centroids):
        for id, cl, cen in zip(ids, classes, Centroids):
            if id >= self.wait_beanID:
                self.id_class[id].append(cl)
                self.id_xCentroid[id] = cen[0]

    def update_time(self):
        #update if considered bean in position
        if  len(self.id_xCentroid) != 0:
            if self.id_xCentroid[self.wait_beanID] >= self.xpoint_ref:
                stack_classes = self.id_class[self.wait_beanID]
                bean_class = max(set(stack_classes), key=stack_classes.count)
                timestamp = time.time() + self.timeadd + (bean_class*self.timebt)
                
                heapq.heappush(self.heap_timestamp, timestamp)

                self.time_info[timestamp] = bean_class

                del self.id_class[self.wait_beanID]
                del self.id_xCentroid[self.wait_beanID]

                self.wait_beanID += 1

        #update timer if heaptimestamp is change
        if len(self.heap_timestamp) != 0 and self.timestampInterupt != self.heap_timestamp[0]:
            
            if self.flag_timer:
                self.timer.cancel()

            self.timer = Timer(self.heap_timestamp[0]-time.time(), self.interupt_Handeler)
            self.timer.start()
            self.timestampInterupt = self.heap_timestamp[0]
            self.idInterupt = self.time_info[self.timestampInterupt]
            self.flag_timer = 1

    def plan(self, ids, classes, xCentroids):
        self.collect( ids, classes, xCentroids)
        self.update_time()
            
    def calculate_data_send(self):
        self.data_send = int(0b1 << self.time_info[self.timestampInterupt])

    def send_UART(self):
        byte_number = self.data_send.bit_length() + 7 // 8
        text = self.data_send.to_bytes(byte_number, 'big').decode()
        self.serial.write(bytearray(text,'ascii'))
        time.sleep(1)

    def interupt_Handeler(self):
        heapq.heappop(self.heap_timestamp)

        self.calculate_data_send()

        print("Class {}".format(self.time_info[self.timestampInterupt]))
        self.send_UART()

        del self.time_info[self.timestampInterupt]

        self.flag_timer = 0
