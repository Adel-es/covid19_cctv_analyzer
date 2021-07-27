import time; 

class FPSCalc : 
    def __init__(self) : 
        self.startTime = 0; 
        self.endTime = 0; 
    
    def start(self): 
        self.startTime = time.perf_counter() 

    def end(self) -> int : 
        self.endTime = time.perf_counter()
        interval = self.endTime - self.startTime
        fps = int(1.0 / interval)
        return fps; 