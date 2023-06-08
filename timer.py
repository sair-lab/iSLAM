import time

class Timer:
    def __init__(self):
        self.st = {}
        self.delta = {}

    def clear(self, name=None):
        if name is None:
            self.st = {}
            self.delta = {}
        elif isinstance(name, list) or isinstance(name, tuple):
            for x in name:
                self.st.pop(x, None)
                self.delta.pop(x, None)
        else:
            self.st.pop(name, None)
            self.delta.pop(name, None)

    def tic(self, name='default'):
        self.st[name] = time.perf_counter()
    
    def toc(self, name='default'):
        if name not in self.st:
            return -1
        t = time.perf_counter()
        dt = t - self.st[name]
        self.st[name] = t
        if name not in self.delta:
            self.delta[name] = [dt]
        else:
            self.delta[name].append(dt)
        return dt

    def avg(self, name='default'):
        if name not in self.delta:
            return -1
        return sum(self.delta[name]) / len(self.delta[name])

    def tot(self, name='default'):
        if name not in self.delta:
            return -1
        return sum(self.delta[name])

    def last(self, name='default'):
        if name not in self.delta:
            return -1
        return self.delta[name][-1]
        