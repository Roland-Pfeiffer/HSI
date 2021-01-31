#!/urs/bin/env python3

class Descriptor:
    def __init__(self, mat=None):
        self.material = mat

class TriDescriptor(Descriptor):
    def __init__(self, start, peak, stop, mat):
        super().__init__(mat)
        self.start = start
        self.peak = peak
        self.stop = stop

class Set:
    def __init__(self, desc: Descriptor):
        self.desc_list = [desc]
        self.material = desc.material

d = TriDescriptor(1, 2, 3, 'foobar')
s = Set(d)
print(s.material)