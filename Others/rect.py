#!/usr/bin/env python

class Rect:
    """Common class for both ground truth objects and hypothesis objects"""

    def __init__(self, box):
        """Constructor from dict with keys width, height, x, y, dco and id"""
        
        assert box["width"]  >= 0 # 0 allowed, since intersection of rects may create empty rect
        assert box["height"] >= 0 # 0 allowed, since intersection of rects may create empty rect

        self.x_ = box["x"]
        self.y_ = box["y"]
        self.w_ = box["width"]
        self.h_ = box["height"]
        
        
    
    def area(self):
        """Area of this instance"""
        return self.w_ * self.h_
    
    def intersect(self, o):
        """Create new Rect from intersection of self and o. Cave: id and dco will be lost."""
        box = {}
        box["x"] = max(self.x_, o.x_)
        box["y"] = max(self.y_, o.y_)
        box["width"] = max(0, min(self.x_ + self.w_, o.x_ + o.w_) - box["x"])
        box["height"] = max(0, min(self.y_ + self.h_, o.y_ + o.h_) - box["y"])
        box["id"] = "intersect"
        return Rect(box)

    def overlap(self, o):
        """Overlap of this and other Rect o"""
        ia = self.intersect(o).area()
        union = self.area() + o.area() - ia
        return float(ia) / union

        

