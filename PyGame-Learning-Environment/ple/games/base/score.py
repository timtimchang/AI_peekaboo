def score(color,time):
    import math
    ret = 0.0
    if color > 2:
        return
    if color == 0:
        ret = 5*math.cos(time)
    elif color == 1:
        sign = 1.0
        tmp = math.cos(time)+math.sin(time)
        if tmp < 0 :
            sign = -1.0
        ret = 5*sign*(tmp**2)/2
    elif color == 2:
        ret = 5*math.sin(time)
    return int(ret)